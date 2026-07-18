from __future__ import annotations
import json
import uuid
from pathlib import Path
import typer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from src.utils.pdf_extractor import extract_pdf
from src.graph.pipeline import run_pipeline
from src.output.report_generator import generate_report
from src.evaluation.hallucination_checker import verify_evidence_chain, hallucination_rate
from src.evaluation.calibration_metrics import calibration_report
from src.evaluation.consistency_runner import run_consistency_experiment
from src.utils.telemetry import setup_telemetry, get_langfuse, get_tracer, shutdown
import config

app = typer.Typer()
console = Console()


def _load_cvs(cv_dir: Path) -> list[dict]:
    pdfs = sorted(cv_dir.glob("*.pdf"))
    if not pdfs:
        console.print(f"[red]No PDF files found in {cv_dir}[/red]")
        raise typer.Exit(1)
    results = []
    for pdf in pdfs:
        extracted = extract_pdf(str(pdf))
        if extracted["extraction_status"] == "low_confidence":
            console.print(f"[yellow]⚠ {pdf.name} — low confidence extraction, flagged[/yellow]")
        extracted["candidate_id"] = pdf.stem
        results.append(extracted)
    return results


@app.command()
def main(
    jd: Path = typer.Option(..., help="Path to job description text file"),
    cv_dir: Path = typer.Option(..., help="Directory of candidate CV PDFs"),
    output: Path = typer.Option(Path("results"), help="Output directory"),
    runs: int = typer.Option(1, help="Number of pipeline runs (use 3 for consistency test)"),
    eval: bool = typer.Option(False, help="Run full evaluation suite"),
    no_cache: bool = typer.Option(False, help="Disable extraction cache"),
    session_id: str | None = typer.Option(None, "--session-id", help="Group this run with others in Langfuse"),
):
    """EvidenceRank ATS — rank IT candidates against a job description."""
    setup_telemetry()

    if not jd.exists():
        console.print(f"[red]JD file not found: {jd}[/red]")
        raise typer.Exit(1)
    if not cv_dir.is_dir():
        console.print(f"[red]CV directory not found: {cv_dir}[/red]")
        raise typer.Exit(1)

    jd_text = jd.read_text(encoding="utf-8")
    cv_raws = _load_cvs(cv_dir)
    console.print(f"[green]Loaded {len(cv_raws)} CVs from {cv_dir}[/green]")

    run_id = str(uuid.uuid4())[:8]
    out_dir = output / f"run_{run_id}"
    lf = get_langfuse()

    try:
        if runs > 1:
            
            tracer = get_tracer()
            with tracer.start_as_current_span(
                "consistency_experiment",
                attributes={"n_runs": runs, "run.id": run_id},
            ):
                console.print(f"[blue]Running {runs}× consistency experiment...[/blue]")
                consistency = run_consistency_experiment(jd_text, cv_raws, n_runs=runs)
                console.print(f"Consistency mean τ: {consistency['mean_tau']:.3f}")

                # Log Kendall's τ on each run's Langfuse trace
                from itertools import combinations
                for i, (a, b) in enumerate(combinations(range(runs), 2)):
                    lf.create_score(
                        trace_id=consistency["otel_trace_ids"][a],
                        name=f"kendall_tau_vs_run{b}",
                        value=consistency["pairwise_taus"][i],
                    )
                lf.create_score(
                    trace_id=consistency["otel_trace_ids"][0],
                    name="mean_tau",
                    value=consistency["mean_tau"],
                    comment=f"session_id={consistency['session_id']}",
                )

                (out_dir / "evaluation").mkdir(parents=True, exist_ok=True)
                (out_dir / "evaluation" / "consistency_metrics.json").write_text(
                    json.dumps(consistency, indent=2), encoding="utf-8"
                )
            return

        phase_labels = {
            1: "[Phase 1] Parsing job description...",
            2: "[Phase 2] Extracting CV profiles...",
            3: "[Phase 3] Judging candidates...",
            4: "[Phase 4] Calibrating final ranking...",
        }

        with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console) as progress:
            task = progress.add_task("Running pipeline...", total=None)
            progress.update(task, description=phase_labels[1])

            def on_phase_complete(entry: dict) -> None:
                next_label = phase_labels.get(entry["phase"] + 1)
                if next_label:
                    progress.update(task, description=next_label)

            state = run_pipeline(
                jd_text, cv_raws, run_id=run_id, use_cache=not no_cache,
                session_id=session_id, on_phase_complete=on_phase_complete,
            )

        otel_trace_id = state.otel_trace_id

        ranking = state.final_ranking
        table = Table(title=f"Ranking — {state.jd_structured.role_title}")
        table.add_column("Rank", style="bold")
        table.add_column("Candidate")
        table.add_column("Name")
        table.add_column("Score")
        table.add_column("Δ Phase4")
        table.add_column("Confidence")
        table.add_column("Seniority")

        profile_map = {p.candidate_id: p for p in state.cv_profiles}
        assessment_map = {a.candidate_id: a for a in state.candidate_assessments}

        for rc in ranking.ranked_candidates:
            p = profile_map.get(rc.candidate_id)
            a = assessment_map.get(rc.candidate_id)
            table.add_row(
                str(rc.rank),
                rc.candidate_id,
                p.basic_info.full_name or "N/A" if p else "N/A",
                str(rc.calibrated_score),
                f"{rc.delta_from_raw:+.1f}",
                a.confidence if a else "N/A",
                a.seniority_alignment if a else "N/A",
            )
        console.print(table)

        cv_text_map = {c["candidate_id"]: c["raw_text"] for c in cv_raws}

        if eval:
            console.print("[blue]Running evaluation suite...[/blue]")
            all_flags = []
            for a in state.candidate_assessments:
                flags = verify_evidence_chain(a, cv_text_map.get(a.candidate_id, ""))
                all_flags.extend(flags)
            state.hallucination_flags = all_flags
            h_rate = hallucination_rate(all_flags)
            console.print(f"Hallucination rate: {h_rate:.1%}")

            cal = calibration_report(state.candidate_assessments, ranking)
            console.print(f"Calibration — raw std: {cal['raw_std']:.1f}, calibrated std: {cal['calibrated_std']:.1f}")
            console.print(f"Mean Phase 4 delta: {cal['mean_abs_delta']:.1f}, rank changes: {cal['rank_changes']}")

            # Log evaluation metrics as Langfuse scores on this run's trace
            fabricated = sum(1 for f in all_flags if f.status == "fabricated")
            lf.create_score(
                trace_id=otel_trace_id,
                name="hallucination_rate",
                value=h_rate,
                comment=f"{fabricated} fabricated / {len(all_flags)} total",
            )
            lf.create_score(trace_id=otel_trace_id, name="calibration_raw_std",       value=cal["raw_std"])
            lf.create_score(trace_id=otel_trace_id, name="calibration_calibrated_std", value=cal["calibrated_std"])
            lf.create_score(trace_id=otel_trace_id, name="calibration_mean_abs_delta", value=cal["mean_abs_delta"])
            lf.create_score(trace_id=otel_trace_id, name="rank_changes",               value=float(cal["rank_changes"]))

            (out_dir / "evaluation").mkdir(parents=True, exist_ok=True)
            (out_dir / "evaluation" / "calibration_report.json").write_text(
                json.dumps(cal, indent=2), encoding="utf-8"
            )

        generate_report(state, out_dir)
        console.print(f"\n[green]Report written to {out_dir}/report.md[/green]")


    finally:
        shutdown()  # explicit flush — atexit is a backup, not guaranteed in CLI


if __name__ == "__main__":
    app()
