# Remove Baselines + Ablation Study Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Strip all baseline comparison and ablation study code from EvidenceRank, reverting ablation-gated pipeline behavior to always-on.

**Architecture:** Pure deletion + simplification. Nine files are deleted entirely. Four files are edited to remove conditional branches that were only needed for ablation variants — the always-on path is kept as-is. No new code is introduced.

**Tech Stack:** Python, LangChain, Typer, Pydantic v2, pytest (uv run pytest)

---

## File Map

| File | Action |
|---|---|
| `src/evaluation/baselines.py` | Delete |
| `src/evaluation/ablation.py` | Delete |
| `src/prompts/judge_no_grounding.py` | Delete |
| `tests/test_baselines.py` | Delete |
| `tests/test_ablation.py` | Delete |
| `tests/test_pipeline_ablation.py` | Delete |
| `tests/test_candidate_judge.py` | Delete |
| `tests/test_main_flags.py` | Delete |
| `docs/superpowers/specs/2026-07-17-evaluation-baselines-ablation-design.md` | Delete |
| `docs/superpowers/plans/2026-07-17-evaluation-baselines-ablation.md` | Delete |
| `src/agents/candidate_judge.py` | Modify — remove `use_evidence_grounding` param, always use grounding prompt |
| `src/graph/nodes.py` | Modify — remove `use_evidence_grounding` param from `phase3_candidate_judge` |
| `src/graph/pipeline.py` | Modify — remove 4 ablation flags + `_ranking_from_raw`, simplify to always-on |
| `main.py` | Modify — remove imports, CLI flags, and execution blocks for baselines + ablation |

---

### Task 1: Delete all pure ablation/baseline files

**Files:**
- Delete: `src/evaluation/baselines.py`
- Delete: `src/evaluation/ablation.py`
- Delete: `src/prompts/judge_no_grounding.py`
- Delete: `tests/test_baselines.py`
- Delete: `tests/test_ablation.py`
- Delete: `tests/test_pipeline_ablation.py`
- Delete: `tests/test_candidate_judge.py`
- Delete: `tests/test_main_flags.py`
- Delete: `docs/superpowers/specs/2026-07-17-evaluation-baselines-ablation-design.md`
- Delete: `docs/superpowers/plans/2026-07-17-evaluation-baselines-ablation.md`

- [ ] **Step 1: Delete the files**

```bash
git rm src/evaluation/baselines.py \
       src/evaluation/ablation.py \
       src/prompts/judge_no_grounding.py \
       tests/test_baselines.py \
       tests/test_ablation.py \
       tests/test_pipeline_ablation.py \
       tests/test_candidate_judge.py \
       tests/test_main_flags.py \
       "docs/superpowers/specs/2026-07-17-evaluation-baselines-ablation-design.md" \
       "docs/superpowers/plans/2026-07-17-evaluation-baselines-ablation.md"
```

- [ ] **Step 2: Commit**

```bash
git commit -m "chore: delete baselines, ablation, and their test files"
```

---

### Task 2: Simplify `CandidateJudgeAgent`

**Files:**
- Modify: `src/agents/candidate_judge.py`

Remove `use_evidence_grounding`, the conditional import, and the conditional prompt selection. Always use the grounding prompt.

- [ ] **Step 1: Replace the file content**

Replace `src/agents/candidate_judge.py` with:

```python
from __future__ import annotations
from langchain_core.messages import SystemMessage, HumanMessage
from src.models.schemas import CandidateAssessment, CandidateProfile, JDRequirements
from src.utils.llm import get_llm, invoke_with_telemetry
from src.prompts import judge as prompts
import config


class CandidateJudgeAgent:
    def __init__(self):
        self._llm = get_llm(config.LARGE_MODEL, CandidateAssessment, config.JUDGE_TEMPERATURE)

    def run(
        self,
        profile: CandidateProfile,
        jd: JDRequirements,
        context_chunks: list[str] | None = None,
        skill_matches: list | None = None,
    ) -> CandidateAssessment:
        return invoke_with_telemetry(
            self._llm,
            [SystemMessage(content=prompts.SYSTEM),
             HumanMessage(content=prompts.human(
                 jd.model_dump_json(indent=2),
                 profile.model_dump_json(indent=2),
                 context_chunks=context_chunks,
                 skill_matches=skill_matches,
             ))],
            run_name="candidate_judge",
        )
```

- [ ] **Step 2: Commit**

```bash
git add src/agents/candidate_judge.py
git commit -m "refactor: remove use_evidence_grounding from CandidateJudgeAgent — always use grounding prompt"
```

---

### Task 3: Simplify `phase3_candidate_judge` in `nodes.py`

**Files:**
- Modify: `src/graph/nodes.py`

Remove `use_evidence_grounding` param and its passthrough to `CandidateJudgeAgent`.

- [ ] **Step 1: Replace `phase3_candidate_judge`**

In `src/graph/nodes.py`, replace the entire `phase3_candidate_judge` function (lines 54–97) with:

```python
def phase3_candidate_judge(
    profiles: list[CandidateProfile],
    jd: JDRequirements,
    vector_store: CVVectorStore | None,
) -> list[CandidateAssessment]:
    with _tracer.start_as_current_span("phase3/candidate_judge") as span:
        span.set_attribute("phase", 3)
        span.set_attribute("n_candidates", len(profiles))

        jd_text = jd.model_dump_json()

        def process(profile: CandidateProfile) -> CandidateAssessment:
            with _tracer.start_as_current_span(
                f"phase3/judge/{profile.candidate_id}"
            ) as cspan:
                try:
                    context_chunks: list[str] = []
                    if vector_store is not None:
                        context_chunks = vector_store.retrieve(
                            profile.candidate_id, jd_text,
                            top_k=10,
                        )
                    candidate_names = {s.canonical_skill for s in profile.skills}
                    skill_matches = [
                        SkillMatchResult(
                            jd_skill=s.skill,
                            best_match=s.skill if s.skill in candidate_names else None,
                            score=1.0 if s.skill in candidate_names else 0.0,
                            is_required=s.is_mandatory,
                        )
                        for s in jd.required_skills + jd.preferred_skills
                    ]
                    return CandidateJudgeAgent().run(
                        profile, jd, context_chunks or None, skill_matches or None
                    )
                except Exception as exc:
                    cspan.record_exception(exc)
                    cspan.set_status(StatusCode.ERROR, str(exc))
                    raise

        return [process(profile) for profile in profiles]
```

- [ ] **Step 2: Commit**

```bash
git add src/graph/nodes.py
git commit -m "refactor: remove use_evidence_grounding from phase3_candidate_judge"
```

---

### Task 4: Simplify `run_pipeline` in `pipeline.py`

**Files:**
- Modify: `src/graph/pipeline.py`

Remove `_ranking_from_raw`, the 4 ablation flags, and their conditional branches. Inline imports cleanup (`FinalRanking`, `RankedCandidate`, `SkillMatchResult` are no longer used here).

- [ ] **Step 1: Replace the file content**

Replace `src/graph/pipeline.py` with:

```python
from __future__ import annotations
import hashlib
import time
import uuid
from typing import Callable
from src.graph.state import ATSState
from src.graph.nodes import (
    phase1_jd_parser, phase2_cv_extractor,
    phase3_candidate_judge, phase4_pool_calibrator,
)
from src.utils.cache import ExtractionCache
from src.utils.vector_store import CVVectorStore
from src.utils.skill_normalizer import normalize_skills
from src.utils.telemetry import setup_telemetry, get_tracer, current_otel_trace_id
from src.models.schemas import CandidateProfile, JDRequirements
import config


def _filter_by_required_skills(
    profiles: list[CandidateProfile],
    jd: JDRequirements,
) -> tuple[list[CandidateProfile], list[str]]:
    if not jd.required_skills:
        return profiles, []
    required_names = {s.skill for s in jd.required_skills}
    passing: list[CandidateProfile] = []
    eliminated: list[str] = []
    for profile in profiles:
        candidate_names = {s.canonical_skill for s in profile.skills}
        if required_names & candidate_names:
            passing.append(profile)
        else:
            eliminated.append(profile.candidate_id)
    return passing, eliminated


def run_pipeline(
    jd_raw: str,
    cv_raws: list[dict],
    run_id: str | None = None,
    use_cache: bool = True,
    session_id: str | None = None,
    on_phase_complete: Callable[[dict], None] | None = None,
) -> ATSState:
    setup_telemetry()
    run_id = run_id or str(uuid.uuid4())[:8]
    jd_hash = hashlib.sha256(jd_raw.encode()).hexdigest()[:12]
    cache        = ExtractionCache(config.CACHE_DB_PATH) if use_cache else None
    vector_store = CVVectorStore(config.CHROMA_DB_PATH)  if use_cache else None

    tracer = get_tracer()
    with tracer.start_as_current_span(
        "pipeline",
        attributes={
            "run.id": run_id,
            "run.n_candidates": len(cv_raws),
            "run.jd_hash": jd_hash,
            "run.session_id": session_id or "",
            "llm.small_model": config.SMALL_MODEL,
            "llm.large_model": config.LARGE_MODEL,
        },
    ):
        otel_trace_id = current_otel_trace_id()
        trace_log: list[dict] = []

        def _run_phase(n: int, fn, *args, **extra_log):
            t0 = time.time()
            result = fn(*args)
            entry = {"phase": n, "duration_s": round(time.time() - t0, 2), **extra_log}
            trace_log.append(entry)
            if on_phase_complete:
                on_phase_complete(entry)
            return result

        jd_structured = _run_phase(1, phase1_jd_parser, jd_raw, cache)

        # Normalize JD skill names through same LLM prompt as CV Phase 2B
        # so both sides are canonical and exact-match comparison works
        jd_skill_names = [s.skill for s in jd_structured.required_skills + jd_structured.preferred_skills]
        if jd_skill_names:
            jd_norm_map = normalize_skills(jd_skill_names)
            for s in jd_structured.required_skills:
                s.skill = jd_norm_map.get(s.skill, s.skill)
            for s in jd_structured.preferred_skills:
                s.skill = jd_norm_map.get(s.skill, s.skill)

        cv_profiles = _run_phase(2, phase2_cv_extractor, cv_raws, cache,
                                 vector_store, candidates=len(cv_raws))

        cv_profiles, eliminated = _filter_by_required_skills(cv_profiles, jd_structured)

        assessments = _run_phase(3, phase3_candidate_judge, cv_profiles,
                                 jd_structured, vector_store)

        final_ranking = _run_phase(4, phase4_pool_calibrator, assessments, jd_structured)

        return ATSState(
            jd_raw=jd_raw,
            cv_raws=cv_raws,
            jd_structured=jd_structured,
            cv_profiles=cv_profiles,
            candidate_assessments=assessments,
            final_ranking=final_ranking,
            run_id=run_id,
            otel_trace_id=otel_trace_id,
            trace_log=trace_log,
            use_cache=use_cache,
            eliminated_candidates=eliminated,
        )
```

- [ ] **Step 2: Commit**

```bash
git add src/graph/pipeline.py
git commit -m "refactor: remove ablation flags and _ranking_from_raw from run_pipeline"
```

---

### Task 5: Clean `main.py`

**Files:**
- Modify: `main.py`

Remove the two import lines, two CLI params, and the two `if baselines:` / `if ablation:` execution blocks.

- [ ] **Step 1: Remove baseline/ablation imports (lines 15–16)**

Delete these two lines from `main.py`:

```python
from src.evaluation.baselines import run_baselines
from src.evaluation.ablation import run_ablation
```

- [ ] **Step 2: Remove CLI params (lines 46–47)**

Delete these two lines from the `main()` function signature:

```python
    baselines: bool = typer.Option(False, "--baselines", help="Run TF-IDF and keyword baseline rankers"),
    ablation: bool = typer.Option(False, "--ablation", help="Run 4-variant ablation study (LLM-heavy)"),
```

- [ ] **Step 3: Remove the `if baselines:` block**

Delete the entire block (from `if baselines:` through the closing Langfuse score call for `baseline_tfidf_vs_keyword_tau`):

```python
        if baselines:
            console.print("[blue]Running baseline comparison...[/blue]")
            evidencerank_ranking = [rc.candidate_id for rc in ranking.ranked_candidates]
            baseline_results = run_baselines(jd_text, cv_raws, evidencerank_ranking)

            bl_table = Table(title="Baseline Comparison")
            bl_table.add_column("Method")
            bl_table.add_column("Ranking (1→N)")
            bl_table.add_column("τ vs EvidenceRank")
            for method in ("tfidf", "keyword"):
                r = baseline_results[method]
                ranking_str = " > ".join(r["ranking"])
                tau = baseline_results["cross_method_tau"][f"{method}_vs_evidencerank"]
                bl_table.add_row(method.upper(), ranking_str, f"{tau:.3f}")
            er = baseline_results["evidencerank"]
            bl_table.add_row("EvidenceRank", " > ".join(er["ranking"]), "1.000")
            console.print(bl_table)

            (out_dir / "evaluation").mkdir(parents=True, exist_ok=True)
            (out_dir / "evaluation" / "baselines.json").write_text(
                json.dumps(baseline_results, indent=2), encoding="utf-8"
            )
            console.print(f"[green]Baselines saved to {out_dir}/evaluation/baselines.json[/green]")

            # Langfuse scores (best-effort, after the important write is done)
            lf.create_score(trace_id=otel_trace_id, name="baseline_tfidf_tau",
                            value=baseline_results["cross_method_tau"]["tfidf_vs_evidencerank"])
            lf.create_score(trace_id=otel_trace_id, name="baseline_keyword_tau",
                            value=baseline_results["cross_method_tau"]["keyword_vs_evidencerank"])
            lf.create_score(trace_id=otel_trace_id, name="baseline_tfidf_vs_keyword_tau",
                            value=baseline_results["cross_method_tau"]["tfidf_vs_keyword"])
```

- [ ] **Step 4: Remove the `if ablation:` block**

Delete the entire block (from `if ablation:` through the closing Langfuse score loop):

```python
        if ablation:
            console.print("[blue]Running ablation study (this will make additional LLM calls)...[/blue]")
            ablation_results = run_ablation(jd_text, cv_raws, state, cv_text_map)

            ab_table = Table(title="Ablation Study")
            ab_table.add_column("Variant")
            ab_table.add_column("Hallucination Rate")
            ab_table.add_column("Score Std")
            ab_table.add_column("LLM Calls")
            ab_table.add_column("τ vs Full")
            for variant_name, metrics in ablation_results.items():
                ab_table.add_row(
                    variant_name,
                    f"{metrics['hallucination_rate']:.1%}",
                    f"{metrics['score_std']:.1f}",
                    str(metrics['llm_calls']),
                    f"{metrics['tau_vs_full']:.3f}",
                )
            console.print(ab_table)

            (out_dir / "evaluation").mkdir(parents=True, exist_ok=True)
            (out_dir / "evaluation" / "ablation.json").write_text(
                json.dumps(ablation_results, indent=2), encoding="utf-8"
            )
            console.print(f"[green]Ablation saved to {out_dir}/evaluation/ablation.json[/green]")

            # Langfuse scores (best-effort, after the important write is done)
            for variant_name, metrics in ablation_results.items():
                lf.create_score(trace_id=otel_trace_id,
                                name=f"ablation_{variant_name}_hallucination_rate",
                                value=metrics["hallucination_rate"])
                lf.create_score(trace_id=otel_trace_id,
                                name=f"ablation_{variant_name}_tau",
                                value=metrics["tau_vs_full"])
```

- [ ] **Step 5: Commit**

```bash
git add main.py
git commit -m "refactor: remove --baselines and --ablation flags from CLI"
```

---

### Task 6: Verify — run the full test suite

- [ ] **Step 1: Run all tests**

```bash
uv run pytest -v
```

Expected: all remaining tests pass (no import errors from the deleted modules, no references to removed params).

If any test fails with an `ImportError` referencing `baselines`, `ablation`, `judge_no_grounding`, or `use_evidence_grounding`, that file was missed in Task 1 — delete it and re-run.

- [ ] **Step 2: Confirm no stale references remain**

```bash
grep -r "baselines\|ablation\|judge_no_grounding\|use_evidence_grounding\|use_vector_store\|use_skill_filter\|use_pool_calibration\|_ranking_from_raw" src/ main.py tests/
```

Expected: no output.
