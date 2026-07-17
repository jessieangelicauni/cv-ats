from __future__ import annotations
import json
from pathlib import Path
from datetime import date
from src.models.schemas import (
    CandidateAssessment, CandidateProfile, HallucinationFlag,
)
from src.graph.state import ATSState
from src.evaluation.hallucination_checker import hallucination_rate


def _profile_for(cid: str, profiles: list[CandidateProfile]) -> CandidateProfile | None:
    return next((p for p in profiles if p.candidate_id == cid), None)


def _assessment_for(cid: str, assessments: list[CandidateAssessment]) -> CandidateAssessment | None:
    return next((a for a in assessments if a.candidate_id == cid), None)


def _render_candidate_block(
    rank: int,
    rc,
    profile: CandidateProfile,
    assessment: CandidateAssessment,
    flags: list[HallucinationFlag],
) -> str:
    bi = profile.basic_info
    lines = [
        f"---\n",
        f"### Rank {rank} — {rc.candidate_id} | {bi.full_name or '(name not found)'}",
        f"**Score:** {rc.calibrated_score} (raw: {assessment.raw_score}, Δ {rc.delta_from_raw:+.1f})"
        f" | **Confidence:** {assessment.confidence.capitalize()}"
        f" | **Seniority:** {assessment.seniority_alignment.capitalize()}",
        "",
        "#### Contact",
        f"- Email: {bi.email or 'N/A'}",
        f"- Phone: {bi.phone or 'N/A'}",
        f"- Location: {bi.location or 'N/A'}",
        f"- LinkedIn: {bi.linkedin_url or 'N/A'}",
        f"- Current Title: {bi.current_title or 'N/A'}",
        "",
        "#### Skills",
        "| Canonical Skill | Raw Mention | Proficiency | Evidence Quote |",
        "|---|---|---|---|",
    ]
    for s in profile.skills:
        lines.append(
            f"| {s.canonical_skill} | {s.raw_mention} | {s.proficiency} | {s.evidence_quote} |"
        )

    lines += ["", "#### Education",
              "| Degree | Field | Institution | Year |",
              "|---|---|---|---|"]
    for e in profile.education:
        lines.append(f"| {e.degree} | {e.field} | {e.institution} | {e.year or 'N/A'} |")

    lines += ["", "#### Work Experience"]
    for w in profile.work_history:
        lines += [
            f"**{w.role} — {w.company}**",
            f"Duration: {w.tenure_months or 'N/A'} months"
            f" | Leadership: {'Yes' if w.has_leadership_indicators else 'No'}",
            f"Technologies: {', '.join(w.technologies)}",
            "Achievements:",
        ]
        for ach in w.achievements:
            lines.append(f'- "{ach}"')
        lines.append("")

    lines += ["#### LLM Judgment"]
    candidate_flags = {f.claim: f.status for f in flags if f.candidate_id == rc.candidate_id}
    for item in assessment.evidence_chain:
        h_status = candidate_flags.get(item.assessment, "unknown")
        lines += [
            f"**{item.dimension} — {item.dimension_score}/10**",
            f"Assessment: {item.assessment}",
            f"Evidence: \"{item.evidence_quote}\"",
            f"Hallucination status: {h_status}",
            "",
        ]

    lines += [
        f"**Holistic Score: {assessment.raw_score} (raw) → {rc.calibrated_score} (calibrated)**",
        f"Key Strengths: {', '.join(assessment.key_strengths)}",
        f"Key Gaps: {', '.join(assessment.key_gaps)}",
        f"Comparative note: {rc.comparative_notes}",
        "",
    ]
    return "\n".join(lines)


def generate_report(state: ATSState, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "candidates").mkdir(exist_ok=True)
    (output_dir / "evaluation").mkdir(exist_ok=True)

    ranking = state.final_ranking
    jd = state.jd_structured
    profiles = state.cv_profiles
    assessments = state.candidate_assessments
    flags = state.hallucination_flags
    run_id = state.run_id

    (output_dir / "ranking.json").write_text(
        ranking.model_dump_json(indent=2), encoding="utf-8"
    )

    for a in assessments:
        (output_dir / "candidates" / f"{a.candidate_id}.json").write_text(
            a.model_dump_json(indent=2), encoding="utf-8"
        )

    h_rate = hallucination_rate(flags)
    fabricated = [f for f in flags if f.status == "fabricated"]
    h_report = {
        "overall_fabrication_rate": round(h_rate, 4),
        "total_claims": len([f for f in flags if f.status != "acknowledged_gap"]),
        "fabricated_count": len(fabricated),
        "fabricated_claims": [f.model_dump() for f in fabricated],
    }
    (output_dir / "evaluation" / "hallucination_report.json").write_text(
        json.dumps(h_report, indent=2), encoding="utf-8"
    )

    md_lines = [
        "# ATS Evaluation Report",
        f"Run: {run_id} | JD: {jd.role_title} ({jd.seniority_level})"
        f" | Candidates: {len(assessments)} | Date: {date.today()}",
        "",
        "---",
        "",
        "## Ranking Summary",
        "",
        "| Rank | File | Name | Calibrated Score | Raw Score | Δ | Confidence | Seniority |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for rc in ranking.ranked_candidates:
        a = _assessment_for(rc.candidate_id, assessments)
        p = _profile_for(rc.candidate_id, profiles)
        name = p.basic_info.full_name if p else "N/A"
        md_lines.append(
            f"| {rc.rank} | {rc.candidate_id} | {name} | {rc.calibrated_score}"
            f" | {a.raw_score if a else 'N/A'} | {rc.delta_from_raw:+.1f}"
            f" | {a.confidence.capitalize() if a else 'N/A'}"
            f" | {a.seniority_alignment.capitalize() if a else 'N/A'} |"
        )

    md_lines += ["", f"Pool calibration rationale: {ranking.calibration_rationale}",
                 "", "---", "", "## Candidate Dossiers", ""]

    for rc in ranking.ranked_candidates:
        profile = _profile_for(rc.candidate_id, profiles)
        assessment = _assessment_for(rc.candidate_id, assessments)
        if profile and assessment:
            md_lines.append(
                _render_candidate_block(rc.rank, rc, profile, assessment, flags)
            )

    if ranking.borderline_pairs:
        md_lines += ["## Pool Calibration Notes", "", "### Borderline Pairs"]
        for bp in ranking.borderline_pairs:
            md_lines.append(str(bp))

    md_lines += [
        "", "---", "", "## Hallucination Summary",
        f"Overall fabrication rate: {h_rate:.1%} ({len(fabricated)} / "
        f"{len([f for f in flags if f.status != 'acknowledged_gap'])} claims)",
    ]
    if fabricated:
        md_lines += ["", "### Flagged Claims"]
        for f in fabricated:
            md_lines += [
                f"**{f.candidate_id} — FABRICATED**",
                f"Claim: {f.claim}",
                f"Quote used: {f.source_quote}",
                "",
            ]

    (output_dir / "report.md").write_text("\n".join(md_lines), encoding="utf-8")
