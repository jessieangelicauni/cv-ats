from __future__ import annotations
from langchain_core.messages import SystemMessage, HumanMessage
from src.models.schemas import (
    CandidateAssessment, FinalRanking, JDRequirements,
    PoolCalibrationResult, RankedCandidate, BorderlinePair,
)
from src.utils.llm import get_llm
from src.prompts import calibrator as prompts
import config


def _summarise_assessment(a: CandidateAssessment) -> str:
    strengths = "; ".join(a.key_strengths[:3])
    gaps = "; ".join(a.key_gaps[:2])
    return (
        f"score={a.raw_score} confidence={a.confidence} "
        f"seniority={a.seniority_alignment}\n"
        f"  strengths: {strengths}\n"
        f"  gaps: {gaps}"
    )


class PoolCalibratorAgent:
    def __init__(self):
        self._llm = get_llm(config.LARGE_MODEL, PoolCalibrationResult, config.JUDGE_TEMPERATURE)

    def run(self, assessments: list[CandidateAssessment], jd: JDRequirements) -> FinalRanking:
        position_map = {i + 1: a.candidate_id for i, a in enumerate(assessments)}

        required_skills = ", ".join(s.skill for s in jd.required_skills)
        preferred_skills = ", ".join(s.skill for s in jd.preferred_skills)
        domain_expertise = ", ".join(jd.domain_expertise)
        role_summary = (
            f"{jd.role_title} ({jd.seniority_level}), min {jd.min_years_experience}y exp\n"
            f"Required skills: {required_skills}\n"
            f"Preferred skills: {preferred_skills}\n"
            f"Domain expertise: {domain_expertise}"
        )

        assessments_summary = "\n\n".join(
            f"Position {i + 1}:\n" + _summarise_assessment(a)
            for i, a in enumerate(assessments)
        )

        result: PoolCalibrationResult = self._llm.invoke(
            [SystemMessage(content=prompts.SYSTEM),
             HumanMessage(content=prompts.human(role_summary, assessments_summary, len(assessments)))]
        )

        entries = [
            (position_map[e.position], e)
            for e in result.calibrated_entries
            if e.position in position_map
        ]
        entries.sort(key=lambda x: x[1].calibrated_score, reverse=True)

        ranked = [
            RankedCandidate(
                rank=rank,
                candidate_id=cid,
                calibrated_score=entry.calibrated_score,
                delta_from_raw=entry.delta_from_raw,
                comparative_notes=entry.comparative_notes,
            )
            for rank, (cid, entry) in enumerate(entries, 1)
        ]

        borderline_pairs = [
            BorderlinePair(
                candidate_a=position_map[bp.position_a],
                candidate_b=position_map[bp.position_b],
                reason=bp.reason,
            )
            for bp in result.borderline_pairs
            if bp.position_a in position_map and bp.position_b in position_map
        ]

        return FinalRanking(
            ranked_candidates=ranked,
            pool_summary=result.pool_summary,
            calibration_rationale=result.calibration_rationale,
            borderline_pairs=borderline_pairs,
        )
