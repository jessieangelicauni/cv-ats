from __future__ import annotations
from langchain_core.messages import SystemMessage, HumanMessage
from src.models.schemas import CandidateAssessment, FinalRanking, JDRequirements
from src.utils.llm import get_llm, invoke_with_telemetry
from src.prompts import calibrator as prompts
import config


def _summarise_assessment(a: CandidateAssessment) -> str:
    strengths = "; ".join(a.key_strengths[:3])
    gaps = "; ".join(a.key_gaps[:2])
    return (
        f"[{a.candidate_id}] score={a.raw_score} confidence={a.confidence} "
        f"seniority={a.seniority_alignment}\n"
        f"  strengths: {strengths}\n"
        f"  gaps: {gaps}"
    )


class PoolCalibratorAgent:
    def __init__(self):
        self._llm = get_llm(config.LARGE_MODEL, FinalRanking, config.JUDGE_TEMPERATURE)

    def run(self, assessments: list[CandidateAssessment], jd: JDRequirements) -> FinalRanking:
        required_skills = ", ".join(s.skill for s in jd.required_skills)
        preferred_skills = ", ".join(s.skill for s in jd.preferred_skills)
        domain_expertise = ", ".join(jd.domain_expertise)
        role_summary = (
            f"{jd.role_title} ({jd.seniority_level}), min {jd.min_years_experience}y exp\n"
            f"Required skills: {required_skills}\n"
            f"Preferred skills: {preferred_skills}\n"
            f"Domain expertise: {domain_expertise}"
        )
        assessments_summary = "\n\n".join(_summarise_assessment(a) for a in assessments)
        return invoke_with_telemetry(
            self._llm,
            [SystemMessage(content=prompts.SYSTEM),
             HumanMessage(content=prompts.human(role_summary, assessments_summary))],
            run_name="pool_calibrator",
        )
