from __future__ import annotations
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from src.models.schemas import CandidateAssessment, CandidateProfile, JDRequirements, HallucinationFlag
from src.utils.llm import get_llm
from src.evaluation.hallucination_checker import verify_evidence_chain
from src.prompts import judge as prompts
import config


def _force_unverified_to_gap(
    assessment: CandidateAssessment, failed: list[HallucinationFlag]
) -> CandidateAssessment:
    failed_claims = {f.claim for f in failed}
    for item in assessment.evidence_chain:
        if item.assessment in failed_claims:
            item.evidence_quote = "NOT FOUND IN CV"
            item.dimension_score = min(item.dimension_score, 3.0)
    return assessment


class CandidateJudgeAgent:
    def __init__(self):
        self._llm = get_llm(config.LARGE_MODEL, CandidateAssessment, config.JUDGE_TEMPERATURE)

    def run(
        self,
        profile: CandidateProfile,
        jd: JDRequirements,
        raw_cv_text: str,
        skill_matches: list | None = None,
    ) -> CandidateAssessment:
        messages = [
            SystemMessage(content=prompts.SYSTEM),
            HumanMessage(content=prompts.human(
                jd.model_dump_json(indent=2),
                profile.model_dump_json(indent=2),
                raw_cv_text,
                skill_matches=skill_matches,
            )),
        ]

        for attempt in range(config.JUDGE_MAX_RETRIES + 1):
            assessment: CandidateAssessment = self._llm.invoke(messages)
            assessment.candidate_id = profile.candidate_id

            flags = verify_evidence_chain(assessment, raw_cv_text)
            failed = [f for f in flags if f.status == "fabricated"]
            if not failed:
                return assessment

            if attempt == config.JUDGE_MAX_RETRIES:
                return _force_unverified_to_gap(assessment, failed)

            dim_by_claim = {item.assessment: item.dimension for item in assessment.evidence_chain}
            failed_items = [(dim_by_claim[f.claim], f.claim, f.source_quote or "") for f in failed]
            messages.append(AIMessage(content=assessment.model_dump_json()))
            messages.append(HumanMessage(content=prompts.retry_human(failed_items)))

        raise AssertionError("unreachable")
