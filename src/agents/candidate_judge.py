from __future__ import annotations
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from src.models.schemas import CandidateAssessment, CandidateProfile, EvidenceItem, JDRequirements
from src.utils.llm import get_llm
from src.evaluation.hallucination_checker import verify_evidence_chain
from src.prompts import judge as prompts
import config


def _force_items_to_gap(items: list[EvidenceItem]) -> None:
    for item in items:
        item.evidence_quote = "NOT FOUND IN CV"
        item.dimension_score = min(item.dimension_score, 3.0)


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

        unresolved: set[str] | None = None

        for attempt in range(config.JUDGE_MAX_RETRIES + 1):
            assessment: CandidateAssessment = self._llm.invoke(messages)
            assessment.candidate_id = profile.candidate_id

            flags = verify_evidence_chain(assessment, raw_cv_text)
            status_by_claim = {f.claim: f.status for f in flags}

            if unresolved is None:
                unresolved = {
                    item.dimension for item in assessment.evidence_chain
                    if status_by_claim.get(item.assessment) == "fabricated"
                }
            else:
                # A dimension resolves only once EVERY evidence_chain item
                # bearing it verifies as genuinely grounded ("inferred") --
                # a dimension can carry more than one item, and a single
                # grounded sibling must not paper over another that's still
                # fabricated. A model that gives up mid-retry by writing its
                # own "NOT FOUND IN CV" comes back "acknowledged_gap", not
                # "inferred" -- that must NOT count as resolved either,
                # otherwise the model can escape further correction just by
                # declaring defeat. Only the exhaustion fallback below may
                # grant a gap. A dimension the model dropped entirely from
                # this attempt has no items to check "all" of and is kept
                # unresolved rather than vacuously passing.
                unresolved = {
                    dim for dim in unresolved
                    if not (
                        any(item.dimension == dim for item in assessment.evidence_chain)
                        and all(
                            status_by_claim.get(item.assessment) == "inferred"
                            for item in assessment.evidence_chain
                            if item.dimension == dim
                        )
                    )
                }

            if not unresolved:
                return assessment

            # Only items that are themselves still ungrounded get corrected
            # or gapped -- an already-genuinely-inferred sibling under the
            # same (still-unresolved) dimension is left untouched.
            unresolved_items = [
                item for item in assessment.evidence_chain
                if item.dimension in unresolved and status_by_claim.get(item.assessment) != "inferred"
            ]

            if attempt == config.JUDGE_MAX_RETRIES:
                _force_items_to_gap(unresolved_items)
                return assessment

            failed_items = [
                (item.dimension, item.assessment, item.evidence_quote)
                for item in unresolved_items
            ]
            messages.append(AIMessage(content=assessment.model_dump_json()))
            messages.append(HumanMessage(content=prompts.retry_human(failed_items)))

        raise AssertionError("unreachable")
