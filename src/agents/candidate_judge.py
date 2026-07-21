from __future__ import annotations
from langchain_core.messages import SystemMessage, HumanMessage
from src.models.schemas import CandidateAssessment, CandidateProfile, JDRequirements
from src.utils.llm import get_llm
from src.prompts import judge as prompts
import config


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
        assessment: CandidateAssessment = self._llm.invoke(
            [SystemMessage(content=prompts.SYSTEM),
             HumanMessage(content=prompts.human(
                 jd.model_dump_json(indent=2),
                 profile.model_dump_json(indent=2),
                 raw_cv_text,
                 skill_matches=skill_matches,
             ))]
        )
        # candidate_id is assigned deterministically by the caller (main.py, from the
        # source filename) and must never be rewritten by the model: when the file
        # name looks like a real person's name, the model "cleans it up" to match the
        # name it extracts, silently breaking the cv_text_map lookup used by
        # hallucination_checker.verify_evidence_chain (see test_cv_extractor.py for
        # the analogous fix in CVExtractorAgent).
        assessment.candidate_id = profile.candidate_id
        return assessment
