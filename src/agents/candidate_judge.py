from __future__ import annotations
from langchain_core.messages import SystemMessage, HumanMessage
from src.models.schemas import CandidateAssessment, CandidateProfile, JDRequirements
from src.utils.llm import get_llm, invoke_with_telemetry
from src.prompts import judge as prompts
from src.prompts import judge_no_grounding as prompts_no_grounding
import config


class CandidateJudgeAgent:
    def __init__(self, use_evidence_grounding: bool = True):
        self._llm = get_llm(config.LARGE_MODEL, CandidateAssessment, config.JUDGE_TEMPERATURE)
        self._use_evidence_grounding = use_evidence_grounding

    def run(
        self,
        profile: CandidateProfile,
        jd: JDRequirements,
        context_chunks: list[str] | None = None,
        skill_matches: list | None = None,
    ) -> CandidateAssessment:
        system = (
            prompts.SYSTEM if self._use_evidence_grounding
            else prompts_no_grounding.SYSTEM
        )
        return invoke_with_telemetry(
            self._llm,
            [SystemMessage(content=system),
             HumanMessage(content=prompts.human(
                 jd.model_dump_json(indent=2),
                 profile.model_dump_json(indent=2),
                 context_chunks=context_chunks,
                 skill_matches=skill_matches,
             ))],
            run_name="candidate_judge",
        )
