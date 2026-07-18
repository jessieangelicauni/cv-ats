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
        skill_matches: list | None = None,
    ) -> CandidateAssessment:
        return self._llm.invoke(
            [SystemMessage(content=prompts.SYSTEM),
             HumanMessage(content=prompts.human(
                 jd.model_dump_json(indent=2),
                 profile.model_dump_json(indent=2),
                 skill_matches=skill_matches,
             ))]
        )
