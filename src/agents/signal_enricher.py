from __future__ import annotations
from langchain_core.messages import SystemMessage, HumanMessage
from src.models.schemas import CandidateProfile, JDRequirements
from src.utils.llm import get_llm, invoke_with_telemetry
from src.prompts import signal_enricher as prompts
import config


class SignalEnricherAgent:
    """
    DEPRECATED: Signal enricher phase has been removed from the pipeline.
    CandidateJudgeAgent now directly consumes CandidateProfile instead of EnrichedProfile.
    """
    def __init__(self):
        raise NotImplementedError("SignalEnricherAgent has been removed from the pipeline.")

    def run(self, profile: CandidateProfile, jd: JDRequirements) -> CandidateProfile:
        raise NotImplementedError("SignalEnricherAgent has been removed from the pipeline.")
