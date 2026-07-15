from __future__ import annotations
from langchain_core.messages import SystemMessage, HumanMessage
from src.models.schemas import CandidateProfile, EnrichedProfile, EnrichmentSignals, JDRequirements
from src.utils.llm import get_llm, invoke_with_telemetry
from src.prompts import signal_enricher as prompts
import config


class SignalEnricherAgent:
    def __init__(self):
        self._llm = get_llm(config.SMALL_MODEL, EnrichmentSignals, config.EXTRACTION_TEMPERATURE)

    def run(self, profile: CandidateProfile, jd: JDRequirements) -> EnrichedProfile:
        signals: EnrichmentSignals = invoke_with_telemetry(
            self._llm,
            [SystemMessage(content=prompts.SYSTEM),
             HumanMessage(content=prompts.human(
                 profile.model_dump_json(indent=2),
                 jd.model_dump_json(indent=2),
             ))],
        )
        return EnrichedProfile(
            **profile.model_dump(),
            **signals.model_dump(),
        )
