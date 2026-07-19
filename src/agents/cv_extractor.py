from __future__ import annotations
from langchain_core.messages import SystemMessage, HumanMessage
from src.models.schemas import CandidateProfile
from src.utils.llm import get_llm
from src.utils.cache import ExtractionCache
from src.prompts import cv_extractor as prompts
import config


class CVExtractorAgent:
    def __init__(self, cache: ExtractionCache | None = None):
        self._extract_llm = get_llm(config.SMALL_MODEL, CandidateProfile, config.EXTRACTION_TEMPERATURE)
        self._cache = cache

    def run(self, cv_raw: dict) -> CandidateProfile:
        cv_text = cv_raw["raw_text"]
        candidate_id = cv_raw["candidate_id"]
        cache_key = ExtractionCache.make_key(cv_text + candidate_id, prefix="cv")

        if self._cache:
            cached = self._cache.get(cache_key)
            if cached:
                return CandidateProfile.model_validate(cached)

        profile: CandidateProfile = self._extract_llm.invoke(
            [SystemMessage(content=prompts.SYSTEM_2A),
             HumanMessage(content=prompts.human_2a(cv_text, candidate_id))]
        )

        if self._cache:
            self._cache.set(cache_key, profile.model_dump())
        return profile
