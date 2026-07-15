from __future__ import annotations
from langchain_core.messages import SystemMessage, HumanMessage
from src.models.schemas import CandidateProfile, SkillNormalizationMap
from src.utils.llm import get_llm
from src.utils.cache import ExtractionCache
from src.prompts import cv_extractor as prompts
import config


class CVExtractorAgent:
    def __init__(self, cache: ExtractionCache | None = None):
        self._extract_llm = get_llm(config.SMALL_MODEL, CandidateProfile, config.EXTRACTION_TEMPERATURE)
        self._norm_llm = get_llm(config.SMALL_MODEL, SkillNormalizationMap, config.EXTRACTION_TEMPERATURE)
        self._cache = cache

    def run(self, cv_raw: dict) -> CandidateProfile:
        cv_text = cv_raw["raw_text"]
        candidate_id = cv_raw["candidate_id"]

        if self._cache:
            key = ExtractionCache.make_key(cv_text + candidate_id, prefix="cv")
            cached = self._cache.get(key)
            if cached:
                return CandidateProfile.model_validate(cached)

        profile: CandidateProfile = self._extract_llm.invoke([
            SystemMessage(content=prompts.SYSTEM_2A),
            HumanMessage(content=prompts.human_2a(cv_text, candidate_id)),
        ])

        raw_mentions = [s.raw_mention for s in profile.skills]
        if raw_mentions:
            norm_map: SkillNormalizationMap = self._norm_llm.invoke([
                SystemMessage(content=prompts.SYSTEM_2B),
                HumanMessage(content=prompts.human_2b(raw_mentions)),
            ])
            for skill in profile.skills:
                skill.canonical_skill = norm_map.mappings.get(
                    skill.raw_mention, skill.raw_mention
                )

        if self._cache:
            key = ExtractionCache.make_key(cv_text + candidate_id, prefix="cv")
            self._cache.set(key, profile.model_dump())

        return profile
