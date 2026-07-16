from __future__ import annotations
from opentelemetry.trace import get_current_span
from langchain_core.messages import SystemMessage, HumanMessage
from src.models.schemas import CandidateProfile, SkillNormalizationMap
from src.utils.llm import get_llm, invoke_with_telemetry
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
        cache_key = ExtractionCache.make_key(cv_text + candidate_id, prefix="cv")

        if self._cache:
            cached = self._cache.get(cache_key)
            if cached:
                get_current_span().set_attribute("cache.hit", True)
                return CandidateProfile.model_validate(cached)

        get_current_span().set_attribute("cache.hit", False)

        # Two LLM calls: extraction then skill normalisation (two separate generations)
        profile: CandidateProfile = invoke_with_telemetry(
            self._extract_llm,
            [SystemMessage(content=prompts.SYSTEM_2A),
             HumanMessage(content=prompts.human_2a(cv_text, candidate_id))],
            run_name="cv_extractor.extract",
        )

        raw_mentions = [s.raw_mention for s in profile.skills]
        if raw_mentions:
            norm_map: SkillNormalizationMap = invoke_with_telemetry(
                self._norm_llm,
                [SystemMessage(content=prompts.SYSTEM_2B),
                 HumanMessage(content=prompts.human_2b(raw_mentions))],
                run_name="cv_extractor.normalize_skills",
            )
            for skill in profile.skills:
                skill.canonical_skill = norm_map.mappings.get(
                    skill.raw_mention, skill.raw_mention
                )

        if self._cache:
            self._cache.set(cache_key, profile.model_dump())
        return profile
