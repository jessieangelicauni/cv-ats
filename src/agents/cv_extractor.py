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
        # Mixing the system prompt into the key ensures a prompt edit (e.g. a fix to
        # how a field is extracted) invalidates old cache entries automatically —
        # otherwise a stale pre-fix extraction keeps being served forever, silently
        # masking the fix. See the "Daniel Adif Nugroho Resume" full_name=null
        # incident: the candidate_id-leak fix in human_2a landed, but the cache
        # entry written before the fix kept being replayed.
        cache_key = ExtractionCache.make_key(
            cv_text + candidate_id + prompts.SYSTEM_2A, prefix="cv"
        )

        if self._cache:
            cached = self._cache.get(cache_key)
            if cached:
                profile = CandidateProfile.model_validate(cached)
                profile.candidate_id = candidate_id
                return profile

        profile: CandidateProfile = self._extract_llm.invoke(
            [SystemMessage(content=prompts.SYSTEM_2A),
             HumanMessage(content=prompts.human_2a(cv_text))]
        )
        # candidate_id is assigned deterministically by the caller (main.py, from the
        # source filename) and must never be rewritten by the model: when the file
        # name looks like a real person's name, the model "cleans it up" to match the
        # name it extracts, silently breaking traceability back to the source file.
        profile.candidate_id = candidate_id

        if self._cache:
            self._cache.set(cache_key, profile.model_dump())
        return profile
