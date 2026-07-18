from __future__ import annotations
import hashlib
from langchain_core.messages import SystemMessage, HumanMessage
from src.models.schemas import JDRequirements
from src.utils.llm import get_llm
from src.utils.cache import ExtractionCache
from src.prompts import jd_parser as prompts
import config


class JDParserAgent:
    def __init__(self, cache: ExtractionCache | None = None):
        self._llm = get_llm(config.SMALL_MODEL, JDRequirements, config.EXTRACTION_TEMPERATURE)
        self._cache = cache

    def run(self, jd_text: str) -> JDRequirements:
        jd_hash = hashlib.sha256(jd_text.encode()).hexdigest()
        cache_key = ExtractionCache.make_key(jd_text, prefix="jd")

        if self._cache:
            cached = self._cache.get(cache_key)
            if cached:
                return JDRequirements.model_validate(cached)

        result: JDRequirements = self._llm.invoke(
            [SystemMessage(content=prompts.SYSTEM),
             HumanMessage(content=prompts.human(jd_text, jd_hash))]
        )
        result.raw_jd_hash = jd_hash

        if self._cache:
            self._cache.set(cache_key, result.model_dump())
        return result
