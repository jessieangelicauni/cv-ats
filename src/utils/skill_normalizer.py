from __future__ import annotations
from langchain_core.messages import SystemMessage, HumanMessage
from src.models.schemas import SkillNormalizationMap
from src.utils.llm import get_llm, invoke_with_telemetry
from src.prompts.cv_extractor import SYSTEM_2B, human_2b
import config


def normalize_skills(raw_names: list[str]) -> dict[str, str]:
    """Normalize skill name strings to canonical form via LLM.

    Returns {raw_name: canonical_name}. At EXTRACTION_TEMPERATURE=0.0 this
    is deterministic — same input always produces same canonical string.
    """
    if not raw_names:
        return {}
    llm = get_llm(config.SMALL_MODEL, SkillNormalizationMap, config.EXTRACTION_TEMPERATURE)
    result: SkillNormalizationMap = invoke_with_telemetry(
        llm,
        [SystemMessage(content=SYSTEM_2B),
         HumanMessage(content=human_2b(raw_names))],
        run_name="skill_normalizer",
    )
    return result.mappings
