from __future__ import annotations
from typing import TypedDict, Annotated
import operator
from src.models.schemas import (
    JDRequirements, CandidateProfile,
    CandidateAssessment, FinalRanking, HallucinationFlag,
)


class ATSState(TypedDict):
    jd_raw: str
    cv_raws: list[dict]
    jd_structured: JDRequirements | None
    cv_profiles: list[CandidateProfile]
    candidate_assessments: list[CandidateAssessment]
    final_ranking: FinalRanking | None
    run_id: str
    otel_trace_id: str      # 32-hex OTel trace ID — used for Langfuse scoring
    trace_log: Annotated[list[dict], operator.add]
    hallucination_flags: Annotated[list[HallucinationFlag], operator.add]
    use_cache: bool
