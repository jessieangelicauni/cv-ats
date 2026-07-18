from __future__ import annotations
from pydantic import BaseModel, Field
from src.models.schemas import (
    JDRequirements, CandidateProfile,
    CandidateAssessment, FinalRanking, HallucinationFlag,
)


class ATSState(BaseModel):
    jd_raw: str
    cv_raws: list[dict]
    jd_structured: JDRequirements | None = None
    cv_profiles: list[CandidateProfile] = Field(default_factory=list)
    candidate_assessments: list[CandidateAssessment] = Field(default_factory=list)
    final_ranking: FinalRanking | None = None
    run_id: str
    trace_log: list[dict] = Field(default_factory=list)
    hallucination_flags: list[HallucinationFlag] = Field(default_factory=list)
    use_cache: bool = True
    eliminated_candidates: list[str] = Field(default_factory=list)
