from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, TypeAlias
from pydantic import BaseModel, Field, ConfigDict


@dataclass
class SkillMatchResult:
    jd_skill: str
    best_match: str | None
    score: float
    is_required: bool


class SkillNormalizationMap(BaseModel):
    mappings: dict[str, str]

Proficiency: TypeAlias = Literal["beginner", "intermediate", "advanced", "expert"]


class SkillRequirement(BaseModel):
    skill: str
    level: Proficiency
    is_mandatory: bool


class EducationRequirement(BaseModel):
    degree: str
    field: str
    is_mandatory: bool


class JDRequirements(BaseModel):
    role_title: str
    seniority_level: Literal["junior", "mid", "senior", "lead", "principal"]
    required_skills: list[SkillRequirement]
    preferred_skills: list[SkillRequirement]
    min_years_experience: int
    education: EducationRequirement
    domain_expertise: list[str]
    leadership_expected: bool
    soft_skills: list[str]
    industry_context: str
    raw_jd_hash: str


class SkillEntry(BaseModel):
    raw_mention: str
    canonical_skill: str
    proficiency: Proficiency
    evidence_quote: str


class WorkEntry(BaseModel):
    company: str
    role: str
    tenure_months: int | None = None
    technologies: list[str]
    achievements: list[str]
    has_leadership_indicators: bool


class EducationEntry(BaseModel):
    degree: str
    field: str
    institution: str
    year: int | None = None


class LanguageEntry(BaseModel):
    language: str
    proficiency: Literal["native", "fluent", "professional", "basic"]


class CandidateBasicInfo(BaseModel):
    full_name: str | None = None
    email: str | None = None
    phone: str | None = None
    location: str | None = None
    linkedin_url: str | None = None
    current_title: str | None = None


class CandidateProfile(BaseModel):
    candidate_id: str
    basic_info: CandidateBasicInfo
    skills: list[SkillEntry]
    work_history: list[WorkEntry]
    education: list[EducationEntry]
    certifications: list[str]
    languages: list[LanguageEntry]
    total_experience_months: int


class EvidenceItem(BaseModel):
    dimension: str
    assessment: str
    evidence_quote: str
    dimension_score: float = Field(ge=0.0, le=10.0)


class HallucinationFlag(BaseModel):
    candidate_id: str
    claim: str
    status: Literal["supported", "inferred", "fabricated", "acknowledged_gap"]
    source_quote: str | None = None


class CandidateAssessment(BaseModel):
    candidate_id: str
    raw_score: float = Field(ge=0.0, le=100.0)
    confidence: Literal["high", "medium", "low"]
    evidence_chain: list[EvidenceItem]
    key_strengths: list[str]
    key_gaps: list[str]
    seniority_alignment: Literal["overqualified", "aligned", "underqualified"]


class RankedCandidate(BaseModel):
    rank: int
    candidate_id: str
    calibrated_score: float = Field(ge=0.0, le=100.0)
    delta_from_raw: float
    comparative_notes: str


class BorderlinePair(BaseModel):
    candidate_a: str
    candidate_b: str
    reason: str


class FinalRanking(BaseModel):
    ranked_candidates: list[RankedCandidate]
    pool_summary: str
    calibration_rationale: str
    borderline_pairs: list[BorderlinePair]
