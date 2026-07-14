from __future__ import annotations
from typing import Literal
from pydantic import BaseModel


class SkillRequirement(BaseModel):
    skill: str
    level: Literal["beginner", "intermediate", "advanced", "expert"]
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


class SkillNormalizationMap(BaseModel):
    mappings: dict[str, str]


class SkillEntry(BaseModel):
    raw_mention: str
    canonical_skill: str
    proficiency: Literal["beginner", "intermediate", "advanced", "expert"]
    evidence_quote: str


class WorkEntry(BaseModel):
    company: str
    role: str
    tenure_months: int | None
    technologies: list[str]
    achievements: list[str]
    has_leadership_indicators: bool


class EducationEntry(BaseModel):
    degree: str
    field: str
    institution: str
    year: int | None


class LanguageEntry(BaseModel):
    language: str
    proficiency: Literal["native", "fluent", "professional", "basic"]


class CandidateBasicInfo(BaseModel):
    full_name: str | None
    email: str | None
    phone: str | None
    location: str | None
    linkedin_url: str | None
    current_title: str | None


class CandidateProfile(BaseModel):
    candidate_id: str
    basic_info: CandidateBasicInfo
    skills: list[SkillEntry]
    work_history: list[WorkEntry]
    education: list[EducationEntry]
    certifications: list[str]
    languages: list[LanguageEntry]
    total_experience_months: int


class EnrichmentSignals(BaseModel):
    company_tiers: list[Literal["tier1_mnc", "tier2_established", "tier3_startup"]]
    highest_prestige_company: str
    career_trajectory: Literal["ascending", "lateral", "stagnant", "descending"]
    leadership_count: int
    measurable_impact_count: int
    tenure_stability: Literal["stable", "moderate", "job_hopper"]
    relevant_experience_months: int


class EnrichedProfile(BaseModel):
    candidate_id: str
    basic_info: CandidateBasicInfo
    skills: list[SkillEntry]
    work_history: list[WorkEntry]
    education: list[EducationEntry]
    certifications: list[str]
    languages: list[LanguageEntry]
    total_experience_months: int
    company_tiers: list[Literal["tier1_mnc", "tier2_established", "tier3_startup"]]
    highest_prestige_company: str
    career_trajectory: Literal["ascending", "lateral", "stagnant", "descending"]
    leadership_count: int
    measurable_impact_count: int
    tenure_stability: Literal["stable", "moderate", "job_hopper"]
    relevant_experience_months: int


class EvidenceItem(BaseModel):
    dimension: str
    assessment: str
    evidence_quote: str
    dimension_score: float


class HallucinationFlag(BaseModel):
    candidate_id: str
    claim: str
    status: Literal["supported", "inferred", "fabricated", "acknowledged_gap"]
    source_quote: str | None


class CandidateAssessment(BaseModel):
    candidate_id: str
    raw_score: float
    confidence: Literal["high", "medium", "low"]
    evidence_chain: list[EvidenceItem]
    key_strengths: list[str]
    key_gaps: list[str]
    seniority_alignment: Literal["overqualified", "aligned", "underqualified"]


class RankedCandidate(BaseModel):
    rank: int
    candidate_id: str
    calibrated_score: float
    delta_from_raw: float
    comparative_notes: str


class FinalRanking(BaseModel):
    ranked_candidates: list[RankedCandidate]
    pool_summary: str
    calibration_rationale: str
    borderline_pairs: list[dict]
