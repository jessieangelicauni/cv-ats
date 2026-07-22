from src.models.schemas import (
    JDRequirements, EducationRequirement,
    CandidateProfile, CandidateBasicInfo,
    CandidateAssessment, EvidenceItem,
)


def make_jd_requirements(**overrides) -> JDRequirements:
    defaults = dict(
        role_title="Engineer", seniority_level="mid",
        required_skills=[], preferred_skills=[], min_years_experience=3,
        education=EducationRequirement(degree="BSc", field="CS", is_mandatory=False),
        domain_expertise=[], leadership_expected=False,
        soft_skills=[], industry_context="IT", raw_jd_hash="abc",
    )
    defaults.update(overrides)
    return JDRequirements(**defaults)


def make_candidate_profile(candidate_id: str = "cv_001", **overrides) -> CandidateProfile:
    defaults = dict(
        candidate_id=candidate_id,
        basic_info=CandidateBasicInfo(),
        skills=[], work_history=[], education=[],
        certifications=[], languages=[], total_experience_months=36,
    )
    defaults.update(overrides)
    return CandidateProfile(**defaults)


def make_candidate_assessment(candidate_id: str = "cv_001", raw_score: float = 75.0, **overrides) -> CandidateAssessment:
    defaults = dict(
        candidate_id=candidate_id, raw_score=raw_score, confidence="high",
        evidence_chain=[EvidenceItem(dimension="Technical Skills Fit", assessment="Good.",
                                      evidence_quote="Python experience", dimension_score=8.0)],
        key_strengths=["Python"], key_gaps=[], seniority_alignment="aligned",
    )
    defaults.update(overrides)
    return CandidateAssessment(**defaults)
