from unittest.mock import MagicMock, patch
from src.agents.candidate_judge import CandidateJudgeAgent
from src.models.schemas import (
    CandidateAssessment, EvidenceItem, EnrichedProfile,
    CandidateBasicInfo, JDRequirements, EducationRequirement,
)

def _make_enriched() -> EnrichedProfile:
    return EnrichedProfile(
        candidate_id="cv_001",
        basic_info=CandidateBasicInfo(
            full_name="Ahmad", email=None, phone=None,
            location=None, linkedin_url=None, current_title="Engineer",
        ),
        skills=[], work_history=[], education=[],
        certifications=[], languages=[], total_experience_months=60,
        company_tiers=["tier1_mnc"], highest_prestige_company="Google",
        career_trajectory="ascending", leadership_count=3,
        measurable_impact_count=4, tenure_stability="stable",
        relevant_experience_months=60,
    )

def _make_jd() -> JDRequirements:
    return JDRequirements(
        role_title="Senior Backend Engineer", seniority_level="senior",
        required_skills=[], preferred_skills=[], min_years_experience=5,
        education=EducationRequirement(degree="BSc", field="CS", is_mandatory=False),
        domain_expertise=[], leadership_expected=True,
        soft_skills=[], industry_context="IT", raw_jd_hash="abc",
    )

def _make_assessment() -> CandidateAssessment:
    return CandidateAssessment(
        candidate_id="cv_001",
        raw_score=87.0,
        confidence="high",
        evidence_chain=[
            EvidenceItem(
                dimension="Technical Skills Fit",
                assessment="Strong Python skills.",
                evidence_quote="5 years Python development",
                dimension_score=9.0,
            )
        ],
        key_strengths=["Tier 1 MNC experience"],
        key_gaps=["Kubernetes not evidenced"],
        seniority_alignment="aligned",
    )

def test_judge_returns_candidate_assessment():
    mock_llm = MagicMock(return_value=_make_assessment())
    with patch("src.agents.candidate_judge.get_llm", return_value=mock_llm):
        agent = CandidateJudgeAgent()
        result = agent.run(_make_enriched(), _make_jd())
    assert isinstance(result, CandidateAssessment)
    assert result.candidate_id == "cv_001"
    assert result.raw_score == 87.0

def test_judge_evidence_chain_has_items():
    mock_llm = MagicMock(return_value=_make_assessment())
    with patch("src.agents.candidate_judge.get_llm", return_value=mock_llm):
        agent = CandidateJudgeAgent()
        result = agent.run(_make_enriched(), _make_jd())
    assert len(result.evidence_chain) > 0
    assert result.evidence_chain[0].evidence_quote != ""
