from unittest.mock import MagicMock, patch
from src.agents.signal_enricher import SignalEnricherAgent
from src.models.schemas import (
    EnrichedProfile, EnrichmentSignals,
    CandidateProfile, CandidateBasicInfo, JDRequirements,
    SkillRequirement, EducationRequirement,
)

def _make_profile() -> CandidateProfile:
    return CandidateProfile(
        candidate_id="cv_001",
        basic_info=CandidateBasicInfo(
            full_name="Ahmad Faris", email=None, phone=None,
            location=None, linkedin_url=None, current_title="Senior Engineer",
        ),
        skills=[], work_history=[], education=[],
        certifications=[], languages=[], total_experience_months=60,
    )

def _make_jd() -> JDRequirements:
    return JDRequirements(
        role_title="Senior Backend Engineer", seniority_level="senior",
        required_skills=[], preferred_skills=[], min_years_experience=5,
        education=EducationRequirement(degree="BSc", field="CS", is_mandatory=False),
        domain_expertise=["backend"], leadership_expected=True,
        soft_skills=[], industry_context="IT", raw_jd_hash="abc123",
    )

def _make_signals() -> EnrichmentSignals:
    return EnrichmentSignals(
        company_tiers=["tier1_mnc"],
        highest_prestige_company="Google",
        career_trajectory="ascending",
        leadership_count=3,
        measurable_impact_count=4,
        tenure_stability="stable",
        relevant_experience_months=60,
    )

def test_signal_enricher_returns_enriched_profile():
    mock_llm = MagicMock(return_value=_make_signals())
    with patch("src.agents.signal_enricher.get_llm", return_value=mock_llm):
        agent = SignalEnricherAgent()
        result = agent.run(_make_profile(), _make_jd())
    assert isinstance(result, EnrichedProfile)
    assert result.highest_prestige_company == "Google"
    assert result.career_trajectory == "ascending"
    assert result.candidate_id == "cv_001"


def test_signal_enricher_preserves_profile_fields():
    mock_llm = MagicMock(return_value=_make_signals())
    with patch("src.agents.signal_enricher.get_llm", return_value=mock_llm):
        agent = SignalEnricherAgent()
        result = agent.run(_make_profile(), _make_jd())
    assert result.basic_info.full_name == "Ahmad Faris"
    assert result.total_experience_months == 60
