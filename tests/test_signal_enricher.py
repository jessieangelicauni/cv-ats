from unittest.mock import MagicMock, patch
from src.agents.signal_enricher import SignalEnricherAgent
from src.models.schemas import (
    CandidateProfile, CandidateBasicInfo, JDRequirements,
    SkillRequirement, EducationRequirement,
)
import pytest

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

def test_signal_enricher_is_deprecated():
    """SignalEnricherAgent is no longer used in the pipeline."""
    with pytest.raises(NotImplementedError):
        agent = SignalEnricherAgent()


def test_signal_enricher_run_is_deprecated():
    """SignalEnricherAgent.run() raises NotImplementedError."""
    with pytest.raises(NotImplementedError):
        # This will fail during __init__, but that's expected
        agent = SignalEnricherAgent()
        agent.run(_make_profile(), _make_jd())
