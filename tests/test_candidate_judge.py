from unittest.mock import MagicMock, patch
from src.agents.candidate_judge import CandidateJudgeAgent
from src.models.schemas import CandidateBasicInfo, EducationRequirement
from tests.conftest import make_candidate_profile, make_jd_requirements, make_candidate_assessment


def _make_profile(candidate_id: str):
    return make_candidate_profile(
        candidate_id,
        basic_info=CandidateBasicInfo(full_name="Daniel Adif Nugroho"),
        total_experience_months=48,
    )


def _make_jd():
    return make_jd_requirements(
        role_title="Data Engineer", min_years_experience=2,
        education=EducationRequirement(degree="Bachelor's", field="CS", is_mandatory=True),
        industry_context="tech", raw_jd_hash="hash123",
    )


def _make_mock_assessment(candidate_id: str):
    return make_candidate_assessment(
        candidate_id,
        key_strengths=[],
    )


def test_agent_run_overrides_llm_candidate_id_with_provided_id():
    profile = _make_profile("Daniel Adif Nugroho Resume")
    mock_assessment = _make_mock_assessment("Daniel Adif Nugroho")
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = mock_assessment

    with patch("src.agents.candidate_judge.get_llm", return_value=mock_llm):
        agent = CandidateJudgeAgent()
        result = agent.run(profile, _make_jd(), raw_cv_text="Python, SQL")

    assert result.candidate_id == "Daniel Adif Nugroho Resume"
