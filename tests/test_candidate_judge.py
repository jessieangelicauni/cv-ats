from unittest.mock import MagicMock, patch
from src.agents.candidate_judge import CandidateJudgeAgent
from src.models.schemas import CandidateBasicInfo, EducationRequirement, HallucinationFlag
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


def _make_fabricated_flag(assessment):
    item = assessment.evidence_chain[0]
    return HallucinationFlag(
        candidate_id=assessment.candidate_id,
        claim=item.assessment,
        status="fabricated",
        source_quote=item.evidence_quote,
    )


@patch("src.agents.candidate_judge.get_llm")
@patch("src.agents.candidate_judge.verify_evidence_chain")
def test_agent_retries_once_then_returns_grounded_assessment(mock_verify, mock_get_llm):
    profile = _make_profile("cv_001")
    fabricated = _make_mock_assessment("cv_001")
    grounded = _make_mock_assessment("cv_001")
    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = [fabricated, grounded]
    mock_get_llm.return_value = mock_llm
    mock_verify.side_effect = [[_make_fabricated_flag(fabricated)], []]

    agent = CandidateJudgeAgent()
    result = agent.run(profile, _make_jd(), raw_cv_text="irrelevant for this test")

    assert mock_llm.invoke.call_count == 2
    assert result.candidate_id == "cv_001"
    assert all(item.evidence_quote != "NOT FOUND IN CV" for item in result.evidence_chain)
    second_call_messages = mock_llm.invoke.call_args_list[1][0][0]
    assert len(second_call_messages) == 4


@patch("src.agents.candidate_judge.get_llm")
@patch("src.agents.candidate_judge.verify_evidence_chain")
def test_agent_forces_gap_after_exhausting_retries(mock_verify, mock_get_llm):
    profile = _make_profile("cv_001")
    fabricated = _make_mock_assessment("cv_001")
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = fabricated
    mock_get_llm.return_value = mock_llm
    mock_verify.return_value = [_make_fabricated_flag(fabricated)]

    with patch("config.JUDGE_MAX_RETRIES", 2):
        agent = CandidateJudgeAgent()
        result = agent.run(profile, _make_jd(), raw_cv_text="irrelevant for this test")

    assert mock_llm.invoke.call_count == 3
    assert result.evidence_chain[0].evidence_quote == "NOT FOUND IN CV"
    assert result.evidence_chain[0].dimension_score <= 3.0
