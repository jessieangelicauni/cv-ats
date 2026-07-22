from unittest.mock import MagicMock, patch
from src.agents.candidate_judge import CandidateJudgeAgent
from src.models.schemas import CandidateBasicInfo, EducationRequirement, EvidenceItem, HallucinationFlag
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


@patch("src.agents.candidate_judge.verify_evidence_chain", return_value=[])
def test_agent_run_overrides_llm_candidate_id_with_provided_id(mock_verify):
    profile = _make_profile("Daniel Adif Nugroho Resume")
    mock_assessment = _make_mock_assessment("Daniel Adif Nugroho")
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = mock_assessment

    with patch("src.agents.candidate_judge.get_llm", return_value=mock_llm):
        agent = CandidateJudgeAgent()
        result = agent.run(profile, _make_jd(), raw_cv_text="Python, SQL")

    assert result.candidate_id == "Daniel Adif Nugroho Resume"
    assert mock_llm.invoke.call_count == 1


def _make_flag(assessment, status: str):
    item = assessment.evidence_chain[0]
    return HallucinationFlag(
        candidate_id=assessment.candidate_id,
        claim=item.assessment,
        status=status,
        source_quote=item.evidence_quote,
    )


def _make_fabricated_flag(assessment):
    return _make_flag(assessment, "fabricated")


@patch("src.agents.candidate_judge.get_llm")
@patch("src.agents.candidate_judge.verify_evidence_chain")
def test_agent_retries_once_then_returns_grounded_assessment(mock_verify, mock_get_llm):
    profile = _make_profile("cv_001")
    fabricated = _make_mock_assessment("cv_001")
    grounded = _make_mock_assessment("cv_001")
    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = [fabricated, grounded]
    mock_get_llm.return_value = mock_llm
    mock_verify.side_effect = [
        [_make_flag(fabricated, "fabricated")],
        [_make_flag(grounded, "inferred")],
    ]

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


@patch("src.agents.candidate_judge.get_llm")
@patch("src.agents.candidate_judge.verify_evidence_chain")
def test_agent_keeps_dimension_unresolved_while_a_sibling_item_is_still_fabricated(mock_verify, mock_get_llm):
    """Regression test: a dimension can have more than one evidence_chain
    item (the judge is free to add several per dimension). If only one of
    two items under the same dimension becomes grounded, the dimension must
    not be considered resolved while its sibling item is still fabricated --
    otherwise that sibling's fabricated content ships untouched."""
    profile = _make_profile("cv_001")

    def _two_item_assessment():
        return make_candidate_assessment(
            "cv_001",
            key_strengths=[],
            evidence_chain=[
                EvidenceItem(dimension="Technical Skills Fit", assessment="Claim A",
                             evidence_quote="Quote A", dimension_score=8.0),
                EvidenceItem(dimension="Technical Skills Fit", assessment="Claim B",
                             evidence_quote="Quote B", dimension_score=8.0),
            ],
        )

    first = _two_item_assessment()
    second = _two_item_assessment()
    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = [first, second, second]
    mock_get_llm.return_value = mock_llm

    def _flags():
        return [
            HallucinationFlag(candidate_id="cv_001", claim="Claim A", status="fabricated", source_quote="Quote A"),
            HallucinationFlag(candidate_id="cv_001", claim="Claim B", status="inferred", source_quote="Quote B"),
        ]

    mock_verify.side_effect = [_flags(), _flags(), _flags()]

    with patch("config.JUDGE_MAX_RETRIES", 2):
        agent = CandidateJudgeAgent()
        result = agent.run(profile, _make_jd(), raw_cv_text="irrelevant for this test")

    assert mock_llm.invoke.call_count == 3
    claim_a_item = next(i for i in result.evidence_chain if i.assessment == "Claim A")
    claim_b_item = next(i for i in result.evidence_chain if i.assessment == "Claim B")
    assert claim_a_item.evidence_quote == "NOT FOUND IN CV"
    assert claim_a_item.dimension_score <= 3.0
    assert claim_b_item.evidence_quote == "Quote B"
    assert claim_b_item.dimension_score == 8.0


@patch("src.agents.candidate_judge.get_llm")
@patch("src.agents.candidate_judge.verify_evidence_chain")
def test_agent_does_not_accept_self_declared_gap_as_resolved(mock_verify, mock_get_llm):
    """Regression test: a model that gives up mid-retry by writing its own
    "NOT FOUND IN CV" (status acknowledged_gap) must NOT be treated as
    resolved -- only our own exhaustion fallback may grant a gap, and it
    must still cap the score, unlike a model's own unguarded rewrite."""
    profile = _make_profile("cv_001")
    fabricated = _make_mock_assessment("cv_001")
    gapped = _make_mock_assessment("cv_001")
    gapped.evidence_chain[0].evidence_quote = "NOT FOUND IN CV"
    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = [fabricated, gapped, gapped]
    mock_get_llm.return_value = mock_llm
    mock_verify.side_effect = [
        [_make_flag(fabricated, "fabricated")],
        [_make_flag(gapped, "acknowledged_gap")],
        [_make_flag(gapped, "acknowledged_gap")],
    ]

    with patch("config.JUDGE_MAX_RETRIES", 2):
        agent = CandidateJudgeAgent()
        result = agent.run(profile, _make_jd(), raw_cv_text="irrelevant for this test")

    assert mock_llm.invoke.call_count == 3
    assert result.evidence_chain[0].evidence_quote == "NOT FOUND IN CV"
    assert result.evidence_chain[0].dimension_score <= 3.0
