# tests/test_candidate_judge.py
from unittest.mock import MagicMock, patch
from src.agents.candidate_judge import CandidateJudgeAgent
from src.models.schemas import (
    CandidateAssessment, EvidenceItem, CandidateProfile,
    CandidateBasicInfo, JDRequirements, EducationRequirement,
)


def _make_assessment() -> CandidateAssessment:
    return CandidateAssessment(
        candidate_id="cv_001", raw_score=80.0, confidence="high",
        evidence_chain=[EvidenceItem(
            dimension="Technical Skills Fit", assessment="Good.",
            evidence_quote="Python", dimension_score=8.0,
        )],
        key_strengths=["Python"], key_gaps=[], seniority_alignment="aligned",
    )


def _make_profile() -> CandidateProfile:
    return CandidateProfile(
        candidate_id="cv_001",
        basic_info=CandidateBasicInfo(
            full_name="Test", email=None, phone=None,
            location=None, linkedin_url=None, current_title=None,
        ),
        skills=[], work_history=[], education=[],
        certifications=[], languages=[], total_experience_months=24,
    )


def _make_jd() -> JDRequirements:
    return JDRequirements(
        role_title="Engineer", seniority_level="mid",
        required_skills=[], preferred_skills=[], min_years_experience=2,
        education=EducationRequirement(degree="BSc", field="CS", is_mandatory=False),
        domain_expertise=[], leadership_expected=False,
        soft_skills=[], industry_context="IT", raw_jd_hash="abc",
    )


def test_judge_default_uses_evidence_grounding_system_prompt():
    from src.prompts import judge
    mock_llm = MagicMock()
    with patch("src.agents.candidate_judge.get_llm", return_value=mock_llm), \
         patch("src.agents.candidate_judge.invoke_with_telemetry",
               return_value=_make_assessment()) as mock_invoke:
        agent = CandidateJudgeAgent()
        agent.run(_make_profile(), _make_jd())
    messages = mock_invoke.call_args[0][1]
    assert messages[0].content == judge.SYSTEM


def test_judge_no_grounding_uses_no_grounding_system_prompt():
    from src.prompts import judge_no_grounding
    mock_llm = MagicMock()
    with patch("src.agents.candidate_judge.get_llm", return_value=mock_llm), \
         patch("src.agents.candidate_judge.invoke_with_telemetry",
               return_value=_make_assessment()) as mock_invoke:
        agent = CandidateJudgeAgent(use_evidence_grounding=False)
        agent.run(_make_profile(), _make_jd())
    messages = mock_invoke.call_args[0][1]
    assert messages[0].content == judge_no_grounding.SYSTEM


def test_judge_no_grounding_human_message_unchanged():
    """The human() function is reused regardless of grounding flag."""
    mock_llm = MagicMock()
    captured = []

    def capture_invoke(chain, msgs, **kw):
        captured.append(msgs)
        return _make_assessment()

    with patch("src.agents.candidate_judge.get_llm", return_value=mock_llm), \
         patch("src.agents.candidate_judge.invoke_with_telemetry", side_effect=capture_invoke):
        CandidateJudgeAgent(use_evidence_grounding=True).run(_make_profile(), _make_jd())
        CandidateJudgeAgent(use_evidence_grounding=False).run(_make_profile(), _make_jd())

    assert captured[0][1].content == captured[1][1].content
