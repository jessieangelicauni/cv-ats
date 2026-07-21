from unittest.mock import MagicMock, patch
from src.agents.candidate_judge import CandidateJudgeAgent
from src.models.schemas import (
    CandidateAssessment, CandidateProfile, CandidateBasicInfo,
    EvidenceItem, JDRequirements, EducationRequirement,
)


def _make_profile(candidate_id: str) -> CandidateProfile:
    return CandidateProfile(
        candidate_id=candidate_id,
        basic_info=CandidateBasicInfo(full_name="Daniel Adif Nugroho"),
        skills=[], work_history=[], education=[], certifications=[],
        languages=[], total_experience_months=48,
    )


def _make_jd() -> JDRequirements:
    return JDRequirements(
        role_title="Data Engineer", seniority_level="mid",
        min_years_experience=2, required_skills=[], preferred_skills=[],
        domain_expertise=[],
        education=EducationRequirement(degree="Bachelor's", field="CS", is_mandatory=True),
        leadership_expected=False, soft_skills=[], industry_context="tech",
        raw_jd_hash="hash123",
    )


def _make_mock_assessment(candidate_id: str) -> CandidateAssessment:
    return CandidateAssessment(
        candidate_id=candidate_id,
        raw_score=75.0,
        confidence="high",
        evidence_chain=[
            EvidenceItem(dimension="Technical Skills Fit", assessment="Good.",
                         evidence_quote="Python, SQL", dimension_score=7.0),
        ],
        key_strengths=[], key_gaps=[], seniority_alignment="aligned",
    )


def test_agent_run_overrides_llm_candidate_id_with_provided_id():
    # Regression test: given the profile's candidate_id
    # ("Daniel Adif Nugroho Resume", derived deterministically from the source
    # filename), the LLM "cleans it up" to the person's actual name ("Daniel Adif
    # Nugroho") instead of copying it verbatim into CandidateAssessment.candidate_id.
    # Downstream, main.py looks up raw CV text via
    # cv_text_map.get(a.candidate_id, "") — a candidate_id mismatch here silently
    # yields an empty string, and verify_evidence_chain then flags every one of
    # the candidate's evidence_chain items as "fabricated", even verbatim quotes.
    # CVExtractorAgent already guards against this exact failure mode for
    # CandidateProfile.candidate_id (see test_cv_extractor.py); CandidateJudgeAgent
    # must do the same for CandidateAssessment.candidate_id.
    profile = _make_profile("Daniel Adif Nugroho Resume")
    mock_assessment = _make_mock_assessment("Daniel Adif Nugroho")  # LLM "cleaned up" the id
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = mock_assessment

    with patch("src.agents.candidate_judge.get_llm", return_value=mock_llm):
        agent = CandidateJudgeAgent()
        result = agent.run(profile, _make_jd(), raw_cv_text="Python, SQL")

    assert result.candidate_id == "Daniel Adif Nugroho Resume"
