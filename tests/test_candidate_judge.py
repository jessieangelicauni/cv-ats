from unittest.mock import MagicMock, patch
from src.agents.candidate_judge import CandidateJudgeAgent
from src.models.schemas import (
    CandidateAssessment, EvidenceItem, CandidateProfile,
    CandidateBasicInfo, JDRequirements, EducationRequirement,
)
from src.utils.skill_matcher import SkillMatchResult


def _make_profile() -> CandidateProfile:
    return CandidateProfile(
        candidate_id="cv_001",
        basic_info=CandidateBasicInfo(
            full_name="Ahmad", email=None, phone=None,
            location=None, linkedin_url=None, current_title="Engineer",
        ),
        skills=[], work_history=[], education=[],
        certifications=[], languages=[], total_experience_months=60,
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
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = _make_assessment()
    with patch("src.agents.candidate_judge.get_llm", return_value=mock_llm):
        agent = CandidateJudgeAgent()
        result = agent.run(_make_profile(), _make_jd())
    assert isinstance(result, CandidateAssessment)
    assert result.candidate_id == "cv_001"
    assert result.raw_score == 87.0


def test_judge_evidence_chain_has_items():
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = _make_assessment()
    with patch("src.agents.candidate_judge.get_llm", return_value=mock_llm):
        agent = CandidateJudgeAgent()
        result = agent.run(_make_profile(), _make_jd())
    assert len(result.evidence_chain) > 0
    assert result.evidence_chain[0].evidence_quote != ""


def test_judge_prompt_includes_skill_table_when_matches_provided():
    from src.prompts import judge as prompts
    skill_matches = [
        SkillMatchResult(jd_skill="PostgreSQL", best_match="Postgres", score=0.97, is_required=True),
        SkillMatchResult(jd_skill="Kubernetes", best_match="K8s", score=0.94, is_required=True),
    ]
    result = prompts.human(
        jd_json='{"role": "Engineer"}',
        profile_json='{"name": "Ahmad"}',
        skill_matches=skill_matches,
    )
    assert "SKILL COVERAGE" in result
    assert "PostgreSQL" in result
    assert "Postgres" in result


def test_judge_prompt_includes_cv_excerpts_when_chunks_provided():
    from src.prompts import judge as prompts
    result = prompts.human(
        jd_json='{"role": "Engineer"}',
        profile_json='{"name": "Ahmad"}',
        context_chunks=["Led backend team of 5 engineers", "Python and Go stack"],
    )
    assert "RELEVANT CV EXCERPTS" in result
    assert "Led backend team" in result


def test_judge_prompt_unchanged_when_no_optional_params():
    from src.prompts import judge as prompts
    result = prompts.human(
        jd_json='{"role": "Engineer"}',
        profile_json='{"name": "Ahmad"}',
    )
    assert "SKILL COVERAGE" not in result
    assert "RELEVANT CV EXCERPTS" not in result
    assert "Assess across these dimensions" in result
