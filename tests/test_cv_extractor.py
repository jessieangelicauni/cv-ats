from datetime import date
from unittest.mock import MagicMock, patch
from src.agents.cv_extractor import CVExtractorAgent
from src.models.schemas import (
    CandidateProfile, CandidateBasicInfo, SkillEntry,
    WorkEntry,
)
from src.prompts.cv_extractor import human_2a

SAMPLE_CV = """
Ahmad Faris Bin Razak
Senior Backend Engineer | Kuala Lumpur
ahmad@email.com

EXPERIENCE
Senior Backend Engineer — Google, Jan 2022 – Present
- Led backend migration using Python and gRPC
- Maintained postgres cluster serving 10M users

SKILLS
Python, postgres, Docker
"""

def _make_mock_profile() -> CandidateProfile:
    return CandidateProfile(
        candidate_id="cv_001",
        basic_info=CandidateBasicInfo(
            full_name="Ahmad Faris Bin Razak",
            email="ahmad@email.com",
            phone=None, location="Kuala Lumpur",
            linkedin_url=None, current_title="Senior Backend Engineer",
        ),
        skills=[
            SkillEntry(raw_mention="Python", proficiency="expert",
                       evidence_quote="Python and gRPC"),
            SkillEntry(raw_mention="postgres", proficiency="advanced",
                       evidence_quote="Maintained postgres cluster"),
            SkillEntry(raw_mention="Docker", proficiency="intermediate",
                       evidence_quote="Docker"),
        ],
        work_history=[
            WorkEntry(
                company="Google", role="Senior Backend Engineer",
                tenure_months=30, technologies=["Python", "gRPC", "postgres"],
                achievements=["Led backend migration using Python and gRPC",
                              "Maintained postgres cluster serving 10M users"],
                has_leadership_indicators=True,
            )
        ],
        education=[], certifications=[], languages=[],
        total_experience_months=30,
    )


def test_cv_extractor_returns_candidate_profile():
    mock_extract_llm = MagicMock()
    mock_extract_llm.invoke.return_value = _make_mock_profile()

    with patch("src.agents.cv_extractor.get_llm", return_value=mock_extract_llm):
        agent = CVExtractorAgent()
        result = agent.run({"raw_text": SAMPLE_CV, "candidate_id": "cv_001", "source_file": "cv_001.pdf"})

    assert isinstance(result, CandidateProfile)
    assert result.candidate_id == "cv_001"
    assert result.basic_info.full_name == "Ahmad Faris Bin Razak"


def test_cv_extractor_preserves_raw_mentions():
    mock_extract_llm = MagicMock()
    mock_extract_llm.invoke.return_value = _make_mock_profile()

    with patch("src.agents.cv_extractor.get_llm", return_value=mock_extract_llm):
        agent = CVExtractorAgent()
        result = agent.run({"raw_text": SAMPLE_CV, "candidate_id": "cv_001", "source_file": "cv_001.pdf"})

    postgres_skill = next(s for s in result.skills if s.raw_mention == "postgres")
    assert postgres_skill.raw_mention == "postgres"


def test_agent_run_overrides_llm_candidate_id_with_provided_id():
    mock_profile = _make_mock_profile()
    mock_profile.candidate_id = "Daniel Adif Nugroho"
    mock_extract_llm = MagicMock()
    mock_extract_llm.invoke.return_value = mock_profile

    with patch("src.agents.cv_extractor.get_llm", return_value=mock_extract_llm):
        agent = CVExtractorAgent()
        result = agent.run({
            "raw_text": SAMPLE_CV,
            "candidate_id": "Daniel Adif Nugroho Resume",
            "source_file": "Daniel Adif Nugroho Resume.pdf",
        })

    assert result.candidate_id == "Daniel Adif Nugroho Resume"


def test_agent_run_overrides_candidate_id_on_cache_hit_too():
    mock_cache = MagicMock()
    mock_cache.get.return_value = _make_mock_profile().model_dump()

    with patch("src.agents.cv_extractor.get_llm"):
        agent = CVExtractorAgent(cache=mock_cache)
        result = agent.run({
            "raw_text": SAMPLE_CV,
            "candidate_id": "Daniel Adif Nugroho Resume",
            "source_file": "Daniel Adif Nugroho Resume.pdf",
        })

    assert result.candidate_id == "Daniel Adif Nugroho Resume"


def test_cache_key_changes_when_system_prompt_changes():
    mock_cache = MagicMock()
    mock_cache.get.return_value = None
    cv_raw = {
        "raw_text": SAMPLE_CV,
        "candidate_id": "cv_001",
        "source_file": "cv_001.pdf",
    }

    with patch("src.agents.cv_extractor.get_llm") as mock_get_llm:
        mock_get_llm.return_value.invoke.return_value = _make_mock_profile()
        agent = CVExtractorAgent(cache=mock_cache)
        agent.run(cv_raw)
        key_before = mock_cache.get.call_args[0][0]

        with patch("src.prompts.cv_extractor.SYSTEM_2A", "a different prompt"):
            mock_cache.reset_mock()
            mock_cache.get.return_value = None
            agent.run(cv_raw)
            key_after = mock_cache.get.call_args[0][0]

    assert key_before != key_after


def test_human_2a_includes_reference_date():
    fixed_date = date(2026, 1, 15)
    with patch("src.prompts.cv_extractor.date") as mock_date:
        mock_date.today.return_value = fixed_date
        result = human_2a("CV text here")

    assert "REFERENCE DATE: 2026-01-15" in result


def test_human_2a_does_not_leak_candidate_id_hint():
    result = human_2a("CV text here")
    assert "CANDIDATE_ID" not in result
