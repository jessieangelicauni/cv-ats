from unittest.mock import MagicMock, patch
from src.agents.cv_extractor import CVExtractorAgent
from src.models.schemas import (
    CandidateProfile, CandidateBasicInfo, SkillEntry,
    WorkEntry,
)

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
            SkillEntry(raw_mention="Python", canonical_skill="Python",
                       proficiency="expert", evidence_quote="Python and gRPC"),
            SkillEntry(raw_mention="postgres", canonical_skill="postgres",
                       proficiency="advanced", evidence_quote="Maintained postgres cluster"),
            SkillEntry(raw_mention="Docker", canonical_skill="Docker",
                       proficiency="intermediate", evidence_quote="Docker"),
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
    mock_extract_llm = MagicMock(return_value=_make_mock_profile())

    with patch("src.agents.cv_extractor.get_llm", return_value=mock_extract_llm), \
         patch("src.agents.cv_extractor.invoke_with_telemetry", return_value=_make_mock_profile()):
        agent = CVExtractorAgent()
        result = agent.run({"raw_text": SAMPLE_CV, "candidate_id": "cv_001", "source_file": "cv_001.pdf"})

    assert isinstance(result, CandidateProfile)
    assert result.candidate_id == "cv_001"
    assert result.basic_info.full_name == "Ahmad Faris Bin Razak"


def test_cv_extractor_applies_taxonomy_canonicalization():
    """Taxonomy normalizes 'postgres' -> 'postgresql'."""
    mock_extract_llm = MagicMock(return_value=_make_mock_profile())

    with patch("src.agents.cv_extractor.get_llm", return_value=mock_extract_llm), \
         patch("src.agents.cv_extractor.invoke_with_telemetry", return_value=_make_mock_profile()):
        agent = CVExtractorAgent()
        result = agent.run({"raw_text": SAMPLE_CV, "candidate_id": "cv_001", "source_file": "cv_001.pdf"})

    postgres_skill = next(s for s in result.skills if s.raw_mention == "postgres")
    assert postgres_skill.canonical_skill == "postgresql"
    assert postgres_skill.raw_mention == "postgres"
