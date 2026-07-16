import hashlib
from unittest.mock import MagicMock, patch
from src.agents.jd_parser import JDParserAgent
from src.models.schemas import JDRequirements, SkillRequirement, EducationRequirement

SAMPLE_JD = """
Senior Backend Engineer — required skills: Python (expert, required),
PostgreSQL (advanced, required), Docker (intermediate, nice to have).
Must have 5+ years of experience. BSc Computer Science essential.
Leadership experience preferred.
"""

def _make_mock_jd() -> JDRequirements:
    return JDRequirements(
        role_title="Senior Backend Engineer",
        seniority_level="senior",
        required_skills=[
            SkillRequirement(skill="Python", level="expert", is_mandatory=True),
            SkillRequirement(skill="PostgreSQL", level="advanced", is_mandatory=True),
        ],
        preferred_skills=[
            SkillRequirement(skill="Docker", level="intermediate", is_mandatory=False),
        ],
        min_years_experience=5,
        education=EducationRequirement(degree="BSc", field="Computer Science", is_mandatory=True),
        domain_expertise=["backend", "databases"],
        leadership_expected=True,
        soft_skills=["communication"],
        industry_context="IT software engineering",
        raw_jd_hash=hashlib.sha256(SAMPLE_JD.encode()).hexdigest(),
    )


def test_jd_parser_returns_jd_requirements():
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = _make_mock_jd()
    with patch("src.agents.jd_parser.get_llm", return_value=mock_llm):
        agent = JDParserAgent()
        result = agent.run(SAMPLE_JD)
    assert isinstance(result, JDRequirements)
    assert result.role_title == "Senior Backend Engineer"
    assert result.min_years_experience == 5


def test_jd_parser_uses_cache_on_second_call():
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = _make_mock_jd()
    with patch("src.agents.jd_parser.get_llm", return_value=mock_llm):
        import tempfile
        from pathlib import Path
        from src.utils.cache import ExtractionCache
        with tempfile.TemporaryDirectory() as tmp:
            cache = ExtractionCache(Path(tmp) / "test.db")
            agent = JDParserAgent(cache=cache)
            agent.run(SAMPLE_JD)
            agent.run(SAMPLE_JD)
    assert mock_llm.invoke.call_count == 1
