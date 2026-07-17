from src.utils.skill_matcher import SkillMatcher, SkillMatchResult
from src.models.schemas import SkillRequirement, SkillEntry
from src.utils.embedder import get_embedder


def _req(skill: str, mandatory: bool = True) -> SkillRequirement:
    return SkillRequirement(skill=skill, level="advanced", is_mandatory=mandatory)


def _entry(canonical: str) -> SkillEntry:
    return SkillEntry(
        raw_mention=canonical.lower(),
        canonical_skill=canonical,
        proficiency="advanced",
        evidence_quote="evidence",
    )


def test_synonym_skills_score_high():
    matcher = SkillMatcher(get_embedder())
    results = matcher.match(
        jd_skills=[_req("PostgreSQL")],
        candidate_skills=[_entry("Postgres")],
    )
    assert len(results) == 1
    assert results[0].score >= 0.75


def test_unrelated_skills_score_low():
    matcher = SkillMatcher(get_embedder())
    results = matcher.match(
        jd_skills=[_req("Kubernetes")],
        candidate_skills=[_entry("Microsoft Excel")],
    )
    assert results[0].score < 0.75


def test_is_required_flag_set_correctly():
    matcher = SkillMatcher(get_embedder())
    results = matcher.match(
        jd_skills=[_req("Python", mandatory=True), _req("Figma", mandatory=False)],
        candidate_skills=[_entry("Python")],
    )
    required_result = next(r for r in results if r.jd_skill == "Python")
    optional_result = next(r for r in results if r.jd_skill == "Figma")
    assert required_result.is_required is True
    assert optional_result.is_required is False


def test_empty_candidate_skills_returns_no_match():
    matcher = SkillMatcher(get_embedder())
    results = matcher.match(
        jd_skills=[_req("Python")],
        candidate_skills=[],
    )
    assert results[0].best_match is None
    assert results[0].score == 0.0


def test_returns_one_result_per_jd_skill():
    matcher = SkillMatcher(get_embedder())
    results = matcher.match(
        jd_skills=[_req("Python"), _req("Go"), _req("Rust")],
        candidate_skills=[_entry("Python"), _entry("Golang")],
    )
    assert len(results) == 3
