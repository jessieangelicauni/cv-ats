from src.models.schemas import (
    JDRequirements, SkillRequirement, EducationRequirement,
    CandidateProfile, CandidateBasicInfo, SkillEntry, SkillMatchResult,
    WorkEntry, EducationEntry, LanguageEntry,
    CandidateAssessment, EvidenceItem, HallucinationFlag,
    FinalRanking, RankedCandidate,
)


def test_jd_requirements_rejects_invalid_seniority():
    import pytest
    with pytest.raises(Exception):
        JDRequirements(
            role_title="Dev",
            seniority_level="wizard",
            required_skills=[], preferred_skills=[],
            min_years_experience=3,
            education=EducationRequirement(degree="BSc", field="CS", is_mandatory=False),
            domain_expertise=[], leadership_expected=False,
            soft_skills=[], industry_context="IT", raw_jd_hash="abc",
        )


def test_skill_entry_has_raw_mention():
    entry = SkillEntry(
        raw_mention="postgres",
        proficiency="advanced",
        evidence_quote="managed postgres cluster",
    )
    assert entry.raw_mention == "postgres"


def test_skill_match_result_structure():
    m = SkillMatchResult(jd_skill="Python", best_match="python", score=1.0, is_required=True)
    assert m.jd_skill == "Python"
    assert m.best_match == "python"
    assert m.score == 1.0
    assert m.is_required is True


def test_candidate_assessment_evidence_chain():
    item = EvidenceItem(
        dimension="Technical Skills Fit",
        assessment="Strong Python skills evident.",
        evidence_quote="5 years Python development",
        dimension_score=9.0,
    )
    assert item.dimension_score == 9.0


def test_final_ranking_ranked_candidates():
    rc = RankedCandidate(
        rank=1, candidate_id="cv_001",
        calibrated_score=88.0, delta_from_raw=3.0,
        comparative_notes="Ranks above #2 due to Tier 1 MNC experience.",
    )
    fr = FinalRanking(
        ranked_candidates=[rc],
        pool_summary="Strong pool overall.",
        calibration_rationale="Phase 5 spread scores from 70-88.",
        borderline_pairs=[],
    )
    assert fr.ranked_candidates[0].rank == 1
