from unittest.mock import MagicMock, patch
from src.graph.pipeline import run_pipeline
from src.models.schemas import (
    JDRequirements, EducationRequirement,
    CandidateProfile, CandidateBasicInfo,
    CandidateAssessment, EvidenceItem, FinalRanking, RankedCandidate,
)


def _mock_jd() -> JDRequirements:
    return JDRequirements(
        role_title="Engineer", seniority_level="mid",
        required_skills=[], preferred_skills=[], min_years_experience=3,
        education=EducationRequirement(degree="BSc", field="CS", is_mandatory=False),
        domain_expertise=[], leadership_expected=False,
        soft_skills=[], industry_context="IT", raw_jd_hash="abc",
    )


def _mock_profile(cid: str) -> CandidateProfile:
    return CandidateProfile(
        candidate_id=cid,
        basic_info=CandidateBasicInfo(full_name=None, email=None, phone=None,
                                       location=None, linkedin_url=None, current_title=None),
        skills=[], work_history=[], education=[],
        certifications=[], languages=[], total_experience_months=36,
    )


def _mock_assessment(cid: str) -> CandidateAssessment:
    return CandidateAssessment(
        candidate_id=cid, raw_score=75.0, confidence="medium",
        evidence_chain=[EvidenceItem(dimension="Technical Skills Fit",
                                      assessment="OK", evidence_quote="Python",
                                      dimension_score=7.5)],
        key_strengths=["Python"], key_gaps=[], seniority_alignment="aligned",
    )


def _mock_ranking() -> FinalRanking:
    return FinalRanking(
        ranked_candidates=[
            RankedCandidate(rank=1, candidate_id="cv_001", calibrated_score=77.0,
                            delta_from_raw=2.0, comparative_notes="Best fit."),
            RankedCandidate(rank=2, candidate_id="cv_002", calibrated_score=73.0,
                            delta_from_raw=-2.0, comparative_notes="Weaker skills."),
        ],
        pool_summary="Two candidates evaluated.",
        calibration_rationale="Scores spread from 73 to 77.",
        borderline_pairs=[],
    )


def test_pipeline_produces_final_ranking():
    with (
        patch("src.graph.nodes.JDParserAgent") as MockJD,
        patch("src.graph.nodes.CVExtractorAgent") as MockCV,
        patch("src.graph.nodes.CandidateJudgeAgent") as MockJ,
        patch("src.graph.nodes.PoolCalibratorAgent") as MockPC,
    ):
        MockJD.return_value.run.return_value = _mock_jd()
        MockCV.return_value.run.side_effect = [_mock_profile("cv_001"), _mock_profile("cv_002")]
        MockJ.return_value.run.side_effect = [_mock_assessment("cv_001"), _mock_assessment("cv_002")]
        MockPC.return_value.run.return_value = _mock_ranking()

        result = run_pipeline(
            jd_raw="Software Engineer role",
            cv_raws=[
                {"raw_text": "CV1 text", "candidate_id": "cv_001", "source_file": "cv_001.pdf"},
                {"raw_text": "CV2 text", "candidate_id": "cv_002", "source_file": "cv_002.pdf"},
            ],
            use_cache=False,
        )

    assert result.final_ranking is not None
    assert len(result.final_ranking.ranked_candidates) == 2
    assert result.final_ranking.ranked_candidates[0].candidate_id == "cv_001"
    assert isinstance(result.eliminated_candidates, list)


def test_pipeline_trace_log_has_four_entries():
    with (
        patch("src.graph.nodes.JDParserAgent") as MockJD,
        patch("src.graph.nodes.CVExtractorAgent") as MockCV,
        patch("src.graph.nodes.CandidateJudgeAgent") as MockJ,
        patch("src.graph.nodes.PoolCalibratorAgent") as MockPC,
    ):
        MockJD.return_value.run.return_value = _mock_jd()
        MockCV.return_value.run.side_effect = [_mock_profile("cv_001")]
        MockJ.return_value.run.side_effect = [_mock_assessment("cv_001")]
        MockPC.return_value.run.return_value = FinalRanking(
            ranked_candidates=[RankedCandidate(rank=1, candidate_id="cv_001",
                               calibrated_score=75.0, delta_from_raw=0.0,
                               comparative_notes="Only candidate.")],
            pool_summary="One candidate.", calibration_rationale="N/A", borderline_pairs=[],
        )

        result = run_pipeline(
            jd_raw="Engineer role",
            cv_raws=[{"raw_text": "CV text", "candidate_id": "cv_001", "source_file": "cv.pdf"}],
            use_cache=False,
        )

    assert len(result.trace_log) == 4
