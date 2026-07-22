from unittest.mock import MagicMock, patch
from src.agents.pool_calibrator import PoolCalibratorAgent
from src.models.schemas import FinalRanking, PoolCalibrationResult, CalibratedEntry, CandidateAssessment
from tests.conftest import make_candidate_assessment, make_jd_requirements


def _make_assessments() -> list[CandidateAssessment]:
    return [
        make_candidate_assessment("cv_001", 87.0),
        make_candidate_assessment("cv_002", 74.0),
        make_candidate_assessment("cv_003", 79.0),
    ]


def _make_jd():
    return make_jd_requirements(role_title="Senior Backend Engineer", seniority_level="senior", min_years_experience=5)


def _make_calibration_result() -> PoolCalibrationResult:
    return PoolCalibrationResult(
        calibrated_entries=[
            CalibratedEntry(position=1, calibrated_score=91.0, delta_from_raw=4.0,
                            comparative_notes="Strongest overall."),
            CalibratedEntry(position=3, calibrated_score=79.0, delta_from_raw=0.0,
                            comparative_notes="Solid mid-range."),
            CalibratedEntry(position=2, calibrated_score=70.0, delta_from_raw=-4.0,
                            comparative_notes="Weakest skills fit."),
        ],
        pool_summary="Strong pool with clear differentiation.",
        calibration_rationale="Spread increased from 13 to 21 points.",
        borderline_pairs=[],
    )


def test_calibrator_returns_final_ranking():
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = _make_calibration_result()
    with patch("src.agents.pool_calibrator.get_llm", return_value=mock_llm):
        result = PoolCalibratorAgent().run(_make_assessments(), _make_jd())
    assert isinstance(result, FinalRanking)
    assert len(result.ranked_candidates) == 3
    assert result.ranked_candidates[0].rank == 1
    assert result.ranked_candidates[0].candidate_id == "cv_001"


def test_calibrator_records_delta_from_raw():
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = _make_calibration_result()
    with patch("src.agents.pool_calibrator.get_llm", return_value=mock_llm):
        result = PoolCalibratorAgent().run(_make_assessments(), _make_jd())
    assert result.ranked_candidates[0].delta_from_raw == 4.0


def test_out_of_range_position_is_ignored():
    """LLM returning a position outside 1-N is silently dropped."""
    result_with_invalid = PoolCalibrationResult(
        calibrated_entries=[
            CalibratedEntry(position=1, calibrated_score=91.0, delta_from_raw=4.0,
                            comparative_notes="Best."),
            CalibratedEntry(position=99, calibrated_score=85.0, delta_from_raw=0.0,
                            comparative_notes="Out of range."),
            CalibratedEntry(position=2, calibrated_score=70.0, delta_from_raw=-4.0,
                            comparative_notes="Weakest."),
        ],
        pool_summary="Summary.", calibration_rationale="Rationale.",
        borderline_pairs=[],
    )
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = result_with_invalid
    with patch("src.agents.pool_calibrator.get_llm", return_value=mock_llm):
        result = PoolCalibratorAgent().run(_make_assessments(), _make_jd())

    ids = [rc.candidate_id for rc in result.ranked_candidates]
    assert len(result.ranked_candidates) == 2
    assert "cv_001" in ids
    assert "cv_002" in ids
