from src.evaluation.calibration_metrics import calibration_report
from src.models.schemas import CandidateAssessment, FinalRanking, RankedCandidate
from tests.conftest import make_candidate_assessment

def _make_assessments() -> list[CandidateAssessment]:
    return [
        make_candidate_assessment("cv_001", 87.0, evidence_chain=[], key_strengths=[]),
        make_candidate_assessment("cv_002", 74.0, evidence_chain=[], key_strengths=[]),
        make_candidate_assessment("cv_003", 70.0, evidence_chain=[], key_strengths=[]),
    ]

def _make_ranking() -> FinalRanking:
    return FinalRanking(
        ranked_candidates=[
            RankedCandidate(rank=1, candidate_id="cv_001", calibrated_score=91.0, delta_from_raw=4.0, comparative_notes=""),
            RankedCandidate(rank=2, candidate_id="cv_003", calibrated_score=74.0, delta_from_raw=4.0, comparative_notes=""),
            RankedCandidate(rank=3, candidate_id="cv_002", calibrated_score=65.0, delta_from_raw=-9.0, comparative_notes=""),
        ],
        pool_summary="", calibration_rationale="", borderline_pairs=[],
    )

def test_calibration_report_has_required_keys():
    report = calibration_report(_make_assessments(), _make_ranking())
    for key in ["raw_std", "calibrated_std", "raw_range", "mean_abs_delta", "rank_changes", "score_entropy"]:
        assert key in report

def test_mean_abs_delta_computed_correctly():
    report = calibration_report(_make_assessments(), _make_ranking())
    assert abs(report["mean_abs_delta"] - 17/3) < 0.01

def test_rank_changes_counts_position_changes():
    report = calibration_report(_make_assessments(), _make_ranking())
    assert report["rank_changes"] == 2
