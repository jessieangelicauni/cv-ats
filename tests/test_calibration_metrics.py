from src.evaluation.calibration_metrics import calibration_report
from src.models.schemas import CandidateAssessment, EvidenceItem, FinalRanking, RankedCandidate

def _make_assessments() -> list[CandidateAssessment]:
    def make(cid, score):
        return CandidateAssessment(
            candidate_id=cid, raw_score=score, confidence="high",
            evidence_chain=[], key_strengths=[], key_gaps=[],
            seniority_alignment="aligned",
        )
    return [make("cv_001", 87.0), make("cv_002", 74.0), make("cv_003", 70.0)]

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
    # deltas: 4, 4, -9 → abs: 4, 4, 9 → mean = 17/3
    assert abs(report["mean_abs_delta"] - 17/3) < 0.01

def test_rank_changes_counts_position_changes():
    # raw ranks: cv_001=1, cv_002=2, cv_003=3
    # calibrated ranks: cv_001=1, cv_003=2, cv_002=3
    # cv_002 and cv_003 swapped → 2 rank changes
    report = calibration_report(_make_assessments(), _make_ranking())
    assert report["rank_changes"] == 2
