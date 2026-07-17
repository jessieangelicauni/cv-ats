# tests/test_pipeline_ablation.py
from src.graph.pipeline import _ranking_from_raw
from src.models.schemas import CandidateAssessment, EvidenceItem


def _make_assessment(cid: str, score: float) -> CandidateAssessment:
    return CandidateAssessment(
        candidate_id=cid, raw_score=score, confidence="high",
        evidence_chain=[EvidenceItem(
            dimension="Technical Skills Fit", assessment="OK.",
            evidence_quote="Python", dimension_score=7.0,
        )],
        key_strengths=[], key_gaps=[], seniority_alignment="aligned",
    )


def test_ranking_from_raw_sorts_descending():
    assessments = [
        _make_assessment("a", 70.0),
        _make_assessment("b", 90.0),
        _make_assessment("c", 80.0),
    ]
    result = _ranking_from_raw(assessments)
    ids = [r.candidate_id for r in result.ranked_candidates]
    assert ids == ["b", "c", "a"]


def test_ranking_from_raw_calibrated_equals_raw():
    assessments = [_make_assessment("a", 70.0), _make_assessment("b", 90.0)]
    result = _ranking_from_raw(assessments)
    score_map = {r.candidate_id: r.calibrated_score for r in result.ranked_candidates}
    assert score_map["a"] == 70.0
    assert score_map["b"] == 90.0


def test_ranking_from_raw_delta_is_zero():
    assessments = [_make_assessment("a", 70.0), _make_assessment("b", 90.0)]
    result = _ranking_from_raw(assessments)
    assert all(r.delta_from_raw == 0.0 for r in result.ranked_candidates)


def test_ranking_from_raw_assigns_ranks():
    assessments = [_make_assessment("a", 70.0), _make_assessment("b", 90.0)]
    result = _ranking_from_raw(assessments)
    ranks = [r.rank for r in result.ranked_candidates]
    assert ranks == [1, 2]


def test_ranking_from_raw_pool_summary():
    assessments = [_make_assessment("a", 70.0)]
    result = _ranking_from_raw(assessments)
    assert "No calibration" in result.pool_summary
    assert result.borderline_pairs == []
