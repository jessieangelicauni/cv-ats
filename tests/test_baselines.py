# tests/test_baselines.py
import pytest
from src.evaluation.baselines import tfidf_rank, keyword_rank, run_baselines


SAMPLE_JD = "Python machine learning deep learning neural networks"

SAMPLE_CVS = [
    {"candidate_id": "cv_a", "raw_text": "Java spring boot enterprise microservices"},
    {"candidate_id": "cv_b", "raw_text": "Python scikit-learn deep learning PyTorch neural"},
    {"candidate_id": "cv_c", "raw_text": "Python data analytics pandas machine learning"},
]


def test_tfidf_rank_returns_all_candidates():
    ranking, scores = tfidf_rank(SAMPLE_JD, SAMPLE_CVS)
    assert set(ranking) == {"cv_a", "cv_b", "cv_c"}
    assert set(scores.keys()) == {"cv_a", "cv_b", "cv_c"}


def test_tfidf_rank_most_similar_first():
    ranking, scores = tfidf_rank(SAMPLE_JD, SAMPLE_CVS)
    assert ranking[0] in ("cv_b", "cv_c")   # both Python/ML — either can lead
    assert ranking[-1] == "cv_a"            # Java-only should rank last


def test_tfidf_scores_in_zero_one():
    _, scores = tfidf_rank(SAMPLE_JD, SAMPLE_CVS)
    for v in scores.values():
        assert 0.0 <= v <= 1.0


def test_keyword_rank_returns_all_candidates():
    ranking, counts = keyword_rank(SAMPLE_JD, SAMPLE_CVS)
    assert set(ranking) == {"cv_a", "cv_b", "cv_c"}
    assert set(counts.keys()) == {"cv_a", "cv_b", "cv_c"}


def test_keyword_rank_most_matches_first():
    ranking, counts = keyword_rank(SAMPLE_JD, SAMPLE_CVS)
    assert counts[ranking[0]] >= counts[ranking[-1]]


def test_keyword_java_cv_has_fewer_matches():
    _, counts = keyword_rank(SAMPLE_JD, SAMPLE_CVS)
    assert counts["cv_a"] < counts["cv_b"]


def test_run_baselines_structure():
    er_ranking = ["cv_b", "cv_c", "cv_a"]
    result = run_baselines(SAMPLE_JD, SAMPLE_CVS, er_ranking)

    assert "tfidf" in result
    assert "keyword" in result
    assert "evidencerank" in result
    assert "cross_method_tau" in result

    for method in ("tfidf", "keyword"):
        assert "ranking" in result[method]
        assert "scores" in result[method]
        assert "distribution" in result[method]
        dist = result[method]["distribution"]
        assert set(dist.keys()) == {"mean", "std", "min", "max"}

    assert set(result["cross_method_tau"].keys()) == {
        "tfidf_vs_keyword",
        "tfidf_vs_evidencerank",
        "keyword_vs_evidencerank",
    }


def test_run_baselines_evidencerank_ranking_preserved():
    er_ranking = ["cv_b", "cv_c", "cv_a"]
    result = run_baselines(SAMPLE_JD, SAMPLE_CVS, er_ranking)
    assert result["evidencerank"]["ranking"] == er_ranking


def test_run_baselines_tau_values_in_range():
    er_ranking = ["cv_b", "cv_c", "cv_a"]
    result = run_baselines(SAMPLE_JD, SAMPLE_CVS, er_ranking)
    for tau in result["cross_method_tau"].values():
        assert -1.0 <= tau <= 1.0
