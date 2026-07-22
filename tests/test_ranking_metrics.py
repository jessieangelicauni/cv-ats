from src.evaluation.ranking_metrics import kendall_tau_score, ndcg_at_k

def test_kendall_tau_perfect_agreement():
    llm = ["cv_001", "cv_002", "cv_003"]
    ref = ["cv_001", "cv_002", "cv_003"]
    tau, p = kendall_tau_score(llm, ref)
    assert abs(tau - 1.0) < 1e-6

def test_kendall_tau_perfect_disagreement():
    llm = ["cv_001", "cv_002", "cv_003"]
    ref = ["cv_003", "cv_002", "cv_001"]
    tau, p = kendall_tau_score(llm, ref)
    assert abs(tau - (-1.0)) < 1e-6

def test_ndcg_perfect_ranking_is_one():
    ranked = ["cv_001", "cv_002", "cv_003"]
    relevance = {"cv_001": 3.0, "cv_002": 2.0, "cv_003": 1.0}
    assert abs(ndcg_at_k(ranked, relevance) - 1.0) < 1e-6

def test_ndcg_worst_ranking_is_below_one():
    ranked = ["cv_003", "cv_002", "cv_001"]
    relevance = {"cv_001": 3.0, "cv_002": 2.0, "cv_003": 1.0}
    assert ndcg_at_k(ranked, relevance) < 1.0

def test_ndcg_truncates_at_k():
    # Only the top 2 ranked slots count; a low-relevance item pushed to rank 3
    # shouldn't hurt the score once k=2.
    ranked = ["cv_001", "cv_002", "cv_003"]
    relevance = {"cv_001": 3.0, "cv_002": 3.0, "cv_003": 0.0}
    assert abs(ndcg_at_k(ranked, relevance, k=2) - 1.0) < 1e-6

def test_ndcg_all_zero_relevance_is_zero():
    ranked = ["cv_001", "cv_002"]
    relevance = {"cv_001": 0.0, "cv_002": 0.0}
    assert ndcg_at_k(ranked, relevance) == 0.0

def test_ndcg_missing_candidate_treated_as_zero_relevance():
    # cv_999 has no relevance judgment (treated as 0). Ranking it ahead of a
    # known-relevant candidate should be penalized relative to the ideal order.
    ranked = ["cv_999", "cv_001"]
    relevance = {"cv_001": 2.0}
    assert 0.0 < ndcg_at_k(ranked, relevance) < 1.0
