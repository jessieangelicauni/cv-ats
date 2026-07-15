from src.evaluation.ranking_metrics import kendall_tau_score, ndcg_ranking_score

def test_kendall_tau_perfect_agreement():
    llm = ["cv_001", "cv_002", "cv_003"]
    human = ["cv_001", "cv_002", "cv_003"]
    tau, p = kendall_tau_score(llm, human)
    assert abs(tau - 1.0) < 1e-6

def test_kendall_tau_perfect_disagreement():
    llm = ["cv_001", "cv_002", "cv_003"]
    human = ["cv_003", "cv_002", "cv_001"]
    tau, p = kendall_tau_score(llm, human)
    assert abs(tau - (-1.0)) < 1e-6

def test_ndcg_perfect_ranking():
    llm_ranking = ["cv_001", "cv_002", "cv_003"]
    human_ranking = ["cv_001", "cv_002", "cv_003"]
    score = ndcg_ranking_score(llm_ranking, human_ranking)
    assert abs(score - 1.0) < 1e-6
