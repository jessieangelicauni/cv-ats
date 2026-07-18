from src.evaluation.ranking_metrics import kendall_tau_score

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
