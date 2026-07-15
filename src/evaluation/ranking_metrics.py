from __future__ import annotations
import numpy as np
from scipy.stats import kendalltau
from sklearn.metrics import ndcg_score


def kendall_tau_score(llm_ranking: list[str], human_ranking: list[str]) -> tuple[float, float]:
    ids = list(dict.fromkeys(human_ranking + llm_ranking))
    human_pos = {cid: i for i, cid in enumerate(human_ranking)}
    llm_pos = {cid: i for i, cid in enumerate(llm_ranking)}
    human_vec = [human_pos.get(cid, len(human_ranking)) for cid in ids]
    llm_vec = [llm_pos.get(cid, len(llm_ranking)) for cid in ids]
    tau, p = kendalltau(llm_vec, human_vec)
    return float(tau), float(p)


def ndcg_ranking_score(llm_ranking: list[str], human_ranking: list[str]) -> float:
    n = len(human_ranking)
    human_grades = {cid: n - i for i, cid in enumerate(human_ranking)}
    true_relevance = np.array([[human_grades.get(cid, 0) for cid in llm_ranking]])
    scores = np.array([[n - i for i in range(len(llm_ranking))]])
    return float(ndcg_score(true_relevance, scores))
