from __future__ import annotations
import math
from scipy.stats import kendalltau


def kendall_tau_score(llm_ranking: list[str], reference_ranking: list[str]) -> tuple[float, float]:
    ids = list(dict.fromkeys(reference_ranking + llm_ranking))
    ref_pos = {cid: i for i, cid in enumerate(reference_ranking)}
    llm_pos = {cid: i for i, cid in enumerate(llm_ranking)}
    ref_vec = [ref_pos.get(cid, len(reference_ranking)) for cid in ids]
    llm_vec = [llm_pos.get(cid, len(llm_ranking)) for cid in ids]
    tau, p = kendalltau(llm_vec, ref_vec)
    return float(tau), float(p)


def _dcg(gains: list[float]) -> float:
    return sum((2**g - 1) / math.log2(i + 2) for i, g in enumerate(gains))


def ndcg_at_k(ranked_ids: list[str], relevance: dict[str, float], k: int | None = None) -> float:
    if k is not None:
        ranked_ids = ranked_ids[:k]
    gains = [relevance.get(cid, 0.0) for cid in ranked_ids]
    ideal_gains = sorted(relevance.values(), reverse=True)
    if k is not None:
        ideal_gains = ideal_gains[:k]
    idcg = _dcg(ideal_gains)
    if idcg == 0:
        return 0.0
    return _dcg(gains) / idcg