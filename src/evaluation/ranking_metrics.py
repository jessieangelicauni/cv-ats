from __future__ import annotations
from scipy.stats import kendalltau


def kendall_tau_score(llm_ranking: list[str], reference_ranking: list[str]) -> tuple[float, float]:
    ids = list(dict.fromkeys(reference_ranking + llm_ranking))
    ref_pos = {cid: i for i, cid in enumerate(reference_ranking)}
    llm_pos = {cid: i for i, cid in enumerate(llm_ranking)}
    ref_vec = [ref_pos.get(cid, len(reference_ranking)) for cid in ids]
    llm_vec = [llm_pos.get(cid, len(llm_ranking)) for cid in ids]
    tau, p = kendalltau(llm_vec, ref_vec)
    return float(tau), float(p)
