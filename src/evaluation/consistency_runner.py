from __future__ import annotations
import numpy as np
from itertools import combinations
from src.graph.pipeline import run_pipeline
from src.evaluation.ranking_metrics import kendall_tau_score


def run_consistency_experiment(
    jd_raw: str,
    cv_raws: list[dict],
    n_runs: int = 3,
) -> dict:
    rankings: list[list[str]] = []
    score_maps: list[dict[str, float]] = []

    for i in range(n_runs):
        state = run_pipeline(
            jd_raw,
            cv_raws,
            run_id=f"consistency_{i}",
            use_cache=True,
        )
        ranked = state.final_ranking.ranked_candidates
        rankings.append([r.candidate_id for r in ranked])
        score_maps.append({r.candidate_id: r.calibrated_score for r in ranked})

    taus = [
        kendall_tau_score(rankings[i], rankings[j])[0]
        for i, j in combinations(range(n_runs), 2)
    ]

    common_ids = set(score_maps[0].keys())
    for sm in score_maps[1:]:
        common_ids &= set(sm.keys())
    score_variance = {
        cid: float(np.var([sm[cid] for sm in score_maps]))
        for cid in common_ids
    }

    return {
        "mean_tau": float(np.mean(taus)),
        "min_tau": float(np.min(taus)),
        "pairwise_taus": taus,
        "score_variance": score_variance,
    }
