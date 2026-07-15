from __future__ import annotations
import uuid
import numpy as np
from itertools import combinations
from src.graph.pipeline import run_pipeline
from src.evaluation.ranking_metrics import kendall_tau_score


def run_consistency_experiment(
    jd_raw: str,
    cv_raws: list[dict],
    n_runs: int = 3,
) -> dict:
    # Shared session_id groups all runs in Langfuse for side-by-side comparison
    session_id = f"consistency_{str(uuid.uuid4())[:8]}"
    rankings: list[list[str]] = []
    score_maps: list[dict[str, float]] = []
    otel_trace_ids: list[str] = []

    for i in range(n_runs):
        state = run_pipeline(
            jd_raw,
            cv_raws,
            run_id=f"consistency_{i}",
            use_cache=True,
            session_id=session_id,
        )
        ranked = state["final_ranking"].ranked_candidates
        rankings.append([r.candidate_id for r in ranked])
        score_maps.append({r.candidate_id: r.calibrated_score for r in ranked})
        otel_trace_ids.append(state["otel_trace_id"])

    taus = [
        kendall_tau_score(rankings[i], rankings[j])[0]
        for i, j in combinations(range(n_runs), 2)
    ]

    candidate_ids = list(score_maps[0].keys())
    score_variance = {
        cid: float(np.var([sm.get(cid, 0) for sm in score_maps]))
        for cid in candidate_ids
    }

    return {
        "session_id": session_id,
        "otel_trace_ids": otel_trace_ids,
        "mean_tau": float(np.mean(taus)),
        "min_tau": float(np.min(taus)),
        "pairwise_taus": taus,
        "score_variance": score_variance,
    }
