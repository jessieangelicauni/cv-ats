# src/evaluation/ablation.py
from __future__ import annotations
import numpy as np
from src.graph.pipeline import run_pipeline
from src.evaluation.hallucination_checker import verify_evidence_chain, hallucination_rate
from src.evaluation.ranking_metrics import kendall_tau_score
from src.utils.llm import reset_call_count, get_call_count

ABLATION_VARIANTS: list[dict] = [
    {"name": "no_rag",                "use_vector_store": False},
    {"name": "no_evidence_grounding", "use_evidence_grounding": False},
    {"name": "no_skill_filter",       "use_skill_filter": False},
    {"name": "no_calibration",        "use_pool_calibration": False},
]


def _variant_metrics(state, full_ranking: list[str], cv_text_map: dict[str, str], llm_calls: int = 0) -> dict:
    ranking = [r.candidate_id for r in state.final_ranking.ranked_candidates]
    tau, _ = kendall_tau_score(ranking, full_ranking)

    flags = []
    for a in state.candidate_assessments:
        flags.extend(verify_evidence_chain(a, cv_text_map.get(a.candidate_id, "")))
    h_rate = hallucination_rate(flags)

    scores = [r.calibrated_score for r in state.final_ranking.ranked_candidates]
    score_std = float(np.std(scores)) if scores else 0.0

    return {
        "hallucination_rate": round(h_rate, 4),
        "score_std":          round(score_std, 4),
        "llm_calls":          llm_calls,
        "tau_vs_full":        round(float(tau), 4),
    }


def run_ablation(
    jd_raw: str,
    cv_raws: list[dict],
    full_state,
    cv_text_map: dict[str, str],
) -> dict:
    full_ranking = [r.candidate_id for r in full_state.final_ranking.ranked_candidates]
    results = {"full_system": _variant_metrics(full_state, full_ranking, cv_text_map, llm_calls=0)}
    results["full_system"]["tau_vs_full"] = 1.0

    for variant in ABLATION_VARIANTS:
        name = variant["name"]
        kwargs = {k: v for k, v in variant.items() if k != "name"}
        reset_call_count()
        state = run_pipeline(jd_raw, cv_raws, **kwargs)
        results[name] = _variant_metrics(state, full_ranking, cv_text_map, llm_calls=get_call_count())

    return results
