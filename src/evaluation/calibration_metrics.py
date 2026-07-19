from __future__ import annotations
import numpy as np
from scipy.stats import entropy as scipy_entropy
from src.models.schemas import CandidateAssessment, FinalRanking


def calibration_report(
    assessments: list[CandidateAssessment],
    ranking: FinalRanking,
) -> dict:
    shortlisted_ids = {r.candidate_id for r in ranking.ranked_candidates}
    shortlisted = [a for a in assessments if a.candidate_id in shortlisted_ids]
    raw_scores = np.array([a.raw_score for a in shortlisted])
    calibrated_scores = np.array([r.calibrated_score for r in ranking.ranked_candidates])
    abs_deltas = np.array([abs(r.delta_from_raw) for r in ranking.ranked_candidates])

    raw_rank_map = {
        a.candidate_id: i + 1
        for i, a in enumerate(sorted(shortlisted, key=lambda a: a.raw_score, reverse=True))
    }
    rank_changes = sum(
        1 for r in ranking.ranked_candidates
        if r.rank != raw_rank_map.get(r.candidate_id)
    )

    hist, _ = np.histogram(calibrated_scores, bins=10, range=(0, 100))
    hist = hist + 1e-9

    return {
        "raw_std": float(np.std(raw_scores)),
        "calibrated_std": float(np.std(calibrated_scores)),
        "raw_range": float(np.max(raw_scores) - np.min(raw_scores)),
        "mean_abs_delta": float(np.mean(abs_deltas)),
        "rank_changes": rank_changes,
        "score_entropy": float(scipy_entropy(hist)),
    }
