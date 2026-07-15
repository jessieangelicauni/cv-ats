from __future__ import annotations
import numpy as np
from scipy.stats import entropy as scipy_entropy
from src.models.schemas import CandidateAssessment, FinalRanking


def calibration_report(
    assessments: list[CandidateAssessment],
    ranking: FinalRanking,
) -> dict:
    raw_scores = np.array([a.raw_score for a in assessments])
    calibrated_scores = np.array([r.calibrated_score for r in ranking.ranked_candidates])
    deltas = np.array([abs(r.delta_from_raw) for r in ranking.ranked_candidates])

    hist, _ = np.histogram(calibrated_scores, bins=10, range=(0, 100))
    hist = hist + 1e-9  # avoid log(0)

    return {
        "raw_std": float(np.std(raw_scores)),
        "calibrated_std": float(np.std(calibrated_scores)),
        "raw_range": float(np.max(raw_scores) - np.min(raw_scores)),
        "mean_abs_delta": float(np.mean(deltas)),
        "rank_changes": int(np.sum(deltas > 0)),
        "score_entropy": float(scipy_entropy(hist)),
    }
