# src/evaluation/baselines.py
from __future__ import annotations
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity
from src.evaluation.ranking_metrics import kendall_tau_score


def tfidf_rank(jd_text: str, cv_raws: list[dict]) -> tuple[list[str], dict[str, float]]:
    candidate_ids = [cv["candidate_id"] for cv in cv_raws]
    texts = [jd_text] + [cv["raw_text"] for cv in cv_raws]
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(texts)
    sims = cosine_similarity(matrix[0:1], matrix[1:])[0]
    scores = {cid: round(float(s), 4) for cid, s in zip(candidate_ids, sims)}
    ranking = sorted(candidate_ids, key=lambda cid: scores[cid], reverse=True)
    return ranking, scores


def keyword_rank(jd_text: str, cv_raws: list[dict]) -> tuple[list[str], dict[str, float]]:
    candidate_ids = [cv["candidate_id"] for cv in cv_raws]
    jd_tokens = {
        w for w in jd_text.lower().split()
        if w.isalpha() and w not in ENGLISH_STOP_WORDS
    }
    counts: dict[str, float] = {}
    for cv in cv_raws:
        cv_tokens = set(cv["raw_text"].lower().split())
        counts[cv["candidate_id"]] = float(sum(1 for kw in jd_tokens if kw in cv_tokens))
    ranking = sorted(candidate_ids, key=lambda cid: counts[cid], reverse=True)
    return ranking, counts


def _distribution(scores: dict[str, float]) -> dict[str, float]:
    vals = list(scores.values())
    return {
        "mean": round(float(np.mean(vals)), 4),
        "std":  round(float(np.std(vals)), 4),
        "min":  round(float(np.min(vals)), 4),
        "max":  round(float(np.max(vals)), 4),
    }


def run_baselines(
    jd_text: str,
    cv_raws: list[dict],
    evidencerank_ranking: list[str],
) -> dict:
    tfidf_ranking, tfidf_scores = tfidf_rank(jd_text, cv_raws)
    kw_ranking, kw_scores = keyword_rank(jd_text, cv_raws)

    tau_tf_kw, _ = kendall_tau_score(tfidf_ranking, kw_ranking)
    tau_tf_er, _ = kendall_tau_score(tfidf_ranking, evidencerank_ranking)
    tau_kw_er, _ = kendall_tau_score(kw_ranking, evidencerank_ranking)

    return {
        "tfidf": {
            "ranking": tfidf_ranking,
            "scores": tfidf_scores,
            "distribution": _distribution(tfidf_scores),
        },
        "keyword": {
            "ranking": kw_ranking,
            "scores": kw_scores,
            "distribution": _distribution(kw_scores),
        },
        "evidencerank": {
            "ranking": evidencerank_ranking,
        },
        "cross_method_tau": {
            "tfidf_vs_keyword":        round(float(tau_tf_kw), 4),
            "tfidf_vs_evidencerank":   round(float(tau_tf_er), 4),
            "keyword_vs_evidencerank": round(float(tau_kw_er), 4),
        },
    }
