from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from sentence_transformers import SentenceTransformer
from src.models.schemas import SkillRequirement, SkillEntry


@dataclass
class SkillMatchResult:
    jd_skill: str
    best_match: str | None
    score: float
    is_required: bool


class SkillMatcher:
    def __init__(self, embedder: SentenceTransformer) -> None:
        self._embedder = embedder

    def match(
        self,
        jd_skills: list[SkillRequirement],
        candidate_skills: list[SkillEntry],
    ) -> list[SkillMatchResult]:
        if not jd_skills:
            return []
        if not candidate_skills:
            return [
                SkillMatchResult(
                    jd_skill=s.skill,
                    best_match=None,
                    score=0.0,
                    is_required=s.is_mandatory,
                )
                for s in jd_skills
            ]

        jd_names = [s.skill for s in jd_skills]
        cand_names = [s.canonical_skill for s in candidate_skills]

        jd_embs = self._embedder.encode(jd_names, convert_to_numpy=True)
        cand_embs = self._embedder.encode(cand_names, convert_to_numpy=True)

        jd_norm = jd_embs / (np.linalg.norm(jd_embs, axis=1, keepdims=True) + 1e-9)
        cand_norm = cand_embs / (np.linalg.norm(cand_embs, axis=1, keepdims=True) + 1e-9)
        sim = jd_norm @ cand_norm.T  # (n_jd, n_candidate)

        results: list[SkillMatchResult] = []
        for i, jd_skill in enumerate(jd_skills):
            best_idx = int(np.argmax(sim[i]))
            results.append(SkillMatchResult(
                jd_skill=jd_skill.skill,
                best_match=cand_names[best_idx],
                score=round(float(sim[i, best_idx]), 4),
                is_required=jd_skill.is_mandatory,
            ))
        return results
