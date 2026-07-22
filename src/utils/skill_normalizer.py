from __future__ import annotations
import re
from sentence_transformers import util
from src.models.schemas import SkillRequirement, SkillEntry, SkillMatchResult
from src.utils.embedder import get_embedder
import config


_cv_emb_cache: dict[tuple[str, ...], object] = {}
_jd_emb_cache: dict[tuple[str, ...], object] = {}

_SKILL_SPLIT_RE = re.compile(r"[,;/|]|\band\b")


def _split_skill_mentions(profile_skills: list[SkillEntry]) -> list[str]:
    names: list[str] = []
    for entry in profile_skills:
        for part in _SKILL_SPLIT_RE.split(entry.raw_mention):
            part = part.strip()
            if part:
                names.append(part)
    return names


def compute_skill_matches(
    jd_skills: list[SkillRequirement],
    profile_skills: list[SkillEntry],
) -> list[SkillMatchResult]:
    if not jd_skills:
        return []

    candidate_names = _split_skill_mentions(profile_skills)

    if not candidate_names:
        return [
            SkillMatchResult(jd_skill=s.skill, best_match=None, score=0.0, is_required=s.is_mandatory)
            for s in jd_skills
        ]

    embedder = get_embedder()
    cv_key = tuple(candidate_names)
    if cv_key not in _cv_emb_cache:
        _cv_emb_cache[cv_key] = embedder.encode(candidate_names, convert_to_tensor=True)
    emb_cv = _cv_emb_cache[cv_key]

    jd_names = [s.skill for s in jd_skills]
    jd_key = tuple(jd_names)
    if jd_key not in _jd_emb_cache:
        _jd_emb_cache[jd_key] = embedder.encode(jd_names, convert_to_tensor=True)
    emb_jd = _jd_emb_cache[jd_key]

    scores_matrix = util.cos_sim(emb_jd, emb_cv)

    results: list[SkillMatchResult] = []
    for i, s in enumerate(jd_skills):
        scores = scores_matrix[i]
        best_idx = int(scores.argmax())
        best_score = float(scores.max())
        best_match = candidate_names[best_idx] if best_score >= config.SKILL_MATCH_THRESHOLD else None
        results.append(SkillMatchResult(
            jd_skill=s.skill, best_match=best_match, score=best_score, is_required=s.is_mandatory,
        ))

    return results
