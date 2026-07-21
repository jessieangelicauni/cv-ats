from __future__ import annotations
import time
import uuid
from typing import Callable
from src.graph.state import ATSState
from src.graph.nodes import (
    phase1_jd_parser, phase2_cv_extractor,
    phase3_candidate_judge, phase4_pool_calibrator,
)
from src.utils.cache import ExtractionCache
from src.utils.skill_normalizer import compute_skill_matches
from src.models.schemas import CandidateProfile, JDRequirements
import config


def _filter_by_required_skills(
    profiles: list[CandidateProfile],
    jd: JDRequirements,
) -> tuple[list[CandidateProfile], list[str]]:
    if not jd.required_skills:
        return profiles, []
    passing: list[CandidateProfile] = []
    eliminated: list[str] = []
    for profile in profiles:
        matches = compute_skill_matches(jd.required_skills, profile.skills)
        if any(m.best_match is not None for m in matches):
            passing.append(profile)
        else:
            eliminated.append(profile.candidate_id)
    return passing, eliminated


def run_pipeline(
    jd_raw: str,
    cv_raws: list[dict],
    run_id: str | None = None,
    use_cache: bool = True,
    on_phase_complete: Callable[[dict], None] | None = None,
) -> ATSState:
    run_id = run_id or str(uuid.uuid4())[:8]
    cache = ExtractionCache(config.CACHE_DB_PATH) if use_cache else None

    trace_log: list[dict] = []

    def _run_phase(n: int, fn, *args, **extra_log):
        t0 = time.time()
        result = fn(*args)
        entry = {"phase": n, "duration_s": round(time.time() - t0, 2), **extra_log}
        trace_log.append(entry)
        if on_phase_complete:
            on_phase_complete(entry)
        return result

    jd_structured = _run_phase(1, phase1_jd_parser, jd_raw, cache)

    cv_profiles = _run_phase(2, phase2_cv_extractor, cv_raws, cache,
                             candidates=len(cv_raws))

    cv_profiles, eliminated = _filter_by_required_skills(cv_profiles, jd_structured)

    assessments = _run_phase(3, phase3_candidate_judge, cv_profiles, jd_structured, cv_raws)

    # Keep only the top N candidates by raw Phase 3 score for Phase 4.
    # Prevents context window overflow when candidate pools are large (e.g. 200 CVs).
    shortlist = sorted(assessments, key=lambda a: a.raw_score, reverse=True)[
        : config.TOP_N_FOR_CALIBRATION
    ]

    final_ranking = _run_phase(4, phase4_pool_calibrator, shortlist, jd_structured,
                               shortlist_size=len(shortlist), total_assessed=len(assessments))

    return ATSState(
        jd_raw=jd_raw,
        cv_raws=cv_raws,
        jd_structured=jd_structured,
        cv_profiles=cv_profiles,
        candidate_assessments=assessments,
        final_ranking=final_ranking,
        run_id=run_id,
        trace_log=trace_log,
        use_cache=use_cache,
        eliminated_candidates=eliminated,
        shortlisted_for_calibration=[a.candidate_id for a in shortlist],
    )
