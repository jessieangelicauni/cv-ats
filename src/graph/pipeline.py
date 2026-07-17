from __future__ import annotations
import hashlib
import time
import uuid
from typing import Callable
from src.graph.state import ATSState
from src.graph.nodes import (
    phase1_jd_parser, phase2_cv_extractor,
    phase3_candidate_judge, phase4_pool_calibrator,
)
from src.utils.cache import ExtractionCache
from src.utils.vector_store import CVVectorStore
from src.utils.skill_taxonomy import normalize
from src.utils.telemetry import setup_telemetry, get_tracer, current_otel_trace_id
from src.models.schemas import (
    CandidateProfile, CandidateAssessment, JDRequirements,
    FinalRanking, RankedCandidate, SkillMatchResult,
)
import config


def _filter_by_required_skills(
    profiles: list[CandidateProfile],
    jd: JDRequirements,
) -> tuple[list[CandidateProfile], list[str]]:
    if not jd.required_skills:
        return profiles, []
    required_ids = {normalize(s.skill) for s in jd.required_skills}
    passing: list[CandidateProfile] = []
    eliminated: list[str] = []
    for profile in profiles:
        candidate_ids = {normalize(s.canonical_skill) for s in profile.skills}
        if required_ids & candidate_ids:
            passing.append(profile)
        else:
            eliminated.append(profile.candidate_id)
    return passing, eliminated


def _ranking_from_raw(assessments: list[CandidateAssessment]) -> FinalRanking:
    sorted_a = sorted(assessments, key=lambda a: a.raw_score, reverse=True)
    return FinalRanking(
        ranked_candidates=[
            RankedCandidate(
                rank=i + 1,
                candidate_id=a.candidate_id,
                calibrated_score=a.raw_score,
                delta_from_raw=0.0,
                comparative_notes="",
            )
            for i, a in enumerate(sorted_a)
        ],
        pool_summary="No calibration applied.",
        calibration_rationale="No calibration applied.",
        borderline_pairs=[],
    )


def run_pipeline(
    jd_raw: str,
    cv_raws: list[dict],
    run_id: str | None = None,
    use_cache: bool = True,
    session_id: str | None = None,
    on_phase_complete: Callable[[dict], None] | None = None,
    use_vector_store: bool = True,
    use_skill_filter: bool = True,
    use_evidence_grounding: bool = True,
    use_pool_calibration: bool = True,
) -> ATSState:
    setup_telemetry()
    run_id = run_id or str(uuid.uuid4())[:8]
    jd_hash = hashlib.sha256(jd_raw.encode()).hexdigest()[:12]
    cache        = ExtractionCache(config.CACHE_DB_PATH)  if use_cache else None
    vector_store = CVVectorStore(config.CHROMA_DB_PATH)   if use_cache else None

    rag_store = vector_store if use_vector_store else None

    tracer = get_tracer()
    with tracer.start_as_current_span(
        "pipeline",
        attributes={
            "run.id": run_id,
            "run.n_candidates": len(cv_raws),
            "run.jd_hash": jd_hash,
            "run.session_id": session_id or "",
            "llm.small_model": config.SMALL_MODEL,
            "llm.large_model": config.LARGE_MODEL,
        },
    ):
        otel_trace_id = current_otel_trace_id()
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
        cv_profiles   = _run_phase(2, phase2_cv_extractor, cv_raws, cache,
                                   rag_store, candidates=len(cv_raws))

        if use_skill_filter:
            cv_profiles, eliminated = _filter_by_required_skills(
                cv_profiles, jd_structured)
        else:
            eliminated = []

        assessments = _run_phase(3, phase3_candidate_judge, cv_profiles,
                                 jd_structured, rag_store,
                                 use_evidence_grounding)

        if use_pool_calibration:
            final_ranking = _run_phase(4, phase4_pool_calibrator,
                                       assessments, jd_structured)
        else:
            final_ranking = _ranking_from_raw(assessments)

        return ATSState(
            jd_raw=jd_raw,
            cv_raws=cv_raws,
            jd_structured=jd_structured,
            cv_profiles=cv_profiles,
            candidate_assessments=assessments,
            final_ranking=final_ranking,
            run_id=run_id,
            otel_trace_id=otel_trace_id,
            trace_log=trace_log,
            use_cache=use_cache,
            eliminated_candidates=eliminated,
        )
