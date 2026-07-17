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
from src.utils.telemetry import setup_telemetry, get_tracer, current_otel_trace_id
import config


def run_pipeline(
    jd_raw: str,
    cv_raws: list[dict],
    run_id: str | None = None,
    use_cache: bool = True,
    session_id: str | None = None,
    on_phase_complete: Callable[[dict], None] | None = None,
) -> ATSState:
    setup_telemetry()
    run_id = run_id or str(uuid.uuid4())[:8]
    jd_hash = hashlib.sha256(jd_raw.encode()).hexdigest()[:12]
    cache = ExtractionCache(config.CACHE_DB_PATH) if use_cache else None

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
                                   candidates=len(cv_raws))
        assessments   = _run_phase(3, phase3_candidate_judge, cv_profiles, jd_structured)
        final_ranking = _run_phase(4, phase4_pool_calibrator, assessments, jd_structured)

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
        )
