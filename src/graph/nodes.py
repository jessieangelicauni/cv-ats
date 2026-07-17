from __future__ import annotations
import time
from opentelemetry.trace import StatusCode
from src.graph.state import ATSState
from src.models.schemas import CandidateProfile, CandidateAssessment
from src.agents.jd_parser import JDParserAgent
from src.agents.cv_extractor import CVExtractorAgent
from src.agents.candidate_judge import CandidateJudgeAgent
from src.agents.pool_calibrator import PoolCalibratorAgent
from src.utils.cache import ExtractionCache
from src.utils.telemetry import get_tracer
import config

_tracer = get_tracer()


def _get_cache(state: ATSState) -> ExtractionCache | None:
    if not state.use_cache:
        return None
    return ExtractionCache(config.CACHE_DB_PATH)


def phase1_jd_parser(state: ATSState) -> dict:
    t0 = time.time()
    with _tracer.start_as_current_span("phase1/jd_parser") as span:
        span.set_attribute("phase", 1)
        try:
            cache = _get_cache(state)
            span.set_attribute("cache.enabled", cache is not None)
            result = JDParserAgent(cache=cache).run(state.jd_raw)
        except Exception as exc:
            span.record_exception(exc)
            span.set_status(StatusCode.ERROR, str(exc))
            raise
    return {
        "jd_structured": result,
        "trace_log": [{"phase": 1, "duration_s": round(time.time() - t0, 2)}],
    }


def phase2_cv_extractor(state: ATSState) -> dict:
    t0 = time.time()
    cache = _get_cache(state)

    with _tracer.start_as_current_span("phase2/cv_extractor") as span:
        span.set_attribute("phase", 2)
        span.set_attribute("n_candidates", len(state.cv_raws))

        def process(cv_raw: dict) -> CandidateProfile:
            with _tracer.start_as_current_span(
                f"phase2/cv/{cv_raw['candidate_id']}"
            ) as cspan:
                try:
                    return CVExtractorAgent(cache=cache).run(cv_raw)
                except Exception as exc:
                    cspan.record_exception(exc)
                    cspan.set_status(StatusCode.ERROR, str(exc))
                    raise

        profiles = [process(cv_raw) for cv_raw in state.cv_raws]

    return {
        "cv_profiles": profiles,
        "trace_log": [{"phase": 2, "candidates": len(profiles),
                       "duration_s": round(time.time() - t0, 2)}],
    }


def phase3_signal_enricher(state: ATSState) -> dict:
    t0 = time.time()

    with _tracer.start_as_current_span("phase3/signal_enricher") as span:
        span.set_attribute("phase", 3)
        span.set_attribute("n_candidates", len(state.cv_profiles))
        span.set_attribute("status", "skipped")

    return {
        "trace_log": [{"phase": 3, "status": "skipped", "duration_s": round(time.time() - t0, 2)}],
    }


def phase4_candidate_judge(state: ATSState) -> dict:
    t0 = time.time()
    jd = state.jd_structured

    with _tracer.start_as_current_span("phase4/candidate_judge") as span:
        span.set_attribute("phase", 4)
        span.set_attribute("n_candidates", len(state.cv_profiles))

        def process(profile: CandidateProfile) -> CandidateAssessment:
            with _tracer.start_as_current_span(
                f"phase4/judge/{profile.candidate_id}"
            ) as cspan:
                try:
                    return CandidateJudgeAgent().run(profile, jd)
                except Exception as exc:
                    cspan.record_exception(exc)
                    cspan.set_status(StatusCode.ERROR, str(exc))
                    raise

        assessments = [process(profile) for profile in state.cv_profiles]

    return {
        "candidate_assessments": assessments,
        "trace_log": [{"phase": 4, "duration_s": round(time.time() - t0, 2)}],
    }


def phase5_pool_calibrator(state: ATSState) -> dict:
    t0 = time.time()
    with _tracer.start_as_current_span("phase5/pool_calibrator") as span:
        span.set_attribute("phase", 5)
        span.set_attribute("n_candidates", len(state.candidate_assessments))
        try:
            ranking = PoolCalibratorAgent().run(
                state.candidate_assessments,
                state.jd_structured,
            )
        except Exception as exc:
            span.record_exception(exc)
            span.set_status(StatusCode.ERROR, str(exc))
            raise
    return {
        "final_ranking": ranking,
        "trace_log": [{"phase": 5, "duration_s": round(time.time() - t0, 2)}],
    }

