from __future__ import annotations
import time
from concurrent.futures import ThreadPoolExecutor
from src.graph.state import ATSState
from src.models.schemas import CandidateProfile, EnrichedProfile, CandidateAssessment
from src.agents.jd_parser import JDParserAgent
from src.agents.cv_extractor import CVExtractorAgent
from src.agents.signal_enricher import SignalEnricherAgent
from src.agents.candidate_judge import CandidateJudgeAgent
from src.agents.pool_calibrator import PoolCalibratorAgent
from src.utils.cache import ExtractionCache
import config


def _get_cache(state: ATSState) -> ExtractionCache | None:
    if not state.get("use_cache", True):
        return None
    return ExtractionCache(config.CACHE_DB_PATH)


def phase1_jd_parser(state: ATSState) -> dict:
    t0 = time.time()
    cache = _get_cache(state)
    agent = JDParserAgent(cache=cache)
    result = agent.run(state["jd_raw"])
    return {
        "jd_structured": result,
        "trace_log": [{"phase": 1, "duration_s": round(time.time() - t0, 2)}],
    }


def phase2_cv_extractor(state: ATSState) -> dict:
    t0 = time.time()
    cache = _get_cache(state)

    def process(cv_raw: dict) -> CandidateProfile:
        return CVExtractorAgent(cache=cache).run(cv_raw)

    with ThreadPoolExecutor(max_workers=config.MAX_PARALLEL_WORKERS) as ex:
        profiles = list(ex.map(process, state["cv_raws"]))

    return {
        "cv_profiles": profiles,
        "trace_log": [{"phase": 2, "candidates": len(profiles),
                       "duration_s": round(time.time() - t0, 2)}],
    }


def phase3_signal_enricher(state: ATSState) -> dict:
    t0 = time.time()
    jd = state["jd_structured"]

    def process(profile: CandidateProfile) -> EnrichedProfile:
        return SignalEnricherAgent().run(profile, jd)

    with ThreadPoolExecutor(max_workers=config.MAX_PARALLEL_WORKERS) as ex:
        enriched = list(ex.map(process, state["cv_profiles"]))

    return {
        "enriched_profiles": enriched,
        "trace_log": [{"phase": 3, "duration_s": round(time.time() - t0, 2)}],
    }


def phase4_candidate_judge(state: ATSState) -> dict:
    t0 = time.time()
    jd = state["jd_structured"]

    def process(profile: EnrichedProfile) -> CandidateAssessment:
        return CandidateJudgeAgent().run(profile, jd)

    with ThreadPoolExecutor(max_workers=config.MAX_PARALLEL_WORKERS) as ex:
        assessments = list(ex.map(process, state["enriched_profiles"]))

    return {
        "candidate_assessments": assessments,
        "trace_log": [{"phase": 4, "duration_s": round(time.time() - t0, 2)}],
    }


def phase5_pool_calibrator(state: ATSState) -> dict:
    t0 = time.time()
    ranking = PoolCalibratorAgent().run(state["candidate_assessments"], state["jd_structured"])
    return {
        "final_ranking": ranking,
        "trace_log": [{"phase": 5, "duration_s": round(time.time() - t0, 2)}],
    }
