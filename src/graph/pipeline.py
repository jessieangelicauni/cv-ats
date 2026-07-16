from __future__ import annotations
import hashlib
import uuid
from typing import Callable
from langgraph.graph import StateGraph, START, END
from src.graph.state import ATSState
from src.graph.nodes import (
    phase1_jd_parser, phase2_cv_extractor, phase3_signal_enricher,
    phase4_candidate_judge, phase5_pool_calibrator,
)
from src.utils.telemetry import setup_telemetry, get_tracer, current_otel_trace_id
import config


def build_pipeline() -> StateGraph:
    builder = StateGraph(ATSState)
    builder.add_node("phase1", phase1_jd_parser)
    builder.add_node("phase2", phase2_cv_extractor)
    builder.add_node("phase3", phase3_signal_enricher)
    builder.add_node("phase4", phase4_candidate_judge)
    builder.add_node("phase5", phase5_pool_calibrator)
    builder.add_edge(START, "phase1")
    builder.add_edge("phase1", "phase2")
    builder.add_edge("phase2", "phase3")
    builder.add_edge("phase3", "phase4")
    builder.add_edge("phase4", "phase5")
    builder.add_edge("phase5", END)
    return builder.compile()


def run_pipeline(
    jd_raw: str,
    cv_raws: list[dict],
    run_id: str | None = None,
    use_cache: bool = True,
    session_id: str | None = None,
    on_phase_complete: Callable[[dict], None] | None = None,
) -> ATSState:
    """
    session_id: shared across multiple run_pipeline() calls (e.g. consistency
    experiments) so Langfuse groups them as one session for side-by-side comparison.

    on_phase_complete: called with each trace_log entry (e.g. {"phase": 1,
    "duration_s": 2.1}) as soon as that phase finishes, so callers (e.g. the CLI's
    progress spinner) can report real progress instead of a static label.
    """
    setup_telemetry()

    run_id = run_id or str(uuid.uuid4())[:8]
    jd_hash = hashlib.sha256(jd_raw.encode()).hexdigest()[:12]

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
        # Capture the OTel trace_id while we are inside the root span.
        # This is the ID Langfuse uses for this trace — needed for post-run scoring.
        otel_trace_id = current_otel_trace_id()

        graph = build_pipeline()
        initial_state: ATSState = {
            "jd_raw": jd_raw,
            "cv_raws": cv_raws,
            "jd_structured": None,
            "cv_profiles": [],
            "enriched_profiles": [],
            "candidate_assessments": [],
            "final_ranking": None,
            "run_id": run_id,
            "otel_trace_id": otel_trace_id,
            "trace_log": [],
            "hallucination_flags": [],
            "use_cache": use_cache,
        }

        final_state = initial_state
        seen_log_len = 0
        for state in graph.stream(initial_state, stream_mode="values"):
            final_state = state
            log = state.get("trace_log", [])
            if on_phase_complete and len(log) > seen_log_len:
                on_phase_complete(log[-1])
                seen_log_len = len(log)
        return final_state
