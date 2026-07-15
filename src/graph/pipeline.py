from __future__ import annotations
import uuid
from langgraph.graph import StateGraph, START, END
from src.graph.state import ATSState
from src.graph.nodes import (
    phase1_jd_parser, phase2_cv_extractor, phase3_signal_enricher,
    phase4_candidate_judge, phase5_pool_calibrator,
)


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
) -> ATSState:
    graph = build_pipeline()
    initial_state: ATSState = {
        "jd_raw": jd_raw,
        "cv_raws": cv_raws,
        "jd_structured": None,
        "cv_profiles": [],
        "enriched_profiles": [],
        "candidate_assessments": [],
        "final_ranking": None,
        "run_id": run_id or str(uuid.uuid4()),
        "trace_log": [],
        "hallucination_flags": [],
        "use_cache": use_cache,
    }
    return graph.invoke(initial_state)
