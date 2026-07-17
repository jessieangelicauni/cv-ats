# Design: Remove LangGraph from EvidenceRank Pipeline

**Date:** 2026-07-17
**Status:** Approved

## Problem

The EvidenceRank pipeline uses LangGraph as a sequential executor over 5 phases with no branching, no cycles, no parallelism, and no human-in-the-loop checkpoints. The graph wiring adds dependency weight and forces all state to flow through a `TypedDict` dict-of-dicts interface that only exists to satisfy LangGraph's API. The `Annotated[list, operator.add]` reducers on `trace_log` and `hallucination_flags` solve a concurrent state-merge problem that never occurs in a linear graph.

## Goal

Remove LangGraph entirely and replace the pipeline with a plain sequential function. As a byproduct, convert `ATSState` from a TypedDict to a Pydantic model (consistent with the rest of the codebase) and give each phase function an explicit, honest signature.

## Design

### 1. `src/graph/state.py` — TypedDict → Pydantic model

`ATSState` becomes a `BaseModel`. The `Annotated` reducers and `operator` import are removed. All fields keep the same names and types; mutable fields get `Field(default_factory=list)`.

```python
class ATSState(BaseModel):
    jd_raw: str
    cv_raws: list[dict]
    jd_structured: JDRequirements | None = None
    cv_profiles: list[CandidateProfile] = Field(default_factory=list)
    enriched_profiles: list[EnrichedProfile] = Field(default_factory=list)
    candidate_assessments: list[CandidateAssessment] = Field(default_factory=list)
    final_ranking: FinalRanking | None = None
    run_id: str
    otel_trace_id: str = ""
    trace_log: list[dict] = Field(default_factory=list)
    hallucination_flags: list[HallucinationFlag] = Field(default_factory=list)
    use_cache: bool = True
```

### 2. `src/graph/nodes.py` — explicit inputs and outputs

Each phase function drops the `ATSState` bag and takes only what it needs, returning only what it produces. The `_get_cache()` helper is removed — the caller passes the cache in directly. OTel spans and internal agent calls are unchanged.

| Function | Signature |
|---|---|
| `phase1_jd_parser` | `(jd_raw: str, cache: ExtractionCache \| None) -> JDRequirements` |
| `phase2_cv_extractor` | `(cv_raws: list[dict], cache: ExtractionCache \| None) -> list[CandidateProfile]` |
| `phase3_signal_enricher` | `(cv_profiles: list[CandidateProfile], jd: JDRequirements) -> list[EnrichedProfile]` |
| `phase4_candidate_judge` | `(enriched_profiles: list[EnrichedProfile], jd: JDRequirements) -> list[CandidateAssessment]` |
| `phase5_pool_calibrator` | `(candidate_assessments: list[CandidateAssessment], jd: JDRequirements) -> FinalRanking` |

The `return {"key": value, "trace_log": [...]}` dict at the end of each function is replaced by `return <result>`. Everything else inside each function body is unchanged.

### 3. `src/graph/pipeline.py` — sequential runner

`build_pipeline()` is deleted. `run_pipeline()` calls each phase directly, builds `trace_log` entries around each call with a local `_run_phase()` helper, fires `on_phase_complete` after each phase, then constructs and returns an `ATSState`:

```python
def _run_phase(n, fn, *args, **extra_log):
    t0 = time.time()
    result = fn(*args)
    entry = {"phase": n, "duration_s": round(time.time() - t0, 2), **extra_log}
    trace_log.append(entry)
    if on_phase_complete:
        on_phase_complete(entry)
    return result

jd_structured = _run_phase(1, phase1_jd_parser, jd_raw, cache)
cv_profiles   = _run_phase(2, phase2_cv_extractor, cv_raws, cache, candidates=len(cv_raws))
enriched      = _run_phase(3, phase3_signal_enricher, cv_profiles, jd_structured)
assessments   = _run_phase(4, phase4_candidate_judge, enriched, jd_structured)
final_ranking = _run_phase(5, phase5_pool_calibrator, assessments, jd_structured)

return ATSState(jd_raw=jd_raw, cv_raws=cv_raws, ...)
```

All LangGraph imports are removed from this file. The public signature of `run_pipeline()` is unchanged.

### 4. `main.py` — attribute access

All dict-style state access is updated to attribute access:

| Before | After |
|---|---|
| `state["final_ranking"]` | `state.final_ranking` |
| `state["jd_structured"].role_title` | `state.jd_structured.role_title` |
| `state["cv_profiles"]` | `state.cv_profiles` |
| `state["candidate_assessments"]` | `state.candidate_assessments` |
| `state["otel_trace_id"]` | `state.otel_trace_id` |
| `state["hallucination_flags"] = all_flags` | `state.hallucination_flags = all_flags` |

### 5. `pyproject.toml` — remove dependency

`langgraph>=1.2.9` is removed. `langchain` and `langchain-core` remain because `langchain-ollama` and the agents import from `langchain_core` directly.

### 6. `tests/test_pipeline.py` — minor updates

- Remove `build_pipeline` from the import line
- `result["final_ranking"]` → `result.final_ranking`
- `result["trace_log"]` → `result.trace_log`
- All agent patches (`patch("src.graph.nodes.JDParserAgent")` etc.) are unchanged

No other test files are affected — the rest of the suite patches agents in isolation and does not touch `ATSState` or the pipeline.

## Files Changed

| File | Change |
|---|---|
| `src/graph/state.py` | TypedDict → Pydantic BaseModel, remove Annotated reducers |
| `src/graph/nodes.py` | Explicit function signatures, remove `_get_cache()`, return bare results |
| `src/graph/pipeline.py` | Remove LangGraph, sequential runner with `_run_phase()` helper |
| `main.py` | Dict access → attribute access (8 sites) |
| `pyproject.toml` | Remove `langgraph` dependency |
| `tests/test_pipeline.py` | Remove `build_pipeline` import, update 2 result access lines |

## Files Not Changed

All other files under `src/agents/`, `src/prompts/`, `src/evaluation/`, `src/output/`, `src/utils/`, `src/models/`, and the rest of `tests/` are untouched.
