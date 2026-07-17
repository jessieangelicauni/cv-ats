# Design: Remove LangGraph and Signal Enricher from EvidenceRank Pipeline

**Date:** 2026-07-17
**Status:** Approved

## Problem

The EvidenceRank pipeline has two issues:

1. **LangGraph is unnecessary.** The pipeline is a pure linear chain (phase1 → phase2 → phase3 → phase4 → phase5) with no branching, no cycles, no parallelism, and no human-in-the-loop checkpoints. The graph wiring adds dependency weight and forces all state to flow through a `TypedDict` dict-of-dicts interface that only exists to satisfy LangGraph's API. The `Annotated[list, operator.add]` reducers on `trace_log` and `hallucination_flags` solve a concurrent state-merge problem that never occurs in a linear graph.

2. **Phase 3 (signal enricher) duplicates work the judge already does.** The enricher uses the small model to pre-compute structured career signals (`company_tiers`, `career_trajectory`, `tenure_stability`, `relevant_experience_months`). The judge prompt already describes tier-1 companies by name and concept and derives the same signals from raw work history. The two-step approach adds an LLM call per candidate without a measurable quality benefit.

## Goal

Remove LangGraph entirely and collapse the pipeline to 4 phases. Remove the signal enricher agent, its prompt, and all schema types that only existed to carry its output (`EnrichmentSignals`, `EnrichedProfile`). Convert `ATSState` from a TypedDict to a Pydantic model (consistent with the rest of the codebase) and give each phase function an explicit, honest signature.

## Design

### 1. `src/graph/state.py` — TypedDict → Pydantic model

`ATSState` becomes a `BaseModel`. The `Annotated` reducers and `operator` import are removed. `enriched_profiles` is removed (no longer produced by any phase). All remaining fields keep the same names and types; mutable fields get `Field(default_factory=list)`.

```python
class ATSState(BaseModel):
    jd_raw: str
    cv_raws: list[dict]
    jd_structured: JDRequirements | None = None
    cv_profiles: list[CandidateProfile] = Field(default_factory=list)
    candidate_assessments: list[CandidateAssessment] = Field(default_factory=list)
    final_ranking: FinalRanking | None = None
    run_id: str
    otel_trace_id: str = ""
    trace_log: list[dict] = Field(default_factory=list)
    hallucination_flags: list[HallucinationFlag] = Field(default_factory=list)
    use_cache: bool = True
```

### 2. `src/models/schemas.py` — remove enricher types

`EnrichmentSignals` and `EnrichedProfile` are deleted. The three type aliases used only by those classes (`CompanyTier`, `CareerTrajectory`, `TenureStability`) are also removed. `Proficiency` stays — it is used by `SkillEntry`.

### 3. `src/graph/nodes.py` — explicit inputs and outputs, phase 3 removed

Phase 3 (`phase3_signal_enricher`) is deleted. Each remaining phase function drops the `ATSState` bag and takes only what it needs, returning only what it produces. The `_get_cache()` helper is removed — the caller passes the cache in directly. OTel spans and internal agent calls are unchanged.

| Function | Signature |
|---|---|
| `phase1_jd_parser` | `(jd_raw: str, cache: ExtractionCache \| None) -> JDRequirements` |
| `phase2_cv_extractor` | `(cv_raws: list[dict], cache: ExtractionCache \| None) -> list[CandidateProfile]` |
| `phase3_candidate_judge` | `(profiles: list[CandidateProfile], jd: JDRequirements) -> list[CandidateAssessment]` |
| `phase4_pool_calibrator` | `(candidate_assessments: list[CandidateAssessment], jd: JDRequirements) -> FinalRanking` |

The former phase 4 and 5 are renumbered to 3 and 4. The `return {"key": value, "trace_log": [...]}` dict at the end of each function is replaced by `return <result>`.

### 4. `src/agents/candidate_judge.py` — input type change

`CandidateJudgeAgent.run()` changes its parameter from `EnrichedProfile` to `CandidateProfile`. The judge prompt is unchanged — it already identifies tier-1 companies by name from the raw work history text.

### 5. `src/graph/pipeline.py` — sequential runner, 4 phases

`build_pipeline()` is deleted. `run_pipeline()` calls each phase directly via a `_run_phase()` helper that handles timing and the `on_phase_complete` callback. Phase 3 (enricher) is skipped; `cv_profiles` feeds directly into the judge.

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
assessments   = _run_phase(3, phase3_candidate_judge, cv_profiles, jd_structured)
final_ranking = _run_phase(4, phase4_pool_calibrator, assessments, jd_structured)

return ATSState(jd_raw=jd_raw, cv_raws=cv_raws, jd_structured=jd_structured,
                cv_profiles=cv_profiles, candidate_assessments=assessments,
                final_ranking=final_ranking, run_id=run_id,
                otel_trace_id=otel_trace_id, trace_log=trace_log, use_cache=use_cache)
```

All LangGraph imports are removed. The public signature of `run_pipeline()` is unchanged.

### 6. `main.py` — attribute access

All dict-style state access is updated to attribute access:

| Before | After |
|---|---|
| `state["final_ranking"]` | `state.final_ranking` |
| `state["jd_structured"].role_title` | `state.jd_structured.role_title` |
| `state["cv_profiles"]` | `state.cv_profiles` |
| `state["candidate_assessments"]` | `state.candidate_assessments` |
| `state["otel_trace_id"]` | `state.otel_trace_id` |
| `state["hallucination_flags"] = all_flags` | `state.hallucination_flags = all_flags` |

### 7. `src/output/report_generator.py` — remove enriched profile usage

The "Signal Summary" block per candidate and the per-role company tier annotations in "Work Experience" are removed — they were populated from `EnrichedProfile` fields that no longer exist. `_enriched_for()`, the `enriched` parameter on `_render_candidate_block()`, and the `enriched_list` local in `generate_report()` are all deleted. All dict-style state access is updated to attribute access (same as `main.py`). The rest of the report — ranking table, contact info, skills, education, work history, LLM judgment, hallucination summary — is unaffected.

### 8. `pyproject.toml` — remove dependency

`langgraph>=1.2.9` is removed. `langchain` and `langchain-core` remain because `langchain-ollama` and the agents import from `langchain_core` directly.

### 9. Tests

**Deleted entirely:**
- `tests/test_signal_enricher.py`

**Updated:**

| File | Change |
|---|---|
| `tests/test_pipeline.py` | Remove `build_pipeline` import, `MockSE`, `_mock_enriched()`, `EnrichedProfile` import; phase 4 mock now patches `CandidateJudgeAgent` with `CandidateProfile` input; update `result["final_ranking"]` → `result.final_ranking`, `result["trace_log"]` → `result.trace_log`; trace log assertion changes from 5 entries to 4 |
| `tests/test_candidate_judge.py` | `_make_enriched()` → `_make_profile()` returning `CandidateProfile` (drop all enrichment fields); update both test call sites |
| `tests/test_report_generator.py` | Remove `EnrichedProfile` construction and `"enriched_profiles"` key from state dict passed to `generate_report()` |
| `tests/test_schemas.py` | Remove `EnrichedProfile` and `EnrichmentSignals` from imports and any tests that construct them |

## Files Deleted

| File | Reason |
|---|---|
| `src/agents/signal_enricher.py` | Phase removed |
| `src/prompts/signal_enricher.py` | Phase removed |
| `tests/test_signal_enricher.py` | Agent deleted |

## Files Changed

| File | Change |
|---|---|
| `src/graph/state.py` | TypedDict → Pydantic BaseModel; remove `enriched_profiles` field and Annotated reducers |
| `src/models/schemas.py` | Remove `EnrichmentSignals`, `EnrichedProfile`, `CompanyTier`, `CareerTrajectory`, `TenureStability` |
| `src/graph/nodes.py` | Remove phase 3; explicit signatures for remaining phases; remove `_get_cache()`; return bare results; renumber phase 4→3, 5→4 |
| `src/agents/candidate_judge.py` | Input type `EnrichedProfile` → `CandidateProfile` |
| `src/graph/pipeline.py` | Remove LangGraph; 4-phase sequential runner with `_run_phase()` helper |
| `main.py` | Dict access → attribute access |
| `src/output/report_generator.py` | Remove Signal Summary block, enriched profile parameter, tier annotations; dict access → attribute access |
| `pyproject.toml` | Remove `langgraph` dependency |
| `tests/test_pipeline.py` | Remove enricher mock; 4-phase setup; attribute access on result |
| `tests/test_candidate_judge.py` | `EnrichedProfile` → `CandidateProfile` fixture |
| `tests/test_report_generator.py` | Remove `EnrichedProfile` and `enriched_profiles` from state fixture |
| `tests/test_schemas.py` | Remove `EnrichedProfile` and `EnrichmentSignals` |

## Files Not Changed

All files under `src/evaluation/`, `src/utils/`, `src/prompts/` (except `signal_enricher.py`), and the remaining test files are untouched.
