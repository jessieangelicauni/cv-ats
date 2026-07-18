---
name: remove-otel-langfuse
description: Remove all OpenTelemetry and Langfuse telemetry from EvidenceRank code and paper; invoke_with_telemetry renamed to plain chain.invoke at each call site
metadata:
  type: project
---

# Remove OpenTelemetry + Langfuse

## Goal

Strip all OTel tracing and Langfuse observability from the codebase. Each agent invokes its LLM chain directly via `chain.invoke(messages)`. No new observability layer is introduced.

---

## Files to Delete (1)

| File | Reason |
|---|---|
| `src/utils/telemetry.py` | Entire telemetry module — Langfuse init, OTel tracer, callback handler, shutdown, trace ID extraction |

---

## Files to Modify (11 code + 1 test + 2 config + 1 paper)

### `src/utils/llm.py`
- Remove `RunnableConfig` import
- Remove `_llm_call_count`, `reset_call_count`, `get_call_count`
- Remove `invoke_with_telemetry` function entirely
- Keep only `get_llm`

### `src/agents/jd_parser.py`
- Remove `from opentelemetry.trace import get_current_span`
- Remove both `get_current_span().set_attribute(...)` calls (cache hit/miss)
- Change `invoke_with_telemetry(self._llm, [...], run_name="jd_parser")` → `self._llm.invoke([...])`
- Remove `invoke_with_telemetry` from import

### `src/agents/cv_extractor.py`
- Remove `from opentelemetry.trace import get_current_span`
- Remove both `get_current_span().set_attribute(...)` calls
- Change `invoke_with_telemetry(self._extract_llm, [...], run_name="cv_extractor.extract")` → `self._extract_llm.invoke([...])`
- Remove `invoke_with_telemetry` from import

### `src/agents/candidate_judge.py`
- Remove `invoke_with_telemetry` from import
- Change `invoke_with_telemetry(self._llm, [...], run_name="candidate_judge")` → `self._llm.invoke([...])`

### `src/agents/pool_calibrator.py`
- Remove `invoke_with_telemetry` from import
- Change `invoke_with_telemetry(self._llm, [...], run_name="pool_calibrator")` → `self._llm.invoke([...])`

### `src/utils/skill_normalizer.py`
- Remove `invoke_with_telemetry` from import
- Change `invoke_with_telemetry(llm, [...], run_name="skill_normalizer")` → `llm.invoke([...])`

### `src/graph/nodes.py`
- Remove `from opentelemetry.trace import StatusCode`
- Remove `from src.utils.telemetry import get_tracer`
- Remove `_tracer = get_tracer()` module-level call
- Simplify each phase function — remove all `with _tracer.start_as_current_span(...) as span:` blocks, `span.set_attribute()`, `span.record_exception()`, `span.set_status()` calls. Each function becomes a direct agent call (see invariants).

### `src/graph/pipeline.py`
- Remove `setup_telemetry`, `get_tracer`, `current_otel_trace_id` from telemetry import (import line deleted entirely)
- Remove `session_id: str | None = None` parameter — was Langfuse-only
- Remove `setup_telemetry()` call
- Remove the outer `with tracer.start_as_current_span("pipeline", ...) as ...:` block — inline its body directly
- Remove `otel_trace_id = current_otel_trace_id()` and `otel_trace_id` from `ATSState(...)` construction

### `src/graph/state.py`
- Remove `otel_trace_id: str = ""` field

### `src/evaluation/consistency_runner.py`
- Remove `session_id` generation and the `session_id=session_id` arg to `run_pipeline`
- Remove `otel_trace_ids: list[str] = []` and `otel_trace_ids.append(state.otel_trace_id)`
- Remove `"session_id"` and `"otel_trace_ids"` from return dict

### `main.py`
- Remove `from src.utils.telemetry import setup_telemetry, get_langfuse, get_tracer, shutdown`
- Remove `--session-id` CLI param (`session_id: str | None = typer.Option(...)`)
- Remove `setup_telemetry()` call
- Remove `lf = get_langfuse()`
- Remove `otel_trace_id = state.otel_trace_id` (and all references to `otel_trace_id`)
- Remove the entire Langfuse score block inside `if runs > 1:` (the `from itertools import combinations` import + the two `lf.create_score` loops)
- Remove the `tracer = get_tracer()` and `with tracer.start_as_current_span("consistency_experiment", ...):` wrapper in the `runs > 1` branch — inline its body
- Remove all `lf.create_score(...)` calls inside `if eval:`
- Remove the `try: ... finally: shutdown()` wrapper entirely — `shutdown()` is the only thing in the `finally:` block, so remove both `try:` and `finally: shutdown()` and unindent the pipeline body one level
- Remove `session_id=session_id` from `run_pipeline(...)` call

### `tests/test_cv_extractor.py`
- Replace `patch("src.agents.cv_extractor.invoke_with_telemetry", return_value=_make_mock_profile())` with setting `mock_extract_llm.invoke.return_value = _make_mock_profile()` on the mock LLM
- Both tests need this update (the second test also patches `invoke_with_telemetry`)

### `pyproject.toml`
- Remove `"langfuse>=4.12.0"`
- Remove `"opentelemetry-sdk>=1.43.0"`
- Remove `"opentelemetry-exporter-otlp-proto-http>=1.43.0"`

### `.env.example`
- Remove the `# ── Langfuse ──` section (3 vars: `LANGFUSE_SECRET_KEY`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_HOST`)
- Remove the `# ── OpenTelemetry ──` section (`OTEL_SERVICE_NAME`, `OTEL_SERVICE_VERSION`)
- Remove the `TELEMETRY_ENABLED` line and its comment

---

## Paper Changes (`docs/paper/evidencerank.tex`)

1. **Abstract contribution 1** — Remove `with full OpenTelemetry tracing via Langfuse`
2. **Architecture Overview** (Section 3.1) — Remove `and OpenTelemetry span` from the sentence about each phase's implementation
3. **Hallucination Detection** (Section 3.6) — Remove `and stored as a Langfuse score` from the sentence about hallucination rate reporting
4. **System Configuration and Observability** (Section 3.8) — Keep the thresholds paragraph; remove the entire observability paragraph (the one starting `Every pipeline run produces an OpenTelemetry trace...`)

---

## Invariants After Removal

- The four phase functions in `nodes.py` become direct agent calls with no span wrapping
- `run_pipeline` loses `session_id` but retains `run_id` (still used for output directory naming)
- `ATSState` loses `otel_trace_id`; `consistency_runner` return dict loses `session_id` and `otel_trace_ids`
- `pytest` must pass cleanly after all changes
- `uv run python main.py --help` must show no `--session-id` flag and no import errors
