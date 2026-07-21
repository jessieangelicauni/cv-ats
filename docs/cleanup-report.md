# Repo Cleanup Report — 2026-07-21

Cleanup sweep of the `cv-ats` repo, scoped to `src/`, `tests/`, `config.py`,
`main.py`, `pyproject.toml`/`uv.lock`, and `README.md`/`.env.example`.
`docs/paper/evidencerank.tex` was explicitly out of scope.
Full design: `docs/superpowers/specs/2026-07-21-repo-cleanup-design.md`.

## Pre-existing bug fixed

- `tests/test_report_generator.py` — `HallucinationFlag(status="supported", ...)`
  used an invalid literal (schema only allows `inferred`/`fabricated`/
  `acknowledged_gap`), failing 4 tests. Changed to `status="inferred"`, matching
  the semantics of a verbatim-backed claim. Unrelated to the RAG/ChromaDB
  removal that motivated this cleanup, but blocked establishing a clean test
  baseline, so fixed as part of this sweep at the user's request.

## Dead code removed

- `main.py:15` — unused `import config` (side effects it relied on already
  fire via the transitive `src.graph.pipeline` import).
- `src/evaluation/ranking_metrics.py:2` — unused `import numpy as np`.
- `tests/test_calibration_metrics.py`, `test_pipeline.py`,
  `test_pool_calibrator.py`, `test_schemas.py` — 11 unused imports total
  (`EvidenceItem`, `MagicMock`, `BorderlinePairByPosition`, and 8 unused
  schema classes in `test_schemas.py`), found via
  `uvx ruff check --select F401,F841`.
- `config.BORDERLINE_SCORE_THRESHOLD` (`config.py`, `.env.example`) — defined
  and read from the environment but never consumed anywhere in `src/`; every
  sibling config constant is referenced exactly once, this one zero times.

## Not removed (reviewed, false positives / genuinely in use)

- `src/utils/embedder.py` — still imported by `skill_normalizer.py` and
  `hallucination_checker.py`.
- All `vulture`-flagged Pydantic fields on `src/models/schemas.py` models —
  confirmed used as constructor kwargs throughout the test suite and pipeline.
- No orphaned/dead whole files were found — every `src/` module is imported
  by at least one other file.
- No duplicate/redundant logic was found on manual review of `evaluation/`,
  `utils/`, and `agents/` — modules are small (under 90 lines each) and the
  cosine-similarity usage in `skill_normalizer.py` vs `hallucination_checker.py`
  serves different call shapes, not a clean extraction target.

## Unused dependency removed

- `scikit-learn` (`pyproject.toml`) — zero `sklearn` references anywhere in
  the codebase. `uv.lock` regenerated accordingly.

## Stale documentation fixed

- `README.md` — removed an entire "Telemetry" section describing
  Langfuse/OpenTelemetry integration that doesn't exist in the code (no
  `telemetry.py` file, no `langfuse`/`opentelemetry` references anywhere
  except the README itself); removed a phantom `--session-id` CLI flag not
  present in `main.py`'s actual Typer options; removed ChromaDB
  vector-store/indexing references from the architecture description,
  `--no-cache` flag description, and Project Structure tree; corrected two
  wrong defaults (`SKILL_MATCH_THRESHOLD` documented as `0.75`, actually
  `0.80`; embedding model documented as `all-MiniLM-L6-v2`, actually
  `BAAI/bge-small-en-v1.5`).
- `.env.example` — removed the `BORDERLINE_SCORE_THRESHOLD` entry alongside
  its config-code removal.

## Verification

`uv run pytest -q` passed with 41 tests after every commit in this sweep.
