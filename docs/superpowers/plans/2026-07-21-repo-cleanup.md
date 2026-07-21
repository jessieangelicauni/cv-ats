# Repo Cleanup Sweep Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove unnecessary code from the `cv-ats` repo — dead code, an unused dependency, and stale documentation left over from the earlier RAG/ChromaDB removal refactor — while keeping the test suite green throughout.

**Architecture:** No architectural changes. This is a subtractive cleanup: delete unused imports and a dead config constant, drop an unused dependency, correct one pre-existing test bug that blocks a clean baseline, and rewrite the stale parts of README.md/.env.example. Each task is validated by running `uv run pytest` before moving to the next.

**Tech Stack:** Python 3.11, `uv` for dependency/env management, `pytest` for the test suite, `ruff`/`vulture` (run ephemerally via `uvx`, not added as project dependencies) for detection only.

## Global Constraints

- Docs scope: `docs/paper/evidencerank.tex` is out of scope — do not touch it.
- No new features, no unrelated refactors. Only remove what's confirmed unused, plus the one test-bug fix below.
- Test suite must be run after every individual change (not batched) — `uv run pytest -q`.
- Baseline established in Task 2 is **41 passed** — every later task must reproduce that exact result before its commit.
- Do not add `vulture`/`ruff` to `pyproject.toml` — invoke via `uvx <tool> ...` so they leave no permanent trace in the project.

---

## Investigation findings (reference for all tasks)

Verified during planning — every item below was cross-checked against actual usage, not just tool output, because static analysis tools produce false positives on Pydantic model fields and dynamically-registered functions:

- `src/utils/embedder.py` **must not be touched** — still imported by `src/utils/skill_normalizer.py` and `src/evaluation/hallucination_checker.py`.
- All `vulture`-flagged Pydantic fields in `src/models/schemas.py` (e.g. `leadership_expected`, `soft_skills`, `total_experience_months`, etc.) are false positives — confirmed used as constructor kwargs across the test suite and pipeline. Not touched by this plan.
- `config.BORDERLINE_SCORE_THRESHOLD` is a genuine dead constant: defined, read from `os.getenv`, and never referenced anywhere else in `src/`, `tests/`, or `main.py` (every sibling constant in `config.py` — `SKILL_MATCH_THRESHOLD`, `EMBEDDING_MODEL`, etc. — is referenced exactly once via `config.X`; this one is referenced zero times).
- No fully orphaned files exist — every module under `src/` is imported by at least one other file.
- No commented-out dead code blocks exist in `src/` or `tests/` — all `#` comments reviewed are genuine explanatory comments.
- No duplicate/redundant logic was found on manual review of `src/evaluation/`, `src/utils/`, and `src/agents/` (all under 90 lines each; cosine-similarity usage in `skill_normalizer.py` and `hallucination_checker.py` serves different call shapes and isn't a clean extraction target). **This plan has no "consolidate duplicate logic" task** — the design's category 3 review is satisfied by this documented finding, recorded in the cleanup report in Task 6.
- `scikit-learn` is an unused dependency — zero references to `sklearn` anywhere in `src/`, `tests/`, `main.py`, or `config.py`.
- README.md staleness is broader than the ChromaDB mentions found during brainstorming — it also has an entire fictional "Telemetry" section (Langfuse/OpenTelemetry), a phantom `--session-id` CLI flag, and two wrong default values. Full list is in Task 5.

---

### Task 1: Commit in-progress work

**Files:** none created/modified by this task — commits the pre-existing working-tree changes as-is.

**Interfaces:** N/A (git operation only).

- [ ] **Step 1: Review what's currently modified**

Run: `git status --short`
Expected output (order may vary):
```
 M .env.example
 M README.md
 M config.py
 M src/agents/candidate_judge.py
 M src/evaluation/hallucination_checker.py
 M src/graph/nodes.py
 M src/graph/pipeline.py
 M src/prompts/judge.py
 M tests/test_hallucination_checker.py
?? resume/... (various sample PDFs)
```

- [ ] **Step 2: Stage and commit the modified files (not the untracked sample PDFs)**

```bash
git add .env.example README.md config.py \
  src/agents/candidate_judge.py src/evaluation/hallucination_checker.py \
  src/graph/nodes.py src/graph/pipeline.py src/prompts/judge.py \
  tests/test_hallucination_checker.py
git commit -m "wip: in-progress pipeline/judge changes before cleanup sweep"
```

- [ ] **Step 3: Verify a clean-enough tree**

Run: `git status --short`
Expected: only the untracked `resume/*.pdf` sample files remain (those are sample data, not part of this cleanup — leave untouched).

---

### Task 2: Fix pre-existing test bug and establish a clean baseline

**Files:**
- Modify: `tests/test_report_generator.py:53`

**Interfaces:** N/A — internal test fixture fix, no production code touched.

**Context:** Running `uv run pytest -q` on the current tree fails 4 tests in `test_report_generator.py` with `pydantic_core._pydantic_core.ValidationError: 1 validation error for HallucinationFlag / status / Input should be 'inferred', 'fabricated' or 'acknowledged_gap'`. The fixture at line 52-54 constructs a `HallucinationFlag` with `status="supported"`, which isn't a valid literal on the schema. The claim in that fixture ("Strong Python.") is backed by the quote "Python dev", which is verbatim-present in the raw CV text ("CV text" — actually not verbatim, see below), so semantically the value should be `"inferred"` (the status `hallucination_checker.py` assigns to claims that are verbatim or near-verbatim matches, as opposed to `"fabricated"`).

- [ ] **Step 1: Confirm the current failure**

Run: `uv run pytest tests/test_report_generator.py -q`
Expected: `4 failed` with the `ValidationError` shown above.

- [ ] **Step 2: Fix the invalid literal**

In `tests/test_report_generator.py`, change:

```python
        hallucination_flags=[HallucinationFlag(candidate_id="cv_001",
                                               claim="Strong Python.", status="supported",
                                               source_quote="Python dev")],
```

to:

```python
        hallucination_flags=[HallucinationFlag(candidate_id="cv_001",
                                               claim="Strong Python.", status="inferred",
                                               source_quote="Python dev")],
```

- [ ] **Step 3: Run the full suite to confirm a clean baseline**

Run: `uv run pytest -q`
Expected: `41 passed`

- [ ] **Step 4: Commit**

```bash
git add tests/test_report_generator.py
git commit -m "fix: correct invalid HallucinationFlag status literal in test fixture"
```

---

### Task 3: Remove dead code (unused imports + dead config constant)

**Files:**
- Modify: `main.py:15`
- Modify: `src/evaluation/ranking_metrics.py:2`
- Modify: `tests/test_calibration_metrics.py:2`
- Modify: `tests/test_pipeline.py:1`
- Modify: `tests/test_pool_calibrator.py:4`
- Modify: `tests/test_schemas.py:1-7`
- Modify: `config.py:14`
- Modify: `.env.example:18-19`

**Interfaces:** N/A — pure removals, no signatures change.

**Context:** Every removal below was confirmed unused via `uvx ruff check --select F401,F841 src/ tests/ config.py main.py` and manual verification (shown in the Investigation findings section above). `import config` in `main.py` is safe to remove even though `config.py` has import-time side effects (`load_dotenv()`, `ensure_dirs()`) — those side effects already fire because `main.py` imports `src.graph.pipeline`, which imports `config` itself, before `main.py`'s own `import config` line runs.

- [ ] **Step 1: Remove the unused `import config` from `main.py`**

Change:
```python
from src.evaluation.consistency_runner import run_consistency_experiment
import config

app = typer.Typer()
```
to:
```python
from src.evaluation.consistency_runner import run_consistency_experiment

app = typer.Typer()
```

- [ ] **Step 2: Remove the unused numpy import from `src/evaluation/ranking_metrics.py`**

Change:
```python
from __future__ import annotations
import numpy as np
from scipy.stats import kendalltau
```
to:
```python
from __future__ import annotations
from scipy.stats import kendalltau
```

- [ ] **Step 3: Remove the unused `EvidenceItem` import from `tests/test_calibration_metrics.py`**

Change:
```python
from src.evaluation.calibration_metrics import calibration_report
from src.models.schemas import CandidateAssessment, EvidenceItem, FinalRanking, RankedCandidate
```
to:
```python
from src.evaluation.calibration_metrics import calibration_report
from src.models.schemas import CandidateAssessment, FinalRanking, RankedCandidate
```

- [ ] **Step 4: Remove the unused `MagicMock` import from `tests/test_pipeline.py`**

Change:
```python
from unittest.mock import MagicMock, patch
```
to:
```python
from unittest.mock import patch
```

- [ ] **Step 5: Remove the unused `BorderlinePairByPosition` import from `tests/test_pool_calibrator.py`**

Change:
```python
from src.models.schemas import (
    FinalRanking, PoolCalibrationResult, CalibratedEntry, BorderlinePairByPosition,
    CandidateAssessment, EvidenceItem, JDRequirements, EducationRequirement,
)
```
to:
```python
from src.models.schemas import (
    FinalRanking, PoolCalibrationResult, CalibratedEntry,
    CandidateAssessment, EvidenceItem, JDRequirements, EducationRequirement,
)
```

- [ ] **Step 6: Remove the 8 unused imports from `tests/test_schemas.py`**

Change:
```python
from src.models.schemas import (
    JDRequirements, SkillRequirement, EducationRequirement,
    CandidateProfile, CandidateBasicInfo, SkillEntry, SkillMatchResult,
    WorkEntry, EducationEntry, LanguageEntry,
    CandidateAssessment, EvidenceItem, HallucinationFlag,
    FinalRanking, RankedCandidate,
)
```
to:
```python
from src.models.schemas import (
    JDRequirements, EducationRequirement,
    SkillEntry, SkillMatchResult,
    EvidenceItem,
    FinalRanking, RankedCandidate,
)
```

(`SkillRequirement`, `CandidateProfile`, `CandidateBasicInfo`, `WorkEntry`, `EducationEntry`, `LanguageEntry`, `CandidateAssessment`, `HallucinationFlag` are never referenced in this file's test bodies.)

- [ ] **Step 7: Remove the dead `BORDERLINE_SCORE_THRESHOLD` constant from `config.py`**

Change:
```python
EXTRACTION_TEMPERATURE = float(os.getenv("EXTRACTION_TEMPERATURE", "0.0"))
JUDGE_TEMPERATURE = float(os.getenv("JUDGE_TEMPERATURE", "0.1"))

BORDERLINE_SCORE_THRESHOLD = float(os.getenv("BORDERLINE_SCORE_THRESHOLD", "5.0"))
SKILL_MATCH_THRESHOLD = float(os.getenv("SKILL_MATCH_THRESHOLD", "0.80"))
```
to:
```python
EXTRACTION_TEMPERATURE = float(os.getenv("EXTRACTION_TEMPERATURE", "0.0"))
JUDGE_TEMPERATURE = float(os.getenv("JUDGE_TEMPERATURE", "0.1"))

SKILL_MATCH_THRESHOLD = float(os.getenv("SKILL_MATCH_THRESHOLD", "0.80"))
```

- [ ] **Step 8: Remove the corresponding entry from `.env.example`**

Change:
```
# ── Evaluation ────────────────────────────────────────────────────────────────
# Scores within ±N points of the pool mean are candidates for Phase 4 calibration
BORDERLINE_SCORE_THRESHOLD=5.0

# Sentence-Transformers model used for embeddings (skill matching + hallucination checker)
```
to:
```
# ── Evaluation ────────────────────────────────────────────────────────────────
# Sentence-Transformers model used for embeddings (skill matching + hallucination checker)
```

- [ ] **Step 9: Run the full suite**

Run: `uv run pytest -q`
Expected: `41 passed`

- [ ] **Step 10: Run ruff again to confirm the flagged imports are gone**

Run: `uvx ruff check --select F401,F841 src/ tests/ config.py main.py`
Expected: `All checks passed!`

- [ ] **Step 11: Commit**

```bash
git add main.py src/evaluation/ranking_metrics.py tests/test_calibration_metrics.py \
  tests/test_pipeline.py tests/test_pool_calibrator.py tests/test_schemas.py \
  config.py .env.example
git commit -m "refactor: remove unused imports and dead BORDERLINE_SCORE_THRESHOLD config"
```

---

### Task 4: Remove unused dependency (scikit-learn)

**Files:**
- Modify: `pyproject.toml`
- Modify: `uv.lock` (regenerated, not hand-edited)

**Interfaces:** N/A.

**Context:** `scikit-learn>=1.8.0` is declared in `pyproject.toml` but nothing in `src/`, `tests/`, `main.py`, or `config.py` imports `sklearn` (confirmed by grep across the whole project). `numpy` and `scipy` are genuinely used (by `src/evaluation/calibration_metrics.py`, `consistency_runner.py`) and stay.

- [ ] **Step 1: Remove the dependency line**

In `pyproject.toml`, change:
```toml
dependencies = [
    "langchain>=1.3.13",
    "langchain-core>=1.4.9",
    "langchain-ollama>=1.1.0",
    "pydantic>=2.11.7,<3.0",
    "pymupdf>=1.26.3",
    "pdfplumber>=0.11.10",
    "numpy>=1.26.4,<2.0",
    "scipy>=1.15.3",
    "scikit-learn>=1.8.0",
    "sentence-transformers>=5.6.0",
    "rich>=14.3.3",
    "typer>=0.24.1",
    "python-dotenv>=1.0.1",
]
```
to:
```toml
dependencies = [
    "langchain>=1.3.13",
    "langchain-core>=1.4.9",
    "langchain-ollama>=1.1.0",
    "pydantic>=2.11.7,<3.0",
    "pymupdf>=1.26.3",
    "pdfplumber>=0.11.10",
    "numpy>=1.26.4,<2.0",
    "scipy>=1.15.3",
    "sentence-transformers>=5.6.0",
    "rich>=14.3.3",
    "typer>=0.24.1",
    "python-dotenv>=1.0.1",
]
```

- [ ] **Step 2: Regenerate the lockfile and sync the environment**

```bash
uv lock
uv sync --link-mode=copy
```
Expected: `uv lock` reports the lockfile updated (scikit-learn and its exclusive sub-dependencies dropped); `uv sync` completes without error.

- [ ] **Step 3: Confirm scikit-learn is actually gone from the environment**

Run: `uv run python -c "import sklearn"`
Expected: `ModuleNotFoundError: No module named 'sklearn'`

- [ ] **Step 4: Run the full suite**

Run: `uv run pytest -q`
Expected: `41 passed`

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "chore: remove unused scikit-learn dependency"
```

---

### Task 5: Fix stale documentation in README.md

**Files:**
- Modify: `README.md`

**Interfaces:** N/A — documentation only.

**Context:** README.md describes ChromaDB-based retrieval, a Langfuse/OpenTelemetry "Telemetry" section, and a `--session-id` CLI flag, none of which exist in the current code (confirmed: no `telemetry.py` file anywhere in the repo, no `langfuse`/`opentelemetry` references outside README.md itself, `main.py`'s Typer options are only `--jd`, `--cv-dir`, `--output`, `--runs`, `--eval`, `--no-cache`). It also states two config defaults that don't match `config.py`: `SKILL_MATCH_THRESHOLD` default is documented as `0.75` but `config.py:15` defaults to `0.80`, and the embedding model is documented as `all-MiniLM-L6-v2` but `config.py:16` defaults to `BAAI/bge-small-en-v1.5`. Each edit below is independent; apply them in order (line numbers shift after each edit, so re-locate by the surrounding text shown, not by line number).

- [ ] **Step 1: Fix the intro paragraph (remove Langfuse/OpenTelemetry claim)**

Change:
```markdown
EvidenceRank ranks IT job candidates against a job description using a local-first,
4-phase LLM pipeline. Every score is traced end-to-end via Langfuse/OpenTelemetry,
and an evaluation suite checks outputs for hallucination, calibration, and
run-to-run consistency.
```
to:
```markdown
EvidenceRank ranks IT job candidates against a job description using a local-first,
4-phase LLM pipeline. An evaluation suite checks outputs for hallucination,
calibration, and run-to-run consistency.
```

- [ ] **Step 2: Fix the Phase 2 table row (remove ChromaDB claim)**

Change:
```markdown
| 2 | `src/agents/cv_extractor.py` | `src/prompts/cv_extractor.py` | Extract structured candidate profiles from CV PDFs; CV text is first indexed into ChromaDB and three targeted queries (experience, skills, education) retrieve focused chunks for the extractor |
```
to:
```markdown
| 2 | `src/agents/cv_extractor.py` | `src/prompts/cv_extractor.py` | Extract structured candidate profiles from CV PDFs in a single LLM call |
```

- [ ] **Step 3: Fix the pre-judge filter threshold default**

Change:
```markdown
Between Phase 2 and Phase 3, a **pre-judge filter** (`_filter_by_required_skills`) eliminates
candidates where no required skill exceeds the cosine similarity threshold
(`SKILL_MATCH_THRESHOLD`, default `0.75`), avoiding unnecessary LLM calls.
```
to:
```markdown
Between Phase 2 and Phase 3, a **pre-judge filter** (`_filter_by_required_skills`) eliminates
candidates where no required skill exceeds the cosine similarity threshold
(`SKILL_MATCH_THRESHOLD`, default `0.80`), avoiding unnecessary LLM calls.
```

- [ ] **Step 4: Fix the `utils/` module layout bullet**

Change:
```markdown
- `utils/` — PDF extraction, extraction cache (`cache.py`), ChromaDB vector store
  (`vector_store.py`), semantic skill matcher (`skill_matcher.py`), shared MiniLM
  embedder singleton (`embedder.py`), telemetry setup, and LLM client
```
to:
```markdown
- `utils/` — PDF extraction (`pdf_extractor.py`), extraction cache (`cache.py`),
  semantic skill matcher (`skill_normalizer.py`), shared Sentence-Transformers
  embedder singleton (`embedder.py`), and LLM client (`llm.py`)
```

- [ ] **Step 5: Fix the embedding model description in Setup**

Change:
```markdown
   The MiniLM embedding model (`all-MiniLM-L6-v2`) is downloaded automatically from
   HuggingFace on first run. Set `HF_TOKEN` in `.env` to avoid unauthenticated rate
   limits if you hit them.
```
to:
```markdown
   The embedding model (`EMBEDDING_MODEL` in `.env`, default `BAAI/bge-small-en-v1.5`)
   is downloaded automatically from HuggingFace on first run. Set `HF_TOKEN` in `.env`
   to avoid unauthenticated rate limits if you hit them.
```

- [ ] **Step 6: Fix the `--no-cache` flag description and remove the phantom `--session-id` row**

Change:
```markdown
| `--no-cache` | Disable the PDF extraction cache and ChromaDB vector store |
| `--session-id` | Tag this run with a session ID to group related runs in Langfuse |
```
to:
```markdown
| `--no-cache` | Disable the PDF extraction cache |
```

- [ ] **Step 7: Fix the hallucination threshold reference**

Change:
```markdown
- **Hallucination rate** (`src/evaluation/hallucination_checker.py`) — for each judge
  claim, verifies it against the source CV text using sentence-embedding similarity.
  Claims below `HALLUCINATION_SIMILARITY_THRESHOLD` (default `0.85`) are flagged as
  `fabricated`; claims above it but not verbatim are `inferred`.
```
to:
```markdown
- **Hallucination rate** (`src/evaluation/hallucination_checker.py`) — for each judge
  claim, verifies it against the source CV text using sentence-embedding similarity.
  Claims below the similarity threshold (`SIMILARITY_THRESHOLD = 0.85`, a module
  constant, not env-configurable) are flagged as `fabricated`; claims above it but
  not verbatim are `inferred`.
```

- [ ] **Step 8: Fix the calibration description (remove reference to the now-deleted config constant)**

Change:
```markdown
- **Calibration** (`src/evaluation/calibration_metrics.py`) — compares raw Phase 3
  scores to the Phase 4 pool-calibrated scores (std dev, mean absolute delta, and how
  many candidates changed rank). Scores within `BORDERLINE_SCORE_THRESHOLD` (default
  `5.0`) points of the pool mean are the ones Phase 4 is most likely to adjust.
```
to:
```markdown
- **Calibration** (`src/evaluation/calibration_metrics.py`) — compares raw Phase 3
  scores to the Phase 4 pool-calibrated scores (std dev, mean absolute delta, and how
  many candidates changed rank).
```

- [ ] **Step 9: Remove the entire Telemetry section**

Change:
```markdown
## Telemetry

Every run is traced via OpenTelemetry and exported to [Langfuse](https://langfuse.com)
(`src/utils/telemetry.py`). Configure in `.env`:

- `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` / `LANGFUSE_HOST` — from Langfuse Cloud
  (`https://cloud.langfuse.com`) or a self-hosted instance. `LANGFUSE_HOST` is the only
  thing that controls where traces are sent — the Langfuse SDK builds its own OTLP
  exporter from it (`{LANGFUSE_HOST}/api/public/otel/v1/traces`, authenticated with the
  public/secret key pair); it does not read the standard `OTEL_EXPORTER_OTLP_ENDPOINT`
  env var, so setting that has no effect.
- `OTEL_SERVICE_NAME` / `OTEL_SERVICE_VERSION` — picked up automatically by the
  OpenTelemetry SDK's resource detection and attached to every trace.
- `TELEMETRY_ENABLED` — set to `false` to disable tracing entirely (offline runs, CI,
  unit tests).

Evaluation metrics (hallucination rate, calibration stats, consistency τ) are also
logged back onto each run's trace as Langfuse scores. Consistency experiment sub-runs
are nested under a parent `consistency_experiment` span and share a `session_id` for
side-by-side comparison in Langfuse.

## Testing
```
to:
```markdown
## Testing
```

- [ ] **Step 10: Fix the `.cache/` line in Project Structure**

Change:
```markdown
├── .cache/                  # extraction cache + ChromaDB vector store (gitignored)
```
to:
```markdown
├── .cache/                  # extraction cache (gitignored)
```

- [ ] **Step 11: Fix the `utils/` line in the Project Structure tree**

Change:
```markdown
│   └── utils/               # PDF extraction, cache, embedder, vector store,
│                            #   skill matcher, telemetry, LLM client
```
to:
```markdown
│   └── utils/               # PDF extraction, cache, embedder, skill matcher, LLM client
```

- [ ] **Step 12: Run the full suite (docs-only change, but confirms nothing else broke in this commit)**

Run: `uv run pytest -q`
Expected: `41 passed`

- [ ] **Step 13: Commit**

```bash
git add README.md
git commit -m "docs: remove stale ChromaDB/Langfuse/OpenTelemetry references from README"
```

---

### Task 6: Write the cleanup report and do a final verification pass

**Files:**
- Create: `docs/cleanup-report.md`

**Interfaces:** N/A — documentation deliverable.

- [ ] **Step 1: Write the report**

Create `docs/cleanup-report.md`:

```markdown
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
```

- [ ] **Step 2: Final full-suite run**

Run: `uv run pytest -q`
Expected: `41 passed`

- [ ] **Step 3: Commit**

```bash
git add -f docs/cleanup-report.md
git commit -m "docs: add repo cleanup report"
```

- [ ] **Step 4: Review the full set of cleanup commits**

Run: `git log --oneline -8`
Expected: 6 new commits on top of the pre-cleanup history — the wip commit
(Task 1), the test-bug fix (Task 2), the dead-code removal (Task 3), the
dependency removal (Task 4), the README fix (Task 5), and the cleanup report
(Task 6).
