# CV Extractor Reference Date Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ground the CV extractor's resolution of "Present"/"Current" work-history dates in the actual pipeline run date, instead of the LLM's own guess at what "today" is.

**Architecture:** No architectural changes. `src/prompts/cv_extractor.py` gains a reference date computed at prompt-build time (`datetime.date.today()`), included as a `REFERENCE DATE:` line in the human message, with an explicit system-prompt rule telling the model to resolve "Present"/"Current" against it. No new parameters on any function signature outside this file; no pipeline/state/graph changes.

**Tech Stack:** Python 3.11, `pytest`, `unittest.mock.patch`.

## Global Constraints

- Only `src/prompts/cv_extractor.py` and `tests/test_cv_extractor.py` may change.
- No new parameters on `human_2a()`, `CVExtractorAgent.run()`, `phase2_cv_extractor()`, or `run_pipeline()` — the reference date is computed internally in `human_2a()`, not threaded through as an argument.
- `ExtractionCache`'s cache key (`cv_text + candidate_id`, in `src/agents/cv_extractor.py:18`) must NOT change — the reference date is prompt content only, not part of what's cached.
- After the change: `uv run pytest -q` must show 42 passed, 0 failed (41 existing + 1 new test).
- `src/prompts/jd_parser.py` and `src/prompts/judge.py` are out of scope (see design doc's Scope section for why).

---

### Task 1: Add reference date to the CV extraction prompt

**Files:**
- Modify: `src/prompts/cv_extractor.py`
- Test: `tests/test_cv_extractor.py`

**Interfaces:**
- Consumes: nothing new — `human_2a(cv_text: str, candidate_id: str) -> str` keeps its existing signature.
- Produces: `human_2a()`'s return value now contains a `REFERENCE DATE: YYYY-MM-DD` line before `RESUME TEXT:`. No other function in the codebase calls `human_2a` except `CVExtractorAgent.run()` in `src/agents/cv_extractor.py:27` (unchanged call site, unchanged signature).

**Context:** Current file content (verified during planning):

```python
SYSTEM_2A = """Extract structured data from this resume. Rules:
1. Extract basic personal info (name, email, phone, location, LinkedIn, current title) \
from the header or contact section. Set fields to null if not present — do not fabricate.
2. All achievement strings must be VERBATIM QUOTES from the resume text.
3. Compute tenure_months from dates given. If dates are missing, set null — never estimate.
4. Never merge two separate jobs into one WorkEntry.
5. Skills must appear exactly as written. Do not infer skills from job titles.
6. Do not normalize skill names — capture them as-is in raw_mention.
7. Set total_experience_months as the sum of all non-null tenure_months.
Return a valid JSON object matching the required schema exactly."""


def human_2a(cv_text: str, candidate_id: str) -> str:
    return (
        f"Extract structured profile from the following resume.\n\n"
        f"CANDIDATE_ID: {candidate_id}\n\n"
        f"RESUME TEXT:\n{cv_text}"
    )
```

`tests/test_cv_extractor.py` currently starts with:

```python
from unittest.mock import MagicMock, patch
from src.agents.cv_extractor import CVExtractorAgent
from src.models.schemas import (
    CandidateProfile, CandidateBasicInfo, SkillEntry,
    WorkEntry,
)
```

- [ ] **Step 1: Write the failing test**

Add to `tests/test_cv_extractor.py` (after the existing imports, add two new imports; add the test function at the end of the file, after `test_cv_extractor_preserves_raw_mentions`):

Change the import block from:
```python
from unittest.mock import MagicMock, patch
from src.agents.cv_extractor import CVExtractorAgent
from src.models.schemas import (
    CandidateProfile, CandidateBasicInfo, SkillEntry,
    WorkEntry,
)
```
to:
```python
from datetime import date
from unittest.mock import MagicMock, patch
from src.agents.cv_extractor import CVExtractorAgent
from src.models.schemas import (
    CandidateProfile, CandidateBasicInfo, SkillEntry,
    WorkEntry,
)
from src.prompts.cv_extractor import human_2a
```

Add this test function at the end of the file:
```python
def test_human_2a_includes_reference_date():
    fixed_date = date(2026, 1, 15)
    with patch("src.prompts.cv_extractor.date") as mock_date:
        mock_date.today.return_value = fixed_date
        result = human_2a("CV text here", "cv_001")

    assert "REFERENCE DATE: 2026-01-15" in result
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cv_extractor.py::test_human_2a_includes_reference_date -v`
Expected: FAIL — either `ImportError: cannot import name 'human_2a'` type error is not expected (the function already exists), but the assertion `assert "REFERENCE DATE: 2026-01-15" in result` fails because `human_2a()` doesn't produce a `REFERENCE DATE` line yet. Also expect `AttributeError` is NOT raised (since `src.prompts.cv_extractor` does not yet import `date`, `patch("src.prompts.cv_extractor.date")` will fail with `AttributeError: <module 'src.prompts.cv_extractor'> does not have the attribute 'date'` — this is the actual expected failure mode at this step, confirming the module doesn't yet import `date`).

- [ ] **Step 3: Add the `date` import and reference date to `human_2a()`**

In `src/prompts/cv_extractor.py`, change:
```python
SYSTEM_2A = """Extract structured data from this resume. Rules:
```
to:
```python
from datetime import date

SYSTEM_2A = """Extract structured data from this resume. Rules:
```

Change:
```python
3. Compute tenure_months from dates given. If dates are missing, set null — never estimate.
```
to:
```python
3. Compute tenure_months from dates given. Treat "Present" or "Current" as ending \
on the reference date provided below. If dates are missing, set null — never estimate.
```

Change:
```python
def human_2a(cv_text: str, candidate_id: str) -> str:
    return (
        f"Extract structured profile from the following resume.\n\n"
        f"CANDIDATE_ID: {candidate_id}\n\n"
        f"RESUME TEXT:\n{cv_text}"
    )
```
to:
```python
def human_2a(cv_text: str, candidate_id: str) -> str:
    return (
        f"Extract structured profile from the following resume.\n\n"
        f"CANDIDATE_ID: {candidate_id}\n\n"
        f"REFERENCE DATE: {date.today().isoformat()}\n\n"
        f"RESUME TEXT:\n{cv_text}"
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_cv_extractor.py::test_human_2a_includes_reference_date -v`
Expected: PASS

- [ ] **Step 5: Run the full test suite**

Run: `uv run pytest -q`
Expected: `42 passed` (41 existing + 1 new)

- [ ] **Step 6: Commit**

```bash
git add src/prompts/cv_extractor.py tests/test_cv_extractor.py
git commit -m "feat: ground CV extractor's Present/Current resolution in reference date"
```

---

## Self-review notes (for the record — already applied above)

- **Spec coverage:** Design doc's two changes (SYSTEM_2A rule 3 rewrite, human_2a reference date) are both in Step 3. Test coverage requirement is Steps 1-2. Non-goals (no cache key change, no schema change, no jd_parser/judge.py change) are satisfied by this plan touching only the two listed files.
- **Placeholder scan:** none found.
- **Type consistency:** `human_2a(cv_text: str, candidate_id: str) -> str` signature is unchanged everywhere it's referenced (this plan, the design doc, and the existing call site in `src/agents/cv_extractor.py:27`).
