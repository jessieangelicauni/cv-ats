# Hallucination Checker Spliced-Quote Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop `hallucination_checker.py` from mis-flagging true claims as `"fabricated"` when the judge splices two real-but-non-adjacent CV bullets into one `evidence_quote` with `"; "`, and reduce how often the judge does this in the first place.

**Architecture:** No architectural changes. Two independent, defense-in-depth edits: (1) `src/evaluation/hallucination_checker.py` gains a `"; "`-split fallback check, tried only after the existing whole-quote verification fails; (2) `src/prompts/judge.py`'s `EVIDENCE_RULE` gains a concrete worked example of the anti-pattern. No schema changes, no threshold changes.

**Tech Stack:** Python 3.11, `pytest`.

## Global Constraints

- `HallucinationFlag` keeps exactly its 3 existing statuses (`inferred`, `fabricated`, `acknowledged_gap`) — no schema change.
- `MAX_WINDOW_LINES` (4) and `SIMILARITY_THRESHOLD` (0.85) in `src/evaluation/hallucination_checker.py` are unchanged.
- The new fallback only splits on `"; "` (semicolon-space) — no broader connector list.
- No change to `src/prompts/jd_parser.py`.
- After all tasks: `uv run pytest -q` must show 44 passed, 0 failed (42 existing + 2 new).

---

### Task 1: Handle spliced evidence quotes in the hallucination checker

**Files:**
- Modify: `src/evaluation/hallucination_checker.py`
- Test: `tests/test_hallucination_checker.py`

**Interfaces:**
- Consumes: nothing new — `verify_evidence_chain(assessment: CandidateAssessment, raw_cv_text: str) -> list[HallucinationFlag]` keeps its existing signature and is still the only entry point called by `main.py` and `src/evaluation/consistency_runner.py` (unchanged call sites).
- Produces: two new internal helpers, `_verify_single_span(quote: str, raw_cv_text: str, normalized_raw: str) -> bool` and `_all_parts_verified(quote: str, raw_cv_text: str, normalized_raw: str) -> bool`. These are private to this module — no other file imports them.

**Context:** Current file content (verified during planning, shown in full):

```python
from __future__ import annotations
import re
from sentence_transformers import util
from src.models.schemas import CandidateAssessment, HallucinationFlag
from src.utils.embedder import get_embedder

SIMILARITY_THRESHOLD = 0.85
MAX_WINDOW_LINES = 4


def _normalize_for_containment(text: str) -> str:
    return re.sub(r"\s+", "", text).lower()


def _line_windows(full_text: str) -> list[str]:
    # PDF-extracted CVs often split one logical entry (e.g. "role, company, dates")
    # across several lines purely due to column layout. Comparing the quote against
    # single lines alone misses these; sliding windows of consecutive lines let a
    # quote that spans 2-4 original lines still find its match.
    lines = [line.strip() for line in full_text.split("\n") if line.strip()]
    windows: list[str] = []
    for size in range(1, MAX_WINDOW_LINES + 1):
        for i in range(len(lines) - size + 1):
            windows.append(" ".join(lines[i : i + size]))
    return windows


def _max_window_similarity(quote: str, full_text: str) -> float:
    windows = _line_windows(full_text)
    if not windows:
        return 0.0
    embedder = get_embedder()
    emb_quote = embedder.encode(quote, convert_to_tensor=True)
    emb_windows = embedder.encode(windows, convert_to_tensor=True)
    scores = util.cos_sim(emb_quote, emb_windows)[0]
    return float(scores.max())


def verify_evidence_chain(
    assessment: CandidateAssessment,
    raw_cv_text: str,
) -> list[HallucinationFlag]:
    flags: list[HallucinationFlag] = []
    normalized_raw = _normalize_for_containment(raw_cv_text)

    for item in assessment.evidence_chain:
        quote = item.evidence_quote.strip().strip('"\'')

        if quote == "NOT FOUND IN CV":
            status = "acknowledged_gap"
        elif _normalize_for_containment(quote) in normalized_raw:
            # Verbatim modulo whitespace/line-break differences from PDF layout.
            status = "inferred"
        elif _max_window_similarity(quote, raw_cv_text) > SIMILARITY_THRESHOLD:
            status = "inferred"
        else:
            status = "fabricated"

        flags.append(HallucinationFlag(
            candidate_id=assessment.candidate_id,
            claim=item.assessment,
            status=status,
            source_quote=quote if quote != "NOT FOUND IN CV" else None,
        ))
    return flags


def hallucination_rate(flags: list[HallucinationFlag]) -> float:
    countable = [f for f in flags if f.status != "acknowledged_gap"]
    if not countable:
        return 0.0
    fabricated = sum(1 for f in countable if f.status == "fabricated")
    return fabricated / len(countable)
```

`tests/test_hallucination_checker.py` currently starts with:

```python
from src.evaluation.hallucination_checker import verify_evidence_chain, hallucination_rate
from src.models.schemas import CandidateAssessment, EvidenceItem, HallucinationFlag

RAW_CV = "Led backend migration using Python and gRPC. Maintained postgres cluster serving 10M users."


def _make_assessment(quotes: list[str]) -> CandidateAssessment:
    return CandidateAssessment(
        candidate_id="cv_001", raw_score=80.0, confidence="high",
        evidence_chain=[
            EvidenceItem(dimension=f"dim_{i}", assessment="Good.",
                         evidence_quote=q, dimension_score=8.0)
            for i, q in enumerate(quotes)
        ],
        key_strengths=[], key_gaps=[], seniority_alignment="aligned",
    )
```

These two new tests use a synthetic multi-job CV (mirroring the real `cv_00014.pdf` structure that surfaced this bug: two job entries, each with its own bullets, separated by a job-title/company/date header) so the two spliced halves are more than `MAX_WINDOW_LINES` (4) lines apart and can never land in the same sliding window. Both fixtures below were verified during planning by running the actual embedding model against them — the exact similarity scores confirm the described pass/fail behavior, so implement exactly as specified rather than substituting different example text.

- [ ] **Step 1: Write the two failing tests**

Add to `tests/test_hallucination_checker.py`, at the end of the file:

```python
MULTI_JOB_CV = (
    "Experience\n"
    "Infrastructure Engineer\n"
    "at CompanyA\n"
    "2024-01 - Present\n"
    "Designed and implemented infrastructure as code using Buildah, managing 100+ servers\n"
    "Optimized cloud infrastructure using Hetzner, reducing costs by 48%\n"
    "Infrastructure Engineer\n"
    "at CompanyB\n"
    "2020-12 - 2024-01\n"
    "Built CI/CD pipelines using Helm, reducing deployment time from 29s to 491ms\n"
)


def test_spliced_quote_from_two_nonadjacent_real_bullets_is_not_fabricated():
    # Regression test for the cv_00014.pdf case: the judge joined two real bullets
    # from two different (non-adjacent) job entries with "; ". Neither the verbatim
    # check nor the whole-quote window-similarity check can verify the joined string
    # (the halves are 5 lines apart, past MAX_WINDOW_LINES), but each half is
    # individually a true, verbatim CV fact and must not be flagged fabricated.
    quote = (
        "Designed and implemented infrastructure as code using Buildah, managing 100+ servers; "
        "Built CI/CD pipelines using Helm, reducing deployment time from 29s to 491ms"
    )
    assessment = _make_assessment([quote])
    flags = verify_evidence_chain(assessment, MULTI_JOB_CV)
    assert flags[0].status == "inferred"


def test_spliced_quote_with_one_fabricated_half_is_still_fabricated():
    # The "; " fallback must not become a loophole: if either half doesn't verify
    # against the CV, the whole quote must still be flagged fabricated.
    quote = (
        "Designed and implemented infrastructure as code using Buildah, managing 100+ servers; "
        "Personally negotiated a merger between two Fortune 500 companies"
    )
    assessment = _make_assessment([quote])
    flags = verify_evidence_chain(assessment, MULTI_JOB_CV)
    assert flags[0].status == "fabricated"
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `uv run pytest tests/test_hallucination_checker.py -k spliced -v`
Expected: both FAIL. `test_spliced_quote_from_two_nonadjacent_real_bullets_is_not_fabricated` fails with `assert 'fabricated' == 'inferred'` (current code has no splice fallback, so the whole-quote checks fail and it falls through to `"fabricated"`). `test_spliced_quote_with_one_fabricated_half_is_still_fabricated` currently PASSES already (current code already produces `"fabricated"` for this case with no fallback logic at all) — this is expected and fine; it becomes a true regression guard once Step 3 adds the fallback, since a careless implementation of the fallback could turn this into a false "inferred".

- [ ] **Step 3: Add the splice-fallback logic**

In `src/evaluation/hallucination_checker.py`, change:

```python
def _max_window_similarity(quote: str, full_text: str) -> float:
    windows = _line_windows(full_text)
    if not windows:
        return 0.0
    embedder = get_embedder()
    emb_quote = embedder.encode(quote, convert_to_tensor=True)
    emb_windows = embedder.encode(windows, convert_to_tensor=True)
    scores = util.cos_sim(emb_quote, emb_windows)[0]
    return float(scores.max())


def verify_evidence_chain(
    assessment: CandidateAssessment,
    raw_cv_text: str,
) -> list[HallucinationFlag]:
    flags: list[HallucinationFlag] = []
    normalized_raw = _normalize_for_containment(raw_cv_text)

    for item in assessment.evidence_chain:
        quote = item.evidence_quote.strip().strip('"\'')

        if quote == "NOT FOUND IN CV":
            status = "acknowledged_gap"
        elif _normalize_for_containment(quote) in normalized_raw:
            # Verbatim modulo whitespace/line-break differences from PDF layout.
            status = "inferred"
        elif _max_window_similarity(quote, raw_cv_text) > SIMILARITY_THRESHOLD:
            status = "inferred"
        else:
            status = "fabricated"
```

to:

```python
def _max_window_similarity(quote: str, full_text: str) -> float:
    windows = _line_windows(full_text)
    if not windows:
        return 0.0
    embedder = get_embedder()
    emb_quote = embedder.encode(quote, convert_to_tensor=True)
    emb_windows = embedder.encode(windows, convert_to_tensor=True)
    scores = util.cos_sim(emb_quote, emb_windows)[0]
    return float(scores.max())


def _verify_single_span(quote: str, raw_cv_text: str, normalized_raw: str) -> bool:
    if _normalize_for_containment(quote) in normalized_raw:
        return True
    return _max_window_similarity(quote, raw_cv_text) > SIMILARITY_THRESHOLD


def _all_parts_verified(quote: str, raw_cv_text: str, normalized_raw: str) -> bool:
    # The judge is instructed never to splice separate CV bullets into one quote
    # with "; ", but it does so occasionally. When it does, each half is often a
    # real, verbatim fact — just not adjacent enough to pass as one contiguous
    # excerpt. Verify each half independently before giving up on the quote.
    parts = [p.strip() for p in quote.split("; ")]
    return all(
        part and _verify_single_span(part, raw_cv_text, normalized_raw)
        for part in parts
    )


def verify_evidence_chain(
    assessment: CandidateAssessment,
    raw_cv_text: str,
) -> list[HallucinationFlag]:
    flags: list[HallucinationFlag] = []
    normalized_raw = _normalize_for_containment(raw_cv_text)

    for item in assessment.evidence_chain:
        quote = item.evidence_quote.strip().strip('"\'')

        if quote == "NOT FOUND IN CV":
            status = "acknowledged_gap"
        elif _verify_single_span(quote, raw_cv_text, normalized_raw):
            status = "inferred"
        elif "; " in quote and _all_parts_verified(quote, raw_cv_text, normalized_raw):
            status = "inferred"
        else:
            status = "fabricated"
```

(The `# Verbatim modulo whitespace/line-break differences from PDF layout.` comment moves with its logic into `_verify_single_span` — it no longer needs a separate inline comment at the call site in `verify_evidence_chain` since the helper name now conveys the same intent.)

- [ ] **Step 4: Run the new tests to verify they pass**

Run: `uv run pytest tests/test_hallucination_checker.py -k spliced -v`
Expected: both PASS.

- [ ] **Step 5: Run the full test suite**

Run: `uv run pytest -q`
Expected: `44 passed` (42 existing + 2 new).

- [ ] **Step 6: Commit**

```bash
git add src/evaluation/hallucination_checker.py tests/test_hallucination_checker.py
git commit -m "fix: verify spliced evidence quotes part-by-part instead of flagging them fabricated"
```

---

### Task 2: Add a concrete anti-pattern example to the judge's EVIDENCE_RULE

**Files:**
- Modify: `src/prompts/judge.py`

**Interfaces:** N/A — prose-only change to the `SYSTEM` prompt constant; no function signatures affected.

**Context:** Current `EVIDENCE RULE` paragraph in `src/prompts/judge.py` (full current file shown; only this paragraph changes):

```python
SYSTEM = """You are an experienced senior technical recruiter with 15 years in IT hiring.
You reason like a human expert: leadership and measurable impact matter more than job \
titles; relevant tenure increases confidence; career changes that represent growth are \
not penalized.

When a candidate has worked at a Tier 1 multinational company, treat this as a \
meaningful positive signal — it indicates exposure to large-scale systems, professional \
engineering culture, and organisational breadth. Weight this in your holistic \
assessment, particularly under Career Trajectory and Experience Depth.

EVIDENCE RULE: Every evidence_chain item MUST include an evidence_quote containing \
exact, verbatim text copied from the RAW CV TEXT block below — never from the \
structured candidate profile, and never paraphrased, reformatted, or summarized. \
The structured profile is provided only to help you understand the candidate faster; \
it is not a quotable source. evidence_quote must be ONE contiguous excerpt (a single \
line or a short run of adjacent lines) copied as-is — never join separate bullets, \
lines, or sections together with "..." or any other connector. If you need evidence \
from two unrelated parts of the CV, add a separate evidence_chain item for each one \
instead of merging them into a single quote. If no supporting sentence exists in the \
raw CV text, set evidence_quote to "NOT FOUND IN CV" and lower the dimension_score \
accordingly. Never state a fact not traceable to the raw CV text.
Return a valid JSON object matching the required schema exactly."""
```

This is a single-task change with no automated test (it's prose steering an LLM's behavior — see the design doc's Testing section for why this isn't independently unit-testable). Verification here is a careful read-through, not a test run.

- [ ] **Step 1: Insert the concrete example**

In `src/prompts/judge.py`, change:

```python
from two unrelated parts of the CV, add a separate evidence_chain item for each one \
instead of merging them into a single quote. If no supporting sentence exists in the \
raw CV text, set evidence_quote to "NOT FOUND IN CV" and lower the dimension_score \
accordingly. Never state a fact not traceable to the raw CV text.
Return a valid JSON object matching the required schema exactly."""
```

to:

```python
from two unrelated parts of the CV, add a separate evidence_chain item for each one \
instead of merging them into a single quote. \
WRONG: evidence_quote: "Led backend migration using Python; Reduced latency by 40%" — \
this splices two bullets from different roles into one quote with "; ". \
RIGHT: two separate evidence_chain items instead — one with evidence_quote "Led \
backend migration using Python", another with evidence_quote "Reduced latency by 40%". \
If no supporting sentence exists in the \
raw CV text, set evidence_quote to "NOT FOUND IN CV" and lower the dimension_score \
accordingly. Never state a fact not traceable to the raw CV text.
Return a valid JSON object matching the required schema exactly."""
```

- [ ] **Step 2: Run the full test suite (prose-only change, but confirms nothing else broke)**

Run: `uv run pytest -q`
Expected: `44 passed` (unchanged from Task 1 — this file has no unit tests exercising prompt content, matching the existing pattern where `src/prompts/*.py` constants are not directly tested).

- [ ] **Step 3: Commit**

```bash
git add src/prompts/judge.py
git commit -m "docs: add concrete anti-pattern example to judge's evidence-quote rule"
```

---

## Self-review notes (for the record — already applied above)

- **Spec coverage:** Design doc's checker-side fix (Task 1) and prompt-side fix (Task 2) are both covered. Testing section's two required test cases are both in Task 1 Step 1, with pass/fail fixtures independently verified against the real embedding model during planning (not guessed).
- **Placeholder scan:** none found.
- **Type consistency:** `_verify_single_span` and `_all_parts_verified` signatures match between their definition (Task 1 Step 3) and their only call sites (both within `verify_evidence_chain`, same task). `verify_evidence_chain`'s public signature is unchanged, matching the design doc's "no schema/signature changes" constraint.
