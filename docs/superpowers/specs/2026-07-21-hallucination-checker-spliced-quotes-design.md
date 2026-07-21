# Hallucination Checker: Handle Spliced Evidence Quotes — Design

## Problem

Investigation of `cv_00014.pdf`'s evaluation output found two evidence-chain
items incorrectly flagged `"fabricated"` by `hallucination_checker.py`, even
though every underlying fact was true and present verbatim in the CV.

Root cause: the judge (`CandidateJudgeAgent`) spliced two bullets from two
*different* job entries into a single `evidence_quote` using `"; "` as a
connector — e.g. joining a Buildah bullet from the candidate's AgentForge
role with a Helm bullet from their (separate, non-adjacent) Twilio role.
This violates `src/prompts/judge.py`'s own `EVIDENCE_RULE`, which already
states: "evidence_quote must be ONE contiguous excerpt... never join
separate bullets, lines, or sections together with '...' or any other
connector. If you need evidence from two unrelated parts of the CV, add a
separate evidence_chain item for each one instead."

Because the two halves of the spliced quote are not adjacent in the raw CV
text (separated by a job-title/company/date header for a different
employer), `hallucination_checker.py`'s verbatim-containment check fails
(the joined string never appears as one contiguous piece of text), and its
sliding-window similarity fallback (`MAX_WINDOW_LINES = 4`) also fails,
because no 4-line window spans both halves. The checker is working exactly
as designed — it correctly rejects a citation it can't verify as one genuine
contiguous excerpt — but the result mischaracterizes a true claim as
fabricated.

Verified computationally against the actual CV text and the actual flagged
quotes:
```
q1: verbatim containment = False, max_window_similarity = 0.815 (threshold 0.85)
q2: verbatim containment = False, max_window_similarity = 0.843 (threshold 0.85)
```

## Scope

Two independent, defense-in-depth changes:

1. `src/evaluation/hallucination_checker.py` — make the checker robust to
   this specific splicing pattern, so the evaluation metric doesn't punish
   a candidate for the judge's citation-formatting mistake.
2. `src/prompts/judge.py` — reduce how often the judge produces spliced
   quotes in the first place, by adding a concrete worked example to the
   existing (already explicit, but apparently insufficient on its own)
   `EVIDENCE_RULE`.

**Out of scope:**
- No `HallucinationFlag` schema change — still exactly the 3 existing
  statuses (`inferred`, `fabricated`, `acknowledged_gap`).
- No change to `MAX_WINDOW_LINES` or `SIMILARITY_THRESHOLD`.
- No broader connector handling beyond `"; "` (semicolon-space) — this is
  the exact pattern observed and the one the judge's own prompt already
  names. Expanding to other hypothetical connectors (` | `, `...`,
  multi-line joins) is deferred until one is actually observed (YAGNI).
- No automated re-run of the live pipeline against `cv_00014.pdf` — that
  requires a real Ollama call and is out of scope for this fix; a manual
  spot-check after merge is optional, not a plan requirement.
- No change to `src/prompts/jd_parser.py` or any other prompt file.

## Design

### `src/evaluation/hallucination_checker.py`

Extract the existing containment-or-similarity check (currently inlined in
`verify_evidence_chain`) into a helper, reused for both the whole-quote
check and a new per-part check:

```python
def _verify_single_span(quote: str, raw_cv_text: str, normalized_raw: str) -> bool:
    if _normalize_for_containment(quote) in normalized_raw:
        return True
    return _max_window_similarity(quote, raw_cv_text) > SIMILARITY_THRESHOLD
```

Add a helper for the splice-fallback check:

```python
def _all_parts_verified(quote: str, raw_cv_text: str, normalized_raw: str) -> bool:
    parts = [p.strip() for p in quote.split("; ")]
    return all(
        part and _verify_single_span(part, raw_cv_text, normalized_raw)
        for part in parts
    )
```

`verify_evidence_chain` gains one new `elif` branch, tried after the
existing whole-quote checks fail and before falling through to
`"fabricated"`:

```python
elif "; " in quote and _all_parts_verified(quote, raw_cv_text, normalized_raw):
    status = "inferred"
```

Guarding on `part and ...` (not just `...`) ensures a quote like
`"Real thing; "` (trailing separator, empty second half) does NOT pass —
an empty string is trivially "contained" in any text via
`_normalize_for_containment`, so this guard prevents that from being
mistaken for a verified span. A quote like `"Real thing; a lie"` also
correctly still fails, because the fabricated half won't verify against
the CV text either via containment or window similarity.

This is a pure addition to the classification logic: quotes that already
verify as a whole (the common case) are unaffected; only quotes that fail
the existing checks AND contain `"; "` AND have every non-empty part
independently verifiable get reclassified from `"fabricated"` to
`"inferred"`.

### `src/prompts/judge.py`

Add a concrete before/after example directly after the existing
`EVIDENCE_RULE` paragraph in `SYSTEM`, illustrating the exact anti-pattern
observed:

- **Wrong:** one `evidence_quote` splicing two bullets from different roles
  with `"; "`.
- **Right:** two separate `evidence_chain` items, one per bullet.

This reinforces the abstract rule (already present) with a worked example
matching the actual failure mode — LLMs generally follow concrete examples
more reliably than abstract prose alone. This is a probabilistic
improvement only; it does not guarantee the judge will stop splicing
quotes, which is exactly why the checker-side fix above exists as a
backstop regardless of prompt compliance.

## Testing

Add to `tests/test_hallucination_checker.py` (synthetic CV text, no LLM
calls — consistent with all existing tests in this file):

1. A quote splicing two real, verbatim-but-non-adjacent bullets (separated
   by unrelated lines in the synthetic CV text, mirroring the actual
   `cv_00014` structure) → asserts status resolves to `"inferred"`.
2. A quote splicing one real bullet with one fabricated half → asserts
   status still resolves to `"fabricated"` (proves the fallback isn't too
   lenient).
3. All existing tests in this file must continue to pass unchanged.

No test is added for the prompt-wording change in `judge.py` — it's a
prose change to LLM instructions, not independently unit-testable, and is
covered qualitatively by the fact that the checker-side fix (which IS
tested) provides correctness regardless of whether the prompt change has
any measurable effect.

## Non-goals

- No change to how `evidence_quote` values are cached, stored, or scored.
- No change to `dimension_score` calculation or `raw_score` aggregation.
- No retroactive re-evaluation of past run outputs (e.g.
  `results/run_4eb519b2/`) — this fix applies to future pipeline runs only.
