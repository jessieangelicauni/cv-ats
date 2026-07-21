# CV Extractor Reference Date — Design

## Problem

`src/prompts/cv_extractor.py` instructs the LLM to compute `tenure_months` and
`total_experience_months` from dates found in the CV text (`SYSTEM_2A` rule 3:
"Compute tenure_months from dates given... never estimate"). Nothing in the
prompt tells the model what "today" is. For a work history entry like
"Jan 2022 – Present," the model has to resolve "Present" against its own
internal notion of the current date (e.g. its training cutoff), which does
not necessarily match the actual date the pipeline is run on. This makes
tenure calculations for ongoing roles unreliable and non-reproducible across
model versions or time.

Confirmed during investigation: no code anywhere in the repo does date
arithmetic for CV extraction — it is entirely delegated to the LLM via the
prompt. The only other use of `date.today()` in the codebase is
`src/output/report_generator.py:128`, which stamps the generated report with
today's date for display and is unrelated to CV parsing.

## Scope

**In scope:** `src/prompts/cv_extractor.py` only — `SYSTEM_2A` and
`human_2a()`.

**Out of scope:**
- `src/prompts/jd_parser.py` — job descriptions have no date-dependent
  fields to extract.
- `src/prompts/judge.py` — the judge consumes the already-computed
  `tenure_months`/`total_experience_months` from the structured
  `CandidateProfile` (produced by Phase 2); it does not independently
  compute tenure from raw dates, so it doesn't need its own reference date.
- Changing *how* tenure is computed (still fully LLM-driven; this fix only
  grounds the reference point the model resolves "Present" against). No
  deterministic date-parsing code is introduced.
- Threading a date value through `run_pipeline()` / `ATSState` / the graph
  nodes — the reference date is a fact available at the point of use, not
  pipeline state that varies between phases or needs to be recorded per run.

## Design

**`human_2a(cv_text, candidate_id)`** — compute `datetime.date.today()`
internally at call time and prepend a `REFERENCE DATE: {today}` line to the
returned message, before the `RESUME TEXT:` section. No new parameters are
added to `human_2a`, `CVExtractorAgent.run()`, or any caller — this matches
the existing pattern in `report_generator.py`, which also calls
`date.today()` directly at its point of use rather than threading a date
through pipeline state.

**`SYSTEM_2A`** — rule 3 is rewritten to explicitly tell the model how to use
the reference date:

> "Compute tenure_months from dates given. Treat 'Present' or 'Current' as
> ending on the reference date provided below. If dates are missing, set
> null — never estimate."

Adding the date alone would not be sufficient — the model would still have
to infer, unprompted, that an unlabeled date fact is meant to anchor
"Present"/"Current" resolution. The rule makes this explicit.

## Testing

Add a test to `tests/test_cv_extractor.py` that imports `human_2a` directly,
patches `datetime.date.today()` (via `unittest.mock.patch` on
`src.prompts.cv_extractor.date`) to a fixed value, and asserts the returned
prompt string contains that fixed date. This verifies the plumbing (the date
actually reaches the prompt text) without requiring a real LLM call — no
existing test in this file exercises prompt content, only the
`CandidateProfile` return value via a fully mocked LLM.

## Non-goals

- No change to `CandidateProfile`/`WorkEntry` schema.
- No change to caching behavior in `ExtractionCache` (the cache key is
  already `cv_text + candidate_id`; note that if `REFERENCE DATE` were ever
  added to the cache key computation, cache hits would change daily — this
  design deliberately does NOT touch the cache key, since the date is prompt
  content only, not part of what's cached or how the cache key is derived).
- No change to `jd_parser.py` or `judge.py`.
