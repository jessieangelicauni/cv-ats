from __future__ import annotations
import re
from sentence_transformers import util
from src.models.schemas import CandidateAssessment, HallucinationFlag
from src.utils.embedder import get_embedder

SIMILARITY_THRESHOLD = 0.85
MAX_WINDOW_LINES = 4
SPLICE_SEPARATOR_RE = re.compile(r"\s*(?:\.\.\.|…|; )\s*")
# Matches a JSON key:value fragment (e.g. `"degree": "B.Sc..."` or a truncated
# `..indicators": true`). Such a quote was copied from the structured candidate
# profile, not the raw CV prose — the EVIDENCE_RULE forbids this regardless of
# whether the underlying fact is true, so it must never be rescued by the
# semantic-similarity fallback below (which exists only to tolerate PDF line-wrap
# reflow, not to bless format-violating quotes).
STRUCTURED_QUOTE_RE = re.compile(r'"\s*:\s*(true|false|null|-?\d|")', re.IGNORECASE)


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


def _verify_single_span(quote: str, raw_cv_text: str, normalized_raw: str) -> bool:
    if _normalize_for_containment(quote) in normalized_raw:
        return True
    if STRUCTURED_QUOTE_RE.search(quote):
        return False
    return _max_window_similarity(quote, raw_cv_text) > SIMILARITY_THRESHOLD


def _all_parts_verified(quote: str, raw_cv_text: str, normalized_raw: str) -> bool:
    # The judge is instructed never to splice separate CV bullets into one quote
    # with "; ", "...", or any other connector, but it does so occasionally. When it
    # does, each half is often a real, verbatim fact — just not adjacent enough to
    # pass as one contiguous excerpt. Verify each half independently before giving
    # up on the quote.
    parts = [p.strip() for p in SPLICE_SEPARATOR_RE.split(quote)]
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
        elif SPLICE_SEPARATOR_RE.search(quote):
            # A spliced quote is non-contiguous by construction, so it must be
            # verified part-by-part. Running the whole spliced string through the
            # single-span fuzzy-similarity check first is unsound: a genuine half
            # can drag the combined embedding above SIMILARITY_THRESHOLD even when
            # the other half is fabricated, letting the fabrication slip through
            # before the part-by-part check ever runs.
            status = "inferred" if _all_parts_verified(quote, raw_cv_text, normalized_raw) else "fabricated"
        elif _verify_single_span(quote, raw_cv_text, normalized_raw):
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
