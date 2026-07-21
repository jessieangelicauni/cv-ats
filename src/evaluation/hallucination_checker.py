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
