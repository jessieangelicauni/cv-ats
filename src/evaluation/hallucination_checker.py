from __future__ import annotations
import re
from sentence_transformers import util
from src.models.schemas import CandidateAssessment, HallucinationFlag
from src.utils.embedder import get_embedder

SIMILARITY_THRESHOLD = 0.85
MAX_WINDOW_LINES = 4
SPLICE_SEPARATOR_RE = re.compile(r"\s*(?:\.\.\.|…|; )\s*")
STRUCTURED_QUOTE_RE = re.compile(r'"\s*:\s*(true|false|null|-?\d|")', re.IGNORECASE)


def _normalize_for_containment(text: str) -> str:
    return re.sub(r"\s+", "", text).lower()


def _line_windows(full_text: str) -> list[str]:
    lines = [line.strip() for line in full_text.split("\n") if line.strip()]
    windows: list[str] = []
    for size in range(1, MAX_WINDOW_LINES + 1):
        for i in range(len(lines) - size + 1):
            windows.append(" ".join(lines[i : i + size]))
    return windows


_window_emb_cache: dict[str, tuple[list[str], object]] = {}


def _windows_and_embeddings(full_text: str) -> tuple[list[str], object]:
    cached = _window_emb_cache.get(full_text)
    if cached is not None:
        return cached
    windows = _line_windows(full_text)
    emb_windows = get_embedder().encode(windows, convert_to_tensor=True) if windows else None
    _window_emb_cache[full_text] = (windows, emb_windows)
    return windows, emb_windows


def _max_window_similarity(quote: str, full_text: str) -> float:
    windows, emb_windows = _windows_and_embeddings(full_text)
    if not windows:
        return 0.0
    embedder = get_embedder()
    emb_quote = embedder.encode(quote, convert_to_tensor=True)
    scores = util.cos_sim(emb_quote, emb_windows)[0]
    return float(scores.max())


def _verify_single_span(quote: str, raw_cv_text: str, normalized_raw: str) -> bool:
    if _normalize_for_containment(quote) in normalized_raw:
        return True
    if STRUCTURED_QUOTE_RE.search(quote):
        return False
    return _max_window_similarity(quote, raw_cv_text) > SIMILARITY_THRESHOLD


def _all_parts_verified(quote: str, raw_cv_text: str, normalized_raw: str) -> bool:
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
