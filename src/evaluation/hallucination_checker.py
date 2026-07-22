from __future__ import annotations
import re
from src.models.schemas import CandidateAssessment, HallucinationFlag
from src.utils.embedder import get_nli_model

ENTAILMENT_THRESHOLD = 0.5
MAX_WINDOW_LINES = 4
SPLICE_SEPARATOR_RE = re.compile(r"\s*(?:\.\.\.|…|; |\n+)\s*")
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


_window_cache: dict[str, list[str]] = {}


def _cached_line_windows(full_text: str) -> list[str]:
    windows = _window_cache.get(full_text)
    if windows is None:
        windows = _line_windows(full_text)
        _window_cache[full_text] = windows
    return windows


def _max_window_entailment(quote: str, full_text: str) -> float:
    windows = _cached_line_windows(full_text)
    if not windows:
        return 0.0
    scores = get_nli_model().predict(
        [(window, quote) for window in windows], apply_softmax=True, convert_to_numpy=True
    )
    return float(scores[:, 1].max())


def _verify_single_span(quote: str, raw_cv_text: str, normalized_raw: str) -> bool:
    if _normalize_for_containment(quote) in normalized_raw:
        return True
    if STRUCTURED_QUOTE_RE.search(quote):
        return False
    return _max_window_entailment(quote, raw_cv_text) > ENTAILMENT_THRESHOLD


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
        elif _verify_single_span(quote, raw_cv_text, normalized_raw):
            status = "inferred"
        elif SPLICE_SEPARATOR_RE.search(quote):
            status = "inferred" if _all_parts_verified(quote, raw_cv_text, normalized_raw) else "fabricated"
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


def severity_weighted_hallucination_rate(
    assessments: list[CandidateAssessment],
    flags: list[HallucinationFlag],
) -> float:
    total_weight = 0.0
    fabricated_weight = 0.0
    for assessment in assessments:
        claim_status = {
            f.claim: f.status
            for f in flags
            if f.candidate_id == assessment.candidate_id
        }
        for item in assessment.evidence_chain:
            status = claim_status.get(item.assessment)
            if status is None or status == "acknowledged_gap":
                continue
            total_weight += item.dimension_score
            if status == "fabricated":
                fabricated_weight += item.dimension_score
    if total_weight == 0:
        return 0.0
    return fabricated_weight / total_weight
