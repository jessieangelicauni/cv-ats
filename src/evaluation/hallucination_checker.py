from __future__ import annotations
import re
from sentence_transformers import util
from src.models.schemas import CandidateAssessment, HallucinationFlag
from src.utils.embedder import get_embedder


def _max_sentence_similarity(quote: str, full_text: str) -> float:
    """Max cosine similarity between the quote and any sentence in the CV text."""
    sentences = [s.strip() for s in re.split(r"[.!?\n]+", full_text) if s.strip()]
    if not sentences:
        return 0.0
    embedder = get_embedder()
    emb_quote = embedder.encode(quote, convert_to_tensor=True)
    emb_sentences = embedder.encode(sentences, convert_to_tensor=True)
    scores = util.cos_sim(emb_quote, emb_sentences)[0]
    return float(scores.max())


def verify_evidence_chain(
    assessment: CandidateAssessment,
    raw_cv_text: str,
) -> list[HallucinationFlag]:
    flags: list[HallucinationFlag] = []
    for item in assessment.evidence_chain:
        quote = item.evidence_quote.strip()

        if quote == "NOT FOUND IN CV":
            status = "acknowledged_gap"
        elif quote.lower() in raw_cv_text.lower():
            status = "supported"
        elif _max_sentence_similarity(quote, raw_cv_text) > 0.9:
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
