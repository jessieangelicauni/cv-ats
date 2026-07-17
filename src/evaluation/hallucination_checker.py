from __future__ import annotations
from sentence_transformers import util
from src.models.schemas import CandidateAssessment, HallucinationFlag
from src.utils.embedder import get_embedder
import config


def _semantic_similarity(quote: str, full_text: str) -> float:
    embedder = get_embedder()
    emb_quote = embedder.encode(quote, convert_to_tensor=True)
    emb_text = embedder.encode(full_text, convert_to_tensor=True)
    return float(util.cos_sim(emb_quote, emb_text))


def verify_evidence_chain(
    assessment: CandidateAssessment,
    raw_cv_text: str,
) -> list[HallucinationFlag]:
    flags: list[HallucinationFlag] = []
    for item in assessment.evidence_chain:
        quote = item.evidence_quote.strip()
        if quote == "NOT FOUND IN CV":
            status = "acknowledged_gap"
        elif quote in raw_cv_text:
            status = "supported"
        elif _semantic_similarity(quote, raw_cv_text) > config.HALLUCINATION_SIMILARITY_THRESHOLD:
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
