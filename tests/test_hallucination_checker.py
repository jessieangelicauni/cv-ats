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


def test_not_found_in_cv_is_acknowledged_gap():
    assessment = _make_assessment(["NOT FOUND IN CV"])
    flags = verify_evidence_chain(assessment, RAW_CV)
    assert flags[0].status == "acknowledged_gap"


def test_fabricated_claim_detected():
    assessment = _make_assessment(["Managed a team of 50 engineers globally"])
    flags = verify_evidence_chain(assessment, RAW_CV)
    assert flags[0].status == "fabricated"


def test_hallucination_rate_zero_when_no_fabricated():
    flags = [
        HallucinationFlag(candidate_id="cv_001", claim="c", status="inferred", source_quote="q"),
        HallucinationFlag(candidate_id="cv_001", claim="c", status="inferred", source_quote="q"),
    ]
    assert hallucination_rate(flags) == 0.0


def test_hallucination_rate_calculated_correctly():
    flags = [
        HallucinationFlag(candidate_id="cv_001", claim="c", status="inferred", source_quote="q"),
        HallucinationFlag(candidate_id="cv_001", claim="c", status="fabricated", source_quote=None),
        HallucinationFlag(candidate_id="cv_001", claim="c", status="inferred", source_quote="q"),
        HallucinationFlag(candidate_id="cv_001", claim="c", status="acknowledged_gap", source_quote=None),
    ]
    # fabricated=1 / (inferred=2 + fabricated=1) = 1/3
    assert abs(hallucination_rate(flags) - 1/3) < 1e-6
