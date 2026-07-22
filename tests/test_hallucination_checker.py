from unittest.mock import patch
from src.evaluation.hallucination_checker import verify_evidence_chain, hallucination_rate
from src.models.schemas import CandidateAssessment, EvidenceItem, HallucinationFlag
from tests.conftest import make_candidate_assessment

RAW_CV = "Led backend migration using Python and gRPC. Maintained postgres cluster serving 10M users."


def _make_assessment(quotes: list[str]) -> CandidateAssessment:
    return make_candidate_assessment(
        "cv_001", 80.0,
        evidence_chain=[
            EvidenceItem(dimension=f"dim_{i}", assessment="Good.",
                         evidence_quote=q, dimension_score=8.0)
            for i, q in enumerate(quotes)
        ],
        key_strengths=[],
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
    assert abs(hallucination_rate(flags) - 1/3) < 1e-6


def test_verbatim_quote_from_raw_cv_is_not_fabricated():
    raw_cv = (
        "EDUCATION\n"
        "B.Sc. in Artificial Intelligence, TU Delft, 2019\n"
        "B.Sc. in Artificial Intelligence, Carnegie Mellon University, 2018\n"
    )
    assessment = _make_assessment(["B.Sc. in Artificial Intelligence, Carnegie Mellon University, 2018"])
    flags = verify_evidence_chain(assessment, raw_cv)
    assert flags[0].status != "fabricated"


def test_structured_profile_style_quote_is_flagged_even_when_true():
    raw_cv = (
        "EDUCATION\n"
        "B.Sc. in Artificial Intelligence, TU Delft, 2019\n"
    )
    assessment = _make_assessment(
        ['"degree": "B.Sc. in Artificial Intelligence", "institution": "TU Delft", "year": 2019']
    )
    flags = verify_evidence_chain(assessment, raw_cv)
    assert flags[0].status == "fabricated"


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
    quote = (
        "Designed and implemented infrastructure as code using Buildah, managing 100+ servers; "
        "Built CI/CD pipelines using Helm, reducing deployment time from 29s to 491ms"
    )
    assessment = _make_assessment([quote])
    flags = verify_evidence_chain(assessment, MULTI_JOB_CV)
    assert flags[0].status == "inferred"


def test_spliced_quote_with_one_fabricated_half_is_still_fabricated():
    quote = (
        "Designed and implemented infrastructure as code using Buildah, managing 100+ servers; "
        "Personally negotiated a merger between two Fortune 500 companies"
    )
    assessment = _make_assessment([quote])
    flags = verify_evidence_chain(assessment, MULTI_JOB_CV)
    assert flags[0].status == "fabricated"


def test_ellipsis_spliced_quote_from_two_nonadjacent_real_bullets_is_not_fabricated():
    quote = (
        "Designed and implemented infrastructure as code using Buildah, managing 100+ servers... "
        "Built CI/CD pipelines using Helm, reducing deployment time from 29s to 491ms"
    )
    assessment = _make_assessment([quote])
    with patch("src.evaluation.hallucination_checker._max_window_similarity", return_value=0.0):
        flags = verify_evidence_chain(assessment, MULTI_JOB_CV)
    assert flags[0].status == "inferred"
