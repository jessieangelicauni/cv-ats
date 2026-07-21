import json
import tempfile
from pathlib import Path
from src.output.report_generator import generate_report
from src.graph.state import ATSState
from src.models.schemas import (
    FinalRanking, RankedCandidate, CandidateAssessment,
    EvidenceItem, CandidateProfile, CandidateBasicInfo,
    JDRequirements, EducationRequirement, HallucinationFlag,
)


def _make_state() -> ATSState:
    profile = CandidateProfile(
        candidate_id="cv_001",
        basic_info=CandidateBasicInfo(full_name="Ahmad Faris", email="a@b.com",
                                       phone=None, location="KL", linkedin_url=None,
                                       current_title="Engineer"),
        skills=[], work_history=[], education=[],
        certifications=[], languages=[], total_experience_months=60,
    )
    assessment = CandidateAssessment(
        candidate_id="cv_001", raw_score=87.0, confidence="high",
        evidence_chain=[EvidenceItem(dimension="Technical Skills Fit",
                                      assessment="Strong Python.", evidence_quote="Python dev",
                                      dimension_score=9.0)],
        key_strengths=["Python"], key_gaps=["Kubernetes"], seniority_alignment="aligned",
    )
    ranking = FinalRanking(
        ranked_candidates=[RankedCandidate(rank=1, candidate_id="cv_001",
                           calibrated_score=91.0, delta_from_raw=4.0,
                           comparative_notes="Best candidate.")],
        pool_summary="Strong pool.", calibration_rationale="Spread increased.",
        borderline_pairs=[],
    )
    jd = JDRequirements(
        role_title="Senior Backend Engineer", seniority_level="senior",
        required_skills=[], preferred_skills=[], min_years_experience=5,
        education=EducationRequirement(degree="BSc", field="CS", is_mandatory=False),
        domain_expertise=[], leadership_expected=True,
        soft_skills=[], industry_context="IT", raw_jd_hash="abc",
    )
    return ATSState(
        jd_raw="Senior Backend Engineer job description",
        jd_structured=jd,
        cv_raws=[{"raw_text": "CV text", "candidate_id": "cv_001", "source_file": "cv_001.pdf"}],
        cv_profiles=[profile],
        candidate_assessments=[assessment],
        final_ranking=ranking,
        run_id="test_run_001",
        trace_log=[{"phase": 1, "duration_s": 1.2}],
        hallucination_flags=[HallucinationFlag(candidate_id="cv_001",
                                               claim="Strong Python.", status="inferred",
                                               source_quote="Python dev")],
        use_cache=False,
    )


def test_generate_report_creates_expected_files():
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        generate_report(_make_state(), out_dir)
        assert (out_dir / "report.md").exists()
        assert (out_dir / "ranking.json").exists()
        assert (out_dir / "candidates" / "cv_001.json").exists()
        assert (out_dir / "evaluation" / "hallucination_report.json").exists()


def test_report_md_contains_candidate_name():
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        generate_report(_make_state(), out_dir)
        content = (out_dir / "report.md").read_text()
        assert "Ahmad Faris" in content


def test_report_md_contains_evidence_quote():
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        generate_report(_make_state(), out_dir)
        content = (out_dir / "report.md").read_text()
        assert "Python dev" in content


def test_ranking_json_is_valid():
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        generate_report(_make_state(), out_dir)
        data = json.loads((out_dir / "ranking.json").read_text())
        assert data["ranked_candidates"][0]["candidate_id"] == "cv_001"


def test_report_md_contains_eliminated_section_when_candidates_filtered():
    profile = CandidateProfile(
        candidate_id="cv_001",
        basic_info=CandidateBasicInfo(full_name="Ahmad Faris", email=None,
                                       phone=None, location=None, linkedin_url=None,
                                       current_title="Engineer"),
        skills=[], work_history=[], education=[],
        certifications=[], languages=[], total_experience_months=60,
    )
    assessment = CandidateAssessment(
        candidate_id="cv_001", raw_score=87.0, confidence="high",
        evidence_chain=[EvidenceItem(dimension="Technical Skills Fit",
                                      assessment="Strong.", evidence_quote="Python",
                                      dimension_score=9.0)],
        key_strengths=["Python"], key_gaps=[], seniority_alignment="aligned",
    )
    ranking = FinalRanking(
        ranked_candidates=[RankedCandidate(rank=1, candidate_id="cv_001",
                           calibrated_score=90.0, delta_from_raw=3.0,
                           comparative_notes="Best.")],
        pool_summary="One candidate.", calibration_rationale="N/A", borderline_pairs=[],
    )
    jd = JDRequirements(
        role_title="Engineer", seniority_level="senior",
        required_skills=[], preferred_skills=[], min_years_experience=5,
        education=EducationRequirement(degree="BSc", field="CS", is_mandatory=False),
        domain_expertise=[], leadership_expected=True,
        soft_skills=[], industry_context="IT", raw_jd_hash="abc",
    )
    state = ATSState(
        jd_raw="Engineer JD",
        jd_structured=jd,
        cv_raws=[{"raw_text": "CV", "candidate_id": "cv_001", "source_file": "cv.pdf"}],
        cv_profiles=[profile],
        candidate_assessments=[assessment],
        final_ranking=ranking,
        run_id="test_run",
        eliminated_candidates=["cv_002", "cv_003"],
        use_cache=False,
    )
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        generate_report(state, out_dir)
        content = (out_dir / "report.md").read_text()
        assert "Filtered Candidates" in content
        assert "cv_002" in content
        assert "cv_003" in content
