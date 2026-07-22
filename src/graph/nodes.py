from __future__ import annotations
from src.models.schemas import (
    JDRequirements, CandidateProfile, CandidateAssessment, FinalRanking,
)
from src.agents.jd_parser import JDParserAgent
from src.agents.cv_extractor import CVExtractorAgent
from src.agents.candidate_judge import CandidateJudgeAgent
from src.agents.pool_calibrator import PoolCalibratorAgent
from src.utils.cache import ExtractionCache
from src.utils.skill_normalizer import compute_skill_matches
from src.utils.pdf_extractor import raw_text_by_candidate_id


def phase1_jd_parser(jd_raw: str, cache: ExtractionCache | None) -> JDRequirements:
    return JDParserAgent(cache=cache).run(jd_raw)


def phase2_cv_extractor(
    cv_raws: list[dict],
    cache: ExtractionCache | None,
) -> list[CandidateProfile]:
    agent = CVExtractorAgent(cache=cache)
    return [agent.run(cv_raw) for cv_raw in cv_raws]


def phase3_candidate_judge(
    profiles: list[CandidateProfile],
    jd: JDRequirements,
    cv_raws: list[dict],
) -> list[CandidateAssessment]:
    raw_text_by_id = raw_text_by_candidate_id(cv_raws)
    agent = CandidateJudgeAgent()

    def process(profile: CandidateProfile) -> CandidateAssessment:
        skill_matches = compute_skill_matches(
            jd.required_skills + jd.preferred_skills,
            profile.skills,
        )
        raw_cv_text = raw_text_by_id.get(profile.candidate_id, "")
        return agent.run(
            profile, jd, raw_cv_text, skill_matches=skill_matches or None
        )

    return [process(profile) for profile in profiles]


def phase4_pool_calibrator(
    candidate_assessments: list[CandidateAssessment], jd: JDRequirements
) -> FinalRanking:
    return PoolCalibratorAgent().run(candidate_assessments, jd)
