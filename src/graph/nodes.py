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


def phase1_jd_parser(jd_raw: str, cache: ExtractionCache | None) -> JDRequirements:
    return JDParserAgent(cache=cache).run(jd_raw)


def phase2_cv_extractor(
    cv_raws: list[dict],
    cache: ExtractionCache | None,
) -> list[CandidateProfile]:
    def process(cv_raw: dict) -> CandidateProfile:
        return CVExtractorAgent(cache=cache).run(cv_raw)
    return [process(cv_raw) for cv_raw in cv_raws]


def phase3_candidate_judge(
    profiles: list[CandidateProfile],
    jd: JDRequirements,
) -> list[CandidateAssessment]:
    def process(profile: CandidateProfile) -> CandidateAssessment:
        skill_matches = compute_skill_matches(
            jd.required_skills + jd.preferred_skills,
            profile.skills,
        )
        return CandidateJudgeAgent().run(
            profile, jd, skill_matches=skill_matches or None
        )

    return [process(profile) for profile in profiles]


def phase4_pool_calibrator(
    candidate_assessments: list[CandidateAssessment], jd: JDRequirements
) -> FinalRanking:
    return PoolCalibratorAgent().run(candidate_assessments, jd)
