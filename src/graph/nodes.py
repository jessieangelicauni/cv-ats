from __future__ import annotations
from src.models.schemas import (
    JDRequirements, CandidateProfile, CandidateAssessment, FinalRanking,
    SkillMatchResult,
)
from src.agents.jd_parser import JDParserAgent
from src.agents.cv_extractor import CVExtractorAgent
from src.agents.candidate_judge import CandidateJudgeAgent
from src.agents.pool_calibrator import PoolCalibratorAgent
from src.utils.cache import ExtractionCache


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
        candidate_names = {s.canonical_skill for s in profile.skills}
        skill_matches = [
            SkillMatchResult(
                jd_skill=s.skill,
                best_match=s.skill if s.skill in candidate_names else None,
                score=1.0 if s.skill in candidate_names else 0.0,
                is_required=s.is_mandatory,
            )
            for s in jd.required_skills + jd.preferred_skills
        ]
        return CandidateJudgeAgent().run(
            profile, jd, skill_matches=skill_matches or None
        )

    return [process(profile) for profile in profiles]


def phase4_pool_calibrator(
    candidate_assessments: list[CandidateAssessment], jd: JDRequirements
) -> FinalRanking:
    return PoolCalibratorAgent().run(candidate_assessments, jd)
