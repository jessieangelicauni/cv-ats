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
from src.utils.vector_store import CVVectorStore


def phase1_jd_parser(jd_raw: str, cache: ExtractionCache | None) -> JDRequirements:
    return JDParserAgent(cache=cache).run(jd_raw)


def phase2_cv_extractor(
    cv_raws: list[dict],
    cache: ExtractionCache | None,
    vector_store: CVVectorStore | None,
) -> list[CandidateProfile]:
    def process(cv_raw: dict) -> CandidateProfile:
        if vector_store is not None:
            vector_store.index_cv(cv_raw["candidate_id"], cv_raw["raw_text"])
        return CVExtractorAgent(cache=cache).run(cv_raw)
    return [process(cv_raw) for cv_raw in cv_raws]


def phase3_candidate_judge(
    profiles: list[CandidateProfile],
    jd: JDRequirements,
    vector_store: CVVectorStore | None,
) -> list[CandidateAssessment]:
    jd_text = jd.model_dump_json()

    def process(profile: CandidateProfile) -> CandidateAssessment:
        context_chunks: list[str] = []
        if vector_store is not None:
            context_chunks = vector_store.retrieve(
                profile.candidate_id, jd_text,
                top_k=10,
            )
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
            profile, jd, context_chunks or None, skill_matches or None
        )

    return [process(profile) for profile in profiles]


def phase4_pool_calibrator(
    candidate_assessments: list[CandidateAssessment], jd: JDRequirements
) -> FinalRanking:
    return PoolCalibratorAgent().run(candidate_assessments, jd)
