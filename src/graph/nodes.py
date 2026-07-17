from __future__ import annotations
from opentelemetry.trace import StatusCode
from src.models.schemas import (
    JDRequirements, CandidateProfile, CandidateAssessment, FinalRanking,
)
from src.agents.jd_parser import JDParserAgent
from src.agents.cv_extractor import CVExtractorAgent
from src.agents.candidate_judge import CandidateJudgeAgent
from src.agents.pool_calibrator import PoolCalibratorAgent
from src.utils.cache import ExtractionCache
from src.utils.vector_store import CVVectorStore
from src.utils.skill_matcher import SkillMatcher
from src.utils.telemetry import get_tracer
import config

_tracer = get_tracer()


def phase1_jd_parser(jd_raw: str, cache: ExtractionCache | None) -> JDRequirements:
    with _tracer.start_as_current_span("phase1/jd_parser") as span:
        span.set_attribute("phase", 1)
        try:
            return JDParserAgent(cache=cache).run(jd_raw)
        except Exception as exc:
            span.record_exception(exc)
            span.set_status(StatusCode.ERROR, str(exc))
            raise


def phase2_cv_extractor(
    cv_raws: list[dict],
    cache: ExtractionCache | None,
    vector_store: CVVectorStore | None,
) -> list[CandidateProfile]:
    with _tracer.start_as_current_span("phase2/cv_extractor") as span:
        span.set_attribute("phase", 2)
        span.set_attribute("n_candidates", len(cv_raws))

        def process(cv_raw: dict) -> CandidateProfile:
            with _tracer.start_as_current_span(
                f"phase2/cv/{cv_raw['candidate_id']}"
            ) as cspan:
                try:
                    if vector_store is not None:
                        vector_store.index_cv(cv_raw["candidate_id"], cv_raw["raw_text"])
                        queries = [
                            "work experience roles achievements",
                            "technical skills programming languages frameworks",
                            "education degree certifications",
                        ]
                        seen: set[str] = set()
                        chunks: list[str] = []
                        for q in queries:
                            for chunk in vector_store.retrieve(cv_raw["candidate_id"], q):
                                if chunk not in seen:
                                    seen.add(chunk)
                                    chunks.append(chunk)
                        focused = cv_raw.copy()
                        focused["raw_text"] = "\n\n".join(chunks) if chunks else cv_raw["raw_text"]
                        return CVExtractorAgent(cache=cache).run(focused)
                    return CVExtractorAgent(cache=cache).run(cv_raw)
                except Exception as exc:
                    cspan.record_exception(exc)
                    cspan.set_status(StatusCode.ERROR, str(exc))
                    raise

        return [process(cv_raw) for cv_raw in cv_raws]


def phase3_candidate_judge(
    profiles: list[CandidateProfile],
    jd: JDRequirements,
    vector_store: CVVectorStore | None,
    skill_matcher: SkillMatcher | None,
) -> list[CandidateAssessment]:
    with _tracer.start_as_current_span("phase3/candidate_judge") as span:
        span.set_attribute("phase", 3)
        span.set_attribute("n_candidates", len(profiles))

        jd_text = jd.model_dump_json()

        def process(profile: CandidateProfile) -> CandidateAssessment:
            with _tracer.start_as_current_span(
                f"phase3/judge/{profile.candidate_id}"
            ) as cspan:
                try:
                    context_chunks: list[str] = []
                    skill_matches: list = []
                    if vector_store is not None:
                        context_chunks = vector_store.retrieve(
                            profile.candidate_id, jd_text,
                            top_k=config.RETRIEVAL_TOP_K,
                        )
                    if skill_matcher is not None:
                        skill_matches = skill_matcher.match(
                            jd.required_skills + jd.preferred_skills,
                            profile.skills,
                        )
                    return CandidateJudgeAgent().run(
                        profile, jd, context_chunks or None, skill_matches or None
                    )
                except Exception as exc:
                    cspan.record_exception(exc)
                    cspan.set_status(StatusCode.ERROR, str(exc))
                    raise

        return [process(profile) for profile in profiles]


def phase4_pool_calibrator(
    candidate_assessments: list[CandidateAssessment], jd: JDRequirements
) -> FinalRanking:
    with _tracer.start_as_current_span("phase4/pool_calibrator") as span:
        span.set_attribute("phase", 4)
        span.set_attribute("n_candidates", len(candidate_assessments))
        try:
            return PoolCalibratorAgent().run(candidate_assessments, jd)
        except Exception as exc:
            span.record_exception(exc)
            span.set_status(StatusCode.ERROR, str(exc))
            raise
