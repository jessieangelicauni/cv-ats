# Evaluation: Baselines + Ablation Study Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `--baselines` (TF-IDF + keyword-heuristic rankers) and `--ablation` (4 pipeline variants) flags to `main.py`, plus replace fabricated paper sections with real placeholders.

**Architecture:** Baselines live in `src/evaluation/baselines.py` (pure sklearn, no LLM). Ablation lives in `src/evaluation/ablation.py` and reuses `run_pipeline()` with four new boolean flags added to its signature. Each flag disables one system component; the `no_evidence_grounding` variant swaps the judge SYSTEM prompt via a new `src/prompts/judge_no_grounding.py`. Both evaluation modes run after the normal pipeline in `main.py` and write JSON to `run_<id>/evaluation/`.

**Tech Stack:** sklearn (TfidfVectorizer, cosine_similarity), scipy (kendalltau, already used), numpy, LangChain, Pydantic v2, Rich (tables), existing `run_pipeline` / `verify_evidence_chain` / `kendall_tau_score`.

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `src/prompts/judge_no_grounding.py` | Create | Judge SYSTEM prompt without EVIDENCE RULE |
| `src/evaluation/baselines.py` | Create | TF-IDF ranker, keyword ranker, cross-method Kendall's τ |
| `src/evaluation/ablation.py` | Create | Ablation variant runner, per-variant metrics |
| `tests/test_baselines.py` | Create | Tests for baselines module |
| `tests/test_ablation.py` | Create | Tests for ablation module |
| `tests/test_candidate_judge.py` | Create | Tests for `use_evidence_grounding` param |
| `tests/test_pipeline_ablation.py` | Create | Tests for `_ranking_from_raw` helper |
| `src/agents/candidate_judge.py` | Modify | Add `use_evidence_grounding: bool = True` |
| `src/graph/nodes.py` | Modify | Pass `use_evidence_grounding` to `phase3_candidate_judge` |
| `src/graph/pipeline.py` | Modify | Add 4 ablation flags + `_ranking_from_raw` helper |
| `main.py` | Modify | Add `--baselines` and `--ablation` CLI flags + output |
| `docs/paper/evidencerank.tex` | Modify | Replace fabricated Experimental Design + Results sections with placeholders |

---

## Task 1: Create `src/prompts/judge_no_grounding.py`

**Files:**
- Create: `src/prompts/judge_no_grounding.py`

The no-grounding variant uses an identical SYSTEM prompt to `src/prompts/judge.py` but with the EVIDENCE RULE paragraph removed. The LLM still produces `evidence_chain` items (same schema) but without the verbatim-quote constraint, so the hallucination checker will catch more fabricated claims — this is exactly what the ablation measures.

- [ ] **Step 1: Create the file**

```python
# src/prompts/judge_no_grounding.py
SYSTEM = """You are an experienced senior technical recruiter with 15 years in IT hiring.
You reason like a human expert: leadership and measurable impact matter more than job \
titles; relevant tenure increases confidence; career changes that represent growth are \
not penalized.

When a candidate has worked at a Tier 1 multinational company, treat this as a \
meaningful positive signal — it indicates exposure to large-scale systems, professional \
engineering culture, and organisational breadth. Weight this in your holistic \
assessment, particularly under Career Trajectory and Experience Depth.

Return a valid JSON object matching the required schema exactly."""
```

- [ ] **Step 2: Commit**

```bash
git add src/prompts/judge_no_grounding.py
git commit -m "feat: add judge prompt without evidence-chain grounding rule"
```

---

## Task 2: TDD for `src/evaluation/baselines.py`

**Files:**
- Create: `tests/test_baselines.py`
- Create: `src/evaluation/baselines.py`

**Dependencies:** sklearn is already installed (used in `src/evaluation/ranking_metrics.py` via `ndcg_score`).

- [ ] **Step 1: Write failing tests**

```python
# tests/test_baselines.py
import pytest
from src.evaluation.baselines import tfidf_rank, keyword_rank, run_baselines


SAMPLE_JD = "Python machine learning deep learning neural networks"

SAMPLE_CVS = [
    {"candidate_id": "cv_a", "raw_text": "Java spring boot enterprise microservices"},
    {"candidate_id": "cv_b", "raw_text": "Python scikit-learn deep learning PyTorch neural"},
    {"candidate_id": "cv_c", "raw_text": "Python data analytics pandas machine learning"},
]


def test_tfidf_rank_returns_all_candidates():
    ranking, scores = tfidf_rank(SAMPLE_JD, SAMPLE_CVS)
    assert set(ranking) == {"cv_a", "cv_b", "cv_c"}
    assert set(scores.keys()) == {"cv_a", "cv_b", "cv_c"}


def test_tfidf_rank_most_similar_first():
    ranking, scores = tfidf_rank(SAMPLE_JD, SAMPLE_CVS)
    assert ranking[0] in ("cv_b", "cv_c")   # both Python/ML — either can lead
    assert ranking[-1] == "cv_a"            # Java-only should rank last


def test_tfidf_scores_in_zero_one():
    _, scores = tfidf_rank(SAMPLE_JD, SAMPLE_CVS)
    for v in scores.values():
        assert 0.0 <= v <= 1.0


def test_keyword_rank_returns_all_candidates():
    ranking, counts = keyword_rank(SAMPLE_JD, SAMPLE_CVS)
    assert set(ranking) == {"cv_a", "cv_b", "cv_c"}
    assert set(counts.keys()) == {"cv_a", "cv_b", "cv_c"}


def test_keyword_rank_most_matches_first():
    ranking, counts = keyword_rank(SAMPLE_JD, SAMPLE_CVS)
    assert counts[ranking[0]] >= counts[ranking[-1]]


def test_keyword_java_cv_has_fewer_matches():
    _, counts = keyword_rank(SAMPLE_JD, SAMPLE_CVS)
    assert counts["cv_a"] < counts["cv_b"]


def test_run_baselines_structure():
    er_ranking = ["cv_b", "cv_c", "cv_a"]
    result = run_baselines(SAMPLE_JD, SAMPLE_CVS, er_ranking)

    assert "tfidf" in result
    assert "keyword" in result
    assert "evidencerank" in result
    assert "cross_method_tau" in result

    for method in ("tfidf", "keyword"):
        assert "ranking" in result[method]
        assert "scores" in result[method]
        assert "distribution" in result[method]
        dist = result[method]["distribution"]
        assert set(dist.keys()) == {"mean", "std", "min", "max"}

    assert set(result["cross_method_tau"].keys()) == {
        "tfidf_vs_keyword",
        "tfidf_vs_evidencerank",
        "keyword_vs_evidencerank",
    }


def test_run_baselines_evidencerank_ranking_preserved():
    er_ranking = ["cv_b", "cv_c", "cv_a"]
    result = run_baselines(SAMPLE_JD, SAMPLE_CVS, er_ranking)
    assert result["evidencerank"]["ranking"] == er_ranking


def test_run_baselines_tau_values_in_range():
    er_ranking = ["cv_b", "cv_c", "cv_a"]
    result = run_baselines(SAMPLE_JD, SAMPLE_CVS, er_ranking)
    for tau in result["cross_method_tau"].values():
        assert -1.0 <= tau <= 1.0
```

- [ ] **Step 2: Run tests — verify ImportError**

```bash
uv run pytest tests/test_baselines.py -v
```

Expected: `ImportError: cannot import name 'tfidf_rank' from 'src.evaluation.baselines'` (module doesn't exist yet).

- [ ] **Step 3: Implement `src/evaluation/baselines.py`**

```python
# src/evaluation/baselines.py
from __future__ import annotations
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity
from src.evaluation.ranking_metrics import kendall_tau_score


def tfidf_rank(jd_text: str, cv_raws: list[dict]) -> tuple[list[str], dict[str, float]]:
    candidate_ids = [cv["candidate_id"] for cv in cv_raws]
    texts = [jd_text] + [cv["raw_text"] for cv in cv_raws]
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(texts)
    sims = cosine_similarity(matrix[0:1], matrix[1:])[0]
    scores = {cid: round(float(s), 4) for cid, s in zip(candidate_ids, sims)}
    ranking = sorted(candidate_ids, key=lambda cid: scores[cid], reverse=True)
    return ranking, scores


def keyword_rank(jd_text: str, cv_raws: list[dict]) -> tuple[list[str], dict[str, float]]:
    candidate_ids = [cv["candidate_id"] for cv in cv_raws]
    jd_tokens = {
        w for w in jd_text.lower().split()
        if w.isalpha() and w not in ENGLISH_STOP_WORDS
    }
    counts: dict[str, float] = {}
    for cv in cv_raws:
        text_lower = cv["raw_text"].lower()
        counts[cv["candidate_id"]] = float(sum(1 for kw in jd_tokens if kw in text_lower))
    ranking = sorted(candidate_ids, key=lambda cid: counts[cid], reverse=True)
    return ranking, counts


def _distribution(scores: dict[str, float]) -> dict[str, float]:
    vals = list(scores.values())
    return {
        "mean": round(float(np.mean(vals)), 4),
        "std":  round(float(np.std(vals)), 4),
        "min":  round(float(np.min(vals)), 4),
        "max":  round(float(np.max(vals)), 4),
    }


def run_baselines(
    jd_text: str,
    cv_raws: list[dict],
    evidencerank_ranking: list[str],
) -> dict:
    tfidf_ranking, tfidf_scores = tfidf_rank(jd_text, cv_raws)
    kw_ranking, kw_scores = keyword_rank(jd_text, cv_raws)

    tau_tf_kw, _ = kendall_tau_score(tfidf_ranking, kw_ranking)
    tau_tf_er, _ = kendall_tau_score(tfidf_ranking, evidencerank_ranking)
    tau_kw_er, _ = kendall_tau_score(kw_ranking, evidencerank_ranking)

    return {
        "tfidf": {
            "ranking": tfidf_ranking,
            "scores": tfidf_scores,
            "distribution": _distribution(tfidf_scores),
        },
        "keyword": {
            "ranking": kw_ranking,
            "scores": kw_scores,
            "distribution": _distribution(kw_scores),
        },
        "evidencerank": {
            "ranking": evidencerank_ranking,
        },
        "cross_method_tau": {
            "tfidf_vs_keyword":        round(float(tau_tf_kw), 4),
            "tfidf_vs_evidencerank":   round(float(tau_tf_er), 4),
            "keyword_vs_evidencerank": round(float(tau_kw_er), 4),
        },
    }
```

- [ ] **Step 4: Run tests — verify all pass**

```bash
uv run pytest tests/test_baselines.py -v
```

Expected: all 9 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/test_baselines.py src/evaluation/baselines.py
git commit -m "feat: add TF-IDF and keyword-heuristic baseline rankers"
```

---

## Task 3: TDD for `CandidateJudgeAgent(use_evidence_grounding=False)`

**Files:**
- Create: `tests/test_candidate_judge.py`
- Modify: `src/agents/candidate_judge.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_candidate_judge.py
from unittest.mock import MagicMock, patch
from src.agents.candidate_judge import CandidateJudgeAgent
from src.models.schemas import (
    CandidateAssessment, EvidenceItem, CandidateProfile,
    CandidateBasicInfo, JDRequirements, EducationRequirement,
)


def _make_assessment() -> CandidateAssessment:
    return CandidateAssessment(
        candidate_id="cv_001", raw_score=80.0, confidence="high",
        evidence_chain=[EvidenceItem(
            dimension="Technical Skills Fit", assessment="Good.",
            evidence_quote="Python", dimension_score=8.0,
        )],
        key_strengths=["Python"], key_gaps=[], seniority_alignment="aligned",
    )


def _make_profile() -> CandidateProfile:
    return CandidateProfile(
        candidate_id="cv_001",
        basic_info=CandidateBasicInfo(
            full_name="Test", email=None, phone=None,
            location=None, linkedin_url=None, current_title=None,
        ),
        skills=[], work_history=[], education=[],
        certifications=[], languages=[], total_experience_months=24,
    )


def _make_jd() -> JDRequirements:
    return JDRequirements(
        role_title="Engineer", seniority_level="mid",
        required_skills=[], preferred_skills=[], min_years_experience=2,
        education=EducationRequirement(degree="BSc", field="CS", is_mandatory=False),
        domain_expertise=[], leadership_expected=False,
        soft_skills=[], industry_context="IT", raw_jd_hash="abc",
    )


def test_judge_default_uses_evidence_grounding_system_prompt():
    from src.prompts import judge
    mock_llm = MagicMock()
    with patch("src.agents.candidate_judge.get_llm", return_value=mock_llm), \
         patch("src.agents.candidate_judge.invoke_with_telemetry",
               return_value=_make_assessment()) as mock_invoke:
        agent = CandidateJudgeAgent()
        agent.run(_make_profile(), _make_jd())
    messages = mock_invoke.call_args[0][1]
    assert messages[0].content == judge.SYSTEM


def test_judge_no_grounding_uses_no_grounding_system_prompt():
    from src.prompts import judge_no_grounding
    mock_llm = MagicMock()
    with patch("src.agents.candidate_judge.get_llm", return_value=mock_llm), \
         patch("src.agents.candidate_judge.invoke_with_telemetry",
               return_value=_make_assessment()) as mock_invoke:
        agent = CandidateJudgeAgent(use_evidence_grounding=False)
        agent.run(_make_profile(), _make_jd())
    messages = mock_invoke.call_args[0][1]
    assert messages[0].content == judge_no_grounding.SYSTEM


def test_judge_no_grounding_human_message_unchanged():
    """The human() function is reused regardless of grounding flag."""
    from src.prompts import judge
    mock_llm = MagicMock()
    with patch("src.agents.candidate_judge.get_llm", return_value=mock_llm), \
         patch("src.agents.candidate_judge.invoke_with_telemetry",
               return_value=_make_assessment()) as mock_invoke:
        agent_on  = CandidateJudgeAgent(use_evidence_grounding=True)
        agent_off = CandidateJudgeAgent(use_evidence_grounding=False)
        agent_on.run(_make_profile(), _make_jd())
        msgs_on = mock_invoke.call_args[0][1]
        agent_off.run(_make_profile(), _make_jd())
        msgs_off = mock_invoke.call_args[0][1]
    assert msgs_on[1].content == msgs_off[1].content
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
uv run pytest tests/test_candidate_judge.py -v
```

Expected: `TypeError: CandidateJudgeAgent.__init__() got an unexpected keyword argument 'use_evidence_grounding'`

- [ ] **Step 3: Modify `src/agents/candidate_judge.py`**

Replace the full file:

```python
# src/agents/candidate_judge.py
from __future__ import annotations
from langchain_core.messages import SystemMessage, HumanMessage
from src.models.schemas import CandidateAssessment, CandidateProfile, JDRequirements
from src.utils.llm import get_llm, invoke_with_telemetry
from src.prompts import judge as prompts
from src.prompts import judge_no_grounding as prompts_no_grounding
import config


class CandidateJudgeAgent:
    def __init__(self, use_evidence_grounding: bool = True):
        self._llm = get_llm(config.LARGE_MODEL, CandidateAssessment, config.JUDGE_TEMPERATURE)
        self._use_evidence_grounding = use_evidence_grounding

    def run(
        self,
        profile: CandidateProfile,
        jd: JDRequirements,
        context_chunks: list[str] | None = None,
        skill_matches: list | None = None,
    ) -> CandidateAssessment:
        system = (
            prompts.SYSTEM if self._use_evidence_grounding
            else prompts_no_grounding.SYSTEM
        )
        return invoke_with_telemetry(
            self._llm,
            [SystemMessage(content=system),
             HumanMessage(content=prompts.human(
                 jd.model_dump_json(indent=2),
                 profile.model_dump_json(indent=2),
                 context_chunks=context_chunks,
                 skill_matches=skill_matches,
             ))],
            run_name="candidate_judge",
        )
```

- [ ] **Step 4: Run tests — verify all pass**

```bash
uv run pytest tests/test_candidate_judge.py -v
```

Expected: all 3 tests PASS.

- [ ] **Step 5: Run full test suite — no regressions**

```bash
uv run pytest -v
```

Expected: all existing tests still pass.

- [ ] **Step 6: Commit**

```bash
git add tests/test_candidate_judge.py src/agents/candidate_judge.py
git commit -m "feat: add use_evidence_grounding param to CandidateJudgeAgent"
```

---

## Task 4: Add ablation flags to `pipeline.py` and `nodes.py`

**Files:**
- Create: `tests/test_pipeline_ablation.py`
- Modify: `src/graph/pipeline.py`
- Modify: `src/graph/nodes.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_pipeline_ablation.py
from src.graph.pipeline import _ranking_from_raw
from src.models.schemas import CandidateAssessment, EvidenceItem


def _make_assessment(cid: str, score: float) -> CandidateAssessment:
    return CandidateAssessment(
        candidate_id=cid, raw_score=score, confidence="high",
        evidence_chain=[EvidenceItem(
            dimension="Technical Skills Fit", assessment="OK.",
            evidence_quote="Python", dimension_score=7.0,
        )],
        key_strengths=[], key_gaps=[], seniority_alignment="aligned",
    )


def test_ranking_from_raw_sorts_descending():
    assessments = [
        _make_assessment("a", 70.0),
        _make_assessment("b", 90.0),
        _make_assessment("c", 80.0),
    ]
    result = _ranking_from_raw(assessments)
    ids = [r.candidate_id for r in result.ranked_candidates]
    assert ids == ["b", "c", "a"]


def test_ranking_from_raw_calibrated_equals_raw():
    assessments = [_make_assessment("a", 70.0), _make_assessment("b", 90.0)]
    result = _ranking_from_raw(assessments)
    score_map = {r.candidate_id: r.calibrated_score for r in result.ranked_candidates}
    assert score_map["a"] == 70.0
    assert score_map["b"] == 90.0


def test_ranking_from_raw_delta_is_zero():
    assessments = [_make_assessment("a", 70.0), _make_assessment("b", 90.0)]
    result = _ranking_from_raw(assessments)
    assert all(r.delta_from_raw == 0.0 for r in result.ranked_candidates)


def test_ranking_from_raw_assigns_ranks():
    assessments = [_make_assessment("a", 70.0), _make_assessment("b", 90.0)]
    result = _ranking_from_raw(assessments)
    ranks = [r.rank for r in result.ranked_candidates]
    assert ranks == [1, 2]


def test_ranking_from_raw_pool_summary():
    assessments = [_make_assessment("a", 70.0)]
    result = _ranking_from_raw(assessments)
    assert "No calibration" in result.pool_summary
    assert result.borderline_pairs == []
```

- [ ] **Step 2: Run tests — verify ImportError**

```bash
uv run pytest tests/test_pipeline_ablation.py -v
```

Expected: `ImportError: cannot import name '_ranking_from_raw'`

- [ ] **Step 3: Modify `src/graph/pipeline.py`**

Replace the full file with the updated version:

```python
# src/graph/pipeline.py
from __future__ import annotations
import hashlib
import time
import uuid
from typing import Callable
from src.graph.state import ATSState
from src.graph.nodes import (
    phase1_jd_parser, phase2_cv_extractor,
    phase3_candidate_judge, phase4_pool_calibrator,
)
from src.utils.cache import ExtractionCache
from src.utils.embedder import get_embedder
from src.utils.vector_store import CVVectorStore
from src.utils.skill_matcher import SkillMatcher
from src.utils.telemetry import setup_telemetry, get_tracer, current_otel_trace_id
from src.models.schemas import (
    CandidateProfile, CandidateAssessment, JDRequirements,
    FinalRanking, RankedCandidate,
)
import config


def _filter_by_required_skills(
    profiles: list[CandidateProfile],
    jd: JDRequirements,
    skill_matcher: SkillMatcher | None,
) -> tuple[list[CandidateProfile], list[str]]:
    if skill_matcher is None or not jd.required_skills:
        return profiles, []
    passing: list[CandidateProfile] = []
    eliminated: list[str] = []
    for profile in profiles:
        matches = skill_matcher.match(jd.required_skills, profile.skills)
        if any(m.score >= config.SKILL_MATCH_THRESHOLD for m in matches):
            passing.append(profile)
        else:
            eliminated.append(profile.candidate_id)
    return passing, eliminated


def _ranking_from_raw(assessments: list[CandidateAssessment]) -> FinalRanking:
    sorted_a = sorted(assessments, key=lambda a: a.raw_score, reverse=True)
    return FinalRanking(
        ranked_candidates=[
            RankedCandidate(
                rank=i + 1,
                candidate_id=a.candidate_id,
                calibrated_score=a.raw_score,
                delta_from_raw=0.0,
                comparative_notes="",
            )
            for i, a in enumerate(sorted_a)
        ],
        pool_summary="No calibration applied.",
        calibration_rationale="No calibration applied.",
        borderline_pairs=[],
    )


def run_pipeline(
    jd_raw: str,
    cv_raws: list[dict],
    run_id: str | None = None,
    use_cache: bool = True,
    session_id: str | None = None,
    on_phase_complete: Callable[[dict], None] | None = None,
    use_vector_store: bool = True,
    use_skill_filter: bool = True,
    use_evidence_grounding: bool = True,
    use_pool_calibration: bool = True,
) -> ATSState:
    setup_telemetry()
    run_id = run_id or str(uuid.uuid4())[:8]
    jd_hash = hashlib.sha256(jd_raw.encode()).hexdigest()[:12]
    cache         = ExtractionCache(config.CACHE_DB_PATH)  if use_cache else None
    vector_store  = CVVectorStore(config.CHROMA_DB_PATH)   if use_cache else None
    skill_matcher = SkillMatcher(get_embedder())            if use_cache else None

    # Ablation overrides: disable specific components independently of use_cache
    rag_store = vector_store if use_vector_store else None

    tracer = get_tracer()
    with tracer.start_as_current_span(
        "pipeline",
        attributes={
            "run.id": run_id,
            "run.n_candidates": len(cv_raws),
            "run.jd_hash": jd_hash,
            "run.session_id": session_id or "",
            "llm.small_model": config.SMALL_MODEL,
            "llm.large_model": config.LARGE_MODEL,
        },
    ):
        otel_trace_id = current_otel_trace_id()
        trace_log: list[dict] = []

        def _run_phase(n: int, fn, *args, **extra_log):
            t0 = time.time()
            result = fn(*args)
            entry = {"phase": n, "duration_s": round(time.time() - t0, 2), **extra_log}
            trace_log.append(entry)
            if on_phase_complete:
                on_phase_complete(entry)
            return result

        jd_structured = _run_phase(1, phase1_jd_parser, jd_raw, cache)
        cv_profiles   = _run_phase(2, phase2_cv_extractor, cv_raws, cache,
                                   rag_store, candidates=len(cv_raws))

        if use_skill_filter:
            cv_profiles, eliminated = _filter_by_required_skills(
                cv_profiles, jd_structured, skill_matcher)
        else:
            eliminated = []

        assessments = _run_phase(3, phase3_candidate_judge, cv_profiles,
                                 jd_structured, rag_store, skill_matcher,
                                 use_evidence_grounding)

        if use_pool_calibration:
            final_ranking = _run_phase(4, phase4_pool_calibrator,
                                       assessments, jd_structured)
        else:
            final_ranking = _ranking_from_raw(assessments)

        return ATSState(
            jd_raw=jd_raw,
            cv_raws=cv_raws,
            jd_structured=jd_structured,
            cv_profiles=cv_profiles,
            candidate_assessments=assessments,
            final_ranking=final_ranking,
            run_id=run_id,
            otel_trace_id=otel_trace_id,
            trace_log=trace_log,
            use_cache=use_cache,
            eliminated_candidates=eliminated,
        )
```

- [ ] **Step 4: Modify `src/graph/nodes.py` — add `use_evidence_grounding` to `phase3_candidate_judge`**

Replace only the `phase3_candidate_judge` function:

```python
def phase3_candidate_judge(
    profiles: list[CandidateProfile],
    jd: JDRequirements,
    vector_store: CVVectorStore | None,
    skill_matcher: SkillMatcher | None,
    use_evidence_grounding: bool = True,
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
                    return CandidateJudgeAgent(
                        use_evidence_grounding=use_evidence_grounding
                    ).run(
                        profile, jd, context_chunks or None, skill_matches or None
                    )
                except Exception as exc:
                    cspan.record_exception(exc)
                    cspan.set_status(StatusCode.ERROR, str(exc))
                    raise

        return [process(profile) for profile in profiles]
```

- [ ] **Step 5: Run tests — verify all pass**

```bash
uv run pytest tests/test_pipeline_ablation.py tests/test_candidate_judge.py -v
```

Expected: all 8 tests PASS.

- [ ] **Step 6: Run full test suite — no regressions**

```bash
uv run pytest -v
```

Expected: all existing tests still PASS.

- [ ] **Step 7: Commit**

```bash
git add tests/test_pipeline_ablation.py src/graph/pipeline.py src/graph/nodes.py
git commit -m "feat: add ablation flags to run_pipeline and _ranking_from_raw helper"
```

---

## Task 5: TDD for `src/evaluation/ablation.py`

**Files:**
- Create: `tests/test_ablation.py`
- Create: `src/evaluation/ablation.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_ablation.py
from unittest.mock import patch, MagicMock
from src.evaluation.ablation import run_ablation, ABLATION_VARIANTS


def _make_ranked(ids: list[str]):
    return [
        MagicMock(candidate_id=cid, calibrated_score=90.0 - i * 5)
        for i, cid in enumerate(ids)
    ]


def _make_state(ids: list[str]):
    state = MagicMock()
    state.final_ranking.ranked_candidates = _make_ranked(ids)
    state.candidate_assessments = [MagicMock(candidate_id=cid) for cid in ids]
    return state


def test_ablation_variants_list_has_four_entries():
    assert len(ABLATION_VARIANTS) == 4


def test_ablation_variant_names():
    names = {v["name"] for v in ABLATION_VARIANTS}
    assert names == {"no_rag", "no_evidence_grounding", "no_skill_filter", "no_calibration"}


def test_run_ablation_calls_pipeline_once_per_variant():
    full_state = _make_state(["a", "b"])
    cv_text_map = {"a": "text a", "b": "text b"}

    with patch("src.evaluation.ablation.run_pipeline",
               return_value=_make_state(["a", "b"])) as mock_pipe, \
         patch("src.evaluation.ablation.verify_evidence_chain", return_value=[]), \
         patch("src.evaluation.ablation.hallucination_rate", return_value=0.0):
        run_ablation("jd", [], full_state, cv_text_map)

    assert mock_pipe.call_count == len(ABLATION_VARIANTS)


def test_run_ablation_passes_correct_kwargs_to_pipeline():
    full_state = _make_state(["a", "b"])
    cv_text_map = {"a": "text a", "b": "text b"}
    captured_kwargs = []

    def capture(*args, **kwargs):
        captured_kwargs.append(kwargs)
        return _make_state(["a", "b"])

    with patch("src.evaluation.ablation.run_pipeline", side_effect=capture), \
         patch("src.evaluation.ablation.verify_evidence_chain", return_value=[]), \
         patch("src.evaluation.ablation.hallucination_rate", return_value=0.0):
        run_ablation("jd", [], full_state, cv_text_map)

    # no_rag variant must pass use_vector_store=False
    no_rag_call = next(kw for kw in captured_kwargs if kw.get("use_vector_store") is False)
    assert no_rag_call is not None

    # no_evidence_grounding variant must pass use_evidence_grounding=False
    no_eg_call = next(kw for kw in captured_kwargs if kw.get("use_evidence_grounding") is False)
    assert no_eg_call is not None


def test_run_ablation_result_contains_all_variants():
    full_state = _make_state(["a", "b"])
    cv_text_map = {"a": "text a", "b": "text b"}

    with patch("src.evaluation.ablation.run_pipeline",
               return_value=_make_state(["a", "b"])), \
         patch("src.evaluation.ablation.verify_evidence_chain", return_value=[]), \
         patch("src.evaluation.ablation.hallucination_rate", return_value=0.0):
        result = run_ablation("jd", [], full_state, cv_text_map)

    assert "full_system" in result
    for v in ABLATION_VARIANTS:
        assert v["name"] in result


def test_run_ablation_metrics_keys():
    full_state = _make_state(["a", "b"])
    cv_text_map = {"a": "text a", "b": "text b"}

    with patch("src.evaluation.ablation.run_pipeline",
               return_value=_make_state(["a", "b"])), \
         patch("src.evaluation.ablation.verify_evidence_chain", return_value=[]), \
         patch("src.evaluation.ablation.hallucination_rate", return_value=0.0):
        result = run_ablation("jd", [], full_state, cv_text_map)

    for metrics in result.values():
        assert set(metrics.keys()) == {
            "hallucination_rate", "score_std", "llm_calls", "tau_vs_full"
        }


def test_full_system_tau_is_one():
    full_state = _make_state(["a", "b"])
    cv_text_map = {"a": "text a", "b": "text b"}

    with patch("src.evaluation.ablation.run_pipeline",
               return_value=_make_state(["a", "b"])), \
         patch("src.evaluation.ablation.verify_evidence_chain", return_value=[]), \
         patch("src.evaluation.ablation.hallucination_rate", return_value=0.0):
        result = run_ablation("jd", [], full_state, cv_text_map)

    assert result["full_system"]["tau_vs_full"] == 1.0
```

- [ ] **Step 2: Run tests — verify ImportError**

```bash
uv run pytest tests/test_ablation.py -v
```

Expected: `ImportError: cannot import name 'run_ablation'`

- [ ] **Step 3: Implement `src/evaluation/ablation.py`**

```python
# src/evaluation/ablation.py
from __future__ import annotations
import numpy as np
from src.graph.pipeline import run_pipeline
from src.evaluation.hallucination_checker import verify_evidence_chain, hallucination_rate
from src.evaluation.ranking_metrics import kendall_tau_score

ABLATION_VARIANTS: list[dict] = [
    {"name": "no_rag",                "use_vector_store": False},
    {"name": "no_evidence_grounding", "use_evidence_grounding": False},
    {"name": "no_skill_filter",       "use_skill_filter": False},
    {"name": "no_calibration",        "use_pool_calibration": False},
]


def _variant_metrics(state, full_ranking: list[str], cv_text_map: dict[str, str]) -> dict:
    ranking = [r.candidate_id for r in state.final_ranking.ranked_candidates]
    tau, _ = kendall_tau_score(ranking, full_ranking)

    flags = []
    for a in state.candidate_assessments:
        flags.extend(verify_evidence_chain(a, cv_text_map.get(a.candidate_id, "")))
    h_rate = hallucination_rate(flags)

    scores = [r.calibrated_score for r in state.final_ranking.ranked_candidates]
    score_std = float(np.std(scores)) if scores else 0.0

    return {
        "hallucination_rate": round(h_rate, 4),
        "score_std":          round(score_std, 4),
        "llm_calls":          len(state.candidate_assessments),
        "tau_vs_full":        round(float(tau), 4),
    }


def run_ablation(
    jd_raw: str,
    cv_raws: list[dict],
    full_state,
    cv_text_map: dict[str, str],
) -> dict:
    full_ranking = [r.candidate_id for r in full_state.final_ranking.ranked_candidates]
    results = {"full_system": _variant_metrics(full_state, full_ranking, cv_text_map)}
    results["full_system"]["tau_vs_full"] = 1.0  # always 1.0 by definition

    for variant in ABLATION_VARIANTS:
        name = variant["name"]
        kwargs = {k: v for k, v in variant.items() if k != "name"}
        state = run_pipeline(jd_raw, cv_raws, **kwargs)
        results[name] = _variant_metrics(state, full_ranking, cv_text_map)

    return results
```

- [ ] **Step 4: Run tests — verify all pass**

```bash
uv run pytest tests/test_ablation.py -v
```

Expected: all 7 tests PASS.

- [ ] **Step 5: Run full test suite — no regressions**

```bash
uv run pytest -v
```

Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
git add tests/test_ablation.py src/evaluation/ablation.py
git commit -m "feat: add ablation study runner with 4 pipeline variants"
```

---

## Task 6: Add `--baselines` and `--ablation` flags to `main.py`

**Files:**
- Modify: `main.py`

Both flags follow the same pattern as the existing `--eval` flag: they run after the normal pipeline, print a Rich table, save JSON to `run_<id>/evaluation/`, and log metrics to Langfuse.

- [ ] **Step 1: Add the two new parameters to the `main()` function signature**

In `main.py`, find the existing `@app.command()` decorated function and add two new options after `session_id`:

```python
baselines: bool = typer.Option(False, "--baselines", help="Run TF-IDF and keyword-heuristic baseline rankers (fast, no LLM)"),
ablation:  bool = typer.Option(False, "--ablation",  help="Run 4 ablation variants (LLM-heavy — use deliberately)"),
```

- [ ] **Step 2: Add `--baselines` block after the ranking table is printed**

Locate the line `console.print(table)` in `main.py` and add the following block immediately after it (before the `if eval:` block):

```python
        cv_text_map = {c["candidate_id"]: c["raw_text"] for c in cv_raws}
        er_ranking = [r.candidate_id for r in ranking.ranked_candidates]

        if baselines:
            import json
            from src.evaluation.baselines import run_baselines
            bl = run_baselines(jd_text, cv_raws, er_ranking)

            bl_table = Table(title="Baseline Comparison")
            bl_table.add_column("Method")
            bl_table.add_column("Ranking")
            bl_table.add_column("Mean Score")
            bl_table.add_column("Std Score")
            bl_table.add_column("τ vs EvidenceRank")
            for method in ("tfidf", "keyword"):
                d = bl[method]
                bl_table.add_row(
                    method.upper(),
                    " > ".join(d["ranking"]),
                    str(d["distribution"]["mean"]),
                    str(d["distribution"]["std"]),
                    str(bl["cross_method_tau"][f"{method}_vs_evidencerank"]),
                )
            console.print(bl_table)

            (out_dir / "evaluation").mkdir(parents=True, exist_ok=True)
            (out_dir / "evaluation" / "baselines.json").write_text(
                json.dumps(bl, indent=2), encoding="utf-8"
            )
            for key, val in bl["cross_method_tau"].items():
                lf.create_score(trace_id=otel_trace_id, name=f"baseline_{key}", value=val)
            console.print(f"[green]Baselines written to {out_dir}/evaluation/baselines.json[/green]")
```

- [ ] **Step 3: Add `--ablation` block after the `--baselines` block**

```python
        if ablation:
            import json
            from src.evaluation.ablation import run_ablation
            console.print("[blue]Running ablation study (this will run 4 extra pipeline passes)...[/blue]")
            ab = run_ablation(jd_text, cv_raws, state, cv_text_map)

            ab_table = Table(title="Ablation Study")
            ab_table.add_column("Variant")
            ab_table.add_column("Hallucination Rate")
            ab_table.add_column("Score Std")
            ab_table.add_column("LLM Calls")
            ab_table.add_column("τ vs Full")
            for variant_name, metrics in ab.items():
                ab_table.add_row(
                    variant_name,
                    f"{metrics['hallucination_rate']:.1%}",
                    str(metrics["score_std"]),
                    str(metrics["llm_calls"]),
                    str(metrics["tau_vs_full"]),
                )
            console.print(ab_table)

            (out_dir / "evaluation").mkdir(parents=True, exist_ok=True)
            (out_dir / "evaluation" / "ablation.json").write_text(
                json.dumps(ab, indent=2), encoding="utf-8"
            )
            for variant_name, metrics in ab.items():
                lf.create_score(
                    trace_id=otel_trace_id,
                    name=f"ablation_{variant_name}_hallucination_rate",
                    value=metrics["hallucination_rate"],
                )
                lf.create_score(
                    trace_id=otel_trace_id,
                    name=f"ablation_{variant_name}_tau_vs_full",
                    value=metrics["tau_vs_full"],
                )
            console.print(f"[green]Ablation written to {out_dir}/evaluation/ablation.json[/green]")
```

- [ ] **Step 4: Verify the CLI help text is correct**

```bash
uv run python main.py --help
```

Expected output includes:
```
--baselines    Run TF-IDF and keyword-heuristic baseline rankers (fast, no LLM)
--ablation     Run 4 ablation variants (LLM-heavy — use deliberately)
```

- [ ] **Step 5: Commit**

```bash
git add main.py
git commit -m "feat: add --baselines and --ablation CLI flags to main.py"
```

---

## Task 7: Replace fabricated sections in `docs/paper/evidencerank.tex`

**Files:**
- Modify: `docs/paper/evidencerank.tex`

The paper currently has a fabricated `\section{Experimental Design}` (lines ~515–585) and a fully fabricated `\section{Results and Discussion}` (lines ~587–827) with invented numbers and non-existent human annotators. Replace both sections and fix the Conclusion and Acknowledgement.

All placeholders use `%% TODO:` so they are greppable: `grep -n "TODO" docs/paper/evidencerank.tex`

- [ ] **Step 1: Replace `\section{Experimental Design}` content**

Find the block from `\section{Experimental Design}` through to the line before `\section{Results and Discussion}` and replace with:

```latex
%% ─────────────────────────────────────────────────────────────────────────────
\section{Experimental Design}
\label{sec:experiment}

\subsection{Dataset}

%% TODO: Update with final dataset details after running experiments.
%% Planned: 20 anonymised IT candidate CVs across seniority levels and
%% specialisations, evaluated against two job descriptions:
%% JD1 — Senior Backend Engineer (Python/PostgreSQL/Kubernetes, 5+ years)
%% JD2 — Data Engineer (Spark/Airflow/dbt, 3+ years).
%% This will yield 40 evaluation instances (20 candidates × 2 JDs).

\subsection{Baselines}

We compare EvidenceRank against two non-LLM baselines and four ablation variants.

\textbf{Non-LLM Baselines:}

\begin{enumerate}
  \item \textbf{TF-IDF + Cosine (B1)}: TF-IDF vectorisation of CV and JD text
        using \texttt{sklearn.TfidfVectorizer} with English stop-word removal,
        ranked by cosine similarity. Represents the conventional ATS keyword-based
        approach. No LLM involved.
  \item \textbf{Keyword-Heuristic (B2)}: Count of unique JD non-stop-word token
        matches (case-insensitive) found in the CV. Represents the simplest
        possible lexical ATS filter.
\end{enumerate}

\textbf{Ablation Variants} (EvidenceRank with one component disabled):

\begin{enumerate}
  \item \textbf{No-RAG}: ChromaDB self-document retrieval disabled; LLM receives
        full raw CV text instead of focused paragraph excerpts. Isolates the
        contribution of RAG to hallucination rate.
  \item \textbf{No-Evidence-Grounding}: Judge prompt EVIDENCE RULE removed;
        the LLM is not required to cite verbatim text. Isolates the contribution
        of the evidence-chain constraint to hallucination rate.
  \item \textbf{No-Skill-Filter}: Semantic skill pre-filter disabled; all
        candidates proceed to Phase 3 LLM judging. Measures LLM call savings
        from the pre-filter.
  \item \textbf{No-Calibration}: Phase 4 pool calibration skipped; Phase 3 raw
        scores sorted directly into the final ranking. Measures Phase 4's
        contribution to score spread and ranking change.
\end{enumerate}

\subsection{Evaluation Metrics}

\begin{enumerate}
  \item \textbf{Hallucination rate} $H$ (Equation~\ref{eq:hallucination_rate}):
        Measured for EvidenceRank and ablation variants using the post-hoc
        hallucination checker on all candidate assessments.
  \item \textbf{Score standard deviation}: Spread of calibrated scores across
        the candidate pool, measuring how discriminative each configuration is.
  \item \textbf{LLM invocations}: Total Phase 3 judge calls per run, measuring
        compute efficiency relative to the no-filter ablation.
  \item \textbf{Kendall's $\tau$ vs.\ full system}: Rank correlation between
        each ablation variant's ranking and the full EvidenceRank ranking,
        measuring how much each removed component shifts the final order.
  \item \textbf{Cross-method Kendall's $\tau$}: Rank correlation between
        TF-IDF, keyword-heuristic, and EvidenceRank rankings, measuring
        divergence from lexical baselines.
\end{enumerate}

\subsection{Implementation Details}

EvidenceRank is implemented in Python 3.11 using LangChain (v1.3.13+) for LLM
orchestration, ChromaDB (v0.6+) for vector storage, sentence-transformers (v5.6+)
with \texttt{all-MiniLM-L6-v2} for embeddings, and Pydantic v2 for schema
validation. Small-model tasks (Phases 1 and 2) use \texttt{qwen2.5:7b}; large-model
tasks (Phases 3 and 4) use \texttt{llama3.3:70b} (Q4\_K\_M quantisation).
%% TODO: Report actual hardware specs and per-run wall-clock time after experiments.
```

- [ ] **Step 2: Replace `\section{Results and Discussion}` entirely**

Find the block from `\section{Results and Discussion}` through to the line before `\section{Research Novelties}` and replace with:

```latex
%% ─────────────────────────────────────────────────────────────────────────────
\section{Results and Discussion}
\label{sec:results}

%% TODO: Fill in all tables and discussion after running experiments on
%% the full 20-CV × 2-JD dataset.
%% Run: uv run python main.py --jd jd/... --cv-dir resume/ --baselines --ablation --eval
%% Results land in results/run_<id>/evaluation/baselines.json and ablation.json.

\subsection{Baseline Comparison}

%% TODO: Insert Table comparing TF-IDF (B1), Keyword-Heuristic (B2), and
%% EvidenceRank across: ranking order, score distribution (mean, std),
%% and cross-method Kendall's τ.
%%
%% Table skeleton (fill in values):
%%
%% \begin{table}[!t]
%% \caption{Baseline Ranking Comparison}
%% \label{tab:baselines}
%% \centering
%% \begin{tabular}{lccccc}
%% \toprule
%% \textbf{Method} & \textbf{Score Mean} & \textbf{Score Std}
%%   & \textbf{$\tau$ vs B1} & \textbf{$\tau$ vs B2} & \textbf{$\tau$ vs ER} \\
%% \midrule
%% B1: TF-IDF Cosine   & -- & -- & 1.00 & -- & -- \\
%% B2: Keyword-Heuristic & -- & -- & --  & 1.00 & -- \\
%% EvidenceRank (Ours) & -- & -- & --  & --  & 1.00 \\
%% \bottomrule
%% \end{tabular}
%% \end{table}

\subsection{Ablation Study}

%% TODO: Insert Table reporting per-variant metrics.
%%
%% \begin{table}[!t]
%% \caption{Ablation Study: Contribution of Each Pipeline Component}
%% \label{tab:ablation}
%% \centering
%% \begin{tabular}{lcccc}
%% \toprule
%% \textbf{Configuration} & \textbf{$H$ (\%)} & \textbf{Score Std}
%%   & \textbf{LLM Calls} & \textbf{$\tau$ vs Full} \\
%% \midrule
%% Full EvidenceRank         & -- & -- & -- & 1.00 \\
%% -- No RAG                 & -- & -- & -- & -- \\
%% -- No Evidence Grounding  & -- & -- & -- & -- \\
%% -- No Skill Filter        & -- & -- & -- & -- \\
%% -- No Calibration         & -- & -- & -- & -- \\
%% \bottomrule
%% \end{tabular}
%% \end{table}

\subsection{Ranking Stability}

%% TODO: Report Kendall's τ across 3 repeated runs (--runs 3 flag).
%% Run: uv run python main.py --jd ... --cv-dir ... --runs 3

\subsection{Pool Calibration Impact}

%% TODO: Report score std before (Phase 3 raw) and after (Phase 4 calibrated),
%% mean absolute delta, and rank change count from calibration_report.json.
%% Run with: --eval flag.

\subsection{Limitations}

%% TODO: Update after experiments. Anticipated limitations:
The current evaluation dataset is limited in size, which constrains statistical
power. All experiments use locally-served quantised models, which may perform
below cloud API equivalents in raw accuracy. The hallucination checker relies on
MiniLM cosine similarity, which may classify semantically accurate paraphrases
as fabricated (false positives at the 0.85 threshold).
```

- [ ] **Step 3: Fix fabricated numbers in `\section{Conclusion and Future Work}`**

Find the paragraph starting `We presented EvidenceRank` in the Conclusion and replace it:

```latex
We presented EvidenceRank, a four-phase multi-agent LLM pipeline for IT candidate
ranking that simultaneously addresses hallucination, pool-blind scoring, and
ranking instability—three critical gaps unresolved by existing literature. The
system enforces self-document RAG, verbatim evidence-chain grounding, semantic
skill pre-filtering, and pool-aware calibration as architectural primitives for
reliable LLM-based recruitment.
%% TODO: Add quantitative summary sentence after experiments, e.g.:
%% "Experiments on N CVs × 2 JDs show that EvidenceRank reduces hallucination
%% rate from X\% (no grounding) to Y\%, reduces LLM invocations by Z\% via
%% the skill pre-filter, and achieves near-deterministic ranking stability
%% (Kendall's $\tau = ...$, SD = ...) across repeated runs."
```

- [ ] **Step 4: Fix the Acknowledgement section**

Find `\section*{Acknowledgement}` and replace its body:

```latex
\section*{Acknowledgement}

%% TODO: Add acknowledgements after experiments are conducted with real annotators.
%% Remove this placeholder before submission.
```

- [ ] **Step 5: Verify grep finds all TODO markers**

```bash
grep -n "TODO" docs/paper/evidencerank.tex
```

Expected: at least 8 lines containing `TODO:` across the four replaced blocks.

- [ ] **Step 6: Commit**

```bash
git add docs/paper/evidencerank.tex
git commit -m "docs: replace fabricated experiment/results sections with greppable placeholders"
```

---

## Self-Review Checklist

**Spec coverage:**
- ✅ `--baselines` flag → Task 2 (baselines.py) + Task 6 (main.py)
- ✅ `--ablation` flag → Tasks 3–5 (judge, pipeline, ablation.py) + Task 6 (main.py)
- ✅ TF-IDF ranker → Task 2
- ✅ Keyword-heuristic ranker → Task 2
- ✅ 4 ablation variants (no_rag, no_evidence_grounding, no_skill_filter, no_calibration) → Tasks 3–5
- ✅ JSON output to `run_<id>/evaluation/` → Task 6
- ✅ Rich console tables → Task 6
- ✅ Langfuse score logging → Task 6
- ✅ Paper placeholders (dataset, baselines, ablation, results) → Task 7
- ✅ Cross-method Kendall's τ → Task 2 (`run_baselines`)
- ✅ Per-variant metrics (hallucination rate, score std, llm_calls, τ vs full) → Task 5

**Type consistency:**
- `run_baselines(jd_text, cv_raws, er_ranking)` used in Task 2 and Task 6 ✅
- `run_ablation(jd_raw, cv_raws, full_state, cv_text_map)` used in Task 5 and Task 6 ✅
- `_ranking_from_raw(assessments)` defined in Task 4, used in `pipeline.py` Task 4 ✅
- `CandidateJudgeAgent(use_evidence_grounding=...)` defined in Task 3, used in `nodes.py` Task 4 ✅
- `phase3_candidate_judge(..., use_evidence_grounding)` defined in Task 4 (nodes.py), called from `pipeline.py` Task 4 ✅
