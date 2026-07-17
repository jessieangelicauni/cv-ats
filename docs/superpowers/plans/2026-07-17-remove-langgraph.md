# Remove LangGraph and Signal Enricher Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove LangGraph and the signal enricher phase, convert ATSState to Pydantic, and give each phase function an explicit signature — collapsing the pipeline from 5 phases to 4.

**Architecture:** Each phase function in `nodes.py` gets explicit typed inputs and outputs. `pipeline.py` calls them sequentially via a `_run_phase()` helper, builds trace log entries, and constructs an `ATSState` Pydantic model at the end. LangGraph is removed entirely.

**Tech Stack:** Python 3.11+, Pydantic v2, LangChain Ollama, OpenTelemetry, uv

---

## File Map

| File | Action | Responsibility after change |
|---|---|---|
| `src/agents/candidate_judge.py` | Modify | Accept `CandidateProfile` instead of `EnrichedProfile` |
| `tests/test_candidate_judge.py` | Modify | Fixture uses `CandidateProfile` |
| `src/models/schemas.py` | Modify | Remove `EnrichmentSignals`, `EnrichedProfile`, `CompanyTier`, `CareerTrajectory`, `TenureStability` |
| `tests/test_schemas.py` | Modify | Remove import and test for deleted types |
| `src/agents/signal_enricher.py` | Delete | — |
| `src/prompts/signal_enricher.py` | Delete | — |
| `tests/test_signal_enricher.py` | Delete | — |
| `src/graph/state.py` | Modify | Pydantic `BaseModel`, no `enriched_profiles`, no Annotated reducers |
| `src/graph/nodes.py` | Modify | Explicit signatures, 4 phases, no `_get_cache()`, bare return values |
| `src/graph/pipeline.py` | Modify | Sequential `_run_phase()` runner, no LangGraph |
| `tests/test_pipeline.py` | Modify | 4-phase mocks, attribute access on result |
| `src/output/report_generator.py` | Modify | Remove enriched profile, attribute access on `ATSState` |
| `tests/test_report_generator.py` | Modify | State fixture is `ATSState`, no enriched profile |
| `main.py` | Modify | Attribute access on state, 4-entry `phase_labels` |
| `pyproject.toml` | Modify | Remove `langgraph` dependency |

---

### Task 1: Update `CandidateJudgeAgent` to accept `CandidateProfile`

**Files:**
- Modify: `src/agents/candidate_judge.py`
- Modify: `tests/test_candidate_judge.py`

- [ ] **Step 1: Update the test to use `CandidateProfile`**

Replace the entire contents of `tests/test_candidate_judge.py`:

```python
from unittest.mock import MagicMock, patch
from src.agents.candidate_judge import CandidateJudgeAgent
from src.models.schemas import (
    CandidateAssessment, EvidenceItem, CandidateProfile,
    CandidateBasicInfo, JDRequirements, EducationRequirement,
)


def _make_profile() -> CandidateProfile:
    return CandidateProfile(
        candidate_id="cv_001",
        basic_info=CandidateBasicInfo(
            full_name="Ahmad", email=None, phone=None,
            location=None, linkedin_url=None, current_title="Engineer",
        ),
        skills=[], work_history=[], education=[],
        certifications=[], languages=[], total_experience_months=60,
    )


def _make_jd() -> JDRequirements:
    return JDRequirements(
        role_title="Senior Backend Engineer", seniority_level="senior",
        required_skills=[], preferred_skills=[], min_years_experience=5,
        education=EducationRequirement(degree="BSc", field="CS", is_mandatory=False),
        domain_expertise=[], leadership_expected=True,
        soft_skills=[], industry_context="IT", raw_jd_hash="abc",
    )


def _make_assessment() -> CandidateAssessment:
    return CandidateAssessment(
        candidate_id="cv_001",
        raw_score=87.0,
        confidence="high",
        evidence_chain=[
            EvidenceItem(
                dimension="Technical Skills Fit",
                assessment="Strong Python skills.",
                evidence_quote="5 years Python development",
                dimension_score=9.0,
            )
        ],
        key_strengths=["Tier 1 MNC experience"],
        key_gaps=["Kubernetes not evidenced"],
        seniority_alignment="aligned",
    )


def test_judge_returns_candidate_assessment():
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = _make_assessment()
    with patch("src.agents.candidate_judge.get_llm", return_value=mock_llm):
        agent = CandidateJudgeAgent()
        result = agent.run(_make_profile(), _make_jd())
    assert isinstance(result, CandidateAssessment)
    assert result.candidate_id == "cv_001"
    assert result.raw_score == 87.0


def test_judge_evidence_chain_has_items():
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = _make_assessment()
    with patch("src.agents.candidate_judge.get_llm", return_value=mock_llm):
        agent = CandidateJudgeAgent()
        result = agent.run(_make_profile(), _make_jd())
    assert len(result.evidence_chain) > 0
    assert result.evidence_chain[0].evidence_quote != ""
```

- [ ] **Step 2: Run test to verify it fails**

```
uv run pytest tests/test_candidate_judge.py -v
```

Expected: FAIL — `agent.run()` still expects `EnrichedProfile` but `_make_profile()` passes `CandidateProfile`. Type mismatch at runtime or import error.

- [ ] **Step 3: Update `src/agents/candidate_judge.py`**

Replace the entire file:

```python
from __future__ import annotations
from langchain_core.messages import SystemMessage, HumanMessage
from src.models.schemas import CandidateAssessment, CandidateProfile, JDRequirements
from src.utils.llm import get_llm, invoke_with_telemetry
from src.prompts import judge as prompts
import config


class CandidateJudgeAgent:
    def __init__(self):
        self._llm = get_llm(config.LARGE_MODEL, CandidateAssessment, config.JUDGE_TEMPERATURE)

    def run(self, profile: CandidateProfile, jd: JDRequirements) -> CandidateAssessment:
        return invoke_with_telemetry(
            self._llm,
            [SystemMessage(content=prompts.SYSTEM),
             HumanMessage(content=prompts.human(
                 jd.model_dump_json(indent=2),
                 profile.model_dump_json(indent=2),
             ))],
            run_name="candidate_judge",
        )
```

- [ ] **Step 4: Run test to verify it passes**

```
uv run pytest tests/test_candidate_judge.py -v
```

Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```
git add src/agents/candidate_judge.py tests/test_candidate_judge.py
git commit -m "refactor: candidate judge accepts CandidateProfile instead of EnrichedProfile"
```

---

### Task 2: Remove enricher schema types

**Files:**
- Modify: `src/models/schemas.py`
- Modify: `tests/test_schemas.py`

- [ ] **Step 1: Update `tests/test_schemas.py`**

Remove `EnrichedProfile` and `EnrichmentSignals` from the import line, and delete `test_enrichment_signals_career_trajectory`. The file becomes:

```python
from src.models.schemas import (
    JDRequirements, SkillRequirement, EducationRequirement,
    CandidateProfile, CandidateBasicInfo, SkillEntry, SkillNormalizationMap,
    WorkEntry, EducationEntry, LanguageEntry,
    CandidateAssessment, EvidenceItem, HallucinationFlag,
    FinalRanking, RankedCandidate,
)


def test_jd_requirements_rejects_invalid_seniority():
    import pytest
    with pytest.raises(Exception):
        JDRequirements(
            role_title="Dev",
            seniority_level="wizard",
            required_skills=[], preferred_skills=[],
            min_years_experience=3,
            education=EducationRequirement(degree="BSc", field="CS", is_mandatory=False),
            domain_expertise=[], leadership_expected=False,
            soft_skills=[], industry_context="IT", raw_jd_hash="abc",
        )


def test_skill_entry_has_raw_and_canonical():
    entry = SkillEntry(
        raw_mention="postgres",
        canonical_skill="PostgreSQL",
        proficiency="advanced",
        evidence_quote="managed postgres cluster",
    )
    assert entry.raw_mention == "postgres"
    assert entry.canonical_skill == "PostgreSQL"


def test_skill_normalization_map_structure():
    m = SkillNormalizationMap(mappings={"postgres": "PostgreSQL", "vue js": "Vue.js"})
    assert m.mappings["postgres"] == "PostgreSQL"
    assert m.mappings["vue js"] == "Vue.js"


def test_candidate_assessment_evidence_chain():
    item = EvidenceItem(
        dimension="Technical Skills Fit",
        assessment="Strong Python skills evident.",
        evidence_quote="5 years Python development",
        dimension_score=9.0,
    )
    assert item.dimension_score == 9.0


def test_final_ranking_ranked_candidates():
    rc = RankedCandidate(
        rank=1, candidate_id="cv_001",
        calibrated_score=88.0, delta_from_raw=3.0,
        comparative_notes="Ranks above #2 due to Tier 1 MNC experience.",
    )
    fr = FinalRanking(
        ranked_candidates=[rc],
        pool_summary="Strong pool overall.",
        calibration_rationale="Phase 5 spread scores from 70-88.",
        borderline_pairs=[],
    )
    assert fr.ranked_candidates[0].rank == 1
```

- [ ] **Step 2: Run schemas test to verify it still passes**

```
uv run pytest tests/test_schemas.py -v
```

Expected: PASS (5 tests, `test_enrichment_signals_career_trajectory` is gone)

- [ ] **Step 3: Remove deleted types from `src/models/schemas.py`**

Remove these lines:
- The comment `# Shared type aliases — used in both EnrichmentSignals and EnrichedProfile`
- `CompanyTier: TypeAlias = Literal["tier1_mnc", "tier2_established", "tier3_startup"]`
- `CareerTrajectory: TypeAlias = Literal["ascending", "lateral", "stagnant", "descending"]`
- `TenureStability: TypeAlias = Literal["stable", "moderate", "job_hopper"]`
- The entire `EnrichmentSignals` class (lines 90–98)
- The entire `EnrichedProfile` class (lines 100–116)

The `Proficiency` alias and all other classes stay. The top of the file becomes:

```python
from __future__ import annotations
from typing import Literal, TypeAlias
from pydantic import BaseModel, Field, ConfigDict

Proficiency: TypeAlias = Literal["beginner", "intermediate", "advanced", "expert"]
```

- [ ] **Step 4: Run the full test suite to confirm no remaining imports of deleted types**

```
uv run pytest -v
```

Expected: All previously passing tests still pass. Any test that imports `EnrichedProfile` or `EnrichmentSignals` (there should be none left after Step 1 and Task 1) will error — fix those before continuing.

- [ ] **Step 5: Commit**

```
git add src/models/schemas.py tests/test_schemas.py
git commit -m "refactor: remove EnrichedProfile, EnrichmentSignals, and related type aliases"
```

---

### Task 3: Delete signal enricher files

**Files:**
- Delete: `src/agents/signal_enricher.py`
- Delete: `src/prompts/signal_enricher.py`
- Delete: `tests/test_signal_enricher.py`

- [ ] **Step 1: Delete all three files**

```
git rm src/agents/signal_enricher.py src/prompts/signal_enricher.py tests/test_signal_enricher.py
```

- [ ] **Step 2: Run the test suite**

```
uv run pytest -v
```

Expected: All tests pass. No module imports the deleted files at this point.

- [ ] **Step 3: Commit**

```
git commit -m "refactor: delete signal enricher agent, prompt, and tests"
```

---

### Task 4: Convert `ATSState` from TypedDict to Pydantic `BaseModel`

**Files:**
- Modify: `src/graph/state.py`

- [ ] **Step 1: Replace the entire contents of `src/graph/state.py`**

```python
from __future__ import annotations
from pydantic import BaseModel, Field
from src.models.schemas import (
    JDRequirements, CandidateProfile,
    CandidateAssessment, FinalRanking, HallucinationFlag,
)


class ATSState(BaseModel):
    jd_raw: str
    cv_raws: list[dict]
    jd_structured: JDRequirements | None = None
    cv_profiles: list[CandidateProfile] = Field(default_factory=list)
    candidate_assessments: list[CandidateAssessment] = Field(default_factory=list)
    final_ranking: FinalRanking | None = None
    run_id: str
    otel_trace_id: str = ""
    trace_log: list[dict] = Field(default_factory=list)
    hallucination_flags: list[HallucinationFlag] = Field(default_factory=list)
    use_cache: bool = True
```

- [ ] **Step 2: Run the test suite**

```
uv run pytest -v
```

Expected: Tests that directly import `ATSState` may fail if they construct it as a TypedDict. Most tests don't construct `ATSState` directly — the pipeline tests will fail in Task 7 where we address them. Any import-level errors will surface here — fix them before proceeding.

- [ ] **Step 3: Commit**

```
git add src/graph/state.py
git commit -m "refactor: convert ATSState from TypedDict to Pydantic BaseModel"
```

---

### Task 5: Refactor `nodes.py` — explicit signatures, 4 phases

**Files:**
- Modify: `src/graph/nodes.py`

- [ ] **Step 1: Replace the entire contents of `src/graph/nodes.py`**

```python
from __future__ import annotations
import time
from opentelemetry.trace import StatusCode
from src.models.schemas import (
    JDRequirements, CandidateProfile, CandidateAssessment, FinalRanking,
)
from src.agents.jd_parser import JDParserAgent
from src.agents.cv_extractor import CVExtractorAgent
from src.agents.candidate_judge import CandidateJudgeAgent
from src.agents.pool_calibrator import PoolCalibratorAgent
from src.utils.cache import ExtractionCache
from src.utils.telemetry import get_tracer

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
    cv_raws: list[dict], cache: ExtractionCache | None
) -> list[CandidateProfile]:
    with _tracer.start_as_current_span("phase2/cv_extractor") as span:
        span.set_attribute("phase", 2)
        span.set_attribute("n_candidates", len(cv_raws))

        def process(cv_raw: dict) -> CandidateProfile:
            with _tracer.start_as_current_span(
                f"phase2/cv/{cv_raw['candidate_id']}"
            ) as cspan:
                try:
                    return CVExtractorAgent(cache=cache).run(cv_raw)
                except Exception as exc:
                    cspan.record_exception(exc)
                    cspan.set_status(StatusCode.ERROR, str(exc))
                    raise

        return [process(cv_raw) for cv_raw in cv_raws]


def phase3_candidate_judge(
    profiles: list[CandidateProfile], jd: JDRequirements
) -> list[CandidateAssessment]:
    with _tracer.start_as_current_span("phase3/candidate_judge") as span:
        span.set_attribute("phase", 3)
        span.set_attribute("n_candidates", len(profiles))

        def process(profile: CandidateProfile) -> CandidateAssessment:
            with _tracer.start_as_current_span(
                f"phase3/judge/{profile.candidate_id}"
            ) as cspan:
                try:
                    return CandidateJudgeAgent().run(profile, jd)
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
```

- [ ] **Step 2: Run the test suite**

```
uv run pytest -v
```

Expected: Pipeline tests will fail (they still call `build_pipeline` / use the old graph). All other tests should pass.

- [ ] **Step 3: Commit**

```
git add src/graph/nodes.py
git commit -m "refactor: explicit phase signatures in nodes.py, remove phase3 signal enricher"
```

---

### Task 6: Replace LangGraph pipeline with sequential runner

**Files:**
- Modify: `src/graph/pipeline.py`
- Modify: `pyproject.toml`

- [ ] **Step 1: Replace the entire contents of `src/graph/pipeline.py`**

```python
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
from src.utils.telemetry import setup_telemetry, get_tracer, current_otel_trace_id
import config


def run_pipeline(
    jd_raw: str,
    cv_raws: list[dict],
    run_id: str | None = None,
    use_cache: bool = True,
    session_id: str | None = None,
    on_phase_complete: Callable[[dict], None] | None = None,
) -> ATSState:
    setup_telemetry()
    run_id = run_id or str(uuid.uuid4())[:8]
    jd_hash = hashlib.sha256(jd_raw.encode()).hexdigest()[:12]
    cache = ExtractionCache(config.CACHE_DB_PATH) if use_cache else None

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
                                   candidates=len(cv_raws))
        assessments   = _run_phase(3, phase3_candidate_judge, cv_profiles, jd_structured)
        final_ranking = _run_phase(4, phase4_pool_calibrator, assessments, jd_structured)

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
        )
```

- [ ] **Step 2: Remove `langgraph` from `pyproject.toml`**

Remove this line from the `dependencies` list:
```
"langgraph>=1.2.9",
```

- [ ] **Step 3: Sync dependencies**

```
uv sync
```

Expected: uv removes langgraph from the environment.

- [ ] **Step 4: Run the test suite**

```
uv run pytest -v
```

Expected: Pipeline tests fail (they still import `build_pipeline` and use dict access). Other tests pass.

- [ ] **Step 5: Commit**

```
git add src/graph/pipeline.py pyproject.toml uv.lock
git commit -m "refactor: replace LangGraph with sequential _run_phase() runner"
```

---

### Task 7: Update `tests/test_pipeline.py`

**Files:**
- Modify: `tests/test_pipeline.py`

- [ ] **Step 1: Replace the entire contents of `tests/test_pipeline.py`**

```python
from unittest.mock import MagicMock, patch
from src.graph.pipeline import run_pipeline
from src.models.schemas import (
    JDRequirements, EducationRequirement,
    CandidateProfile, CandidateBasicInfo,
    CandidateAssessment, EvidenceItem, FinalRanking, RankedCandidate,
)


def _mock_jd() -> JDRequirements:
    return JDRequirements(
        role_title="Engineer", seniority_level="mid",
        required_skills=[], preferred_skills=[], min_years_experience=3,
        education=EducationRequirement(degree="BSc", field="CS", is_mandatory=False),
        domain_expertise=[], leadership_expected=False,
        soft_skills=[], industry_context="IT", raw_jd_hash="abc",
    )


def _mock_profile(cid: str) -> CandidateProfile:
    return CandidateProfile(
        candidate_id=cid,
        basic_info=CandidateBasicInfo(full_name=None, email=None, phone=None,
                                       location=None, linkedin_url=None, current_title=None),
        skills=[], work_history=[], education=[],
        certifications=[], languages=[], total_experience_months=36,
    )


def _mock_assessment(cid: str) -> CandidateAssessment:
    return CandidateAssessment(
        candidate_id=cid, raw_score=75.0, confidence="medium",
        evidence_chain=[EvidenceItem(dimension="Technical Skills Fit",
                                      assessment="OK", evidence_quote="Python",
                                      dimension_score=7.5)],
        key_strengths=["Python"], key_gaps=[], seniority_alignment="aligned",
    )


def _mock_ranking() -> FinalRanking:
    return FinalRanking(
        ranked_candidates=[
            RankedCandidate(rank=1, candidate_id="cv_001", calibrated_score=77.0,
                            delta_from_raw=2.0, comparative_notes="Best fit."),
            RankedCandidate(rank=2, candidate_id="cv_002", calibrated_score=73.0,
                            delta_from_raw=-2.0, comparative_notes="Weaker skills."),
        ],
        pool_summary="Two candidates evaluated.",
        calibration_rationale="Scores spread from 73 to 77.",
        borderline_pairs=[],
    )


def test_pipeline_produces_final_ranking():
    with (
        patch("src.graph.nodes.JDParserAgent") as MockJD,
        patch("src.graph.nodes.CVExtractorAgent") as MockCV,
        patch("src.graph.nodes.CandidateJudgeAgent") as MockJ,
        patch("src.graph.nodes.PoolCalibratorAgent") as MockPC,
    ):
        MockJD.return_value.run.return_value = _mock_jd()
        MockCV.return_value.run.side_effect = [
            _mock_profile("cv_001"), _mock_profile("cv_002"),
        ]
        MockJ.return_value.run.side_effect = [
            _mock_assessment("cv_001"), _mock_assessment("cv_002"),
        ]
        MockPC.return_value.run.return_value = _mock_ranking()

        result = run_pipeline(
            jd_raw="Software Engineer role",
            cv_raws=[
                {"raw_text": "CV1 text", "candidate_id": "cv_001", "source_file": "cv_001.pdf"},
                {"raw_text": "CV2 text", "candidate_id": "cv_002", "source_file": "cv_002.pdf"},
            ],
            use_cache=False,
        )

    assert result.final_ranking is not None
    assert len(result.final_ranking.ranked_candidates) == 2
    assert result.final_ranking.ranked_candidates[0].candidate_id == "cv_001"


def test_pipeline_trace_log_has_four_entries():
    with (
        patch("src.graph.nodes.JDParserAgent") as MockJD,
        patch("src.graph.nodes.CVExtractorAgent") as MockCV,
        patch("src.graph.nodes.CandidateJudgeAgent") as MockJ,
        patch("src.graph.nodes.PoolCalibratorAgent") as MockPC,
    ):
        MockJD.return_value.run.return_value = _mock_jd()
        MockCV.return_value.run.side_effect = [_mock_profile("cv_001")]
        MockJ.return_value.run.side_effect = [_mock_assessment("cv_001")]
        MockPC.return_value.run.return_value = FinalRanking(
            ranked_candidates=[RankedCandidate(rank=1, candidate_id="cv_001",
                               calibrated_score=75.0, delta_from_raw=0.0,
                               comparative_notes="Only candidate.")],
            pool_summary="One candidate.", calibration_rationale="N/A", borderline_pairs=[],
        )

        result = run_pipeline(
            jd_raw="Engineer role",
            cv_raws=[{"raw_text": "CV text", "candidate_id": "cv_001", "source_file": "cv.pdf"}],
            use_cache=False,
        )

    assert len(result.trace_log) == 4
```

- [ ] **Step 2: Run the pipeline tests**

```
uv run pytest tests/test_pipeline.py -v
```

Expected: PASS (2 tests)

- [ ] **Step 3: Commit**

```
git add tests/test_pipeline.py
git commit -m "test: update pipeline tests for 4-phase sequential runner"
```

---

### Task 8: Update `report_generator.py` and its test

**Files:**
- Modify: `src/output/report_generator.py`
- Modify: `tests/test_report_generator.py`

- [ ] **Step 1: Update the test first — replace `tests/test_report_generator.py`**

```python
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
                                               claim="Strong Python.", status="supported",
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
```

- [ ] **Step 2: Run the report generator tests to verify they fail**

```
uv run pytest tests/test_report_generator.py -v
```

Expected: FAIL — `generate_report()` still uses dict access and references `enriched_profiles`.

- [ ] **Step 3: Replace the entire contents of `src/output/report_generator.py`**

```python
from __future__ import annotations
import json
from pathlib import Path
from datetime import date
from src.models.schemas import (
    CandidateAssessment, CandidateProfile, HallucinationFlag,
)
from src.graph.state import ATSState
from src.evaluation.hallucination_checker import hallucination_rate


def _profile_for(cid: str, profiles: list[CandidateProfile]) -> CandidateProfile | None:
    return next((p for p in profiles if p.candidate_id == cid), None)


def _assessment_for(cid: str, assessments: list[CandidateAssessment]) -> CandidateAssessment | None:
    return next((a for a in assessments if a.candidate_id == cid), None)


def _render_candidate_block(
    rank: int,
    rc,
    profile: CandidateProfile,
    assessment: CandidateAssessment,
    flags: list[HallucinationFlag],
) -> str:
    bi = profile.basic_info
    lines = [
        f"---\n",
        f"### Rank {rank} — {rc.candidate_id} | {bi.full_name or '(name not found)'}",
        f"**Score:** {rc.calibrated_score} (raw: {assessment.raw_score}, Δ {rc.delta_from_raw:+.1f})"
        f" | **Confidence:** {assessment.confidence.capitalize()}"
        f" | **Seniority:** {assessment.seniority_alignment.capitalize()}",
        "",
        "#### Contact",
        f"- Email: {bi.email or 'N/A'}",
        f"- Phone: {bi.phone or 'N/A'}",
        f"- Location: {bi.location or 'N/A'}",
        f"- LinkedIn: {bi.linkedin_url or 'N/A'}",
        f"- Current Title: {bi.current_title or 'N/A'}",
        "",
        "#### Skills",
        "| Canonical Skill | Raw Mention | Proficiency | Evidence Quote |",
        "|---|---|---|---|",
    ]
    for s in profile.skills:
        lines.append(
            f"| {s.canonical_skill} | {s.raw_mention} | {s.proficiency} | {s.evidence_quote} |"
        )

    lines += ["", "#### Education",
              "| Degree | Field | Institution | Year |",
              "|---|---|---|---|"]
    for e in profile.education:
        lines.append(f"| {e.degree} | {e.field} | {e.institution} | {e.year or 'N/A'} |")

    lines += ["", "#### Work Experience"]
    for w in profile.work_history:
        lines += [
            f"**{w.role} — {w.company}**",
            f"Duration: {w.tenure_months or 'N/A'} months"
            f" | Leadership: {'Yes' if w.has_leadership_indicators else 'No'}",
            f"Technologies: {', '.join(w.technologies)}",
            "Achievements:",
        ]
        for ach in w.achievements:
            lines.append(f'- "{ach}"')
        lines.append("")

    lines += ["#### LLM Judgment"]
    candidate_flags = {f.claim: f.status for f in flags if f.candidate_id == rc.candidate_id}
    for item in assessment.evidence_chain:
        h_status = candidate_flags.get(item.assessment, "unknown")
        lines += [
            f"**{item.dimension} — {item.dimension_score}/10**",
            f"Assessment: {item.assessment}",
            f"Evidence: \"{item.evidence_quote}\"",
            f"Hallucination status: {h_status}",
            "",
        ]

    lines += [
        f"**Holistic Score: {assessment.raw_score} (raw) → {rc.calibrated_score} (calibrated)**",
        f"Key Strengths: {', '.join(assessment.key_strengths)}",
        f"Key Gaps: {', '.join(assessment.key_gaps)}",
        f"Comparative note: {rc.comparative_notes}",
        "",
    ]
    return "\n".join(lines)


def generate_report(state: ATSState, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "candidates").mkdir(exist_ok=True)
    (output_dir / "evaluation").mkdir(exist_ok=True)

    ranking = state.final_ranking
    jd = state.jd_structured
    profiles = state.cv_profiles
    assessments = state.candidate_assessments
    flags = state.hallucination_flags
    run_id = state.run_id

    (output_dir / "ranking.json").write_text(
        ranking.model_dump_json(indent=2), encoding="utf-8"
    )

    for a in assessments:
        (output_dir / "candidates" / f"{a.candidate_id}.json").write_text(
            a.model_dump_json(indent=2), encoding="utf-8"
        )

    h_rate = hallucination_rate(flags)
    fabricated = [f for f in flags if f.status == "fabricated"]
    h_report = {
        "overall_fabrication_rate": round(h_rate, 4),
        "total_claims": len([f for f in flags if f.status != "acknowledged_gap"]),
        "fabricated_count": len(fabricated),
        "fabricated_claims": [f.model_dump() for f in fabricated],
    }
    (output_dir / "evaluation" / "hallucination_report.json").write_text(
        json.dumps(h_report, indent=2), encoding="utf-8"
    )

    md_lines = [
        "# ATS Evaluation Report",
        f"Run: {run_id} | JD: {jd.role_title} ({jd.seniority_level})"
        f" | Candidates: {len(assessments)} | Date: {date.today()}",
        "",
        "---",
        "",
        "## Ranking Summary",
        "",
        "| Rank | File | Name | Calibrated Score | Raw Score | Δ | Confidence | Seniority |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for rc in ranking.ranked_candidates:
        a = _assessment_for(rc.candidate_id, assessments)
        p = _profile_for(rc.candidate_id, profiles)
        name = p.basic_info.full_name if p else "N/A"
        md_lines.append(
            f"| {rc.rank} | {rc.candidate_id} | {name} | {rc.calibrated_score}"
            f" | {a.raw_score if a else 'N/A'} | {rc.delta_from_raw:+.1f}"
            f" | {a.confidence.capitalize() if a else 'N/A'}"
            f" | {a.seniority_alignment.capitalize() if a else 'N/A'} |"
        )

    md_lines += ["", f"Pool calibration rationale: {ranking.calibration_rationale}",
                 "", "---", "", "## Candidate Dossiers", ""]

    for rc in ranking.ranked_candidates:
        profile = _profile_for(rc.candidate_id, profiles)
        assessment = _assessment_for(rc.candidate_id, assessments)
        if profile and assessment:
            md_lines.append(
                _render_candidate_block(rc.rank, rc, profile, assessment, flags)
            )

    if ranking.borderline_pairs:
        md_lines += ["## Pool Calibration Notes", "", "### Borderline Pairs"]
        for bp in ranking.borderline_pairs:
            md_lines.append(str(bp))

    md_lines += [
        "", "---", "", "## Hallucination Summary",
        f"Overall fabrication rate: {h_rate:.1%} ({len(fabricated)} / "
        f"{len([f for f in flags if f.status != 'acknowledged_gap'])} claims)",
    ]
    if fabricated:
        md_lines += ["", "### Flagged Claims"]
        for f in fabricated:
            md_lines += [
                f"**{f.candidate_id} — FABRICATED**",
                f"Claim: {f.claim}",
                f"Quote used: {f.source_quote}",
                "",
            ]

    (output_dir / "report.md").write_text("\n".join(md_lines), encoding="utf-8")
```

- [ ] **Step 4: Run the report generator tests**

```
uv run pytest tests/test_report_generator.py -v
```

Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```
git add src/output/report_generator.py tests/test_report_generator.py
git commit -m "refactor: remove enriched profile from report generator, use ATSState attribute access"
```

---

### Task 9: Update `main.py`

**Files:**
- Modify: `main.py`

- [ ] **Step 1: Replace the `main()` function body in `main.py`**

Update the `phase_labels` dict and all state access. The full updated `main()` function:

```python
@app.command()
def main(
    jd: Path = typer.Option(..., help="Path to job description text file"),
    cv_dir: Path = typer.Option(..., help="Directory of candidate CV PDFs"),
    output: Path = typer.Option(Path("results"), help="Output directory"),
    runs: int = typer.Option(1, help="Number of pipeline runs (use 3 for consistency test)"),
    eval: bool = typer.Option(False, help="Run full evaluation suite"),
    no_cache: bool = typer.Option(False, help="Disable extraction cache"),
):
    """LangGraph ATS — rank IT candidates against a job description."""
    setup_telemetry()

    if not jd.exists():
        console.print(f"[red]JD file not found: {jd}[/red]")
        raise typer.Exit(1)
    if not cv_dir.is_dir():
        console.print(f"[red]CV directory not found: {cv_dir}[/red]")
        raise typer.Exit(1)

    jd_text = jd.read_text(encoding="utf-8")
    cv_raws = _load_cvs(cv_dir)
    console.print(f"[green]Loaded {len(cv_raws)} CVs from {cv_dir}[/green]")

    run_id = str(uuid.uuid4())[:8]
    out_dir = output / f"run_{run_id}"
    lf = get_langfuse()

    try:
        if runs > 1:
            console.print(f"[blue]Running {runs}× consistency experiment...[/blue]")
            consistency = run_consistency_experiment(jd_text, cv_raws, n_runs=runs)
            console.print(f"Consistency mean τ: {consistency['mean_tau']:.3f}")

            from itertools import combinations
            for i, (a, b) in enumerate(combinations(range(runs), 2)):
                lf.create_score(
                    trace_id=consistency["otel_trace_ids"][a],
                    name=f"kendall_tau_vs_run{b}",
                    value=consistency["pairwise_taus"][i],
                )
            lf.create_score(
                trace_id=consistency["otel_trace_ids"][0],
                name="mean_tau",
                value=consistency["mean_tau"],
                comment=f"session_id={consistency['session_id']}",
            )

            import json
            (out_dir / "evaluation").mkdir(parents=True, exist_ok=True)
            (out_dir / "evaluation" / "consistency_metrics.json").write_text(
                json.dumps(consistency, indent=2), encoding="utf-8"
            )
            return

        phase_labels = {
            1: "[Phase 1] Parsing job description...",
            2: "[Phase 2] Extracting CV profiles...",
            3: "[Phase 3] Judging candidates...",
            4: "[Phase 4] Calibrating final ranking...",
        }

        with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console) as progress:
            task = progress.add_task("Running pipeline...", total=None)
            progress.update(task, description=phase_labels[1])

            def on_phase_complete(entry: dict) -> None:
                next_label = phase_labels.get(entry["phase"] + 1)
                if next_label:
                    progress.update(task, description=next_label)

            state = run_pipeline(
                jd_text, cv_raws, run_id=run_id, use_cache=not no_cache,
                on_phase_complete=on_phase_complete,
            )

        otel_trace_id = state.otel_trace_id

        ranking = state.final_ranking
        table = Table(title=f"Ranking — {state.jd_structured.role_title}")
        table.add_column("Rank", style="bold")
        table.add_column("Candidate")
        table.add_column("Name")
        table.add_column("Score")
        table.add_column("Δ Phase4")
        table.add_column("Confidence")
        table.add_column("Seniority")

        profile_map = {p.candidate_id: p for p in state.cv_profiles}
        assessment_map = {a.candidate_id: a for a in state.candidate_assessments}

        for rc in ranking.ranked_candidates:
            p = profile_map.get(rc.candidate_id)
            a = assessment_map.get(rc.candidate_id)
            table.add_row(
                str(rc.rank),
                rc.candidate_id,
                p.basic_info.full_name or "N/A" if p else "N/A",
                str(rc.calibrated_score),
                f"{rc.delta_from_raw:+.1f}",
                a.confidence if a else "N/A",
                a.seniority_alignment if a else "N/A",
            )
        console.print(table)

        if eval:
            console.print("[blue]Running evaluation suite...[/blue]")
            all_flags = []
            cv_text_map = {c["candidate_id"]: c["raw_text"] for c in cv_raws}
            for a in state.candidate_assessments:
                flags = verify_evidence_chain(a, cv_text_map.get(a.candidate_id, ""))
                all_flags.extend(flags)
            state.hallucination_flags = all_flags
            h_rate = hallucination_rate(all_flags)
            console.print(f"Hallucination rate: {h_rate:.1%}")

            cal = calibration_report(state.candidate_assessments, ranking)
            console.print(f"Calibration — raw std: {cal['raw_std']:.1f}, calibrated std: {cal['calibrated_std']:.1f}")
            console.print(f"Mean Phase 4 delta: {cal['mean_abs_delta']:.1f}, rank changes: {cal['rank_changes']}")

            fabricated = sum(1 for f in all_flags if f.status == "fabricated")
            lf.create_score(
                trace_id=otel_trace_id,
                name="hallucination_rate",
                value=h_rate,
                comment=f"{fabricated} fabricated / {len(all_flags)} total",
            )
            lf.create_score(trace_id=otel_trace_id, name="calibration_raw_std",       value=cal["raw_std"])
            lf.create_score(trace_id=otel_trace_id, name="calibration_calibrated_std", value=cal["calibrated_std"])
            lf.create_score(trace_id=otel_trace_id, name="calibration_mean_abs_delta", value=cal["mean_abs_delta"])
            lf.create_score(trace_id=otel_trace_id, name="rank_changes",               value=float(cal["rank_changes"]))

            import json
            (out_dir / "evaluation").mkdir(parents=True, exist_ok=True)
            (out_dir / "evaluation" / "calibration_report.json").write_text(
                json.dumps(cal, indent=2), encoding="utf-8"
            )

        generate_report(state, out_dir)
        console.print(f"\n[green]Report written to {out_dir}/report.md[/green]")

    finally:
        shutdown()
```

- [ ] **Step 2: Run the full test suite**

```
uv run pytest -v
```

Expected: All tests pass.

- [ ] **Step 3: Commit**

```
git add main.py
git commit -m "refactor: update main.py — attribute access on ATSState, 4-phase labels"
```

---

### Task 10: Final verification

- [ ] **Step 1: Run the complete test suite with coverage**

```
uv run pytest --cov=src -v
```

Expected: All tests pass. No imports of `langgraph`, `EnrichedProfile`, `EnrichmentSignals`, `SignalEnricher`, or `build_pipeline` anywhere in the codebase.

- [ ] **Step 2: Confirm langgraph is gone**

```
uv run python -c "import langgraph" 2>&1
```

Expected: `ModuleNotFoundError: No module named 'langgraph'`

- [ ] **Step 3: Confirm pipeline runs end-to-end (smoke test with mocks)**

```
uv run pytest tests/test_pipeline.py -v
```

Expected: PASS (2 tests)

- [ ] **Step 4: Commit final state**

```
git add -A
git commit -m "chore: verify all tests pass after LangGraph and signal enricher removal"
```
