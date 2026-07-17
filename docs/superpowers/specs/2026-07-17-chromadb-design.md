# Design: ChromaDB Semantic Retrieval + Skill Matching

**Date:** 2026-07-17
**Status:** Approved
**Depends on:** `2026-07-17-remove-langgraph-design.md` (must be implemented first)

## Problem

The current pipeline has three inefficiencies:

1. **CV chunk embeddings are recomputed every run.** The hallucination checker already uses `sentence-transformers` for semantic similarity, but embeddings are thrown away after each run.
2. **The CV extractor and judge receive the full raw CV text.** For verbose CVs, this fills the LLM context with irrelevant content (cover letters, personal statements, repeated formatting).
3. **Skill matching between JD requirements and candidate profiles is implicit.** The judge has to infer whether "Postgres" satisfies "PostgreSQL" from unstructured text. Candidates with zero relevant skills still go through the expensive judge phase.

## Goal

Add ChromaDB as a persistent chunk embedding store. Use it to (a) retrieve focused CV context for the extractor and judge, and (b) pre-compute a skill match table passed to the judge as structured evidence. Add a pre-judge filter that eliminates candidates with no required-skill matches, reducing latency for clearly irrelevant CVs.

## Architecture Overview

Three components share a single `SentenceTransformer` instance (MiniLM, `config.EMBEDDING_MODEL`):

| Component | Storage | Purpose |
|---|---|---|
| `CVVectorStore` | ChromaDB on disk | Chunk CV text, persist embeddings, retrieve JD-relevant chunks |
| `SkillMatcher` | In-memory only | Cosine similarity between JD skill names and candidate canonical skills |
| `get_embedder()` | Module singleton | One model instance shared by all three consumers |

`--no-cache` sets all three to `None`, restoring full original pipeline behaviour.

## Design

### 1. `src/utils/embedder.py` — shared embedder singleton

Moves the `_get_embedder()` singleton out of `hallucination_checker.py` into a shared module. All embedding consumers import from here.

```python
_model: SentenceTransformer | None = None

def get_embedder() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(config.EMBEDDING_MODEL)
    return _model
```

`hallucination_checker.py` drops its own `_get_embedder()` and imports `get_embedder` from here instead.

### 2. `src/utils/vector_store.py` — `CVVectorStore`

Wraps a ChromaDB `PersistentClient` with a single `"cv_chunks"` collection. Uses pre-computed embeddings via the shared embedder (not ChromaDB's built-in embedding function) — avoids loading a second model instance.

**Chunker** (private):
```python
def _chunk_text(text: str, min_len: int = 20) -> list[str]:
    chunks = re.split(r'\n\s*\n', text)
    return [c.strip() for c in chunks if len(c.strip()) >= min_len]
```

Splits on blank lines (paragraph/section breaks). Filters chunks shorter than 20 characters.

**Public interface:**
```python
class CVVectorStore:
    def __init__(self, path: Path): ...

    def index_cv(self, candidate_id: str, raw_text: str) -> None:
        # Compute cv_hash = sha256(raw_text)[:12]
        # If collection already has chunks for candidate_id with same cv_hash: skip
        # If cv_hash changed: delete old chunks, re-index
        # Chunk raw_text, embed all chunks, add to collection with metadata:
        #   {candidate_id, chunk_index, cv_hash}

    def retrieve(self, candidate_id: str, query: str, top_k: int = 5) -> list[str]:
        # Embed query, query collection filtered by candidate_id, return top-k chunk strings
```

Storage path: `config.CHROMA_DB_PATH` (`.cache/chroma/`).

### 3. `src/utils/skill_matcher.py` — `SkillMatcher`

Pure in-memory skill matching. ChromaDB is not used here — the skill set is tiny (10–30 JD skills, 5–20 candidate skills) and nothing needs to be persisted.

```python
@dataclass
class SkillMatchResult:
    jd_skill: str
    best_match: str | None   # candidate canonical_skill with highest similarity
    score: float             # cosine similarity 0.0–1.0
    is_required: bool

class SkillMatcher:
    def __init__(self, embedder: SentenceTransformer): ...

    def match(
        self,
        jd_skills: list[SkillRequirement],
        candidate_skills: list[SkillEntry],
    ) -> list[SkillMatchResult]:
        # Encode jd_skills[].skill and candidate_skills[].canonical_skill
        # Compute cosine similarity matrix
        # For each JD skill: find best-matching candidate skill
        # Return SkillMatchResult for each JD skill
```

### 4. Phase 2 — CV extractor integration (`src/graph/nodes.py`)

Signature:
```python
def phase2_cv_extractor(
    cv_raws: list[dict],
    cache: ExtractionCache | None,
    vector_store: CVVectorStore | None,
) -> list[CandidateProfile]
```

For each CV (when `vector_store` is not `None`):
1. `vector_store.index_cv(candidate_id, raw_text)` — index chunks (skips if cached).
2. Retrieve focused chunks using three section-focused queries:
   - `"work experience roles achievements"`
   - `"technical skills programming languages frameworks"`
   - `"education degree certifications"`
3. Deduplicate and join retrieved chunks; replace `cv_raw["raw_text"]` with the focused text before passing to `CVExtractorAgent`.

Falls back to full raw text when `vector_store` is `None`.

### 5. Candidate filter — `_filter_by_required_skills()` (`src/graph/pipeline.py`)

Called between Phase 2 and Phase 3. Uses `SkillMatcher` only (no ChromaDB).

```python
def _filter_by_required_skills(
    profiles: list[CandidateProfile],
    jd: JDRequirements,
    skill_matcher: SkillMatcher | None,
) -> tuple[list[CandidateProfile], list[str]]:
    # Returns (passing_profiles, eliminated_candidate_ids)
    # Uses jd.required_skills only (not preferred_skills) — filter is strict-required only
    # A candidate passes if any required JD skill has score >= config.SKILL_MATCH_THRESHOLD
    # If skill_matcher is None: all candidates pass (no filtering)
```

Eliminated candidate IDs are stored in `ATSState.eliminated_candidates` and rendered as a short section in the report.

### 6. Phase 3 — judge integration (`src/graph/nodes.py`)

Signature:
```python
def phase3_candidate_judge(
    profiles: list[CandidateProfile],
    jd: JDRequirements,
    vector_store: CVVectorStore | None,
    skill_matcher: SkillMatcher | None,
) -> list[CandidateAssessment]
```

For each candidate (when not `None`):
1. `context_chunks = vector_store.retrieve(candidate_id, jd_text, top_k=config.RETRIEVAL_TOP_K)`
2. `skill_matches = skill_matcher.match(jd.required_skills + jd.preferred_skills, profile.skills)`
3. Pass both to `CandidateJudgeAgent.run(profile, jd, context_chunks, skill_matches)`.

### 7. `src/prompts/judge.py` — updated human prompt

```python
def human(
    jd_json: str,
    profile_json: str,
    context_chunks: list[str] = [],
    skill_matches: list[SkillMatchResult] = [],
) -> str:
```

Two optional sections appended before the scoring instructions:

**CV excerpts section** (when `context_chunks` non-empty):
```
RELEVANT CV EXCERPTS (raw text — use for evidence_quote verification):
--- chunk ---
<chunk>
```

**Skill coverage table** (when `skill_matches` non-empty):
```
SKILL COVERAGE (pre-computed — use as grounding for Technical Skills Fit):
| JD Skill    | Candidate Match | Score | Required |
|-------------|-----------------|-------|----------|
| PostgreSQL  | Postgres        | 0.97  | Yes      |
| Kubernetes  | K8s             | 0.94  | Yes      |
| React       | (no match)      | 0.00  | No       |
```

Existing scoring instructions are unchanged.

`CandidateJudgeAgent.run()` gains matching optional parameters and passes them through.

### 8. `src/graph/pipeline.py` — full updated flow

```python
cache         = ExtractionCache(config.CACHE_DB_PATH) if use_cache else None
vector_store  = CVVectorStore(config.CHROMA_DB_PATH)  if use_cache else None
skill_matcher = SkillMatcher(get_embedder())           if use_cache else None

jd_structured            = _run_phase(1, phase1_jd_parser, jd_raw, cache)
cv_profiles              = _run_phase(2, phase2_cv_extractor, cv_raws, cache, vector_store)
cv_profiles, eliminated  = _filter_by_required_skills(cv_profiles, jd_structured, skill_matcher)
assessments              = _run_phase(3, phase3_candidate_judge,
                                         cv_profiles, jd_structured, vector_store, skill_matcher)
final_ranking            = _run_phase(4, phase4_pool_calibrator, assessments, jd_structured)

return ATSState(..., eliminated_candidates=eliminated)
```

### 9. `src/graph/state.py` — new field

`ATSState` gains one field:
```python
eliminated_candidates: list[str] = Field(default_factory=list)
```

### 10. `config.py` — new settings

```python
CHROMA_DB_PATH         = BASE_DIR / ".cache" / "chroma"
SKILL_MATCH_THRESHOLD  = float(os.getenv("SKILL_MATCH_THRESHOLD", "0.75"))
RETRIEVAL_TOP_K        = int(os.getenv("RETRIEVAL_TOP_K", "5"))
```

`ensure_dirs()` gains `CHROMA_DB_PATH`.

### 11. `pyproject.toml` — new dependency

```toml
"chromadb>=0.6.0",
```

### 12. `src/output/report_generator.py` — eliminated candidates section

`generate_report()` renders a short section at the bottom of `report.md` listing eliminated candidates and the reason (`"No required skill match found"`).

### 13. Tests

**New files:**

| File | Coverage |
|---|---|
| `tests/test_vector_store.py` | `index_cv()` skips re-index when hash unchanged; re-indexes on change; `retrieve()` returns non-empty list for matching query |
| `tests/test_skill_matcher.py` | High score for "PostgreSQL"/"Postgres"; low score for unrelated pair; `is_required` flag correct |

**Updated files:**

| File | Change |
|---|---|
| `tests/test_pipeline.py` | Pass `use_cache=False` (already done); assert `eliminated_candidates` is a list on result |
| `tests/test_candidate_judge.py` | Add test variant with `context_chunks` and `skill_matches` populated; assert judge prompt contains skill table |
| `tests/test_report_generator.py` | Add `eliminated_candidates` to state fixture; assert eliminated section appears in report |

## Files Added

| File | Purpose |
|---|---|
| `src/utils/embedder.py` | Shared MiniLM singleton |
| `src/utils/vector_store.py` | `CVVectorStore` — ChromaDB chunk store |
| `src/utils/skill_matcher.py` | `SkillMatcher` — in-memory JD/candidate skill matching |
| `tests/test_vector_store.py` | Vector store tests |
| `tests/test_skill_matcher.py` | Skill matcher tests |

## Files Changed

| File | Change |
|---|---|
| `src/utils/embedder.py` | New (moved from `hallucination_checker.py`) |
| `src/evaluation/hallucination_checker.py` | Replace `_get_embedder()` with import from `embedder.py` |
| `src/graph/nodes.py` | Add `vector_store` to Phase 2; add `vector_store` + `skill_matcher` to Phase 3 |
| `src/agents/candidate_judge.py` | Add `context_chunks` and `skill_matches` optional params |
| `src/prompts/judge.py` | Add CV excerpts and skill coverage sections to `human()` |
| `src/graph/pipeline.py` | Instantiate `CVVectorStore` and `SkillMatcher`; add filter step |
| `src/graph/state.py` | Add `eliminated_candidates` field |
| `src/output/report_generator.py` | Render eliminated candidates section |
| `config.py` | Add `CHROMA_DB_PATH`, `SKILL_MATCH_THRESHOLD`, `RETRIEVAL_TOP_K` |
| `pyproject.toml` | Add `chromadb>=0.6.0` |
| `tests/test_pipeline.py` | Assert `eliminated_candidates` field |
| `tests/test_candidate_judge.py` | Test with context chunks and skill matches |
| `tests/test_report_generator.py` | Add eliminated candidates to fixture |

## Files Not Changed

`src/agents/jd_parser.py`, `src/agents/cv_extractor.py`, `src/agents/pool_calibrator.py`, `src/evaluation/`, `src/models/schemas.py`, `src/utils/cache.py`, `src/utils/llm.py`, `src/utils/pdf_extractor.py`, `src/utils/telemetry.py`, and all other test files are untouched.
