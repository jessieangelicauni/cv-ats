# Remove ChromaDB RAG Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Strip ChromaDB indexing and retrieval from the pipeline entirely so the judge sees only the structured `CandidateProfile` JSON, then update the paper to reflect the actual architecture.

**Architecture:** The code change is purely subtractive — remove `vector_store` parameters and calls from Phase 2 and Phase 3, delete `src/utils/vector_store.py` and its tests, remove `CHROMA_DB_PATH` from config, and drop the `chromadb` package. The paper change rewrites all RAG/ChromaDB claims to describe structured extraction.

**Tech Stack:** Python 3.11, LaTeX (IEEEtran), `uv` for package management.

---

## File Map

| File | Action |
|---|---|
| `src/graph/nodes.py` | Remove `vector_store` param from `phase2_cv_extractor` and `phase3_candidate_judge`; remove `index_cv` and `retrieve` calls |
| `src/graph/pipeline.py` | Remove `CVVectorStore` import, `vector_store` instantiation, and `vector_store` args |
| `src/agents/candidate_judge.py` | Remove `context_chunks` parameter from `run()` |
| `src/prompts/judge.py` | Remove the `if context_chunks:` block from `human()` |
| `src/utils/vector_store.py` | **Delete** |
| `tests/test_vector_store.py` | **Delete** |
| `config.py` | Remove `CHROMA_DB_PATH` and its `ensure_dirs()` entry |
| `pyproject.toml` | Remove `chromadb>=0.6.0` |
| `docs/paper/evidencerank.tex` | Rewrite all RAG/ChromaDB references across 8 locations |

---

### Task 1: Clean nodes.py — remove vector_store from Phase 2 and Phase 3

**Files:**
- Modify: `src/graph/nodes.py`

- [ ] **Step 1: Replace the full file content**

Write `src/graph/nodes.py` with this content:

```python
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
```

- [ ] **Step 2: Verify the import of CVVectorStore is gone**

```bash
grep -n "vector_store\|CVVectorStore\|context_chunks\|index_cv\|retrieve" src/graph/nodes.py
```

Expected: no output.

- [ ] **Step 3: Commit**

```bash
git add src/graph/nodes.py
git commit -m "refactor: remove vector_store from Phase 2 and Phase 3 nodes"
```

---

### Task 2: Clean pipeline.py — remove CVVectorStore instantiation and args

**Files:**
- Modify: `src/graph/pipeline.py`

- [ ] **Step 1: Replace the full file content**

Write `src/graph/pipeline.py` with this content:

```python
from __future__ import annotations
import time
import uuid
from typing import Callable
from src.graph.state import ATSState
from src.graph.nodes import (
    phase1_jd_parser, phase2_cv_extractor,
    phase3_candidate_judge, phase4_pool_calibrator,
)
from src.utils.cache import ExtractionCache
from src.utils.skill_normalizer import normalize_skills
from src.models.schemas import CandidateProfile, JDRequirements
import config


def _filter_by_required_skills(
    profiles: list[CandidateProfile],
    jd: JDRequirements,
) -> tuple[list[CandidateProfile], list[str]]:
    if not jd.required_skills:
        return profiles, []
    required_names = {s.skill for s in jd.required_skills}
    passing: list[CandidateProfile] = []
    eliminated: list[str] = []
    for profile in profiles:
        candidate_names = {s.canonical_skill for s in profile.skills}
        if required_names & candidate_names:
            passing.append(profile)
        else:
            eliminated.append(profile.candidate_id)
    return passing, eliminated


def run_pipeline(
    jd_raw: str,
    cv_raws: list[dict],
    run_id: str | None = None,
    use_cache: bool = True,
    on_phase_complete: Callable[[dict], None] | None = None,
) -> ATSState:
    run_id = run_id or str(uuid.uuid4())[:8]
    cache = ExtractionCache(config.CACHE_DB_PATH) if use_cache else None

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

    # Normalize JD skill names through same LLM prompt as CV Phase 2B
    # so both sides are canonical and exact-match comparison works
    jd_skill_names = [s.skill for s in jd_structured.required_skills + jd_structured.preferred_skills]
    if jd_skill_names:
        jd_norm_map = normalize_skills(jd_skill_names)
        for s in jd_structured.required_skills:
            s.skill = jd_norm_map.get(s.skill, s.skill)
        for s in jd_structured.preferred_skills:
            s.skill = jd_norm_map.get(s.skill, s.skill)

    cv_profiles = _run_phase(2, phase2_cv_extractor, cv_raws, cache,
                             candidates=len(cv_raws))

    cv_profiles, eliminated = _filter_by_required_skills(cv_profiles, jd_structured)

    assessments = _run_phase(3, phase3_candidate_judge, cv_profiles, jd_structured)

    final_ranking = _run_phase(4, phase4_pool_calibrator, assessments, jd_structured)

    return ATSState(
        jd_raw=jd_raw,
        cv_raws=cv_raws,
        jd_structured=jd_structured,
        cv_profiles=cv_profiles,
        candidate_assessments=assessments,
        final_ranking=final_ranking,
        run_id=run_id,
        trace_log=trace_log,
        use_cache=use_cache,
        eliminated_candidates=eliminated,
    )
```

- [ ] **Step 2: Verify CVVectorStore is gone**

```bash
grep -n "CVVectorStore\|vector_store\|chromadb" src/graph/pipeline.py
```

Expected: no output.

- [ ] **Step 3: Commit**

```bash
git add src/graph/pipeline.py
git commit -m "refactor: remove CVVectorStore from pipeline — no vector store instantiation"
```

---

### Task 3: Clean candidate_judge.py and judge.py — remove context_chunks

**Files:**
- Modify: `src/agents/candidate_judge.py`
- Modify: `src/prompts/judge.py`

- [ ] **Step 1: Replace candidate_judge.py**

Write `src/agents/candidate_judge.py` with this content:

```python
from __future__ import annotations
from langchain_core.messages import SystemMessage, HumanMessage
from src.models.schemas import CandidateAssessment, CandidateProfile, JDRequirements
from src.utils.llm import get_llm
from src.prompts import judge as prompts
import config


class CandidateJudgeAgent:
    def __init__(self):
        self._llm = get_llm(config.LARGE_MODEL, CandidateAssessment, config.JUDGE_TEMPERATURE)

    def run(
        self,
        profile: CandidateProfile,
        jd: JDRequirements,
        skill_matches: list | None = None,
    ) -> CandidateAssessment:
        return self._llm.invoke(
            [SystemMessage(content=prompts.SYSTEM),
             HumanMessage(content=prompts.human(
                 jd.model_dump_json(indent=2),
                 profile.model_dump_json(indent=2),
                 skill_matches=skill_matches,
             ))]
        )
```

- [ ] **Step 2: Replace judge.py**

Write `src/prompts/judge.py` with this content:

```python
SYSTEM = """You are an experienced senior technical recruiter with 15 years in IT hiring.
You reason like a human expert: leadership and measurable impact matter more than job \
titles; relevant tenure increases confidence; career changes that represent growth are \
not penalized.

When a candidate has worked at a Tier 1 multinational company, treat this as a \
meaningful positive signal — it indicates exposure to large-scale systems, professional \
engineering culture, and organisational breadth. Weight this in your holistic \
assessment, particularly under Career Trajectory and Experience Depth.

EVIDENCE RULE: Every evidence_chain item MUST include an evidence_quote containing \
exact text from the candidate profile. If supporting text is not found, set \
evidence_quote to "NOT FOUND IN CV" and lower the dimension_score accordingly. \
Never state a fact not traceable to the profile.
Return a valid JSON object matching the required schema exactly."""


def human(
    jd_json: str,
    profile_json: str,
    skill_matches: list | None = None,
) -> str:
    parts: list[str] = [
        f"Assess the following candidate against the job requirements.\n\n"
        f"JOB REQUIREMENTS:\n{jd_json}\n\n"
        f"CANDIDATE PROFILE:\n{profile_json}\n\n"
    ]

    if skill_matches:
        rows = [
            "SKILL COVERAGE (pre-computed — use as grounding for Technical Skills Fit):",
            "| JD Skill | Candidate Match | Score | Required |",
            "|---|---|---|---|",
        ]
        for m in skill_matches:
            match_str = m.best_match if m.best_match else "(no match)"
            req_str = "Yes" if m.is_required else "No"
            rows.append(f"| {m.jd_skill} | {match_str} | {m.score:.2f} | {req_str} |")
        parts.append("\n".join(rows) + "\n\n")

    parts.append(
        "Assess across these dimensions:\n"
        "1. Technical Skills Fit\n"
        "2. Experience Depth\n"
        "3. Career Trajectory\n"
        "4. Leadership & Impact\n"
        "5. Education & Credentials\n\n"
        "For each: write assessment, cite evidence_quote (exact text or "
        '"NOT FOUND IN CV"), score 0-10.\n\n'
        "Then produce raw_score (0-100) as your HOLISTIC judgment — not a formula. "
        "Weigh dimensions contextually based on what this specific role requires."
    )

    return "".join(parts)
```

- [ ] **Step 3: Verify context_chunks is gone**

```bash
grep -n "context_chunks" src/agents/candidate_judge.py src/prompts/judge.py
```

Expected: no output.

- [ ] **Step 4: Commit**

```bash
git add src/agents/candidate_judge.py src/prompts/judge.py
git commit -m "refactor: remove context_chunks from CandidateJudgeAgent and judge prompt"
```

---

### Task 4: Delete vector_store.py and test_vector_store.py

**Files:**
- Delete: `src/utils/vector_store.py`
- Delete: `tests/test_vector_store.py`

- [ ] **Step 1: Delete both files**

```bash
rm src/utils/vector_store.py tests/test_vector_store.py
```

- [ ] **Step 2: Verify no remaining imports of CVVectorStore**

```bash
grep -rn "vector_store\|CVVectorStore" src/ tests/
```

Expected: no output.

- [ ] **Step 3: Commit**

```bash
git add -u src/utils/vector_store.py tests/test_vector_store.py
git commit -m "refactor: delete CVVectorStore and its tests — ChromaDB removed"
```

---

### Task 5: Clean config.py and pyproject.toml

**Files:**
- Modify: `config.py`
- Modify: `pyproject.toml`

- [ ] **Step 1: Replace config.py**

Write `config.py` with this content:

```python
from __future__ import annotations
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

SMALL_MODEL = os.getenv("SMALL_MODEL", "qwen2.5:7b")
LARGE_MODEL = os.getenv("LARGE_MODEL", "llama3.3:70b")

EXTRACTION_TEMPERATURE = float(os.getenv("EXTRACTION_TEMPERATURE", "0.0"))
JUDGE_TEMPERATURE = float(os.getenv("JUDGE_TEMPERATURE", "0.1"))

BORDERLINE_SCORE_THRESHOLD = float(os.getenv("BORDERLINE_SCORE_THRESHOLD", "5.0"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "results"
JD_DIR = BASE_DIR / "jd"
CACHE_DB_PATH = BASE_DIR / ".cache" / "extractions.db"


def ensure_dirs() -> None:
    for _d in [RESULTS_DIR, JD_DIR, CACHE_DB_PATH.parent]:
        _d.mkdir(parents=True, exist_ok=True)


ensure_dirs()
```

- [ ] **Step 2: Remove chromadb from pyproject.toml**

In `pyproject.toml`, find and replace:

Find:
```
    "sentence-transformers>=5.6.0",
    "chromadb>=0.6.0",
```

Replace with:
```
    "sentence-transformers>=5.6.0",
```

- [ ] **Step 3: Verify**

```bash
grep -n "CHROMA_DB_PATH\|chromadb" config.py pyproject.toml
```

Expected: no output.

- [ ] **Step 4: Commit**

```bash
git add config.py pyproject.toml
git commit -m "refactor: remove CHROMA_DB_PATH from config and chromadb from dependencies"
```

---

### Task 6: Verify all code changes — run full test suite

**Files:**
- Read: test output

- [ ] **Step 1: Run the full test suite**

```bash
uv run pytest -v
```

Expected: all tests pass. The 4 tests from `test_vector_store.py` are gone (deleted in Task 4); all remaining tests should be green. If any test fails, fix it before proceeding.

- [ ] **Step 2: Confirm no stale references in src/**

```bash
grep -rn "vector_store\|CVVectorStore\|context_chunks\|CHROMA_DB_PATH\|chromadb" src/ config.py pyproject.toml
```

Expected: no output.

---

### Task 7: Paper — Abstract, Introduction, and Keywords

**Files:**
- Modify: `docs/paper/evidencerank.tex`

- [ ] **Step 1: Fix the abstract — remove RAG contribution, renumber**

Find:
```latex
introduces (1) a \emph{retrieval-augmented assessment} mechanism that indexes
candidate CV text into ChromaDB as paragraph-level chunks and retrieves focused
context using the job requirements as the retrieval query; (2) an \emph{evidence-chain grounding}
constraint requiring every dimension score to cite a verbatim text quote, backed by
post-hoc semantic verification using all-MiniLM-L6-v2; (3) a \emph{semantic skill
pre-filter} that eliminates candidates lacking required-skill matches through exact
canonical name matching after LLM-based skill normalisation, before expensive LLM judging; and (4) a \emph{pool-aware holistic
calibration} phase that resolves borderline pairs through comparative reasoning.
```

Replace with:
```latex
introduces (1) an \emph{evidence-chain grounding}
constraint requiring every dimension score to cite a verbatim text quote, backed by
post-hoc semantic verification using all-MiniLM-L6-v2; (2) a \emph{semantic skill
pre-filter} that eliminates candidates lacking required-skill matches through exact
canonical name matching after LLM-based skill normalisation, before expensive LLM judging; and (3) a \emph{pool-aware holistic
calibration} phase that resolves borderline pairs through comparative reasoning.
```

- [ ] **Step 2: Remove "retrieval-augmented generation" from keywords**

Find:
```latex
applicant tracking systems, large language models, retrieval-augmented generation,
hallucination detection, multi-agent systems, candidate ranking, natural language
processing, semantic similarity, IT recruitment
```

Replace with:
```latex
applicant tracking systems, large language models, hallucination detection, multi-agent systems, candidate ranking, natural language
processing, semantic similarity, IT recruitment
```

- [ ] **Step 3: Fix Introduction contribution 1 — remove "RAG-augmented"**

Find:
```latex
        (JD parsing → RAG-augmented CV extraction → evidence-grounded candidate
```

Replace with:
```latex
        (JD parsing → structured CV extraction → evidence-grounded candidate
```

- [ ] **Step 4: Delete Introduction contribution 2 (the RAG enumerate item)**

Find:
```latex
  \item We introduce \textbf{retrieval-augmented self-document assessment}: each
        candidate's CV is paragraph-chunked, indexed in ChromaDB, and queried
        with three dimension-focused queries (experience, skills, education) to
        provide focused context to the LLM assessor, reducing hallucination by
        providing grounding evidence in-context.

```

Replace with (empty string — delete the item entirely):
```latex
```

- [ ] **Step 5: Fix Introduction contribution 4 — remove "MiniLM-based cosine similarity"**

Find:
```latex
  \item We implement a \textbf{semantic skill pre-filter} using MiniLM-based
        cosine similarity that eliminates candidates below a required-skill
        match threshold before LLM judging, reducing unnecessary LLM invocations.
```

Replace with:
```latex
  \item We implement a \textbf{semantic skill pre-filter} using LLM-normalised
        exact canonical name matching that eliminates candidates with no
        required-skill overlap before LLM judging, reducing unnecessary LLM invocations.
```

- [ ] **Step 6: Verify**

```bash
grep -n "RAG-augmented\|retrieval-augmented generation\|paragraph-chunked\|indexed in ChromaDB\|MiniLM-based" docs/paper/evidencerank.tex
```

Expected: no output.

- [ ] **Step 7: Commit**

```bash
git add docs/paper/evidencerank.tex
git commit -m "refactor(paper): update abstract, keywords, and Introduction contributions — remove RAG claim"
```

---

### Task 8: Paper — Research Gaps G2 and G3

**Files:**
- Modify: `docs/paper/evidencerank.tex`

- [ ] **Step 1: Delete G2 (self-document retrieval blind spot)**

Find:
```latex
\textbf{G2 — Self-document retrieval blind spot:} RAG is applied to external
corpora, not the candidate's own document at paragraph granularity
\cite{tran2025, lo2025}.

```

Replace with (empty — delete the paragraph):
```latex
```

- [ ] **Step 2: Fix G3 — replace "cosine similarity" with "exact canonical name matching"**

Find:
```latex
\textbf{G3 — Coarse-grained skill matching:} Skill alignment uses full-document
similarity or rigid ontologies; no paper applies targeted per-skill cosine
similarity as a binary pre-filter \cite{alonso2025, ajjam2026}.
```

Replace with:
```latex
\textbf{G3 — Coarse-grained skill matching:} Skill alignment uses full-document
similarity or rigid ontologies; no paper applies LLM-normalised exact canonical
name matching as a targeted per-skill pre-filter \cite{alonso2025, ajjam2026}.
```

- [ ] **Step 3: Fix the "addresses all five gaps" sentence**

Find:
```latex
EvidenceRank directly addresses all five gaps.
```

Replace with:
```latex
EvidenceRank directly addresses all remaining gaps.
```

- [ ] **Step 4: Verify**

```bash
grep -n "G2\|Self-document retrieval blind spot\|all five gaps\|per-skill cosine" docs/paper/evidencerank.tex
```

Expected: no output.

- [ ] **Step 5: Commit**

```bash
git add docs/paper/evidencerank.tex
git commit -m "refactor(paper): remove G2 gap, fix G3 to describe exact name matching"
```

---

### Task 9: Paper — Section 3.1 Overview and Architecture Diagram

**Files:**
- Modify: `docs/paper/evidencerank.tex`

- [ ] **Step 1: Fix Section 3.1 overview sentence — remove ChromaDB from embedding subsystems**

Find:
```latex
A shared
all-MiniLM-L6-v2 singleton provides embeddings to three sub-systems: the
ChromaDB vector store, the semantic skill matcher, and the post-hoc hallucination
checker. All intermediate state is carried in a typed \texttt{ATSState} model.
```

Replace with:
```latex
A shared all-MiniLM-L6-v2 singleton provides embeddings to the post-hoc
hallucination checker. All intermediate state is carried in a typed \texttt{ATSState} model.
```

- [ ] **Step 2: Remove the chroma and rag TikZ nodes**

Find:
```latex
\node[subbox, right=1.6cm of p2] (chroma) {ChromaDB\\Paragraph\\Chunks};
\node[subbox, right=1.6cm of p3] (rag) {RAG\\Retrieval\\(Top-$K$)};
\node[subbox, below=0.3cm of rag] (skills) {Skill Coverage\\Table (MiniLM\\Cosine)};
```

Replace with:
```latex
\node[subbox, right=1.6cm of p3] (skills) {Skill Coverage\\Table (Exact\\Match)};
```

- [ ] **Step 3: Remove the three RAG arrows**

Find:
```latex
\draw[dasharrow] (p2) -- (chroma);
\draw[dasharrow] (chroma) -- (rag);
\draw[dasharrow] (rag) -- (p3);
\draw[dasharrow] (prefilter) -- (skills);
```

Replace with:
```latex
\draw[dasharrow] (prefilter) -- (skills);
```

- [ ] **Step 4: Fix the figure caption**

Find:
```latex
\caption{EvidenceRank four-phase pipeline. Dashed arrows indicate data flows
from the shared MiniLM embedding sub-systems. Candidates rejected by the
pre-filter bypass Phase 3 entirely.}
```

Replace with:
```latex
\caption{EvidenceRank four-phase pipeline. Dashed arrows indicate auxiliary data flows.
Candidates rejected by the pre-filter bypass Phase 3 entirely.}
```

- [ ] **Step 5: Verify**

```bash
grep -n "chroma\|RAG\|Top-\$K\$\|MiniLM.Cosine\|three sub-systems" docs/paper/evidencerank.tex
```

Expected: no output (the remaining MiniLM references are for the hallucination checker, which is fine — but "MiniLM\\Cosine" specifically for the skill table node is gone).

- [ ] **Step 6: Commit**

```bash
git add docs/paper/evidencerank.tex
git commit -m "refactor(paper): remove ChromaDB/RAG from architecture diagram and Section 3.1 overview"
```

---

### Task 10: Paper — Phase 2 body, Pre-Filter section, Phase 3 inputs

**Files:**
- Modify: `docs/paper/evidencerank.tex`

- [ ] **Step 1: Rename Phase 2 heading and rewrite body — remove RAG paragraphs**

Find:
```latex
\subsection{Phase 2: RAG-Augmented CV Extraction}
\label{subsec:phase2}

Phase 2 introduces self-document retrieval-augmented generation. Each candidate
CV undergoes two operations before LLM extraction:

\textbf{Indexing.} The raw CV text is split into paragraphs using the regex
pattern \verb|r'\n\s*\n'| (blank-line boundaries), yielding semantically
coherent chunks. Chunks shorter than 10 characters are discarded. Each chunk is
encoded with all-MiniLM-L6-v2 and stored in a ChromaDB \texttt{PersistentClient}
collection keyed by \texttt{candidate\_id}. Cache invalidation uses the first 12
hexadecimal characters of the SHA-256 hash of the raw text: if the stored hash
matches, indexing is skipped.

\textbf{Focused retrieval.} Three queries are executed against the ChromaDB
collection for each candidate:
\begin{align}
  q_1 &= \textit{``work experience roles achievements''} \notag \\
  q_2 &= \textit{``technical skills programming languages frameworks''} \notag \\
  q_3 &= \textit{``education degree certifications''} \notag
\end{align}
The top-$K$ (default $K=5$) chunks per query are retrieved using embedded cosine
similarity, deduplicated, and concatenated as a focused CV text that replaces the
raw full document for extraction. This ensures the LLM extractor receives
structured, section-relevant content rather than noisy full-document text.

The extraction itself uses two sequential LLM calls. The first (\texttt{SYSTEM\_2A})
extracts a structured \texttt{CandidateProfile} with strict verbatim quoting:
achievement strings must be exact quotes from the CV, tenure months computed
precisely from dates, and skill names recorded as-is without inference. The second
call (\texttt{SYSTEM\_2B}) normalises raw skill mentions to canonical forms (e.g.,
``postgre'' $\to$ ``PostgreSQL'', ``k8s'' $\to$ ``Kubernetes'') using a dedicated
\texttt{SkillNormalizationMap}.
```

Replace with:
```latex
\subsection{Phase 2: Structured CV Extraction}
\label{subsec:phase2}

Phase 2 extracts structured candidate data from each raw CV text through two
sequential LLM calls. The first (\texttt{SYSTEM\_2A}) extracts a structured
\texttt{CandidateProfile} with strict verbatim quoting:
achievement strings must be exact quotes from the CV, tenure months computed
precisely from dates, and skill names recorded as-is without inference. The second
call (\texttt{SYSTEM\_2B}) normalises raw skill mentions to canonical forms (e.g.,
``postgre'' $\to$ ``PostgreSQL'', ``k8s'' $\to$ ``Kubernetes'') using a dedicated
\texttt{SkillNormalizationMap}.
```

- [ ] **Step 2: Rewrite the Semantic Skill Pre-Filter section — replace cosine math with set-intersection**

Find:
```latex
Between Phase 2 and Phase 3, a pre-judge filter eliminates candidates who match
no required skill above a cosine similarity threshold $\theta$ (default 0.75).

Let $\mathcal{R} = \{r_1, \ldots, r_m\}$ be the required skills from the JD, and
$\mathcal{C}_i = \{s_1, \ldots, s_n\}$ the canonical skills from candidate $i$.
For each required skill $r_j$, the best candidate skill match score is:
\begin{equation}
  \text{score}(r_j, \mathcal{C}_i) = \max_{s_k \in \mathcal{C}_i}
    \cos\bigl(\mathbf{e}_{r_j},\; \mathbf{e}_{s_k}\bigr)
  \label{eq:skill_score}
\end{equation}
where $\mathbf{e}_{r_j}$ and $\mathbf{e}_{s_k}$ are all-MiniLM-L6-v2 embeddings
of the skill name strings. Candidate $i$ is \emph{eliminated} if:
\begin{equation}
  \forall r_j \in \mathcal{R}:\; \text{score}(r_j, \mathcal{C}_i) < \theta
  \label{eq:filter}
\end{equation}
That is, a candidate passes if at least \emph{one} required skill exceeds the
threshold. This conservative filter avoids false negatives: only candidates with
zero required-skill overlap are removed. Eliminated candidates are recorded in
\texttt{ATSState.eliminated\_candidates} and appear in the final report under a
``Filtered Candidates'' section.
```

Replace with:
```latex
Between Phase 2 and Phase 3, a pre-judge filter eliminates candidates who share
no required skill name with the JD after LLM-based normalisation.

Both the JD skill names (Phase 1) and CV skill names (Phase 2B) are normalised
to canonical forms by the same LLM prompt before the filter runs—so ``k8s'' and
``Kubernetes'' converge to the same string. The filter then computes the set
intersection:
\begin{equation}
  \mathcal{R} \cap \mathcal{C}_i \neq \emptyset
  \label{eq:filter}
\end{equation}
where $\mathcal{R} = \{r_1, \ldots, r_m\}$ is the set of required skill names
from the JD and $\mathcal{C}_i = \{s_1, \ldots, s_n\}$ is the set of canonical
skill names extracted from candidate $i$'s CV. A candidate is \emph{eliminated}
if the intersection is empty—that is, they share no required skill name after
normalisation. This conservative filter avoids false negatives: only candidates
with zero required-skill overlap are removed. Eliminated candidates are recorded
in \texttt{ATSState.eliminated\_candidates} and appear in the final report under
a ``Filtered Candidates'' section.
```

- [ ] **Step 3: Remove the ChromaDB bullet from Phase 3 inputs**

Find:
```latex
\begin{enumerate}
  \item The structured \texttt{CandidateProfile} from Phase 2 (JSON).
  \item \textbf{Relevant CV excerpts}: top-$K$ ChromaDB chunks retrieved using
        the serialised \texttt{JDRequirements} as the query embedding.
  \item \textbf{Skill coverage table}: a Markdown table of all JD skills with
        their best candidate match and cosine similarity score.
\end{enumerate}
```

Replace with:
```latex
\begin{enumerate}
  \item The structured \texttt{CandidateProfile} from Phase 2 (JSON).
  \item \textbf{Skill coverage table}: a Markdown table of all JD skills with
        their best candidate match and exact-match score.
\end{enumerate}
```

- [ ] **Step 4: Verify**

```bash
grep -n "RAG-Augmented\|self-document retrieval\|ChromaDB collection\|eq:skill_score\|cosine similarity threshold\|top-\$K\$ ChromaDB" docs/paper/evidencerank.tex
```

Expected: no output.

- [ ] **Step 5: Commit**

```bash
git add docs/paper/evidencerank.tex
git commit -m "refactor(paper): rewrite Phase 2, Pre-Filter, and Phase 3 sections — remove all RAG content"
```

---

### Task 11: Paper — Implementation Details, Research Novelties, Conclusion

**Files:**
- Modify: `docs/paper/evidencerank.tex`

- [ ] **Step 1: Remove ChromaDB from Implementation Details**

Find:
```latex
EvidenceRank is implemented in Python 3.11 using LangChain (v1.3.13+) for LLM
orchestration, ChromaDB (v0.6+) for vector storage, sentence-transformers (v5.6+)
with \texttt{all-MiniLM-L6-v2} for embeddings, and Pydantic v2 for schema
validation.
```

Replace with:
```latex
EvidenceRank is implemented in Python 3.11 using LangChain (v1.3.13+) for LLM
orchestration, sentence-transformers (v5.6+) with \texttt{all-MiniLM-L6-v2} for
hallucination-check embeddings, and Pydantic v2 for schema validation.
```

- [ ] **Step 2: Delete Research Novelty N1**

Find:
```latex
\textbf{N1 — Self-Document Retrieval-Augmented Assessment (G2):}
Rather than retrieving from external job knowledge bases as in
\cite{tran2025,lo2025}, EvidenceRank indexes the candidate's own CV at paragraph
granularity and retrieves dimension-focused excerpts into the LLM's context window.
This is the first application of within-document RAG to recruitment assessment.

```

Replace with (empty — delete entirely):
```latex
```

- [ ] **Step 3: Rewrite Research Novelty N3 — cosine → exact match**

Find:
```latex
\textbf{N3 — Per-Skill Cosine Similarity Pre-Filter (G3):}
Cosine similarity is applied at the individual skill-name level (JD
\texttt{SkillRequirement.skill} vs.\ candidate \texttt{SkillEntry.canonical\_skill})
rather than at full-document level, providing fine-grained required-skill coverage
verification that eliminates clearly unqualified candidates before LLM judging.
```

Replace with:
```latex
\textbf{N3 — LLM-Normalised Exact-Match Skill Pre-Filter (G3):}
After both JD and CV skill names are normalised to canonical forms by the same LLM
prompt, a Python set intersection on the resulting canonical name strings eliminates
candidates with no required-skill overlap before LLM judging. This inter-normalised
exact matching provides targeted required-skill coverage verification that reduces
unnecessary LLM invocations.
```

- [ ] **Step 4: Fix Conclusion first paragraph — remove "self-document RAG"**

Find:
```latex
The architecture integrates self-document RAG, evidence-chain grounding,
semantic skill pre-filtering, and pool-aware calibration.
```

Replace with:
```latex
The architecture integrates evidence-chain grounding, LLM-normalised
exact-match skill pre-filtering, and pool-aware calibration.
```

- [ ] **Step 5: Fix Conclusion key-insight paragraph — remove RAG reference**

Find:
```latex
The key insight is that hallucination in LLM assessment pipelines is not solely an
intrinsic model deficiency but an architectural one: by providing focused
self-document context (RAG), enforcing verbatim evidence citation, and verifying
claims post-generation, the fabrication rate is reduced by grounding each claim in focused
self-document context rather than relying on the model's parametric memory.
```

Replace with:
```latex
The key insight is that hallucination in LLM assessment pipelines is not solely an
intrinsic model deficiency but an architectural one: by enforcing verbatim evidence
citation from the structured candidate profile and verifying each claim post-generation,
the fabrication rate is reduced without relying on the model's parametric memory.
```

- [ ] **Step 6: Fix Conclusion last paragraph — remove "self-document RAG"**

Find:
```latex
EvidenceRank demonstrates that rigorous architectural choices—self-document RAG,
evidence chain constraints, semantic pre-filtering, and pool-aware calibration—can
```

Replace with:
```latex
EvidenceRank demonstrates that rigorous architectural choices—evidence-chain
constraints, normalised exact-match skill pre-filtering, and pool-aware calibration—can
```

- [ ] **Step 7: Verify**

```bash
grep -n "self-document RAG\|ChromaDB\|RAG\|N1\|Per-Skill Cosine\|vector storage\|paragraph granularity" docs/paper/evidencerank.tex
```

Expected: no output (remaining references to "RAG" in the Related Work section describing *other papers'* RAG use are acceptable — only EvidenceRank's own RAG claims must be gone).

- [ ] **Step 8: Commit**

```bash
git add docs/paper/evidencerank.tex
git commit -m "refactor(paper): remove N1 novelty, fix N3, update Implementation Details and Conclusion"
```

---

### Task 12: Final verification

**Files:**
- Read: `docs/paper/evidencerank.tex`, test output

- [ ] **Step 1: Run full test suite one more time**

```bash
uv run pytest -v
```

Expected: all tests pass.

- [ ] **Step 2: Confirm no EvidenceRank-specific RAG/ChromaDB claims remain in paper**

```bash
grep -n "self-document RAG\|ChromaDB\|paragraph-level chunks\|paragraph granularity\|RAG-augmented\|retrieval-augmented generation\|N1 —\|Per-Skill Cosine\|cosine similarity threshold\|eq:skill_score\|vector storage\|all five gaps" docs/paper/evidencerank.tex
```

Expected: no output.

- [ ] **Step 3: Confirm no stale code references**

```bash
grep -rn "CVVectorStore\|vector_store\|context_chunks\|CHROMA_DB_PATH\|chromadb" src/ config.py pyproject.toml tests/
```

Expected: no output.

- [ ] **Step 4: Confirm cross-references are intact**

```bash
grep -n "\\\\ref{eq:" docs/paper/evidencerank.tex
```

Expected: only `eq:filter` and `eq:hallucination_threshold` — no `eq:skill_score` (deleted in Task 10).
