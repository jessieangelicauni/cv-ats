---
name: remove-chromadb-rag
description: Remove ChromaDB vector store and RAG retrieval from Phase 3; judge sees only structured CandidateProfile JSON
metadata:
  type: project
---

# Remove ChromaDB RAG — Design

## Goal

Strip ChromaDB indexing (Phase 2) and retrieval (Phase 3) entirely. The candidate judge already receives the full structured `CandidateProfile` JSON (skills with evidence quotes, work history with achievements, education, certifications) — top-K chunk retrieval adds fragmentation risk without adding coverage.

---

## Scope

Code-only changes plus corresponding paper updates. No schema changes. No new abstractions.

---

## Code Changes

### Files to delete

| File | Reason |
|---|---|
| `src/utils/vector_store.py` | Entire class (`CVVectorStore`) becomes unused |
| `tests/test_vector_store.py` | Tests for deleted module |

### Files to modify

| File | Change |
|---|---|
| `src/graph/nodes.py` | Phase 2: remove `vector_store` param + `index_cv()` call. Phase 3: remove `vector_store` param, `context_chunks` variable, and `retrieve()` call. |
| `src/graph/pipeline.py` | Remove `CVVectorStore` import, `vector_store` instantiation, and `vector_store` args passed to phase2/phase3. |
| `src/agents/candidate_judge.py` | Remove `context_chunks` parameter from `run()`. |
| `src/prompts/judge.py` | Remove the `if context_chunks:` block from `human()`. |
| `config.py` | Remove `CHROMA_DB_PATH` constant and its `ensure_dirs()` entry. |
| `pyproject.toml` | Remove `chromadb>=0.6.0` from dependencies. |

### What is NOT removed

- `src/utils/embedder.py` — kept; still used by `src/evaluation/hallucination_checker.py`
- `config.EMBEDDING_MODEL` — kept; still needed by hallucination checker
- `skill_matches` parameter on `CandidateJudgeAgent.run()` — kept; this is the deterministic skill coverage table (set intersection), not RAG

---

## Resulting Phase 3 Flow

After this change, `phase3_candidate_judge` in `nodes.py`:

1. Receives `profiles: list[CandidateProfile]` and `jd: JDRequirements`
2. For each profile, computes `skill_matches` (exact canonical-name set intersection — unchanged)
3. Calls `CandidateJudgeAgent().run(profile, jd, skill_matches=skill_matches)`
4. Judge sees: full structured `CandidateProfile` JSON + skill coverage table

No vector store parameter. No ChromaDB. No context chunks.

---

## Paper Changes (evidencerank.tex)

### Abstract
Remove "self-document RAG" from the contribution list. Replace with a statement about structured extraction grounding.

### Section 3.1 (System Overview)
Remove ChromaDB from the pipeline overview sentence. Remove `ChromaDB vector store` from the component list.

### Architecture Diagram (TikZ)
- Remove `chroma` node (ChromaDB Paragraph Chunks)
- Remove `rag` node (RAG Retrieval Top-K)
- Remove both `dasharrow` edges connecting p2→chroma→rag→p3
- Remove "MiniLM Cosine" label from the Skill Coverage Table node (skill matching is exact string match, not cosine)

### Section 3.2 (Phase 2)
- Rename heading: "RAG-Augmented CV Extraction" → "Structured CV Extraction"
- Remove the Indexing and Focused Retrieval paragraphs (ChromaDB indexing, MiniLM encoding, paragraph chunking)
- Keep: LLM-based structured extraction, Phase 2B skill normalisation

### Section 3.3 (Phase 3 — Candidate Judge)
- Remove bullet: "Relevant CV excerpts: top-K ChromaDB chunks retrieved using the full JD text as query"
- Keep bullet: "Skill coverage table"

### Research Gaps (Section 2, G2)
Remove or rewrite G2: "Self-document retrieval blind spot" — this was the motivating gap for the RAG feature being deleted.

### Research Novelties (Section 6, N1)
Remove or rewrite N1: "Self-Document Retrieval-Augmented Assessment" — this was the novelty claim for ChromaDB.

### Implementation (Section 4)
Remove `chromadb (v0.6+)` from the dependency list.

### Conclusion
Remove "self-document RAG" from the architectural summary. The conclusion's grounding claim ("fabrication rate is reduced by grounding each claim in focused self-document context") should be rewritten to reference the structured CandidateProfile extraction + evidence quotes instead of RAG context.

---

## Invariants After Changes

- All existing tests pass (`uv run pytest`)
- No import of `CVVectorStore` or `chromadb` anywhere in `src/`
- `grep -r "vector_store\|ChromaDB\|context_chunks" src/` returns no results
- Paper contains no reference to ChromaDB, paragraph chunks, RAG retrieval, or MiniLM cosine for skill matching
- `phase3_candidate_judge` signature: `(profiles, jd)` — no `vector_store` param
- `CandidateJudgeAgent.run()` signature: `(profile, jd, skill_matches=None)` — no `context_chunks` param
