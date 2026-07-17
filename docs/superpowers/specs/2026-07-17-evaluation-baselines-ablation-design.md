---
name: evaluation-baselines-ablation
description: Design for adding TF-IDF/keyword baselines and 4-variant ablation study to EvidenceRank, integrated as --baselines and --ablation flags on main.py
metadata:
  type: project
---

# Evaluation: Baselines + Ablation Study

## Context

EvidenceRank currently has three internal evaluation metrics (`--eval`): hallucination rate, calibration stats, and run-to-run consistency (Kendall's τ). These measure internal system properties but cannot show improvement over simpler methods or attribute gains to specific components. This design adds two new evaluation modes to close those gaps.

No human-annotated ground truth exists. The real experiment will run on 20 CVs × 2 JDs (not yet collected). Code must be data-agnostic and work on any CV directory + JD file.

---

## CLI Interface

Two new flags on `main.py`, consistent with the existing `--eval` pattern. Both run **after** the normal pipeline completes, using `state` (EvidenceRank's ranking) already in memory.

```
--baselines    Run TF-IDF and keyword-heuristic rankers on the same CVs/JD.
               Fast — no LLM. Safe to run every time.

--ablation     Run 4 pipeline variants with one component disabled at a time.
               LLM-heavy — use deliberately.
```

Neither flag changes the normal pipeline output or report.

---

## Baseline Implementations

**New file:** `src/evaluation/baselines.py`

### TF-IDF Ranker
- `sklearn.TfidfVectorizer` fits on JD text + all CV raw texts
- Cosine similarity between JD vector and each CV vector
- Candidates ranked by descending similarity score

### Keyword-Heuristic Ranker
- Tokenize JD into unique non-stopword terms (sklearn English stop-word list)
- Count case-insensitive exact matches per CV
- Candidates ranked by descending match count

### Metrics (no ground truth required)
- Score distribution per method: mean, std, min, max
- Cross-method Kendall's τ:
  - TF-IDF vs Keyword
  - TF-IDF vs EvidenceRank
  - Keyword vs EvidenceRank

### Output
- `run_<id>/evaluation/baselines.json`
- Rich console table: columns = Method | Rank 1 | Rank 2 | ... | τ vs EvidenceRank
- Langfuse scores: `baseline_tfidf_tau`, `baseline_keyword_tau`, `baseline_tfidf_vs_keyword_tau`

```json
{
  "tfidf":        { "ranking": [...], "scores": {}, "distribution": { "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0 } },
  "keyword":      { "ranking": [...], "scores": {}, "distribution": { "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0 } },
  "evidencerank": { "ranking": [...] },
  "cross_method_tau": {
    "tfidf_vs_keyword":       0.0,
    "tfidf_vs_evidencerank":  0.0,
    "keyword_vs_evidencerank": 0.0
  }
}
```

---

## Ablation Study

### Pipeline Changes

`run_pipeline()` gains four new boolean parameters (all default `True`):

| Parameter | Default | What disabling does |
|---|---|---|
| `use_vector_store` | `True` | CV extractor gets full raw text; judge gets no context chunks |
| `use_skill_filter` | `True` | All candidates pass to Phase 3 regardless of skill score |
| `use_evidence_grounding` | `True` | Judge uses `judge_no_grounding.py` prompt without EVIDENCE RULE |
| `use_pool_calibration` | `True` | Raw Phase 3 scores sorted directly into `FinalRanking`; Phase 4 skipped |

Existing `use_cache` flag remains unchanged and continues to control the extraction cache. The new flags are independent of it.

### Four Named Variants

| Variant name | Disabled parameter | What it isolates |
|---|---|---|
| `no_rag` | `use_vector_store=False` | Contribution of focused RAG retrieval to hallucination rate |
| `no_evidence_grounding` | `use_evidence_grounding=False` | Contribution of verbatim-quote rule to fabricated claim rate |
| `no_skill_filter` | `use_skill_filter=False` | LLM call savings from the pre-filter |
| `no_calibration` | `use_pool_calibration=False` | Whether Phase 4 meaningfully shifts rankings |

### New Files
- **`src/evaluation/ablation.py`** — thin wrapper that calls `run_pipeline()` with each variant config and collects metrics
- **`src/prompts/judge_no_grounding.py`** — judge system prompt without the EVIDENCE RULE; used only by the `no_evidence_grounding` variant

### Metrics per Variant
- Hallucination rate (requires running `verify_evidence_chain` on each variant's assessments)
- Score std (spread of calibrated or raw scores)
- LLM invocation count (tracked via a counter incremented in `invoke_with_telemetry`)
- Kendall's τ vs full EvidenceRank ranking

### Output
- `run_<id>/evaluation/ablation.json`
- Rich console table: columns = Variant | Hallucination Rate | Score Std | LLM Calls | τ vs Full
- Langfuse scores: `ablation_<variant>_hallucination_rate`, `ablation_<variant>_tau`

```json
{
  "full_system":          { "hallucination_rate": 0.0, "score_std": 0.0, "llm_calls": 0, "tau_vs_full": 1.0 },
  "no_rag":               { "hallucination_rate": 0.0, "score_std": 0.0, "llm_calls": 0, "tau_vs_full": 0.0 },
  "no_evidence_grounding":{ "hallucination_rate": 0.0, "score_std": 0.0, "llm_calls": 0, "tau_vs_full": 0.0 },
  "no_skill_filter":      { "hallucination_rate": 0.0, "score_std": 0.0, "llm_calls": 0, "tau_vs_full": 0.0 },
  "no_calibration":       { "hallucination_rate": 0.0, "score_std": 0.0, "llm_calls": 0, "tau_vs_full": 0.0 }
}
```

---

## Paper Placeholders

Three blocks added to `evidencerank.tex` using `%% TODO:` comment style:

1. **`\section{Experimental Setup}`** — dataset description with placeholders for final CV/JD counts and anonymisation method
2. **`\subsection{Baseline Methods}`** — TF-IDF and keyword-heuristic descriptions + placeholder results table
3. **`\subsection{Ablation Study}`** — one paragraph per variant + placeholder results table (4 variants × 4 metrics)

All `%% TODO:` markers are greppable: `grep -n "TODO" docs/paper/evidencerank.tex`

---

## Files Changed

| File | Change |
|---|---|
| `src/evaluation/baselines.py` | New — TF-IDF ranker, keyword ranker, cross-method metrics |
| `src/evaluation/ablation.py` | New — ablation variant runner, metrics aggregator |
| `src/prompts/judge_no_grounding.py` | New — judge prompt without EVIDENCE RULE |
| `src/graph/pipeline.py` | Add `use_vector_store`, `use_skill_filter`, `use_evidence_grounding`, `use_pool_calibration` params |
| `src/utils/llm.py` | Add optional LLM invocation counter to `invoke_with_telemetry` |
| `main.py` | Add `--baselines` and `--ablation` flags, output tables, Langfuse logging |
| `docs/paper/evidencerank.tex` | Add placeholder sections for experimental setup, baselines, ablation |

---

## Constraints

- No human ground truth — all metrics are internal or cross-method
- Ablation runs are LLM-heavy; `--ablation` must never be triggered by `--eval` automatically
- `use_cache` and the four new ablation flags are independent; ablation variants always use the extraction cache (no point re-extracting CVs)
- Adding a new ablation variant later = one new dict entry in `ablation.py` + no pipeline changes
