---
name: remove-baselines-ablation
description: Remove all baseline comparison and ablation study code from EvidenceRank; revert ablation-gated behavior to always-on
metadata:
  type: project
---

# Remove Baselines + Ablation Study

## Goal

Strip all baseline (TF-IDF, keyword-heuristic) and ablation study code from the codebase. Ablation-gated pipeline behavior reverts to always-on (the full-system path is the only path).

---

## Files to Delete (7)

| File | Reason |
|---|---|
| `src/evaluation/baselines.py` | TF-IDF + keyword rankers |
| `src/evaluation/ablation.py` | Ablation variant runner |
| `src/prompts/judge_no_grounding.py` | Only exists for `no_evidence_grounding` variant |
| `tests/test_baselines.py` | Baseline tests |
| `tests/test_ablation.py` | Ablation tests |
| `tests/test_pipeline_ablation.py` | Pipeline ablation flag tests |
| `docs/superpowers/specs/2026-07-17-evaluation-baselines-ablation-design.md` | Superseded |
| `docs/superpowers/plans/2026-07-17-evaluation-baselines-ablation.md` | Superseded |
| `tests/test_candidate_judge.py` | All 3 tests are ablation-specific (`use_evidence_grounding`) — delete entirely |
| `tests/test_main_flags.py` | All tests are for `--baselines`/`--ablation` flags — delete entirely |

---

## Files to Modify (6)

### `main.py`
- Remove imports: `run_baselines`, `run_ablation`
- Remove CLI params: `baselines: bool`, `ablation: bool`
- Remove the entire `if baselines:` block (lines covering baseline execution, Rich table, JSON write, Langfuse scores)
- Remove the entire `if ablation:` block (lines covering ablation execution, Rich table, JSON write, Langfuse scores)

### `src/graph/pipeline.py`
- Remove `_ranking_from_raw()` helper function
- Remove the 4 ablation params from `run_pipeline()`: `use_vector_store`, `use_skill_filter`, `use_evidence_grounding`, `use_pool_calibration`
- Revert conditional paths to always-on:
  - Always pass `vector_store` (not `None`) to extractor
  - Always apply skill filter
  - Always pass `use_evidence_grounding=True` to judge (then remove param entirely)
  - Always call pool calibrator (remove raw-score fallback branch)

### `src/graph/nodes.py`
- Remove `use_evidence_grounding` parameter from the node that calls `CandidateJudgeAgent`
- Remove its passthrough into `CandidateJudgeAgent.__init__`

### `src/agents/candidate_judge.py`
- Remove `use_evidence_grounding: bool = True` constructor param
- Remove `self._use_evidence_grounding` attribute
- Remove the conditional prompt selection — always use the grounding prompt (`prompts.SYSTEM`)

### `tests/test_candidate_judge.py` + `tests/test_main_flags.py`
- Both files are 100% ablation/baseline-specific — deleted entirely (covered in the delete list above)

---

## Invariants

- The normal pipeline output (`--eval`, `--runs`, report generation) is unchanged
- No new parameters are introduced — this is pure deletion
- After removal, `pytest` must pass cleanly
