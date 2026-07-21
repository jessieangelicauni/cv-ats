# Repo Cleanup Sweep — Design

## Purpose

Remove unnecessary code from the `cv-ats` repo: dead code, unused dependencies,
redundant/duplicate logic, and stale comments/docs. The repo recently went
through a refactor removing RAG/ChromaDB from the pipeline (see git log:
"remove ChromaDB/RAG from architecture diagram", "remove all RAG content",
etc.), which is the likely source of leftover unused code and stale
documentation.

## Scope

**In scope:**
- `src/` (agents, evaluation, graph, models, output, prompts, utils)
- `tests/`
- `config.py`, `main.py`
- `pyproject.toml` / `uv.lock` (dependency pruning)
- `README.md` and any other repo-root docs describing removed features

**Out of scope:**
- `docs/paper/evidencerank.tex` — the research paper is kept in sync with
  code separately and is not touched by this cleanup.

## Precondition: clean tree

Before cleanup starts, commit the 8 currently-modified files (config.py,
src/graph/nodes.py, src/graph/pipeline.py, src/prompts/judge.py,
src/agents/candidate_judge.py, src/evaluation/hallucination_checker.py,
tests/test_hallucination_checker.py, README.md, .env.example) as their own
commit, describing the in-progress work they represent. This isolates the
cleanup diff from unrelated in-flight changes.

## Detection method

Static analysis tools, run ephemerally (no permanent pyproject.toml changes):

- `uvx ruff check --select F401,F841` — unused imports, unused local variables
- `uvx vulture src/` — unused functions, classes, and whole dead files
- Manual grep cross-check of `pyproject.toml` dependencies against actual
  imports under `src/`/`tests/`, to catch unused third-party packages
  (e.g. `sentence-transformers`, `scikit-learn`, etc.)

Every tool-flagged item is manually reviewed before removal. Vulture and ruff
both produce false positives on dynamically-invoked code — e.g. Pydantic
validators/field configs in `src/models/schemas.py`, and functions registered
by reference in the LangGraph pipeline (`src/graph/pipeline.py`,
`src/graph/nodes.py`) rather than called directly. Nothing is deleted on tool
output alone; each candidate is traced by hand to confirm it's actually
unreferenced.

Known non-obvious case already checked during brainstorming: `src/utils/embedder.py`
looks RAG-adjacent but is still imported by `src/utils/skill_normalizer.py`
and `src/evaluation/hallucination_checker.py` — it must NOT be removed.

## Fix categories and commit plan

One commit per category, applied in this order:

1. **Dead code** — unused imports, unused functions/classes, unused whole files
2. **Unused dependencies** — prune `pyproject.toml` and regenerate `uv.lock`
3. **Duplicate/redundant logic** — consolidate near-identical functions/logic
   found during the review pass
4. **Stale comments & docs** — remove commented-out code blocks and update
   README/docs sections referencing removed RAG/ChromaDB behavior. Confirmed
   stale references exist at README.md lines 16, 35, 85, and 150 (ChromaDB
   vector store mentions, `--no-cache` flag description, directory structure
   comment).

## Validation

`uv run pytest` runs after each individual change (not just once per
category-commit). A regression is caught at the specific edit that caused
it, not after a batch of changes.

## Deliverable: cleanup report

`docs/cleanup-report.md` — a short markdown file listing every item removed
or changed, grouped by category, with a one-line reason each (e.g. "unused
after RAG removal", "duplicate of X", "unused dependency — no imports
found"). This serves as an audit trail for the research project.

## Non-goals

- No new features, no refactors beyond what's needed to remove
  redundancy/duplication.
- No changes to `docs/paper/evidencerank.tex` or paper content.
- No changes to test *behavior* — only removal of dead test code if any is
  found; test coverage should not decrease as a result of this work.
