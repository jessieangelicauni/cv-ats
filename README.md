# EvidenceRank — LLM-Powered ATS for IT Candidate Ranking

EvidenceRank ranks IT job candidates against a job description using a local-first,
multi-phase LLM pipeline built on LangGraph. Every score is traced end-to-end via
Langfuse/OpenTelemetry, and an evaluation suite checks the pipeline's outputs for
hallucination, calibration, and run-to-run consistency.

## Architecture

The pipeline is a 5-phase LangGraph graph (`src/graph/pipeline.py`), where each phase
is a node backed by its own agent and prompt:

| Phase | Node | Agent | Prompt | Purpose |
|---|---|---|---|---|
| 1 | `phase1` | `src/agents/jd_parser.py` | `src/prompts/jd_parser.py` | Parse the raw job description into structured requirements |
| 2 | `phase2` | `src/agents/cv_extractor.py` | `src/prompts/cv_extractor.py` | Extract structured candidate profiles from CV PDFs |
| 3 | `phase3` | `src/agents/signal_enricher.py` | `src/prompts/signal_enricher.py` | Enrich profiles with derived signals |
| 4 | `phase4` | `src/agents/candidate_judge.py` | `src/prompts/judge.py` | Score each candidate against the job description |
| 5 | `phase5` | `src/agents/pool_calibrator.py` | `src/prompts/calibrator.py` | Calibrate scores across the whole candidate pool and produce the final ranking |

State flows through the graph as an `ATSState` (`src/graph/state.py`).

Module layout under `src/`:

- `agents/` — one module per pipeline phase, above
- `prompts/` — the prompt templates each agent uses
- `graph/` — LangGraph state, nodes, and pipeline assembly
- `models/schemas.py` — Pydantic models shared across phases
- `evaluation/` — post-hoc quality checks: `hallucination_checker.py`,
  `calibration_metrics.py`, `consistency_runner.py`, `ranking_metrics.py`
- `output/report_generator.py` — writes the final Markdown ranking report
- `utils/` — PDF extraction, an extraction cache, telemetry setup, and the LLM client

## Setup

1. Install [uv](https://docs.astral.sh/uv/):

   ```powershell
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

2. Install dependencies (uv reads `.python-version` and creates `.venv` automatically):

   ```bash
   uv sync
   ```

3. Copy the environment template and fill in your values:

   ```bash
   cp .env.example .env
   ```

4. Pull the required Ollama models:

   ```bash
   ollama pull qwen2.5:7b     # extraction phases (1-3) — recommended, ~6 GB VRAM
   ollama pull llama3.3:70b   # judge + calibration phases (4-5) — ~48 GB VRAM, or use a Q4 quant
   ```

## Usage

```bash
uv run python main.py --jd jd/machine_learning_engineer.txt --cv-dir resume --output results
```

| Flag | Description |
|---|---|
| `--jd` | Path to the job description text file (required) |
| `--cv-dir` | Directory of candidate CV PDFs (required) |
| `--output` | Output directory for the run's report (default: `results`) |
| `--runs` | Number of pipeline runs; use `3`+ to also compute run-to-run consistency |
| `--eval` | Also run the evaluation suite (hallucination + calibration) |
| `--no-cache` | Disable the PDF extraction cache |

Each run writes to `results/run_<id>/report.md`, with a ranked candidate table printed
to the console.

## Evaluation

Pass `--eval` to compute:

- **Hallucination rate** (`src/evaluation/hallucination_checker.py`) — for each judge
  claim, verifies it against the source CV text using sentence-embedding similarity.
  Claims below `HALLUCINATION_SIMILARITY_THRESHOLD` (default `0.85`) are flagged as
  `fabricated`; claims above it but not verbatim are `inferred`.
- **Calibration** (`src/evaluation/calibration_metrics.py`) — compares raw Phase 4
  scores to the Phase 5 pool-calibrated scores (std dev, mean absolute delta, and how
  many candidates changed rank). Scores within `BORDERLINE_SCORE_THRESHOLD` (default
  `5.0`) points of the pool mean are the ones Phase 5 is most likely to adjust.

Pass `--runs 3` (or higher) to run a **consistency experiment**
(`src/evaluation/consistency_runner.py`): the same JD/CV set is run multiple times and
rankings are compared pairwise with Kendall's τ to measure how stable the ranking is
across runs.

## Telemetry

Every run is traced via OpenTelemetry and exported to [Langfuse](https://langfuse.com)
(`src/utils/telemetry.py`). Configure in `.env`:

- `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` / `LANGFUSE_HOST` — from Langfuse Cloud
  (`https://cloud.langfuse.com`) or a self-hosted instance. `LANGFUSE_HOST` is the only
  thing that controls where traces are sent — the Langfuse SDK builds its own OTLP
  exporter from it (`{LANGFUSE_HOST}/api/public/otel/v1/traces`, authenticated with the
  public/secret key pair); it does not read the standard `OTEL_EXPORTER_OTLP_ENDPOINT`
  env var, so setting that has no effect. To send traces elsewhere (e.g. Jaeger/Tempo),
  point `LANGFUSE_HOST` at a self-hosted Langfuse instance that forwards them, rather
  than trying to redirect the exporter directly.
- `OTEL_SERVICE_NAME` / `OTEL_SERVICE_VERSION` — picked up automatically by the
  OpenTelemetry SDK's resource detection and attached to every trace.
- `TELEMETRY_ENABLED` — set to `false` to disable tracing entirely (offline runs, CI,
  unit tests).

Evaluation metrics (hallucination rate, calibration stats, consistency τ) are also
logged back onto each run's trace as Langfuse scores.

## Testing

```bash
uv run pytest
uv run pytest --cov=src   # with coverage
```

## Project Structure

```
cv-ats/
├── config.py               # env-driven configuration
├── main.py                 # Typer CLI entry point
├── pyproject.toml
├── uv.lock
├── .env.example
├── jd/                      # sample job descriptions
├── resume/                  # sample candidate CV PDFs
├── results/                 # pipeline run outputs (gitignored)
├── src/
│   ├── agents/              # one module per pipeline phase
│   ├── prompts/              # prompt templates
│   ├── graph/                 # LangGraph state, nodes, pipeline
│   ├── models/                # Pydantic schemas
│   ├── evaluation/            # hallucination, calibration, consistency, ranking metrics
│   ├── output/                 # report generation
│   └── utils/                  # PDF extraction, cache, telemetry, LLM client
└── tests/                    # pytest suite, mirrors src/ layout
```
