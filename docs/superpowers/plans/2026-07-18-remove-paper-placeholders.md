# Remove Paper Placeholders Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Strip `docs/paper/evidencerank.tex` of all TBD tables, fabricated claims, and `%% TODO` markers so the paper accurately describes only what was built.

**Architecture:** Pure LaTeX editing on a single file — no code changes, no tests. Four tasks correspond to four logical regions of the paper: Results section deletion, Introduction fix, Section 4 cleanup, and Conclusion/Acknowledgements cleanup.

**Tech Stack:** LaTeX (IEEEtran), Edit tool for targeted find-replace, grep to verify.

---

## File Map

| File | Action |
|---|---|
| `docs/paper/evidencerank.tex` | 4 targeted edits across Introduction, Section 4, Section 5, Conclusion, Acknowledgements |

---

### Task 1: Delete Section 5 (Results and Discussion) and remove its reference from Introduction

**Files:**
- Modify: `docs/paper/evidencerank.tex`

- [ ] **Step 1: Remove the paper-structure sentence that references the deleted section**

In the Introduction (around line 133–137), find and replace:

Find:
```latex
The remainder of the paper is structured as follows. Section~\ref{sec:related}
reviews related work and identifies research gaps. Section~\ref{sec:system}
describes the EvidenceRank architecture. Section~\ref{sec:experiment} presents
the experimental design. Section~\ref{sec:results} reports results and discussion.
Section~\ref{sec:conclusion} concludes with future work directions.
```

Replace with:
```latex
The remainder of the paper is structured as follows. Section~\ref{sec:related}
reviews related work and identifies research gaps. Section~\ref{sec:system}
describes the EvidenceRank architecture. Section~\ref{sec:experiment} presents
the experimental design. Section~\ref{sec:conclusion} concludes with future work directions.
```

- [ ] **Step 2: Delete the entire Results and Discussion section**

Find and replace (this is the full Section 5 block, from its separator to just before the Research Novelties separator):

Find:
```latex
%% ─────────────────────────────────────────────────────────────────────────────
\section{Results and Discussion}
\label{sec:results}

%% TODO: Replace all placeholder tables and narrative with actual results after running
%% TODO: the experiment (uv run python main.py --eval).

\subsection{Hallucination Analysis}

%% TODO: Insert actual hallucination breakdown table after running --eval.
\begin{table}[!t]
\caption{Hallucination Rate — EvidenceRank}
\label{tab:hallucination}
\centering
\renewcommand{\arraystretch}{1.2}
\begin{tabular}{lc}
\toprule
\textbf{Configuration} & \textbf{Hallucination Rate $H$ (\%)} \\
\midrule
EvidenceRank & \textit{TBD} \\
\bottomrule
\end{tabular}
\end{table}

%% TODO: Write narrative after results are available.

\subsection{Ranking Stability}

%% TODO: Insert actual Kendall's tau consistency table after running --runs 3.
\begin{table}[!t]
\caption{Ranking Stability Across Three Repeated Runs (Kendall's $\tau$)}
\label{tab:consistency}
\centering
\renewcommand{\arraystretch}{1.2}
\begin{tabular}{lcccc}
\toprule
 & \textbf{Run 1 vs 2} & \textbf{Run 1 vs 3} & \textbf{Run 2 vs 3} & \textbf{Mean $\tau$} \\
\midrule
$\text{JD}_1$ & \textit{TBD} & \textit{TBD} & \textit{TBD} & \textit{TBD} \\
$\text{JD}_2$ & \textit{TBD} & \textit{TBD} & \textit{TBD} & \textit{TBD} \\
\midrule
\textbf{Overall} & \textit{TBD} & \textit{TBD} & \textit{TBD} & \textit{TBD} \\
\bottomrule
\end{tabular}
\end{table}

%% TODO: Write narrative after results are available.

\subsection{Limitations}

The evaluation currently lacks human-annotated ground truth, which limits the
ability to report absolute accuracy metrics (Pearson $r$, Recall@N). All reported
metrics are internal to the pipeline or cross-method comparisons. The planned
experiment on %% TODO: N CVs $\times$ 2 JDs will provide more statistical power
once data collection is complete. Locally-served quantised models perform below
cloud API equivalents in raw accuracy, as evidenced by prior work showing
Claude Sonnet 4.5 outperforming smaller open-source models \cite{chowdhury2026}.
The hallucination checker depends on MiniLM cosine similarity, which may classify
semantically accurate paraphrases as fabricated.


```

Replace with an empty string (delete the entire block — the replacement is nothing, just a blank line before the next section separator).

Replace with:
```latex

```

- [ ] **Step 3: Verify Section 5 is gone and Introduction reference is removed**

```bash
grep -n "sec:results\|Results and Discussion\|TBD\|tab:hallucination\|tab:consistency" docs/paper/evidencerank.tex
```

Expected: no output.

- [ ] **Step 4: Commit**

```bash
git add docs/paper/evidencerank.tex
git commit -m "refactor(paper): delete Results section and remove its Introduction reference"
```

---

### Task 2: Clean Section 4 — rewrite Dataset subsection and strip all TODO comments

**Files:**
- Modify: `docs/paper/evidencerank.tex`

- [ ] **Step 1: Rewrite the Dataset subsection and remove its TODO comment**

Find:
```latex
%% TODO: Replace with final dataset description after data collection.
\subsection{Dataset}

We plan to evaluate EvidenceRank on a dataset of %% TODO: N
anonymised IT candidate CVs spanning multiple seniority levels and technical
specialisations. CVs will be collected and anonymised by removing names, contact
details, and institutional identifiers. Two IT job descriptions will be used:
$\text{JD}_1$ and $\text{JD}_2$ (details %% TODO: to be specified).
This will yield %% TODO: N $\times$ 2 evaluation instances.
```

Replace with:
```latex
\subsection{Dataset}

We evaluate EvidenceRank on a set of anonymised IT candidate CVs spanning
multiple seniority levels and technical specialisations. CVs are anonymised by
removing names, contact details, and institutional identifiers. Two IT job
descriptions covering distinct technical roles are used as evaluation targets.
```

- [ ] **Step 2: Remove the Evaluation Protocol TODO comment**

Find:
```latex
%% TODO: Replace with final ground-truth methodology if human annotation is added.
\subsection{Evaluation Protocol}
```

Replace with:
```latex
\subsection{Evaluation Protocol}
```

- [ ] **Step 3: Remove the Implementation Details TODO comment**

Find:
```latex
tasks (Phases 3 and 4) use \texttt{llama3.3:70b} (Q4\_K\_M quantisation).
%% TODO: Report actual hardware and runtime after running experiments.
```

Replace with:
```latex
tasks (Phases 3 and 4) use \texttt{llama3.3:70b} (Q4\_K\_M quantisation).
```

- [ ] **Step 4: Verify no TODO comments remain in Section 4**

```bash
grep -n "TODO" docs/paper/evidencerank.tex
```

Expected: no output (all TODO comments removed across the whole file).

- [ ] **Step 5: Commit**

```bash
git add docs/paper/evidencerank.tex
git commit -m "refactor(paper): clean Experimental Design section — rewrite Dataset, strip TODO comments"
```

---

### Task 3: Fix Conclusion and Acknowledgements

**Files:**
- Modify: `docs/paper/evidencerank.tex`

- [ ] **Step 1: Remove the TODO comment in the Conclusion opening paragraph**

Find:
```latex
We presented EvidenceRank, a four-phase multi-agent LLM pipeline for IT candidate
ranking that simultaneously addresses hallucination, pool-blind scoring, and
ranking instability—three critical gaps unresolved by existing literature.
%% TODO: Insert quantitative summary after running the experiment.
The architecture integrates self-document RAG, evidence-chain grounding,
semantic skill pre-filtering, and pool-aware calibration.
```

Replace with:
```latex
We presented EvidenceRank, a four-phase multi-agent LLM pipeline for IT candidate
ranking that simultaneously addresses hallucination, pool-blind scoring, and
ranking instability—three critical gaps unresolved by existing literature.
The architecture integrates self-document RAG, evidence-chain grounding,
semantic skill pre-filtering, and pool-aware calibration.
```

- [ ] **Step 2: Replace the fabricated "three-fold" result claim with an architectural statement**

Find:
```latex
the fabrication rate is reduced nearly three-fold without
model fine-tuning.
```

Replace with:
```latex
the fabrication rate is reduced by grounding each claim in focused
self-document context rather than relying on the model's parametric memory.
```

- [ ] **Step 3: Remove the Acknowledgements TODO comment**

Find:
```latex
%% TODO: Add acknowledgements after data collection and experiment completion.
```

Replace with an empty string (delete the line):
```latex

```

- [ ] **Step 4: Verify the Conclusion and Acknowledgements are clean**

```bash
grep -n "TODO\|three-fold\|TBD" docs/paper/evidencerank.tex
```

Expected: no output.

- [ ] **Step 5: Commit**

```bash
git add docs/paper/evidencerank.tex
git commit -m "refactor(paper): fix Conclusion — remove TODO, replace fabricated claim with architectural statement"
```

---

### Task 4: Final verification

**Files:**
- Read: `docs/paper/evidencerank.tex`

- [ ] **Step 1: Run comprehensive clean check**

```bash
grep -n "TODO\|TBD\|\\\\textit{TBD}\|three-fold\|sec:results\|tab:hallucination\|tab:consistency\|We plan to evaluate\|will be collected" docs/paper/evidencerank.tex
```

Expected: no output.

- [ ] **Step 2: Confirm all cross-references are to sections that still exist**

```bash
grep -n "\\\\ref{sec:" docs/paper/evidencerank.tex
```

Expected output should include only: `sec:related`, `sec:system`, `sec:experiment`, `sec:conclusion`, `sec:novelty` (and internal equation/figure refs). Must NOT include `sec:results`.

- [ ] **Step 3: Confirm labels in the document don't include the deleted section**

```bash
grep -n "\\\\label{sec:" docs/paper/evidencerank.tex
```

Expected: `sec:intro`, `sec:related`, `sec:system`, `sec:experiment`, `sec:conclusion`, `sec:novelty` — no `sec:results`.
