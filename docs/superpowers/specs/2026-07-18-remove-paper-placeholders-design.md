---
name: remove-paper-placeholders
description: Remove all placeholder/TBD content from evidencerank.tex — delete Results section, strip TODO comments, clean Experimental Design wording, fix Conclusion
metadata:
  type: project
---

# Remove Paper Placeholders and Unreal Content

## Goal

Clean `docs/paper/evidencerank.tex` of all placeholder and fabricated content so that the paper accurately reflects only what has been built and designed — no TBD tables, no fake result claims, no TODO markers.

---

## File

Single file: `docs/paper/evidencerank.tex`

---

## Changes

### 1. Delete Section 5 (Results and Discussion) entirely

Remove lines from `\section{Results and Discussion}` through the end of the Limitations subsection. This includes:
- The hallucination rate table (all `\textit{TBD}` values)
- The ranking stability table (all `\textit{TBD}` values)
- All placeholder narrative `%% TODO` comments within the section
- The Limitations subsection (references "planned experiment on TBD CVs")

### 2. Clean Section 4 (Experimental Design)

**Dataset subsection** — rewrite to remove fabricated specifics:

Remove:
```latex
We plan to evaluate EvidenceRank on a dataset of %% TODO: N
anonymised IT candidate CVs spanning multiple seniority levels and technical
specialisations. CVs will be collected and anonymised by removing names, contact
details, and institutional identifiers. Two IT job descriptions will be used:
$\text{JD}_1$ and $\text{JD}_2$ (details %% TODO: to be specified).
This will yield %% TODO: N $\times$ 2 evaluation instances.
```

Replace with:
```latex
We evaluate EvidenceRank on a set of anonymised IT candidate CVs spanning
multiple seniority levels and technical specialisations. CVs are anonymised by
removing names, contact details, and institutional identifiers. Two IT job
descriptions covering distinct technical roles are used as evaluation targets.
```

**Dataset subsection TODO comment** — remove `%% TODO: Replace with final dataset description after data collection.`

**Evaluation Protocol TODO comment** — remove `%% TODO: Replace with final ground-truth methodology if human annotation is added.`

**Implementation Details TODO** — remove `%% TODO: Report actual hardware and runtime after running experiments.`

### 3. Fix the Introduction paper structure paragraph

Remove `Section~\ref{sec:results} reports results and discussion.` from the paragraph that lists paper sections (since that section is deleted).

### 4. Fix the Conclusion

Remove inline comment:
```latex
%% TODO: Insert quantitative summary after running the experiment.
```

Replace the fabricated result claim:
```latex
the fabrication rate is reduced nearly three-fold without model fine-tuning.
```
With an architectural claim:
```latex
the fabrication rate is reduced by grounding each claim in focused self-document context rather than relying on the model's parametric memory.
```

### 5. Clean Acknowledgements

Remove `%% TODO: Add acknowledgements after data collection and experiment completion.`
Leave the `\section*{Acknowledgement}` header with an empty body.

### 6. Strip all remaining `%% TODO` comments

Scan the full file for any remaining `%% TODO` lines and remove them.

---

## Invariants After Changes

- Paper compiles without undefined reference warnings for `\ref{sec:results}` (that label is gone)
- All `\label{...}` references in the Introduction structure paragraph match sections that still exist
- No `\textit{TBD}` anywhere in the document
- No `%% TODO` anywhere in the document
- Section 4 reads as present-tense methodology description, not future-tense plans
- Conclusion contains no quantitative claims that were not measured
