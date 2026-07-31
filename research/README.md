# Research Paper — Retail Demand Forecasting System

This folder contains an IEEE conference-format research paper describing the
retail forecasting system implemented in `../backend` and `../frontend`.

## Contents

```
research/
├── main.tex                Thin wrapper (\input{paper.tex}) — see note below
├── paper.tex               Actual document content (\input's every section)
├── references.bib          BibTeX bibliography (37 entries, all cited)
├── sections/               One .tex file per paper section
├── figures/                TikZ diagrams (architecture, pipeline, results charts)
├── tables/                 Table snippets included from sections/
├── images/                 (reserved for raster images, currently empty —
│                            every figure in this paper is a native TikZ
│                            diagram, so no external image files are required)
└── README.md               This file
```

### Why both `main.tex` and `paper.tex` exist
Overleaf's default compiler looks for a file literally named `main.tex`.
The actual paper content lives in `paper.tex` (its name per the original
project spec); `main.tex` is a one-line wrapper (`\input{paper.tex}`) so an
Overleaf project compiles immediately without needing to change the
"Main document" project setting. If you'd rather compile `paper.tex`
directly (e.g. locally, or by explicitly setting it as Overleaf's main
document via the project menu), that works identically — both approaches
compile the same content.

## How to compile

Requires a TeX distribution that includes `IEEEtran.cls` and `pgf/tikz`
(e.g. TeX Live, MacTeX, or MiKTeX).

```bash
cd research
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

(Substitute `paper` for `main` in all four commands if compiling
`paper.tex` directly instead.) Two `pdflatex` passes after `bibtex` are
required for citations, cross-references, and the bibliography to all
resolve correctly.

## Important note on verification

**No LaTeX distribution (`pdflatex`/`bibtex`) was available in the sandboxed
environment used to write this paper.** Every `.tex` file was written and
manually checked for internal consistency — every `\label` has a
corresponding `\ref`/`\eqref`, every `\cite` key exists in `references.bib`
and every `references.bib` entry is cited at least once, brace and
`\begin`/`\end` pairs are balanced, and table row column counts match their
declared column specifications — but **actual compilation with `pdflatex`
and `bibtex` has not been run or confirmed**. Please compile locally using
the commands above and report back (or open an issue) if any LaTeX errors
occur; they can be fixed directly in the corresponding `sections/*.tex` or
`figures/*.tex` file.

## Content accuracy

Every claim in this paper about what is implemented (the 19 registered
models, the composite scoring formula, hyperparameters, the absence of a
database, the absence of SHAP, etc.) was verified directly against the
source code in `../backend` before being written, not assumed from a
generic template. All figures reporting numeric results (Table IV / the
model comparison table, the RMSE and feature-importance charts) use **real
output** from running the actual system's `/upload` and `/forecast`
endpoints end-to-end on a real retail transaction dataset — no result in
this paper is fabricated or illustrative placeholder data. The specific
dataset, run configuration, and raw JSON response used to produce these
numbers are described in Section V (Dataset Description) and Section X
(Results).

One deliberate deviation from a generic forecasting-paper template is worth
flagging explicitly: the original request assumed a SHAP-based
explainability section. The codebase does not use SHAP anywhere — the real
explainability mechanisms are (1) native XGBoost/Random Forest feature
importance and (2) a retrieval-augmented LLM explanation layer (chatbot +
AI Insights Report). Section VII (Explainable AI) documents these two real
mechanisms instead of fabricating a SHAP integration that does not exist,
and states this substitution explicitly.

## Author information

The `\author{}` block in `paper.tex` is filled in with the four authors
(Ranjith Kumar N, Pooja Ammalajeri, Gayathri M, Preetham Kumar S),
Department of Computer Science and Engineering, PES University, Bengaluru,
and project guide Dr. Jyothi R. No placeholder values remain.
