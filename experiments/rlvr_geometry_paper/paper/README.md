# The Geometry and Transfer of RLVR in Small Reasoning Models — TMLR / arXiv draft

Four-phase paper combining Phase 1 (transfer matrix), Phase 2 (ARS / atomic curriculum),
Phase 3 (weight geometry), Phase 4 (task-vector merging).

## Files
- `main.tex` — full draft, uses the **official TMLR template**
- `tmlr.sty`, `tmlr.bst`, `fancyhdr.sty` — official TMLR style files (from JmlrOrg/tmlr-style-file)
- `references.bib` — bibliography (verify entries before submission)
- `figures/` — every figure the paper uses (PDF, plus PNG previews for the `fig4_*`/`fig5_*` panels)
- `scripts/make_figures.py` — regenerates the Phase-2/3 panels (`fig4_*`, `fig5_*`)
- `scripts/make_extra_figures.py` — regenerates the Phase-1 transfer heatmap, Phase-4 panels and `fig_p2_sp`/`fig_p3_pertask_depth`
- `scripts/fix_emdashes.py` — one-off script that replaced em-dashes with contextual punctuation
- `scripts/` (other) — analysis behind the paper (ARS linking, bootstrap / held-out predictor, merge baselines) and their GPU pipelines; `scripts/local/` holds local variants of the linking/merge streams
- `drafts/` — per-phase section drafts and `results_summary.md`
- `data/` — collected analysis outputs read by the scripts (`analysis/`, `ars/`, `linking/`)

## Build
```bash
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

## TMLR template modes
Set the class option on `\usepackage{...}{tmlr}` in the preamble:
- `\usepackage{tmlr}` — submission (anonymous, "Under review" banner)
- `\usepackage[preprint]{tmlr}` — **arXiv preprint** (named authors, no banner) ← current
- `\usepackage[accepted]{tmlr}` — camera-ready (named authors + OpenReview line)

## Status
Complete draft on the TMLR template: 32 pages including appendices, compiles clean (no errors, no undefined refs),
author-year citations. All six appendices (A–F) filled with real data; metric definitions in
Appendix G; all 10 figures present. No em-dashes (AI-tell removed).

### Remaining before submission
- Fill in author emails / Chaitanya's affiliation.
- Verify every `references.bib` entry against the canonical source (entries are standard but unchecked).
- For camera-ready, switch to `[accepted]` and set `\openreview{...}`.
- Optional polish: a few cosmetic overfull-hbox warnings on wide tables.
- Spot-check numbers against `../reports/report_*.md` once more before posting.
