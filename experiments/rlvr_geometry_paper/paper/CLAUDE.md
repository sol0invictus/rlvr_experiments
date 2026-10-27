# CLAUDE.md — paper/

Guidance for working on the paper **"The Geometry and Transfer of RLVR in Small Reasoning Models"**
(TMLR + arXiv). Everything paper-specific lives in this folder: the LaTeX source, every figure, the
scripts that make figures and paper-level analyses, section drafts, and the analysis data those
scripts read.

The study that produced the results is the parent folder (`experiments/rlvr_geometry_paper/`:
training configs, run/eval/analysis scripts, per-phase reports). The training/eval framework is at
the repo root; read the root `CLAUDE.md` before touching training/eval code.

## Layout

```
main.tex, references.bib       the paper
tmlr.sty, tmlr.bst, fancyhdr.sty   official TMLR template files
main.pdf, main.bbl             build outputs, tracked (.bbl is needed for arXiv)
figures/                       the ONLY figure directory (\graphicspath{{figures/}})
scripts/make_figures.py        Phase-2/3 panels (fig4_*, fig5_*) -> figures/
scripts/make_extra_figures.py  Phase-1 heatmap, Phase-4 panels, fig_p2_sp, fig_p3_pertask_depth -> figures/
scripts/fix_emdashes.py        one-off em-dash cleanup of main.tex
scripts/*.py, scripts/*.sh     paper-level analyses: ARS vs merge/sign-agreement, bootstrap and
                               held-out predictor, replicate merges, TIES/DARE baselines, and the
                               GPU pipelines that ran them
scripts/local/                 local variants of the linking / merge / attn-kvo streams
drafts/                        per-phase section drafts + results_summary.md (from consolidate.py)
data/{analysis,ars,linking}/   analysis outputs the scripts write and read
```

## Building

Run from `paper/` (TeX Live is installed system-wide; matplotlib/numpy available for figures):

```bash
pdflatex main && bibtex main && pdflatex main && pdflatex main   # -> main.pdf
python3 scripts/make_extra_figures.py                            # regenerate extra figures
python3 scripts/make_figures.py                                  # regenerate fig4_*/fig5_* panels
```

The figure scripts resolve paths from their own location, so they work from any directory.
`make_extra_figures.py` reads `../results/weight_analysis/{scale}/` (the study's tracked
weight-geometry results); other data is transcribed inline from the reports.

The analysis scripts use `BASE` = the experiment folder (`experiments/rlvr_geometry_paper/`; a
hardcoded cluster checkout when it exists) and read/write `{BASE}/paper/data/...`. The `.sh`
pipelines are run from the repo root. Absolute `/code-fsx/...` / `/scratch/...` paths are the
cluster checkouts the analyses ran on.

Non-obvious facts that will bite you otherwise:

- **Official TMLR template.** `main.tex` uses `\usepackage[preprint]{tmlr}`. Modes are switched via
  the class option: `{tmlr}` = anonymous submission, `[preprint]` = arXiv (current), `[accepted]` =
  camera-ready (also needs `\openreview{...}`). `tmlr.sty` already loads natbib, fontenc, lmodern,
  fancyhdr, and page geometry. Do **not** re-add those packages, and citations are **author-year**
  (use `\citep`/`\citet`, not numeric).
- **One figure directory.** Figure scripts write straight into `figures/`; there is no separate
  staging copy. Before the 2026-10-26 consolidation there were two, and the paper was silently
  picking up a stale `fig5_6` (single-task) that did not match its caption; the current one is the
  four-task Qwen2.5/Qwen3 depth-migration panel.
- **No em-dashes by design.** Keep new prose em-dash-free (use commas / colons / parentheses /
  sentence breaks).
- LaTeX build intermediates (`*.aux`, `*.log`, `*.out`, `*.blg`, ...) are gitignored.

## Paper structure and where numbers come from

The paper is organized as four "phases". Every number in `main.tex` is transcribed from the
authoritative reports in `../reports/`. **When editing results, reconcile against these, not
from memory:**

| Paper phase | Section | Source report(s) in `../reports/` |
|---|---|---|
| Phase 1: cross-task transfer matrix | §5 | `report_transfer_matrix.md` |
| Phase 2: ARS / atomic curriculum | §4 | `report_phase2_complete.md`, `report_atomic_curriculum.md` |
| Phase 3: weight-space geometry | §6 | `report_phase3_task_distances.md` |
| Phase 4: task-vector merging | §7 | `report_phase4_merge_scaling.md` |

`../reports/MASTER_RLVR_dynamics.md` is the consolidated source of truth across all phases
(includes a paper-section mapping table). Section drafts are in `drafts/`. The configs behind each
paper section, with provenance headers, are in `../configs_by_section/`.

The paper's framing is the **"Path A" honest empirical characterization**: it foregrounds the
robust findings *and* the n=19 null-predictor results. Two predictor claims (sign-agreement
r=−0.81, L1-ARS r=−0.76) were **falsified at larger n** and are presented as negatives. Do not
reintroduce them as positive results. Likewise, Phase-1 *sequential-transfer* and Phase-4
*weight-merging* are different operations and do not predict each other (Pearson r ≈ 0); keep that
distinction.

## Conventions

- Appendices A–F hold the full data tables; Appendix G defines the weight-space metrics referenced
  in §6. Keep heavy per-pair / per-scale tables in appendices, summaries in the body.
- Section titles are descriptive and phase-free (e.g., "The Cross-Task Transfer Matrix"); the
  "four phases" survive only as a conceptual framing in prose and as internal `\label{sec:phaseN}`
  anchors (kept so the many `\S\ref{sec:phaseN}` cross-references resolve to section numbers). Do not
  reintroduce `Phase N:` title prefixes.
- `README.md` documents build + template modes and the remaining pre-submission TODOs
  (author emails, `references.bib` verification, camera-ready switch).
