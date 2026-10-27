# Experiments

Each folder is one self-contained study built on the shared framework at the repo root
(`train_grpo.py`, `train_sft.py`, `evaluate*.py`, `environments/`, `callbacks/`, `data_gen/`).

| Folder | What it is | Status |
|---|---|---|
| `rlvr_geometry_paper/` | Four-phase study behind the paper *"The Geometry and Transfer of RLVR in Small Reasoning Models"*: cross-task transfer matrix, atomic curriculum (ARS), weight-space geometry, task-vector merging. Includes the LaTeX paper. | Complete; paper in pre-submission |
| `countdown_lora/` | LoRA + GRPO on countdown with Qwen3-4B-Instruct-2507 on a single RTX 3090 Ti (vLLM rollouts + FlashAttention-2). | Smoke-tested; full run not yet done (see `notes/`) |
| `cer_project/` | Competence-Edge Routing: per-token SFT/RL loss routing, advantage-gated GRPO variants. | See its `README.md` |

## Conventions

- **Run everything from the repo root**, e.g.
  `python train_grpo.py experiments/countdown_lora/config.yaml` or
  `bash experiments/rlvr_geometry_paper/scripts/run/run_phase2_countdown.sh`.
  All relative paths in configs and scripts are written against the repo root.
- Scripts in subfolders import the root packages (`environments`, `callbacks`, `data_gen`), which
  needs the repo installed in editable mode once per environment:
  `uv pip install --python .venv-vllm/bin/python --no-deps -e .` (or `pip install -e .`).
- Each experiment writes checkpoints and logs to its own `results/` / `outputs/`, which are
  gitignored. Small analysis results the paper depends on are whitelisted in `.gitignore`.
- Absolute `/scratch/...` and `/code-fsx/...` paths are the cluster checkouts the runs used;
  checkpoint paths there are kept as-run.

## rlvr_geometry_paper layout

```
paper/               everything paper-specific: LaTeX, figures/, figure + paper-analysis scripts/,
                     drafts/, data/ (see paper/CLAUDE.md for build + editing rules)
reports/             per-phase reports + MASTER_RLVR_dynamics.md: source of truth for every number
configs/             phase2_*/ and transfer_*/ training configs, plus pilot configs and eval/
configs_by_section/  frozen snapshot of the configs grouped by paper section (MANIFEST.csv maps
                     each to its current path under configs/)
scripts/analysis     weight-space analysis (analyze_*, dump_*, visualize_*, ...)
scripts/eval         evaluation (eval_*, probe_atomics, explore_*)
scripts/merge        task-vector merging (merge_*)
scripts/data         SFT/RFT data generation (generate_*)
scripts/debug        one-off debugging scripts
scripts/run          shell pipelines that chain training + eval (run_*.sh)
results/, outputs/   run artifacts (gitignored except weight_analysis/ and transfer_matrix*/)
```
