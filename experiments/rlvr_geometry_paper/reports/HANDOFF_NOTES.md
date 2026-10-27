# RLVR Phase 2 — Handoff Notes

**Date:** 2026-04-23
**Pod:** `sunguha-rlvr-phase2-qbgw` (namespace `green`, Green cluster P6-B200 8xGPU)
**Status:** Pod is RUNNING and idle. All experiments complete.

---

## Active Pod

```bash
# Update kubeconfig
kraken -p green jobs update-kubeconfig --job-name sunguha-rlvr-phase2-qbgw

# Connect
kraken -p green jobs connect --job-name sunguha-rlvr-phase2-qbgw

# kubectl exec
kubectl -n green exec -it sunguha-rlvr-phase2-qbgw-worker-0 -- bash

# Working directory on pod
cd /scratch/sunguha/rlvr_experiments
```

---

## What's Done

All code is at `/Users/sunguha/llm3/aiCoder/src/rlvr_experiments/` (local) and `/scratch/sunguha/rlvr_experiments/` (pod).

### Best Results

| Approach | SP | Arith | Count | Description |
|----------|-----|-------|-------|-------------|
| Base (SFT) | 10% | 72% | 26% | No GRPO |
| Direct-RLVR | 44% | 32% | ~70% | Phase 1 |
| **RFT-Base** | **55.5%** | 70% | 32% | Best single-task SP model |
| **2-task merge (SP+Arith)** | **56%** | **76%** | **40%** | Best multi-task (no training) |
| 9-task merge (scale=0.33) | 45.5% | 72.5% | 44.5% | All tasks, diluted |

### Key Checkpoints on Pod

| Path | Description |
|------|-------------|
| `/scratch/sunguha/sft_qwen_4.0b_instruct` | Base SFT model (Qwen3-4B) |
| `results/phase2_sp/idea_rft_base/grpo` | RFT-Base 55.5% SP model |
| `results/phase2_arith/stage2_l2` | Atomic-Only 78% arith model |
| `results/phase2_sp/idea_b/stage2_full` | Idea B 51.5% SP model |
| `results/phase2_sp/merged/basic` | 2-task merge (SP+Arith, τ_A+τ_B) |
| `results/phase2_sp/merged/all_tasks_s0.33` | 9-task merge at scale=0.33 |
| `results/phase2_sp/idea_rft/rft_data.jsonl` | 328 verified-correct RFT samples |

### Key Scripts on Pod

| Script | Purpose |
|--------|---------|
| `merge_task_vectors.py` | Merge 2 models with multiple strategies |
| `merge_all_tasks.py` | Merge N models with uniform scaling |
| `generate_rft_data.py` | On-policy rejection sampling (8 GPU, file-based gather) |
| `analyze_idea_b_errors.py` | Error analysis (false-infeasible breakdown) |
| `eval_best_of_n_sp.py` | Best-of-N evaluation |
| `evaluate_transfer.py` | Standard multi-task eval |
| `train_grpo.py` | GRPO training (supports multiple environments) |
| `train_sft.py` | SFT training |

---

## What to Run Next

### Option 1: Per-Task Scaled Merge

Instead of uniform scale=0.33 for all 9 tasks, give higher weight to SP and arith:

```bash
# Modify merge_all_tasks.py to accept per-task scales, e.g.:
# SP=1.0, Arith=1.0, Countdown=0.5, others=0.2
# This should preserve the 2-task merge quality while adding countdown benefit
```

### Option 2: Iterative RFT (STaR-style)

RFT-Base hit 55.5%. Run another round:
1. Generate RFT data from the 55.5% checkpoint (it should have higher pass rate → more frontier samples)
2. SFT base on the new data
3. GRPO 300 steps
4. Repeat

```bash
cd /scratch/sunguha/rlvr_experiments
# Generate new RFT data from the 55.5% model
torchrun --nproc_per_node=8 generate_rft_data.py \
  --model results/phase2_sp/idea_rft_base/grpo \
  --num_problems 300 --k 8 --seed 8888 \
  --output results/phase2_sp/idea_rft_r2/rft_data.jsonl
# Then SFT base on it + GRPO 300 steps
```

### Option 3: Dense Step-Level GRPO Reward

Change reward function to give partial credit per correct path step (not binary 0/1). The 25 wrong-path errors from error analysis are often partially correct — giving them gradient signal should help.

---

## Reports

| File | Content |
|------|---------|
| `report_atomic_curriculum.md` | Full Phase 2 report with all experiments A-RFT |
| `report_transfer_matrix.md` | Phase 1 transfer matrix |
| `report_phase3_task_distances.md` | Phase 3 weight analysis |
| `current_projects/rlvr-phase2-atomic-curriculum.md` | Project tracker |

---

## Useful Commands

```bash
# Eval a model on multiple tasks
torchrun --nproc_per_node=8 evaluate_transfer.py \
  --model <path> --tasks shortest_path,basic_arithmetic,countdown \
  --num_samples 200 --seed 9999 --batch_size 8 --output_dir <eval_dir>

# Merge 2 task vectors
python3 merge_task_vectors.py \
  --base /scratch/sunguha/sft_qwen_4.0b_instruct \
  --model_a <model_a_path> --model_b <model_b_path> \
  --output_dir <merge_dir> --strategy basic

# Generate RFT data (8 GPU, file-based gather)
torchrun --nproc_per_node=8 generate_rft_data.py \
  --model <checkpoint> --num_problems 300 --k 8 --seed 7777 \
  --output <output.jsonl>

# GRPO training
torchrun --nproc_per_node=8 train_grpo.py configs/phase2_sp_ideas/<config>.yaml

# SFT training
torchrun --nproc_per_node=8 train_sft.py configs/phase2_sp_ideas/<config>.yaml
```
