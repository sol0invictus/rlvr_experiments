#!/bin/bash
set -euo pipefail
cd /scratch/sunguha/rlvr_experiments

EVAL_SAMPLES=50
EVAL_SEED=9999
R=experiments/rlvr_geometry_paper/results/phase2_countdown

echo "============================================"
echo "PHASE 2: COUNTDOWN — Atomic Curriculum + Direct RLVR"
echo "============================================"

# --- BASE EVAL ---
echo ">>> BASE EVAL"
torchrun --standalone --nproc_per_node=8 evaluate_transfer.py \
    --model /scratch/sunguha/sft_qwen_4.0b_instruct \
    --tasks countdown --num_samples $EVAL_SAMPLES --seed $EVAL_SEED \
    --output_dir $R/base_eval
echo ""

# --- DIRECT RLVR (300 steps) ---
echo ">>> DIRECT RLVR (300 steps on full countdown)"
torchrun --standalone --nproc_per_node=8 train_grpo.py experiments/rlvr_geometry_paper/configs/phase2_countdown/direct_rlvr_4b.yaml
echo ">>> DIRECT RLVR EVAL"
torchrun --standalone --nproc_per_node=8 evaluate_transfer.py \
    --model $R/direct_rlvr --tasks countdown \
    --num_samples $EVAL_SAMPLES --seed $EVAL_SEED \
    --output_dir $R/direct_rlvr_eval
echo ""

# --- STAGE 1: L1 atoms (100 steps) ---
echo ">>> STAGE 1: L1 atoms"
torchrun --standalone --nproc_per_node=8 train_grpo.py experiments/rlvr_geometry_paper/configs/phase2_countdown/stage1_l1_atoms.yaml
echo ">>> STAGE 1 EVAL"
torchrun --standalone --nproc_per_node=8 evaluate_transfer.py \
    --model $R/stage1_l1 --tasks countdown \
    --num_samples $EVAL_SAMPLES --seed $EVAL_SEED \
    --output_dir $R/stage1_eval
echo ""

# --- STAGE 2: L2 atoms (100 steps, init from stage1) ---
echo ">>> STAGE 2: L2 atoms"
torchrun --standalone --nproc_per_node=8 train_grpo.py experiments/rlvr_geometry_paper/configs/phase2_countdown/stage2_l2_atoms.yaml
echo ">>> STAGE 2 EVAL (= Atomic-Only)"
torchrun --standalone --nproc_per_node=8 evaluate_transfer.py \
    --model $R/stage2_l2 --tasks countdown \
    --num_samples $EVAL_SAMPLES --seed $EVAL_SEED \
    --output_dir $R/stage2_eval
echo ""

# --- STAGE 3: Full task (100 steps, init from stage2) ---
echo ">>> STAGE 3: Full countdown task"
torchrun --standalone --nproc_per_node=8 train_grpo.py experiments/rlvr_geometry_paper/configs/phase2_countdown/stage3_full_task.yaml
echo ">>> STAGE 3 EVAL (= Atomic-Warm)"
torchrun --standalone --nproc_per_node=8 evaluate_transfer.py \
    --model $R/stage3_full --tasks countdown \
    --num_samples $EVAL_SAMPLES --seed $EVAL_SEED \
    --output_dir $R/stage3_eval
echo ""

# --- SUMMARY ---
echo "============================================"
echo "COUNTDOWN RESULTS"
echo "============================================"
for d in base_eval direct_rlvr_eval stage1_eval stage2_eval stage3_eval; do
    f=$(ls -t $R/$d/metrics_*.json 2>/dev/null | head -1)
    if [ -n "$f" ]; then
        acc=$(python3 -c "import json; d=json.load(open('$f')); print(f\"{d['tasks']['countdown']['accuracy']*100:.0f}%\")")
        echo "  $d: $acc"
    fi
done
echo "============================================"
