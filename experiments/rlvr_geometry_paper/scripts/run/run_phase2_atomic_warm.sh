#!/bin/bash
# Phase 2: Atomic-Warm curriculum for basic_arithmetic
# L1 atoms (100 steps) → L2 atoms (100 steps) → Full task (100 steps)
# Eval after each stage on: full task + all atoms (to measure AFR)
#
# Usage: bash experiments/rlvr_geometry_paper/scripts/run/run_phase2_atomic_warm.sh
# Requires: 8 GPUs (torchrun DDP)

set -euo pipefail
cd /scratch/sunguha/rlvr_experiments

EVAL_SAMPLES=50
EVAL_SEED=9999
RESULTS_DIR=experiments/rlvr_geometry_paper/results/phase2_arith

echo "============================================"
echo "PHASE 2: ATOMIC-WARM CURRICULUM"
echo "basic_arithmetic | Qwen3-4B | 300 total steps"
echo "============================================"

# --------------------------------------------------------
# STAGE 1: L1 Grounding Atoms (100 steps)
# --------------------------------------------------------
echo ""
echo ">>> STAGE 1: L1 Atoms (number_extraction + term_count + operator_identification)"
echo "    Model: SFT base → experiments/rlvr_geometry_paper/results/phase2_arith/stage1_l1"
echo ""

torchrun --standalone --nproc_per_node=8 \
    train_grpo.py experiments/rlvr_geometry_paper/configs/phase2_arithmetic/stage1_l1_atoms.yaml

echo ""
echo ">>> STAGE 1 EVAL: Full task + atoms"
python3 experiments/rlvr_geometry_paper/scripts/eval/probe_atomics.py \
    --model_path $RESULTS_DIR/stage1_l1 \
    --num_samples $EVAL_SAMPLES \
    --seed $EVAL_SEED \
    --output $RESULTS_DIR/stage1_l1/ars_post_l1.json

echo ">>> STAGE 1: Eval on full basic_arithmetic task"
python3 evaluate_transfer.py \
    --model $RESULTS_DIR/stage1_l1 \
    --tasks basic_arithmetic \
    --num_samples $EVAL_SAMPLES \
    --seed $EVAL_SEED \
    --output_dir $RESULTS_DIR/stage1_l1/eval_full

# --------------------------------------------------------
# STAGE 2: L2 Local Reasoning Atoms (100 steps)
# --------------------------------------------------------
echo ""
echo ">>> STAGE 2: L2 Atoms (order_of_operations + single_op_eval)"
echo "    Model: stage1_l1 → experiments/rlvr_geometry_paper/results/phase2_arith/stage2_l2"
echo ""

torchrun --standalone --nproc_per_node=8 \
    train_grpo.py experiments/rlvr_geometry_paper/configs/phase2_arithmetic/stage2_l2_atoms.yaml

echo ""
echo ">>> STAGE 2 EVAL: Full task + atoms"
python3 experiments/rlvr_geometry_paper/scripts/eval/probe_atomics.py \
    --model_path $RESULTS_DIR/stage2_l2 \
    --num_samples $EVAL_SAMPLES \
    --seed $EVAL_SEED \
    --output $RESULTS_DIR/stage2_l2/ars_post_l2.json

echo ">>> STAGE 2: Eval on full basic_arithmetic task"
python3 evaluate_transfer.py \
    --model_path $RESULTS_DIR/stage2_l2 \
    --tasks basic_arithmetic \
    --num_samples $EVAL_SAMPLES \
    --seed $EVAL_SEED \
    --output_dir $RESULTS_DIR/stage2_l2/eval_full

# --------------------------------------------------------
# STAGE 3: Full Task (100 steps)
# --------------------------------------------------------
echo ""
echo ">>> STAGE 3: Full basic_arithmetic task"
echo "    Model: stage2_l2 → experiments/rlvr_geometry_paper/results/phase2_arith/stage3_full"
echo ""

torchrun --standalone --nproc_per_node=8 \
    train_grpo.py experiments/rlvr_geometry_paper/configs/phase2_arithmetic/stage3_full_task.yaml

echo ""
echo ">>> STAGE 3 EVAL: Full task + atoms (AFR measurement)"
python3 experiments/rlvr_geometry_paper/scripts/eval/probe_atomics.py \
    --model_path $RESULTS_DIR/stage3_full \
    --num_samples $EVAL_SAMPLES \
    --seed $EVAL_SEED \
    --output $RESULTS_DIR/stage3_full/ars_post_full.json

echo ">>> STAGE 3: Eval on full basic_arithmetic task"
python3 evaluate_transfer.py \
    --model_path $RESULTS_DIR/stage3_full \
    --tasks basic_arithmetic \
    --num_samples $EVAL_SAMPLES \
    --seed $EVAL_SEED \
    --output_dir $RESULTS_DIR/stage3_full/eval_full

# --------------------------------------------------------
# SUMMARY
# --------------------------------------------------------
echo ""
echo "============================================"
echo "PHASE 2 COMPLETE — SUMMARY"
echo "============================================"
echo "Phase 1 reference: Base=62%, Direct-RLVR=50%"
echo ""
echo "Stage 1 (L1 atoms):  see $RESULTS_DIR/stage1_l1/eval_full/"
echo "Stage 2 (L2 atoms):  see $RESULTS_DIR/stage2_l2/eval_full/"
echo "Stage 3 (full task):  see $RESULTS_DIR/stage3_full/eval_full/"
echo ""
echo "ARS progression:"
echo "  Base:     see experiments/rlvr_geometry_paper/outputs/ars_4b_arithmetic_v2.json"
echo "  Post-L1:  see $RESULTS_DIR/stage1_l1/ars_post_l1.json"
echo "  Post-L2:  see $RESULTS_DIR/stage2_l2/ars_post_l2.json"
echo "  Post-Full: see $RESULTS_DIR/stage3_full/ars_post_full.json"
echo "============================================"
