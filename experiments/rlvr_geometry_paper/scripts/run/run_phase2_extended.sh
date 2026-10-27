#!/bin/bash
# Phase 2 Extended: Atomic Curriculum for 4 New Tasks
# Run on Green cluster pod: sunguha-rlvr-phase2-ext-otgf-worker-0
#
# This script runs all experiments sequentially:
# 1. ARS probes for all 4 tasks
# 2. Direct-RLVR baselines (300 steps each)
# 3. Atomic-Only training (L1 100 steps -> L2 100 steps)
# 4. Atomic-Warm training (continue from Atomic-Only with 100 steps full task)

set -e
cd /scratch/sunguha/rlvr_experiments

TASKS="syllogism knights_knaves maze mini_sudoku"
BASE_MODEL="/scratch/sunguha/sft_qwen_4.0b_instruct"
LOG_DIR="logs/phase2_ext"
mkdir -p $LOG_DIR

echo "=============================================="
echo "Phase 2 Extended - Atomic Curriculum Study"
echo "Tasks: $TASKS"
echo "Started: $(date)"
echo "=============================================="

# ==============================================================================
# Step 1: ARS Probes (zero-shot atomic evaluation on base model)
# ==============================================================================
echo ""
echo "=== Step 1: Running ARS probes ==="

for task in $TASKS; do
    echo "  Probing $task..."
    python experiments/rlvr_geometry_paper/scripts/eval/probe_atomics.py \
        --model_path $BASE_MODEL \
        --task $task \
        --num_samples 50 \
        --seed 9999 \
        --output experiments/rlvr_geometry_paper/outputs/ars_4b_${task}.json \
        2>&1 | tee $LOG_DIR/ars_${task}.log
done

echo "ARS probes complete. Results in experiments/rlvr_geometry_paper/outputs/ars_4b_*.json"

# ==============================================================================
# Step 2: Direct-RLVR Baselines (300 steps GRPO on full task)
# ==============================================================================
echo ""
echo "=== Step 2: Running Direct-RLVR baselines ==="

for task in $TASKS; do
    echo "  Training Direct-RLVR on $task..."
    torchrun --standalone --nproc_per_node=8 train_grpo.py \
        experiments/rlvr_geometry_paper/configs/phase2_extended/${task}/direct_rlvr.yaml \
        2>&1 | tee $LOG_DIR/direct_rlvr_${task}.log
done

echo "Direct-RLVR baselines complete."

# ==============================================================================
# Step 3: Atomic-Only Training (L1 -> L2, no full task)
# ==============================================================================
echo ""
echo "=== Step 3: Running Atomic-Only training ==="

for task in $TASKS; do
    echo "  Stage 1 (L1 atoms) for $task..."
    torchrun --standalone --nproc_per_node=8 train_grpo.py \
        experiments/rlvr_geometry_paper/configs/phase2_extended/${task}/stage1_l1.yaml \
        2>&1 | tee $LOG_DIR/atomic_l1_${task}.log

    echo "  Stage 2 (L2 atoms) for $task..."
    torchrun --standalone --nproc_per_node=8 train_grpo.py \
        experiments/rlvr_geometry_paper/configs/phase2_extended/${task}/stage2_l2.yaml \
        2>&1 | tee $LOG_DIR/atomic_l2_${task}.log
done

echo "Atomic-Only training complete."

# ==============================================================================
# Step 4: Atomic-Warm Training (L1 -> L2 -> Full Task)
# ==============================================================================
echo ""
echo "=== Step 4: Running Atomic-Warm training ==="

for task in $TASKS; do
    echo "  Stage 3 (Full task) for $task..."
    torchrun --standalone --nproc_per_node=8 train_grpo.py \
        experiments/rlvr_geometry_paper/configs/phase2_extended/${task}/stage3_full.yaml \
        2>&1 | tee $LOG_DIR/atomic_warm_${task}.log
done

echo "Atomic-Warm training complete."

# ==============================================================================
# Step 5: Evaluation
# ==============================================================================
echo ""
echo "=== Step 5: Running evaluations ==="

for task in $TASKS; do
    echo "  Evaluating $task checkpoints..."

    # Evaluate base model
    python evaluate_transfer.py \
        --model $BASE_MODEL \
        --tasks $task \
        --num_samples 200 \
        --seed 9999 \
        --output_dir experiments/rlvr_geometry_paper/outputs/eval_phase2_ext/${task}/base \
        2>&1 | tee -a $LOG_DIR/eval_${task}.log

    # Evaluate Direct-RLVR
    python evaluate_transfer.py \
        --model experiments/rlvr_geometry_paper/results/phase2_ext/${task}/direct_rlvr \
        --tasks $task \
        --num_samples 200 \
        --seed 9999 \
        --output_dir experiments/rlvr_geometry_paper/outputs/eval_phase2_ext/${task}/direct_rlvr \
        2>&1 | tee -a $LOG_DIR/eval_${task}.log

    # Evaluate Atomic-Only (stage2_l2 checkpoint on full task)
    python evaluate_transfer.py \
        --model experiments/rlvr_geometry_paper/results/phase2_ext/${task}/stage2_l2 \
        --tasks $task \
        --num_samples 200 \
        --seed 9999 \
        --output_dir experiments/rlvr_geometry_paper/outputs/eval_phase2_ext/${task}/atomic_only \
        2>&1 | tee -a $LOG_DIR/eval_${task}.log

    # Evaluate Atomic-Warm (stage3_full checkpoint)
    python evaluate_transfer.py \
        --model experiments/rlvr_geometry_paper/results/phase2_ext/${task}/stage3_full \
        --tasks $task \
        --num_samples 200 \
        --seed 9999 \
        --output_dir experiments/rlvr_geometry_paper/outputs/eval_phase2_ext/${task}/atomic_warm \
        2>&1 | tee -a $LOG_DIR/eval_${task}.log
done

echo ""
echo "=============================================="
echo "Phase 2 Extended Complete!"
echo "Finished: $(date)"
echo "=============================================="
echo ""
echo "Results summary:"
for task in $TASKS; do
    echo "  $task:"
    for model in base direct_rlvr atomic_only atomic_warm; do
        if [ -f "experiments/rlvr_geometry_paper/outputs/eval_phase2_ext/${task}/${model}/metrics_${task}.json" ]; then
            acc=$(python -c "import json; print(json.load(open('experiments/rlvr_geometry_paper/outputs/eval_phase2_ext/${task}/${model}/metrics_${task}.json'))['accuracy'])" 2>/dev/null || echo "N/A")
            echo "    $model: $acc"
        fi
    done
done
