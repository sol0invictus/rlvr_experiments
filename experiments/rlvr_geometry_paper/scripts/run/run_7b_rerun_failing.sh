#!/bin/bash
# ============================================================================
# Rerun failing 7B GRPO tasks with corrected configs
# ============================================================================
# Fixes: countdown had harder task_params than the 4B run
# Also reruns: tower_of_hanoi (0% same-task), basic_arithmetic (reward collapsed)
#
# Usage: nohup bash experiments/rlvr_geometry_paper/scripts/run/run_7b_rerun_failing.sh > 7b_rerun_failing.log 2>&1 &
# ============================================================================

set -e
cd /scratch/sunguha/rlvr_experiments

CONFIGS_DIR="experiments/rlvr_geometry_paper/configs/transfer_7b"
TASKS_TO_RERUN=(countdown tower_of_hanoi basic_arithmetic)

echo "============================================"
echo "7B RERUN FAILING TASKS"
echo "Started: $(date)"
echo "Tasks: ${TASKS_TO_RERUN[*]}"
echo "============================================"

# ──────────────────────────────────────────────
# STEP 1: Remove old failing checkpoints
# ──────────────────────────────────────────────
for task in "${TASKS_TO_RERUN[@]}"; do
    OLD_DIR="experiments/rlvr_geometry_paper/results/grpo_qwen25_7b_${task}"
    if [ -d "${OLD_DIR}" ]; then
        echo "Removing old checkpoint: ${OLD_DIR}"
        rm -rf "${OLD_DIR}"
    fi
done

# ──────────────────────────────────────────────
# STEP 2: GRPO Training
# ──────────────────────────────────────────────
for task in "${TASKS_TO_RERUN[@]}"; do
    OUTPUT_DIR="experiments/rlvr_geometry_paper/results/grpo_qwen25_7b_${task}"
    LOG_FILE="grpo_qwen25_7b_${task}_rerun.log"

    echo ""
    echo "--------------------------------------------"
    echo "Training: ${task}"
    echo "Config: ${CONFIGS_DIR}/train_${task}.yaml"
    echo "Output: ${OUTPUT_DIR}"
    echo "Started: $(date)"
    echo "--------------------------------------------"

    torchrun --standalone --nproc_per_node=8 \
        train_grpo.py ${CONFIGS_DIR}/train_${task}.yaml \
        2>&1 | tee ${LOG_FILE}

    echo "Completed: ${task} at $(date)"
done

echo ""
echo "============================================"
echo "ALL RERUN TRAINING COMPLETE at $(date)"
echo "============================================"

# ──────────────────────────────────────────────
# STEP 3: Re-evaluate ALL 7 models (including reruns)
# ──────────────────────────────────────────────
echo ""
echo "============================================"
echo "Re-evaluating transfer matrix"
echo "Started: $(date)"
echo "============================================"

ALL_TASKS=(countdown maze mini_sudoku tower_of_hanoi shortest_path basic_arithmetic syllogism)
EVAL_TASKS="countdown,maze,mini_sudoku,tower_of_hanoi,shortest_path,basic_arithmetic,syllogism"
EVAL_OUTPUT_DIR="experiments/rlvr_geometry_paper/outputs/transfer_matrix_7b"

# Only re-eval the models we retrained
for task in "${TASKS_TO_RERUN[@]}"; do
    MODEL_DIR="experiments/rlvr_geometry_paper/results/grpo_qwen25_7b_${task}"
    MODEL_NAME="grpo_qwen25_7b_${task}"

    # Remove old eval results
    rm -rf "${EVAL_OUTPUT_DIR}/${MODEL_NAME}"

    echo ""
    echo "Evaluating: ${MODEL_NAME} on ALL tasks"
    echo "Started: $(date)"

    python evaluate_transfer.py \
        --model "${MODEL_DIR}" \
        --tasks "${EVAL_TASKS}" \
        --num_samples 100 \
        --output_dir "${EVAL_OUTPUT_DIR}/${MODEL_NAME}" \
        --max_new_tokens 1024 \
        2>&1 | tee "${EVAL_OUTPUT_DIR}/${MODEL_NAME}_eval.log"

    echo "Completed: ${MODEL_NAME} at $(date)"
done

# ──────────────────────────────────────────────
# STEP 4: Rebuild transfer matrix JSON
# ──────────────────────────────────────────────
echo ""
echo "Rebuilding transfer matrix..."

python3 << 'PYEOF'
import json
from pathlib import Path

output_dir = Path("experiments/rlvr_geometry_paper/outputs/transfer_matrix_7b")
tasks = ["countdown", "maze", "mini_sudoku", "tower_of_hanoi", "shortest_path", "basic_arithmetic", "syllogism"]
models = [f"grpo_qwen25_7b_{t}" for t in tasks]

matrix = {}
for model in models:
    model_dir = output_dir / model
    metrics_files = sorted(model_dir.glob("metrics_*.json"))
    if metrics_files:
        data = json.loads(metrics_files[-1].read_text())
        matrix[model] = {task: data["tasks"].get(task, {}).get("accuracy", 0) for task in tasks}
    else:
        print(f"WARNING: No metrics found for {model}")

result = {"matrix": matrix, "tasks": tasks, "models": models}
(output_dir / "transfer_matrix_7x7.json").write_text(json.dumps(result, indent=2))

print("\n" + "="*80)
print("UPDATED 7B TRANSFER MATRIX")
print("="*80)
header = f"{'':>20} | " + " | ".join(f"{t[:8]:>8}" for t in tasks)
print(header)
print("-" * len(header))
for model in models:
    t = model.replace("grpo_qwen25_7b_", "")
    if model in matrix:
        vals = " | ".join(f"{matrix[model].get(tk,0)*100:>7.1f}%" for tk in tasks)
        print(f"{t:>20} | {vals}")
print("="*80)
PYEOF

echo ""
echo "============================================"
echo "RERUN COMPLETE at $(date)"
echo "============================================"
