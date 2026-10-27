#!/bin/bash
# ============================================================================
# Full 7B Transfer Matrix Pipeline
# ============================================================================
# Runs: SFT warmup -> 7x GRPO training -> 7x7 transfer matrix evaluation
#
# Usage (on Green cluster pod):
#   nohup bash experiments/rlvr_geometry_paper/scripts/run/run_7b_transfer_pipeline.sh > 7b_transfer_pipeline.log 2>&1 &
#
# Hardware: 8x B200 (P6)
# Expected time: SFT ~1h + 7x GRPO ~7x3h = ~22h + eval ~4h = ~26h total
# ============================================================================

set -e
cd /scratch/sunguha/rlvr_experiments

CONFIGS_DIR="experiments/rlvr_geometry_paper/configs/transfer_7b"
SFT_MODEL="experiments/rlvr_geometry_paper/outputs/sft_qwen25_7b"

TASKS=(
    "countdown"
    "maze"
    "mini_sudoku"
    "tower_of_hanoi"
    "shortest_path"
    "basic_arithmetic"
    "syllogism"
)

echo "============================================"
echo "7B TRANSFER MATRIX PIPELINE"
echo "Started: $(date)"
echo "============================================"

# ──────────────────────────────────────────────
# STEP 1: SFT Warmup
# ──────────────────────────────────────────────
echo ""
echo "============================================"
echo "STEP 1: SFT Warmup (Qwen2.5-7B-Instruct)"
echo "Started: $(date)"
echo "============================================"

if [ -d "${SFT_MODEL}" ] && [ -f "${SFT_MODEL}/model.safetensors" -o -f "${SFT_MODEL}/model-00001-of-00004.safetensors" ]; then
    echo "SFT model already exists at ${SFT_MODEL}, skipping SFT step."
else
    torchrun --standalone --nproc_per_node=8 \
        train_sft.py ${CONFIGS_DIR}/config_sft_qwen25_7b.yaml \
        2>&1 | tee sft_qwen25_7b_warmup.log

    echo "SFT completed at $(date)"
fi

# Verify SFT model exists
if [ ! -d "${SFT_MODEL}" ]; then
    echo "ERROR: SFT model not found at ${SFT_MODEL}"
    echo "Check sft_qwen25_7b_warmup.log for errors"
    exit 1
fi

# ──────────────────────────────────────────────
# STEP 2: GRPO Training (7 tasks, sequential)
# ──────────────────────────────────────────────
echo ""
echo "============================================"
echo "STEP 2: GRPO Training (7 tasks)"
echo "Started: $(date)"
echo "============================================"

for task in "${TASKS[@]}"; do
    OUTPUT_DIR="experiments/rlvr_geometry_paper/results/grpo_qwen25_7b_${task}"
    LOG_FILE="grpo_qwen25_7b_${task}_train.log"

    # Skip if already trained (check for checkpoint)
    if [ -d "${OUTPUT_DIR}" ] && ls ${OUTPUT_DIR}/checkpoint-* 1>/dev/null 2>&1; then
        echo "  [SKIP] ${task}: checkpoint already exists in ${OUTPUT_DIR}"
        continue
    fi

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
echo "ALL GRPO TRAINING COMPLETE at $(date)"
echo "============================================"

# ──────────────────────────────────────────────
# STEP 3: Transfer Matrix Evaluation
# ──────────────────────────────────────────────
echo ""
echo "============================================"
echo "STEP 3: Transfer Matrix Evaluation (7x7)"
echo "Started: $(date)"
echo "============================================"

EVAL_TASKS="countdown,maze,mini_sudoku,tower_of_hanoi,shortest_path,basic_arithmetic,syllogism"
EVAL_OUTPUT_DIR="experiments/rlvr_geometry_paper/outputs/transfer_matrix_7b"
mkdir -p ${EVAL_OUTPUT_DIR}

for task in "${TASKS[@]}"; do
    MODEL_DIR="experiments/rlvr_geometry_paper/results/grpo_qwen25_7b_${task}"
    MODEL_NAME="grpo_qwen25_7b_${task}"

    if [ ! -d "${MODEL_DIR}" ]; then
        echo "  [SKIP] ${MODEL_NAME}: model not found"
        continue
    fi

    echo ""
    echo "--------------------------------------------"
    echo "Evaluating: ${MODEL_NAME} on ALL tasks"
    echo "Started: $(date)"
    echo "--------------------------------------------"

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
# STEP 4: Generate Summary Matrix
# ──────────────────────────────────────────────
echo ""
echo "============================================"
echo "STEP 4: Generating Transfer Matrix JSON"
echo "============================================"

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
        with open(metrics_files[-1]) as f:
            data = json.load(f)
        matrix[model] = {task: data["tasks"].get(task, {}).get("accuracy", 0) for task in tasks}
    else:
        print(f"WARNING: No metrics found for {model}")

# Print matrix
print("\n" + "="*80)
print("7B TRANSFER MATRIX (rows=trained_on, cols=evaluated_on)")
print("="*80)

header = f"{'Trained On':<30} | " + " | ".join(f"{t[:8]:>8}" for t in tasks)
print(header)
print("-" * len(header))

for model in models:
    trained_task = model.replace("grpo_qwen25_7b_", "")
    if model in matrix:
        row = f"{trained_task:<30} | " + " | ".join(f"{matrix[model].get(t, 0)*100:>7.1f}%" for t in tasks)
    else:
        row = f"{trained_task:<30} | " + " | ".join(f"{'N/A':>8}" for t in tasks)
    print(row)

print("="*80)

# Save
result = {
    "matrix": matrix,
    "tasks": tasks,
    "models": models,
}
with open(output_dir / "transfer_matrix_7x7.json", "w") as f:
    json.dump(result, f, indent=2)
print(f"\nMatrix saved to {output_dir}/transfer_matrix_7x7.json")
PYEOF

echo ""
echo "============================================"
echo "PIPELINE COMPLETE at $(date)"
echo "============================================"
