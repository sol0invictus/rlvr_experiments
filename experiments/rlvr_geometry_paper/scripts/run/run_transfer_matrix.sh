#!/bin/bash
# Run transfer matrix evaluation: evaluate each trained model on ALL tasks
# This produces a 6x6 matrix of (trained_on, evaluated_on) accuracies

set -e
cd /scratch/sunguha/rlvr_experiments

MODELS=(
    "grpo_qwen4b_maze"
    "grpo_qwen4b_mini_sudoku"
    "grpo_qwen4b_tower_of_hanoi"
    "grpo_qwen4b_shortest_path"
    "grpo_qwen4b_basic_arithmetic"
    "grpo_qwen4b_syllogism"
)

TASKS="maze,mini_sudoku,tower_of_hanoi,shortest_path,basic_arithmetic,syllogism"

OUTPUT_DIR="experiments/rlvr_geometry_paper/outputs/transfer_matrix"
mkdir -p $OUTPUT_DIR

echo "============================================"
echo "TRANSFER MATRIX EVALUATION"
echo "Started: $(date)"
echo "Models: ${#MODELS[@]}"
echo "Tasks: $TASKS"
echo "============================================"

for model in "${MODELS[@]}"; do
    echo ""
    echo "============================================"
    echo "Evaluating: $model on ALL tasks"
    echo "Started: $(date)"
    echo "============================================"

    python evaluate_transfer.py \
        --model "experiments/rlvr_geometry_paper/results/${model}" \
        --tasks "$TASKS" \
        --num_samples 100 \
        --output_dir "${OUTPUT_DIR}/${model}" \
        --max_new_tokens 1024 \
        2>&1 | tee "${OUTPUT_DIR}/${model}_eval.log"

    echo "Completed: $model at $(date)"
done

echo ""
echo "============================================"
echo "ALL EVALUATIONS COMPLETE at $(date)"
echo "============================================"

# Generate summary matrix
echo ""
echo "Generating transfer matrix summary..."
python3 << 'PYEOF'
import json
import os
from pathlib import Path

output_dir = Path("experiments/rlvr_geometry_paper/outputs/transfer_matrix")
models = [
    "grpo_qwen4b_maze",
    "grpo_qwen4b_mini_sudoku",
    "grpo_qwen4b_tower_of_hanoi",
    "grpo_qwen4b_shortest_path",
    "grpo_qwen4b_basic_arithmetic",
    "grpo_qwen4b_syllogism"
]
tasks = ["maze", "mini_sudoku", "tower_of_hanoi", "shortest_path", "basic_arithmetic", "syllogism"]

# Build matrix
matrix = {}
for model in models:
    model_dir = output_dir / model
    metrics_files = list(model_dir.glob("metrics_*.json"))
    if metrics_files:
        with open(metrics_files[0]) as f:
            data = json.load(f)
        matrix[model] = {task: data["tasks"].get(task, {}).get("accuracy", 0) for task in tasks}

# Print matrix
print("\n" + "="*80)
print("TRANSFER MATRIX (rows=trained_on, cols=evaluated_on)")
print("="*80)

# Header
header = f"{'Trained On':<30} | " + " | ".join(f"{t[:8]:>8}" for t in tasks)
print(header)
print("-" * len(header))

# Rows
for model in models:
    trained_task = model.replace("grpo_qwen4b_", "")
    if model in matrix:
        row = f"{trained_task:<30} | " + " | ".join(f"{matrix[model].get(t, 0)*100:>7.1f}%" for t in tasks)
    else:
        row = f"{trained_task:<30} | " + " | ".join(f"{'N/A':>8}" for t in tasks)
    print(row)

print("="*80)

# Save to JSON
with open(output_dir / "transfer_matrix.json", "w") as f:
    json.dump(matrix, f, indent=2)
print(f"\nMatrix saved to {output_dir}/transfer_matrix.json")
PYEOF
