#!/bin/bash
# Run transfer matrix evaluation with parallel GPU execution
# Each model evaluation runs on a single GPU, 8 evals in parallel

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
NUM_SAMPLES=50

OUTPUT_DIR="experiments/rlvr_geometry_paper/outputs/transfer_matrix"
mkdir -p $OUTPUT_DIR

echo "============================================"
echo "TRANSFER MATRIX EVALUATION (PARALLEL)"
echo "Started: $(date)"
echo "Models: ${#MODELS[@]}"
echo "Tasks: $TASKS"
echo "Samples per task: $NUM_SAMPLES"
echo "============================================"

# Run all 6 model evaluations in parallel (one per GPU)
pids=()
gpu_id=0

for model in "${MODELS[@]}"; do
    echo "Starting: $model on GPU $gpu_id"
    mkdir -p "${OUTPUT_DIR}/${model}"

    CUDA_VISIBLE_DEVICES=$gpu_id python evaluate_transfer.py \
        --model "experiments/rlvr_geometry_paper/results/${model}" \
        --tasks "$TASKS" \
        --num_samples $NUM_SAMPLES \
        --output_dir "${OUTPUT_DIR}/${model}" \
        --max_new_tokens 1024 \
        > "${OUTPUT_DIR}/${model}_eval.log" 2>&1 &

    pids+=($!)
    gpu_id=$((gpu_id + 1))
done

echo ""
echo "All ${#pids[@]} evaluations started. Waiting for completion..."
echo "PIDs: ${pids[*]}"

# Wait for all to complete
for pid in "${pids[@]}"; do
    wait $pid
    echo "Process $pid completed"
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
task_short = ["maze", "sudoku", "hanoi", "path", "arith", "syllog"]

# Build matrix
matrix = {}
for model in models:
    model_dir = output_dir / model
    metrics_files = list(model_dir.glob("metrics_*.json"))
    if metrics_files:
        with open(sorted(metrics_files)[-1]) as f:
            data = json.load(f)
        matrix[model] = {task: data["tasks"].get(task, {}).get("accuracy", 0) for task in tasks}
    else:
        print(f"Warning: No metrics found for {model}")

# Print matrix
print("\n" + "="*85)
print("TRANSFER MATRIX (rows=trained_on, cols=evaluated_on)")
print("Diagonal = same-task accuracy, off-diagonal = transfer")
print("="*85)

# Header
header = f"{'Trained On':<12} | " + " | ".join(f"{t:>7}" for t in task_short)
print(header)
print("-" * len(header))

# Rows
for model in models:
    trained_task = model.replace("grpo_qwen4b_", "")
    trained_short = trained_task[:12]
    if model in matrix:
        row_vals = []
        for i, t in enumerate(tasks):
            val = matrix[model].get(t, 0) * 100
            # Highlight diagonal (same task)
            if t == trained_task or (t == "mini_sudoku" and trained_task == "mini_sudoku"):
                row_vals.append(f"{val:>6.1f}*")
            else:
                row_vals.append(f"{val:>7.1f}")
        row = f"{trained_short:<12} | " + " | ".join(row_vals)
    else:
        row = f"{trained_short:<12} | " + " | ".join(f"{'N/A':>7}" for _ in tasks)
    print(row)

print("="*85)
print("* = diagonal (same-task performance)")

# Compute transfer stats
print("\n--- Transfer Statistics ---")
total_transfer = 0
count = 0
for model in models:
    trained_task = model.replace("grpo_qwen4b_", "")
    if model in matrix:
        for t in tasks:
            if t != trained_task:
                total_transfer += matrix[model].get(t, 0)
                count += 1

if count > 0:
    avg_transfer = total_transfer / count * 100
    print(f"Average off-diagonal transfer: {avg_transfer:.1f}%")

# Save to JSON
result = {
    "matrix": matrix,
    "tasks": tasks,
    "models": models,
}
with open(output_dir / "transfer_matrix.json", "w") as f:
    json.dump(result, f, indent=2)
print(f"\nMatrix saved to {output_dir}/transfer_matrix.json")
PYEOF
