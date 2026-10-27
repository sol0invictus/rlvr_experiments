#!/bin/bash
# Transfer Matrix Evaluation with DDP (8 GPUs)
# Evaluates each trained model on all 6 tasks

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

TASKS=("maze" "mini_sudoku" "tower_of_hanoi" "shortest_path" "basic_arithmetic" "syllogism")
NUM_SAMPLES=50

OUTPUT_BASE="experiments/rlvr_geometry_paper/outputs/transfer_matrix"
EVAL_CONFIG_DIR="configs/eval_transfer"
mkdir -p $OUTPUT_BASE $EVAL_CONFIG_DIR

echo "============================================"
echo "TRANSFER MATRIX EVALUATION (8-GPU DDP)"
echo "Started: $(date)"
echo "Models: ${#MODELS[@]}"
echo "Tasks: ${#TASKS[@]}"
echo "Samples per task: $NUM_SAMPLES"
echo "============================================"

# Create eval configs for each task
for task in "${TASKS[@]}"; do
    cat > "${EVAL_CONFIG_DIR}/eval_${task}.yaml" << EOF
model:
  name_or_path: PLACEHOLDER
  torch_dtype: bfloat16

environment:
  name: reasoning_gym
  task_name: ${task}
  samples: ${NUM_SAMPLES}

evaluation:
  num_samples: ${NUM_SAMPLES}
  seed: 9999

generation:
  max_new_tokens: 1024
  temperature: 0.0
  batch_size: 4

system_prompt: |
  You are a helpful assistant. You must always think step-by-step. Put your thinking process inside <think> and </think> tags, and put your final answer inside <answer> and </answer> tags.

output_dir: PLACEHOLDER
EOF
done

echo "Created ${#TASKS[@]} eval configs"

# Run evaluations
for model in "${MODELS[@]}"; do
    model_path="experiments/rlvr_geometry_paper/results/${model}"
    model_out="${OUTPUT_BASE}/${model}"
    mkdir -p "$model_out"

    echo ""
    echo "============================================"
    echo "Model: $model"
    echo "============================================"

    for task in "${TASKS[@]}"; do
        task_out="${model_out}/${task}"
        mkdir -p "$task_out"

        echo "  Evaluating on: $task"

        # Update config with correct paths
        sed -e "s|name_or_path: PLACEHOLDER|name_or_path: ${model_path}|" \
            -e "s|output_dir: PLACEHOLDER|output_dir: ${task_out}|" \
            "${EVAL_CONFIG_DIR}/eval_${task}.yaml" > "${EVAL_CONFIG_DIR}/eval_${model}_on_${task}.yaml"

        # Run with DDP on 8 GPUs
        torchrun --standalone --nproc_per_node=8 evaluate.py \
            "${EVAL_CONFIG_DIR}/eval_${model}_on_${task}.yaml" \
            --samples $NUM_SAMPLES \
            2>&1 | tee "${task_out}/eval.log"

        echo "    Done: $task"
    done

    echo "Completed all tasks for: $model"
done

echo ""
echo "============================================"
echo "ALL EVALUATIONS COMPLETE at $(date)"
echo "============================================"

# Generate transfer matrix summary
echo ""
echo "Generating transfer matrix..."
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
    matrix[model] = {}
    for task in tasks:
        metrics_path = output_dir / model / task / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path) as f:
                data = json.load(f)
            matrix[model][task] = data.get("correctness_reward", 0)
        else:
            matrix[model][task] = None

# Print matrix
print("\n" + "="*85)
print("TRANSFER MATRIX (rows=trained_on, cols=evaluated_on)")
print("Diagonal = same-task accuracy, off-diagonal = transfer")
print("="*85)

# Header
header = f"{'Trained':<12} | " + " | ".join(f"{t:>7}" for t in task_short)
print(header)
print("-" * len(header))

# Rows
for model in models:
    trained_task = model.replace("grpo_qwen4b_", "")
    trained_short = trained_task[:12]
    row_vals = []
    for i, t in enumerate(tasks):
        val = matrix[model].get(t)
        if val is not None:
            val_pct = val * 100
            # Highlight diagonal (same task)
            if t == trained_task:
                row_vals.append(f"{val_pct:>6.1f}*")
            else:
                row_vals.append(f"{val_pct:>7.1f}")
        else:
            row_vals.append(f"{'N/A':>7}")
    row = f"{trained_short:<12} | " + " | ".join(row_vals)
    print(row)

print("="*85)
print("* = diagonal (same-task performance)")

# Compute stats
print("\n--- Statistics ---")
diag_vals = []
off_diag_vals = []
for model in models:
    trained_task = model.replace("grpo_qwen4b_", "")
    for t in tasks:
        val = matrix[model].get(t)
        if val is not None:
            if t == trained_task:
                diag_vals.append(val)
            else:
                off_diag_vals.append(val)

if diag_vals:
    print(f"Diagonal (same-task) avg:     {sum(diag_vals)/len(diag_vals)*100:.1f}%")
if off_diag_vals:
    print(f"Off-diagonal (transfer) avg:  {sum(off_diag_vals)/len(off_diag_vals)*100:.1f}%")

# Save
result = {"matrix": matrix, "tasks": tasks, "models": models}
with open(output_dir / "transfer_matrix.json", "w") as f:
    json.dump(result, f, indent=2)
print(f"\nSaved to {output_dir}/transfer_matrix.json")
PYEOF
