#!/bin/bash
# Extend transfer matrix to include countdown (7x7)
# 1. Evaluate countdown model on all 7 tasks
# 2. Evaluate all 6 existing models on countdown

set -e
cd /scratch/sunguha/rlvr_experiments

# Use the instruct countdown model (matches Qwen-4B pattern)
COUNTDOWN_MODEL="grpo_qwen_4b_instruct_countdown"

EXISTING_MODELS=(
    "grpo_qwen4b_maze"
    "grpo_qwen4b_mini_sudoku"
    "grpo_qwen4b_tower_of_hanoi"
    "grpo_qwen4b_shortest_path"
    "grpo_qwen4b_basic_arithmetic"
    "grpo_qwen4b_syllogism"
)

ALL_TASKS=("countdown" "maze" "mini_sudoku" "tower_of_hanoi" "shortest_path" "basic_arithmetic" "syllogism")
NUM_SAMPLES=50

OUTPUT_BASE="experiments/rlvr_geometry_paper/outputs/transfer_matrix"
EVAL_CONFIG_DIR="configs/eval_transfer"
mkdir -p $OUTPUT_BASE $EVAL_CONFIG_DIR

echo "============================================"
echo "EXTENDING TRANSFER MATRIX WITH COUNTDOWN"
echo "Started: $(date)"
echo "============================================"

# Create countdown eval config
cat > "${EVAL_CONFIG_DIR}/eval_countdown.yaml" << EOF
model:
  name_or_path: PLACEHOLDER
  torch_dtype: bfloat16

environment:
  name: reasoning_gym
  task_name: countdown
  samples: ${NUM_SAMPLES}
  task_params:
    min_numbers: 4
    max_numbers: 5
    min_target: 50
    max_target: 200

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

echo "Created countdown eval config"

# Part 1: Evaluate countdown model on ALL 7 tasks
echo ""
echo "============================================"
echo "Part 1: Countdown model on all 7 tasks"
echo "============================================"

model_path="experiments/rlvr_geometry_paper/results/${COUNTDOWN_MODEL}"
model_out="${OUTPUT_BASE}/${COUNTDOWN_MODEL}"
mkdir -p "$model_out"

for task in "${ALL_TASKS[@]}"; do
    task_out="${model_out}/${task}"
    mkdir -p "$task_out"

    echo "  Evaluating countdown model on: $task"

    # Use existing config or countdown config
    if [ "$task" == "countdown" ]; then
        config_template="${EVAL_CONFIG_DIR}/eval_countdown.yaml"
    else
        config_template="${EVAL_CONFIG_DIR}/eval_${task}.yaml"
    fi

    sed -e "s|name_or_path: PLACEHOLDER|name_or_path: ${model_path}|" \
        -e "s|output_dir: PLACEHOLDER|output_dir: ${task_out}|" \
        "$config_template" > "${EVAL_CONFIG_DIR}/eval_${COUNTDOWN_MODEL}_on_${task}.yaml"

    torchrun --standalone --nproc_per_node=8 evaluate.py \
        "${EVAL_CONFIG_DIR}/eval_${COUNTDOWN_MODEL}_on_${task}.yaml" \
        --samples $NUM_SAMPLES \
        2>&1 | tee "${task_out}/eval.log"

    echo "    Done: $task"
done

echo "Completed countdown model evaluations"

# Part 2: Evaluate all existing models on countdown
echo ""
echo "============================================"
echo "Part 2: Existing models on countdown task"
echo "============================================"

for model in "${EXISTING_MODELS[@]}"; do
    model_path="experiments/rlvr_geometry_paper/results/${model}"
    task_out="${OUTPUT_BASE}/${model}/countdown"
    mkdir -p "$task_out"

    echo "  Evaluating $model on countdown"

    sed -e "s|name_or_path: PLACEHOLDER|name_or_path: ${model_path}|" \
        -e "s|output_dir: PLACEHOLDER|output_dir: ${task_out}|" \
        "${EVAL_CONFIG_DIR}/eval_countdown.yaml" > "${EVAL_CONFIG_DIR}/eval_${model}_on_countdown.yaml"

    torchrun --standalone --nproc_per_node=8 evaluate.py \
        "${EVAL_CONFIG_DIR}/eval_${model}_on_countdown.yaml" \
        --samples $NUM_SAMPLES \
        2>&1 | tee "${task_out}/eval.log"

    echo "    Done: $model on countdown"
done

echo ""
echo "============================================"
echo "ALL COUNTDOWN EVALUATIONS COMPLETE at $(date)"
echo "============================================"

# Generate updated 7x7 transfer matrix
echo ""
echo "Generating 7x7 transfer matrix..."
python3 << 'PYEOF'
import json
from pathlib import Path

output_dir = Path("experiments/rlvr_geometry_paper/outputs/transfer_matrix")

models = [
    "grpo_qwen_4b_instruct_countdown",
    "grpo_qwen4b_maze",
    "grpo_qwen4b_mini_sudoku",
    "grpo_qwen4b_tower_of_hanoi",
    "grpo_qwen4b_shortest_path",
    "grpo_qwen4b_basic_arithmetic",
    "grpo_qwen4b_syllogism"
]

tasks = ["countdown", "maze", "mini_sudoku", "tower_of_hanoi", "shortest_path", "basic_arithmetic", "syllogism"]
task_short = ["cntdwn", "maze", "sudoku", "hanoi", "path", "arith", "syllog"]

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
print("\n" + "="*95)
print("7x7 TRANSFER MATRIX (rows=trained_on, cols=evaluated_on)")
print("="*95)

# Header
header = f"{'Trained':<14} | " + " | ".join(f"{t:>7}" for t in task_short)
print(header)
print("-" * len(header))

# Rows
for model in models:
    # Extract task name from model name
    if "countdown" in model:
        trained_task = "countdown"
        trained_short = "countdown"
    else:
        trained_task = model.replace("grpo_qwen4b_", "")
        trained_short = trained_task[:14]

    row_vals = []
    for t in tasks:
        val = matrix[model].get(t)
        if val is not None:
            val_pct = val * 100
            # Highlight diagonal
            if t == trained_task:
                row_vals.append(f"{val_pct:>6.1f}*")
            else:
                row_vals.append(f"{val_pct:>7.1f}")
        else:
            row_vals.append(f"{'N/A':>7}")
    row = f"{trained_short:<14} | " + " | ".join(row_vals)
    print(row)

print("="*95)
print("* = diagonal (same-task performance)")

# Stats
print("\n--- Statistics ---")
diag_vals = []
off_diag_vals = []
for model in models:
    if "countdown" in model:
        trained_task = "countdown"
    else:
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
with open(output_dir / "transfer_matrix_7x7.json", "w") as f:
    json.dump(result, f, indent=2)
print(f"\nSaved to {output_dir}/transfer_matrix_7x7.json")
PYEOF
