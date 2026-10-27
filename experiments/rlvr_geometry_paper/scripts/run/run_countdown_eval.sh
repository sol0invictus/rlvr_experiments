#!/bin/bash
# Evaluate countdown model (4to5 version which has proper config)
set -e
cd /scratch/sunguha/rlvr_experiments

COUNTDOWN_MODEL="grpo_qwen_4b_instruct_countdown_4to5"
ALL_TASKS=("countdown" "maze" "mini_sudoku" "tower_of_hanoi" "shortest_path" "basic_arithmetic" "syllogism")
NUM_SAMPLES=50

OUTPUT_BASE="experiments/rlvr_geometry_paper/outputs/transfer_matrix"
EVAL_CONFIG_DIR="configs/eval_transfer"

echo "Evaluating countdown model: $COUNTDOWN_MODEL"

model_path="experiments/rlvr_geometry_paper/results/${COUNTDOWN_MODEL}"
model_out="${OUTPUT_BASE}/${COUNTDOWN_MODEL}"
mkdir -p "$model_out"

for task in "${ALL_TASKS[@]}"; do
    task_out="${model_out}/${task}"
    mkdir -p "$task_out"

    echo "  Evaluating on: $task"

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

    # Extract and print correctness
    if [ -f "${task_out}/metrics.json" ]; then
        echo "    Result: $(cat ${task_out}/metrics.json | python3 -c 'import json,sys; d=json.load(sys.stdin); print(f\"correctness={d[\"correctness_reward\"]:.2f}\")')"
    fi
    echo "    Done: $task"
done

echo ""
echo "Countdown model evaluations complete!"

# Update 7x7 matrix
python3 << 'PYEOF'
import json
from pathlib import Path

output_dir = Path("experiments/rlvr_geometry_paper/outputs/transfer_matrix")
countdown_model = "grpo_qwen_4b_instruct_countdown_4to5"

models = [
    countdown_model,
    "grpo_qwen4b_maze",
    "grpo_qwen4b_mini_sudoku",
    "grpo_qwen4b_tower_of_hanoi",
    "grpo_qwen4b_shortest_path",
    "grpo_qwen4b_basic_arithmetic",
    "grpo_qwen4b_syllogism"
]

tasks = ["countdown", "maze", "mini_sudoku", "tower_of_hanoi", "shortest_path", "basic_arithmetic", "syllogism"]
task_short = ["cntdwn", "maze", "sudoku", "hanoi", "path", "arith", "syllog"]

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

print("\n" + "="*100)
print("7x7 TRANSFER MATRIX")
print("="*100)

header = f"{'Trained':<16} | " + " | ".join(f"{t:>7}" for t in task_short)
print(header)
print("-" * len(header))

for model in models:
    if "countdown" in model:
        trained_task = "countdown"
        trained_short = "countdown"
    else:
        trained_task = model.replace("grpo_qwen4b_", "")
        trained_short = trained_task[:16]

    row_vals = []
    for t in tasks:
        val = matrix[model].get(t)
        if val is not None:
            val_pct = val * 100
            if t == trained_task:
                row_vals.append(f"{val_pct:>6.1f}*")
            else:
                row_vals.append(f"{val_pct:>7.1f}")
        else:
            row_vals.append(f"{'N/A':>7}")
    print(f"{trained_short:<16} | " + " | ".join(row_vals))

print("="*100)

# Stats
diag, off = [], []
for model in models:
    trained_task = "countdown" if "countdown" in model else model.replace("grpo_qwen4b_", "")
    for t in tasks:
        val = matrix[model].get(t)
        if val is not None:
            (diag if t == trained_task else off).append(val)

print(f"Diagonal avg: {sum(diag)/len(diag)*100:.1f}%  |  Off-diagonal avg: {sum(off)/len(off)*100:.1f}%")

with open(output_dir / "transfer_matrix_7x7.json", "w") as f:
    json.dump({"matrix": matrix, "tasks": tasks, "models": models}, f, indent=2)
print(f"Saved to {output_dir}/transfer_matrix_7x7.json")
PYEOF
