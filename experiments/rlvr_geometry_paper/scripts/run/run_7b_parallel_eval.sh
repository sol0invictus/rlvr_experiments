#!/bin/bash
# Run 3 model evals in parallel on different GPU pairs
set -e
cd /scratch/sunguha/rlvr_experiments

EVAL_TASKS="countdown,maze,mini_sudoku,tower_of_hanoi,shortest_path,basic_arithmetic,syllogism"
EVAL_DIR="experiments/rlvr_geometry_paper/outputs/transfer_matrix_7b"

# Clean old results
rm -rf $EVAL_DIR/grpo_qwen25_7b_countdown $EVAL_DIR/grpo_qwen25_7b_tower_of_hanoi $EVAL_DIR/grpo_qwen25_7b_basic_arithmetic

echo "Launching 3 parallel evals at $(date)"

CUDA_VISIBLE_DEVICES=0,1 python evaluate_transfer.py \
    --model experiments/rlvr_geometry_paper/results/grpo_qwen25_7b_countdown \
    --tasks "$EVAL_TASKS" --num_samples 100 \
    --output_dir $EVAL_DIR/grpo_qwen25_7b_countdown \
    --max_new_tokens 1024 --batch_size 8 \
    > $EVAL_DIR/countdown_eval_rerun.log 2>&1 &
PID1=$!

CUDA_VISIBLE_DEVICES=2,3 python evaluate_transfer.py \
    --model experiments/rlvr_geometry_paper/results/grpo_qwen25_7b_tower_of_hanoi \
    --tasks "$EVAL_TASKS" --num_samples 100 \
    --output_dir $EVAL_DIR/grpo_qwen25_7b_tower_of_hanoi \
    --max_new_tokens 1024 --batch_size 8 \
    > $EVAL_DIR/hanoi_eval_rerun.log 2>&1 &
PID2=$!

CUDA_VISIBLE_DEVICES=4,5 python evaluate_transfer.py \
    --model experiments/rlvr_geometry_paper/results/grpo_qwen25_7b_basic_arithmetic \
    --tasks "$EVAL_TASKS" --num_samples 100 \
    --output_dir $EVAL_DIR/grpo_qwen25_7b_basic_arithmetic \
    --max_new_tokens 1024 --batch_size 8 \
    > $EVAL_DIR/arith_eval_rerun.log 2>&1 &
PID3=$!

echo "PIDs: countdown=$PID1 hanoi=$PID2 arith=$PID3"
wait $PID1 $PID2 $PID3
echo "ALL EVALS DONE at $(date)"

# Rebuild transfer matrix
python3 << 'PYEOF'
import json
from pathlib import Path
output_dir = Path("experiments/rlvr_geometry_paper/outputs/transfer_matrix_7b")
tasks = ["countdown","maze","mini_sudoku","tower_of_hanoi","shortest_path","basic_arithmetic","syllogism"]
models = [f"grpo_qwen25_7b_{t}" for t in tasks]
matrix = {}
for model in models:
    model_dir = output_dir / model
    metrics_files = sorted(model_dir.glob("metrics_*.json"))
    if metrics_files:
        data = json.loads(metrics_files[-1].read_text())
        matrix[model] = {task: data["tasks"].get(task, {}).get("accuracy", 0) for task in tasks}
result = {"matrix": matrix, "tasks": tasks, "models": models}
(output_dir / "transfer_matrix_7x7.json").write_text(json.dumps(result, indent=2))
print("Matrix saved")
for m in models:
    t = m.replace("grpo_qwen25_7b_","")
    if m in matrix:
        vals = " ".join(f"{matrix[m].get(tk,0)*100:5.1f}%" for tk in tasks)
        print(f"{t:>20}: {vals}")
PYEOF

echo "REBUILD COMPLETE at $(date)"
