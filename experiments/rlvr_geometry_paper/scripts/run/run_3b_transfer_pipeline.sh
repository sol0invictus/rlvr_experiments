#!/bin/bash
# ============================================================================
# Full 3B Transfer Matrix Pipeline (Qwen2.5-3B-Instruct)
# ============================================================================
# Runs: Download model -> SFT warmup -> 7x GRPO -> 7x7 eval (parallel)
#
# Usage: nohup bash experiments/rlvr_geometry_paper/scripts/run/run_3b_transfer_pipeline.sh > 3b_transfer_pipeline.log 2>&1 &
# Hardware: 8x B200 (P6)
# ============================================================================

set -e
cd /scratch/sunguha/rlvr_experiments

CONFIGS_DIR="experiments/rlvr_geometry_paper/configs/transfer_3b"
SFT_MODEL="experiments/rlvr_geometry_paper/outputs/sft_qwen25_3b"
MODEL_DIR="/scratch/sunguha/models/Qwen2.5-3B-Instruct"

TASKS=(countdown maze mini_sudoku tower_of_hanoi shortest_path basic_arithmetic syllogism)

echo "============================================"
echo "3B TRANSFER MATRIX PIPELINE"
echo "Started: $(date)"
echo "============================================"

# ──────────────────────────────────────────────
# STEP 0: Download model if needed
# ──────────────────────────────────────────────
if [ ! -d "$MODEL_DIR" ]; then
    echo "Downloading Qwen2.5-3B-Instruct..."
    python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen2.5-3B-Instruct', local_dir='$MODEL_DIR')
print('Download complete')
"
    echo "Model downloaded at $(date)"
else
    echo "Model already exists at $MODEL_DIR"
fi

# ──────────────────────────────────────────────
# STEP 1: SFT Warmup
# ──────────────────────────────────────────────
echo ""
echo "============================================"
echo "STEP 1: SFT Warmup (Qwen2.5-3B-Instruct)"
echo "Started: $(date)"
echo "============================================"

if [ -d "${SFT_MODEL}" ] && ls ${SFT_MODEL}/*.safetensors 1>/dev/null 2>&1; then
    echo "SFT model already exists at ${SFT_MODEL}, skipping."
else
    torchrun --standalone --nproc_per_node=8 \
        train_sft.py ${CONFIGS_DIR}/config_sft_qwen25_3b.yaml \
        2>&1 | tee sft_qwen25_3b_warmup.log
    echo "SFT completed at $(date)"
fi

if [ ! -d "${SFT_MODEL}" ]; then
    echo "ERROR: SFT model not found at ${SFT_MODEL}"
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
    OUTPUT_DIR="experiments/rlvr_geometry_paper/results/grpo_qwen25_3b_${task}"

    if [ -d "${OUTPUT_DIR}" ] && ls ${OUTPUT_DIR}/checkpoint-* 1>/dev/null 2>&1; then
        echo "  [SKIP] ${task}: checkpoint already exists"
        continue
    fi

    echo ""
    echo "--------------------------------------------"
    echo "Training: ${task}"
    echo "Started: $(date)"
    echo "--------------------------------------------"

    torchrun --standalone --nproc_per_node=8 \
        train_grpo.py ${CONFIGS_DIR}/train_${task}.yaml \
        2>&1 | tee grpo_qwen25_3b_${task}_train.log

    echo "Completed: ${task} at $(date)"
done

echo ""
echo "ALL GRPO TRAINING COMPLETE at $(date)"

# ──────────────────────────────────────────────
# STEP 3: Parallel Evaluation (3 models per wave)
# ──────────────────────────────────────────────
echo ""
echo "============================================"
echo "STEP 3: Transfer Matrix Evaluation (7x7)"
echo "Started: $(date)"
echo "============================================"

EVAL_TASKS="countdown,maze,mini_sudoku,tower_of_hanoi,shortest_path,basic_arithmetic,syllogism"
EVAL_OUTPUT_DIR="experiments/rlvr_geometry_paper/outputs/transfer_matrix_3b"
mkdir -p ${EVAL_OUTPUT_DIR}

# Wave 1: 3 models in parallel (GPUs 0-1, 2-3, 4-5)
echo "Wave 1: countdown, maze, mini_sudoku"
CUDA_VISIBLE_DEVICES=0,1 python evaluate_transfer.py \
    --model experiments/rlvr_geometry_paper/results/grpo_qwen25_3b_countdown --tasks "$EVAL_TASKS" --num_samples 100 \
    --output_dir $EVAL_OUTPUT_DIR/grpo_qwen25_3b_countdown --max_new_tokens 1024 --batch_size 8 \
    > $EVAL_OUTPUT_DIR/countdown_eval.log 2>&1 &
P1=$!

CUDA_VISIBLE_DEVICES=2,3 python evaluate_transfer.py \
    --model experiments/rlvr_geometry_paper/results/grpo_qwen25_3b_maze --tasks "$EVAL_TASKS" --num_samples 100 \
    --output_dir $EVAL_OUTPUT_DIR/grpo_qwen25_3b_maze --max_new_tokens 1024 --batch_size 8 \
    > $EVAL_OUTPUT_DIR/maze_eval.log 2>&1 &
P2=$!

CUDA_VISIBLE_DEVICES=4,5 python evaluate_transfer.py \
    --model experiments/rlvr_geometry_paper/results/grpo_qwen25_3b_mini_sudoku --tasks "$EVAL_TASKS" --num_samples 100 \
    --output_dir $EVAL_OUTPUT_DIR/grpo_qwen25_3b_mini_sudoku --max_new_tokens 1024 --batch_size 8 \
    > $EVAL_OUTPUT_DIR/mini_sudoku_eval.log 2>&1 &
P3=$!

wait $P1 $P2 $P3
echo "Wave 1 done at $(date)"

# Wave 2: 3 models in parallel
echo "Wave 2: tower_of_hanoi, shortest_path, basic_arithmetic"
CUDA_VISIBLE_DEVICES=0,1 python evaluate_transfer.py \
    --model experiments/rlvr_geometry_paper/results/grpo_qwen25_3b_tower_of_hanoi --tasks "$EVAL_TASKS" --num_samples 100 \
    --output_dir $EVAL_OUTPUT_DIR/grpo_qwen25_3b_tower_of_hanoi --max_new_tokens 1024 --batch_size 8 \
    > $EVAL_OUTPUT_DIR/hanoi_eval.log 2>&1 &
P4=$!

CUDA_VISIBLE_DEVICES=2,3 python evaluate_transfer.py \
    --model experiments/rlvr_geometry_paper/results/grpo_qwen25_3b_shortest_path --tasks "$EVAL_TASKS" --num_samples 100 \
    --output_dir $EVAL_OUTPUT_DIR/grpo_qwen25_3b_shortest_path --max_new_tokens 1024 --batch_size 8 \
    > $EVAL_OUTPUT_DIR/shortest_path_eval.log 2>&1 &
P5=$!

CUDA_VISIBLE_DEVICES=4,5 python evaluate_transfer.py \
    --model experiments/rlvr_geometry_paper/results/grpo_qwen25_3b_basic_arithmetic --tasks "$EVAL_TASKS" --num_samples 100 \
    --output_dir $EVAL_OUTPUT_DIR/grpo_qwen25_3b_basic_arithmetic --max_new_tokens 1024 --batch_size 8 \
    > $EVAL_OUTPUT_DIR/basic_arithmetic_eval.log 2>&1 &
P6=$!

wait $P4 $P5 $P6
echo "Wave 2 done at $(date)"

# Wave 3: last model
echo "Wave 3: syllogism"
CUDA_VISIBLE_DEVICES=0,1 python evaluate_transfer.py \
    --model experiments/rlvr_geometry_paper/results/grpo_qwen25_3b_syllogism --tasks "$EVAL_TASKS" --num_samples 100 \
    --output_dir $EVAL_OUTPUT_DIR/grpo_qwen25_3b_syllogism --max_new_tokens 1024 --batch_size 8 \
    > $EVAL_OUTPUT_DIR/syllogism_eval.log 2>&1 &
P7=$!

wait $P7
echo "Wave 3 done at $(date)"

# ──────────────────────────────────────────────
# STEP 4: Generate Summary Matrix
# ──────────────────────────────────────────────
echo ""
echo "Building transfer matrix..."

python3 << 'PYEOF'
import json
from pathlib import Path

output_dir = Path("experiments/rlvr_geometry_paper/outputs/transfer_matrix_3b")
tasks = ["countdown", "maze", "mini_sudoku", "tower_of_hanoi", "shortest_path", "basic_arithmetic", "syllogism"]
models = [f"grpo_qwen25_3b_{t}" for t in tasks]

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
print("3B TRANSFER MATRIX")
print("="*80)
header = f"{'':>20} | " + " | ".join(f"{t[:8]:>8}" for t in tasks)
print(header)
print("-" * len(header))
for model in models:
    t = model.replace("grpo_qwen25_3b_", "")
    if model in matrix:
        vals = " | ".join(f"{matrix[model].get(tk,0)*100:>7.1f}%" for tk in tasks)
        print(f"{t:>20} | {vals}")
print("="*80)
PYEOF

echo ""
echo "============================================"
echo "PIPELINE COMPLETE at $(date)"
echo "============================================"
