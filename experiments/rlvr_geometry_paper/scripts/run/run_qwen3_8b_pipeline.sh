#!/bin/bash
# ============================================================================
# Full Qwen3-8B Transfer Matrix Pipeline (10x10)
# ============================================================================
# Download model -> SFT -> 10x GRPO -> 10x10 eval (parallel)
# Only saves last checkpoint per task.
#
# Usage: nohup bash experiments/rlvr_geometry_paper/scripts/run/run_qwen3_8b_pipeline.sh > qwen3_8b_pipeline.log 2>&1 &
# Hardware: 8x B200 (P6)
# ============================================================================

set -e
cd /scratch/sunguha/rlvr_experiments

CONFIGS_DIR="experiments/rlvr_geometry_paper/configs/transfer_8b"
SFT_MODEL="experiments/rlvr_geometry_paper/outputs/sft_qwen3_8b"
MODEL_DIR="/scratch/sunguha/models/Qwen3-8B"
PREFIX="grpo_qwen3_8b"

ALL_TASKS=(countdown maze mini_sudoku shortest_path basic_arithmetic syllogism knights_knaves leg_counting caesar_cipher cryptarithm)
EVAL_TASKS=$(IFS=,; echo "${ALL_TASKS[*]}")

echo "============================================"
echo "QWEN3-8B TRANSFER MATRIX PIPELINE"
echo "Started: $(date)"
echo "Tasks: ${ALL_TASKS[*]}"
echo "============================================"

# ──────────────────────────────────────────────
# STEP 0: Install deps + Download model
# ──────────────────────────────────────────────
pip install reasoning-gym trl>=0.27.0 2>/dev/null | tail -1

if [ ! -d "$MODEL_DIR" ]; then
    echo "Downloading Qwen3-8B..."
    python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3-8B', local_dir='$MODEL_DIR')
print('Download complete')
"
    echo "Model downloaded at $(date)"
else
    echo "Model already at $MODEL_DIR"
fi

# ──────────────────────────────────────────────
# STEP 1: SFT Warmup
# ──────────────────────────────────────────────
echo ""
echo "============================================"
echo "STEP 1: SFT Warmup (Qwen3-8B)"
echo "Started: $(date)"
echo "============================================"

if [ -d "${SFT_MODEL}" ] && ls ${SFT_MODEL}/*.safetensors 1>/dev/null 2>&1; then
    echo "SFT model exists, skipping."
else
    # Generate warmup data if needed
    if [ ! -f "data_gen/data/sft_warmup.jsonl" ]; then
        python data_gen/generate_sft_warmup.py --num_samples 1000 --output data_gen/data/sft_warmup.jsonl
    fi
    torchrun --standalone --nproc_per_node=8 \
        train_sft.py ${CONFIGS_DIR}/config_sft_qwen3_8b.yaml \
        2>&1 | tee sft_qwen3_8b_warmup.log
    echo "SFT completed at $(date)"
fi

if [ ! -d "${SFT_MODEL}" ]; then
    echo "ERROR: SFT model not found at ${SFT_MODEL}"
    exit 1
fi

# ──────────────────────────────────────────────
# STEP 2: GRPO Training (10 tasks)
# ──────────────────────────────────────────────
echo ""
echo "============================================"
echo "STEP 2: GRPO Training (10 tasks)"
echo "Started: $(date)"
echo "============================================"

for task in "${ALL_TASKS[@]}"; do
    OUTPUT_DIR="experiments/rlvr_geometry_paper/results/${PREFIX}_${task}"
    if [ -d "${OUTPUT_DIR}" ] && ls ${OUTPUT_DIR}/checkpoint-* 1>/dev/null 2>&1; then
        echo "  [SKIP] ${task}: checkpoint exists"
        continue
    fi
    echo ""
    echo "Training: ${task} — $(date)"
    torchrun --standalone --nproc_per_node=8 \
        train_grpo.py ${CONFIGS_DIR}/train_${task}.yaml \
        2>&1 | tee ${PREFIX}_${task}_train.log
    echo "Completed: ${task} at $(date)"
done

echo ""
echo "ALL GRPO TRAINING COMPLETE at $(date)"

# ──────────────────────────────────────────────
# STEP 3: Parallel Evaluation (10x10)
# ──────────────────────────────────────────────
echo ""
echo "============================================"
echo "STEP 3: 10x10 Evaluation"
echo "Started: $(date)"
echo "============================================"

EVAL_DIR="experiments/rlvr_geometry_paper/outputs/transfer_matrix_8b"
mkdir -p ${EVAL_DIR}

# Eval in waves of 3
idx=0
while [ $idx -lt ${#ALL_TASKS[@]} ]; do
    PIDS=()
    for gpu_pair in "0,1" "2,3" "4,5"; do
        if [ $idx -ge ${#ALL_TASKS[@]} ]; then break; fi
        task=${ALL_TASKS[$idx]}
        model_dir="experiments/rlvr_geometry_paper/results/${PREFIX}_${task}"
        out_dir="${EVAL_DIR}/${PREFIX}_${task}"
        log="${EVAL_DIR}/${task}_eval.log"

        if [ -d "${model_dir}" ]; then
            echo "  Eval ${task} on GPUs ${gpu_pair} — $(date)"
            CUDA_VISIBLE_DEVICES=${gpu_pair} python evaluate_transfer.py \
                --model "${model_dir}" --tasks "${EVAL_TASKS}" --num_samples 100 \
                --output_dir "${out_dir}" --max_new_tokens 1024 --batch_size 8 \
                > "${log}" 2>&1 &
            PIDS+=($!)
        else
            echo "  [MISS] ${model_dir} not found"
        fi
        idx=$((idx + 1))
    done
    for pid in "${PIDS[@]}"; do wait $pid; done
done
echo "Eval done at $(date)"

# ──────────────────────────────────────────────
# STEP 4: Build 10x10 matrix
# ──────────────────────────────────────────────
echo ""
echo "Building transfer matrix..."

python3 << 'PYEOF'
import json
from pathlib import Path

tasks = ["countdown","maze","mini_sudoku","shortest_path","basic_arithmetic",
         "syllogism","knights_knaves","leg_counting","caesar_cipher","cryptarithm"]
prefix = "grpo_qwen3_8b"
output_dir = Path("experiments/rlvr_geometry_paper/outputs/transfer_matrix_8b")

matrix = {}
models = [f"{prefix}_{t}" for t in tasks]

for model in models:
    model_dir = output_dir / model
    metrics_files = sorted(model_dir.glob("metrics_*.json"))
    if metrics_files:
        data = json.loads(metrics_files[-1].read_text())
        matrix[model] = {t: data["tasks"].get(t, {}).get("accuracy", 0) for t in tasks}
    else:
        print(f"  WARNING: no metrics for {model}")

result = {"matrix": matrix, "tasks": tasks, "models": models}
(output_dir / "transfer_matrix_10x10.json").write_text(json.dumps(result, indent=2))

print("\n" + "="*100)
print("QWEN3-8B 10x10 TRANSFER MATRIX")
print("="*100)
header = f"{'':>18} | " + " | ".join(f"{t[:8]:>8}" for t in tasks)
print(header)
print("-" * len(header))
for model in models:
    t = model.replace(prefix + "_", "")
    if model in matrix:
        vals = " | ".join(f"{matrix[model].get(tk,0)*100:>7.1f}%" for tk in tasks)
        print(f"{t:>18} | {vals}")
print("="*100)
PYEOF

echo ""
echo "============================================"
echo "PIPELINE COMPLETE at $(date)"
echo "============================================"
