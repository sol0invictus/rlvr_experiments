#!/bin/bash
# ============================================================================
# Train 4 new tasks on all 3 model sizes + build 10x10 transfer matrices
# ============================================================================
# New tasks: knights_knaves, leg_counting, caesar_cipher, cryptarithm
# Models: 3B, 4B, 7B
# 12 GRPO runs total, then 10x10 eval for each size
#
# Usage: nohup bash experiments/rlvr_geometry_paper/scripts/run/run_new_tasks_all_sizes.sh > new_tasks_all_sizes.log 2>&1 &
# Hardware: 8x B200 (P6)
# ============================================================================

set -e
cd /scratch/sunguha/rlvr_experiments

NEW_TASKS=(knights_knaves leg_counting caesar_cipher cryptarithm)

ALL_TASKS="countdown,maze,mini_sudoku,shortest_path,basic_arithmetic,syllogism,knights_knaves,leg_counting,caesar_cipher,cryptarithm"

echo "============================================"
echo "NEW TASKS TRAINING (ALL SIZES)"
echo "Started: $(date)"
echo "Tasks: ${NEW_TASKS[*]}"
echo "============================================"

# ──────────────────────────────────────────────
# STEP 1: GRPO Training — 3B models
# ──────────────────────────────────────────────
echo ""
echo "============================================"
echo "TRAINING: Qwen2.5-3B (4 new tasks)"
echo "============================================"

for task in "${NEW_TASKS[@]}"; do
    OUTPUT_DIR="experiments/rlvr_geometry_paper/results/grpo_qwen25_3b_${task}"
    if [ -d "${OUTPUT_DIR}" ] && ls ${OUTPUT_DIR}/checkpoint-* 1>/dev/null 2>&1; then
        echo "  [SKIP] 3B ${task}: checkpoint exists"
        continue
    fi
    echo ""
    echo "Training: 3B ${task} — $(date)"
    torchrun --standalone --nproc_per_node=8 \
        train_grpo.py experiments/rlvr_geometry_paper/configs/transfer_3b/train_${task}.yaml \
        2>&1 | tee grpo_qwen25_3b_${task}_train.log
    echo "Completed: 3B ${task} at $(date)"
done

# ──────────────────────────────────────────────
# STEP 2: GRPO Training — 4B models
# ──────────────────────────────────────────────
echo ""
echo "============================================"
echo "TRAINING: Qwen3-4B (4 new tasks)"
echo "============================================"

for task in "${NEW_TASKS[@]}"; do
    OUTPUT_DIR="experiments/rlvr_geometry_paper/results/grpo_qwen4b_${task}"
    if [ -d "${OUTPUT_DIR}" ] && ls ${OUTPUT_DIR}/checkpoint-* 1>/dev/null 2>&1; then
        echo "  [SKIP] 4B ${task}: checkpoint exists"
        continue
    fi
    echo ""
    echo "Training: 4B ${task} — $(date)"
    torchrun --standalone --nproc_per_node=8 \
        train_grpo.py experiments/rlvr_geometry_paper/configs/transfer_4b/train_${task}.yaml \
        2>&1 | tee grpo_qwen4b_${task}_train.log
    echo "Completed: 4B ${task} at $(date)"
done

# ──────────────────────────────────────────────
# STEP 3: GRPO Training — 7B models
# ──────────────────────────────────────────────
echo ""
echo "============================================"
echo "TRAINING: Qwen2.5-7B (4 new tasks)"
echo "============================================"

for task in "${NEW_TASKS[@]}"; do
    OUTPUT_DIR="experiments/rlvr_geometry_paper/results/grpo_qwen25_7b_${task}"
    if [ -d "${OUTPUT_DIR}" ] && ls ${OUTPUT_DIR}/checkpoint-* 1>/dev/null 2>&1; then
        echo "  [SKIP] 7B ${task}: checkpoint exists"
        continue
    fi
    echo ""
    echo "Training: 7B ${task} — $(date)"
    torchrun --standalone --nproc_per_node=8 \
        train_grpo.py experiments/rlvr_geometry_paper/configs/transfer_7b/train_${task}.yaml \
        2>&1 | tee grpo_qwen25_7b_${task}_train.log
    echo "Completed: 7B ${task} at $(date)"
done

echo ""
echo "============================================"
echo "ALL TRAINING COMPLETE at $(date)"
echo "============================================"

# ──────────────────────────────────────────────
# STEP 4: 10x10 Evaluation — all 3 sizes
# ──────────────────────────────────────────────
echo ""
echo "============================================"
echo "10x10 TRANSFER MATRIX EVALUATION"
echo "============================================"

# --- Helper function: eval one model ---
eval_model() {
    local model_dir=$1
    local output_dir=$2
    local log_file=$3
    local gpus=$4

    if [ -d "${output_dir}" ] && ls ${output_dir}/metrics_* 1>/dev/null 2>&1; then
        echo "  [SKIP] ${output_dir}: metrics exist"
        return 0
    fi

    CUDA_VISIBLE_DEVICES=${gpus} python evaluate_transfer.py \
        --model "${model_dir}" \
        --tasks "${ALL_TASKS}" \
        --num_samples 100 \
        --output_dir "${output_dir}" \
        --max_new_tokens 1024 \
        --batch_size 8 \
        > "${log_file}" 2>&1
}

# Eval task lists (10 tasks per model, 10 models per size = 30 models total)
# We have the original 7 + 4 new = 11 tasks, but dropping tower_of_hanoi → 10 tasks
# Actually keeping all for completeness — eval all trained models

ORIGINAL_TASKS=(countdown maze mini_sudoku shortest_path basic_arithmetic syllogism)

for SIZE in 3b 7b; do
    if [ "$SIZE" = "3b" ]; then
        PREFIX="grpo_qwen25_3b"
        EVAL_DIR="experiments/rlvr_geometry_paper/outputs/transfer_matrix_3b_10x10"
    else
        PREFIX="grpo_qwen25_7b"
        EVAL_DIR="experiments/rlvr_geometry_paper/outputs/transfer_matrix_7b_10x10"
    fi
    mkdir -p ${EVAL_DIR}

    echo ""
    echo "Evaluating ${SIZE} models (parallel waves)..."

    # Wave: eval 3 models at a time
    ALL_MODELS=()
    for t in "${ORIGINAL_TASKS[@]}" "${NEW_TASKS[@]}"; do
        ALL_MODELS+=("${t}")
    done

    idx=0
    while [ $idx -lt ${#ALL_MODELS[@]} ]; do
        PIDS=()
        for gpu_pair in "0,1" "2,3" "4,5"; do
            if [ $idx -ge ${#ALL_MODELS[@]} ]; then break; fi
            task=${ALL_MODELS[$idx]}
            model_dir="experiments/rlvr_geometry_paper/results/${PREFIX}_${task}"
            out_dir="${EVAL_DIR}/${PREFIX}_${task}"
            log="${EVAL_DIR}/${task}_eval.log"

            if [ -d "${model_dir}" ]; then
                echo "  Eval ${SIZE} ${task} on GPUs ${gpu_pair}"
                eval_model "${model_dir}" "${out_dir}" "${log}" "${gpu_pair}" &
                PIDS+=($!)
            else
                echo "  [MISS] ${model_dir} not found"
            fi
            idx=$((idx + 1))
        done
        for pid in "${PIDS[@]}"; do wait $pid; done
    done
    echo "${SIZE} eval done at $(date)"
done

# 4B models (different naming)
echo ""
echo "Evaluating 4B models..."
EVAL_DIR_4B="experiments/rlvr_geometry_paper/outputs/transfer_matrix_4b_10x10"
mkdir -p ${EVAL_DIR_4B}

ALL_4B_MODELS=()
for t in "${ORIGINAL_TASKS[@]}"; do
    # 4B original models have different naming
    if [ "$t" = "countdown" ]; then
        ALL_4B_MODELS+=("grpo_qwen_4b_instruct_countdown_4to5")
    else
        ALL_4B_MODELS+=("grpo_qwen4b_${t}")
    fi
done
for t in "${NEW_TASKS[@]}"; do
    ALL_4B_MODELS+=("grpo_qwen4b_${t}")
done

idx=0
while [ $idx -lt ${#ALL_4B_MODELS[@]} ]; do
    PIDS=()
    for gpu_pair in "0,1" "2,3" "4,5"; do
        if [ $idx -ge ${#ALL_4B_MODELS[@]} ]; then break; fi
        model=${ALL_4B_MODELS[$idx]}
        model_dir="experiments/rlvr_geometry_paper/results/${model}"
        out_dir="${EVAL_DIR_4B}/${model}"
        log="${EVAL_DIR_4B}/${model}_eval.log"

        if [ -d "${model_dir}" ]; then
            echo "  Eval 4B ${model} on GPUs ${gpu_pair}"
            eval_model "${model_dir}" "${out_dir}" "${log}" "${gpu_pair}" &
            PIDS+=($!)
        else
            echo "  [MISS] ${model_dir} not found"
        fi
        idx=$((idx + 1))
    done
    for pid in "${PIDS[@]}"; do wait $pid; done
done
echo "4B eval done at $(date)"

# ──────────────────────────────────────────────
# STEP 5: Build 10x10 matrices
# ──────────────────────────────────────────────
echo ""
echo "Building 10x10 transfer matrices..."

python3 << 'PYEOF'
import json
from pathlib import Path

all_tasks = ["countdown","maze","mini_sudoku","shortest_path","basic_arithmetic",
             "syllogism","knights_knaves","leg_counting","caesar_cipher","cryptarithm"]

sizes = {
    "3b": ("experiments/rlvr_geometry_paper/outputs/transfer_matrix_3b_10x10", "grpo_qwen25_3b"),
    "4b": ("experiments/rlvr_geometry_paper/outputs/transfer_matrix_4b_10x10", "grpo_qwen4b"),
    "7b": ("experiments/rlvr_geometry_paper/outputs/transfer_matrix_7b_10x10", "grpo_qwen25_7b"),
}

# Special 4B countdown name
COUNTDOWN_4B = "grpo_qwen_4b_instruct_countdown_4to5"

for size, (eval_dir, prefix) in sizes.items():
    output_dir = Path(eval_dir)
    matrix = {}
    models = []

    for task in all_tasks:
        if size == "4b" and task == "countdown":
            model_name = COUNTDOWN_4B
        else:
            model_name = f"{prefix}_{task}"
        models.append(model_name)

        model_dir = output_dir / model_name
        metrics_files = sorted(model_dir.glob("metrics_*.json"))
        if metrics_files:
            data = json.loads(metrics_files[-1].read_text())
            matrix[model_name] = {t: data["tasks"].get(t, {}).get("accuracy", 0) for t in all_tasks}
        else:
            print(f"  WARNING ({size}): no metrics for {model_name}")

    result = {"matrix": matrix, "tasks": all_tasks, "models": models}
    out_file = output_dir / "transfer_matrix_10x10.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(result, indent=2))

    print(f"\n{'='*100}")
    print(f"{size.upper()} 10x10 TRANSFER MATRIX")
    print(f"{'='*100}")
    header = f"{'':>18} | " + " | ".join(f"{t[:8]:>8}" for t in all_tasks)
    print(header)
    print("-" * len(header))
    for model_name in models:
        t = model_name.replace(prefix + "_", "").replace("grpo_qwen_4b_instruct_", "")
        if model_name in matrix:
            vals = " | ".join(f"{matrix[model_name].get(tk,0)*100:>7.1f}%" for tk in all_tasks)
            print(f"{t:>18} | {vals}")
    print(f"{'='*100}")

print("\nAll matrices saved.")
PYEOF

echo ""
echo "============================================"
echo "PIPELINE COMPLETE at $(date)"
echo "============================================"
