#!/bin/bash
cd /scratch/sunguha/rlvr_experiments

MODELS=(
    "experiments/rlvr_geometry_paper/outputs/sft_qwen25_3b|3B-baseline"
    "experiments/rlvr_geometry_paper/results/grpo_qwen25_3b_basic_arithmetic|3B-GRPO"
    "/scratch/sunguha/sft_qwen_4.0b_instruct|4B-baseline"
    "experiments/rlvr_geometry_paper/results/grpo_qwen4b_basic_arithmetic|4B-GRPO"
    "experiments/rlvr_geometry_paper/outputs/sft_qwen25_7b|7B-baseline"
    "experiments/rlvr_geometry_paper/results/grpo_qwen25_7b_basic_arithmetic|7B-GRPO"
    "experiments/rlvr_geometry_paper/outputs/sft_qwen3_8b|8B-baseline"
    "experiments/rlvr_geometry_paper/results/grpo_qwen3_8b_basic_arithmetic|8B-GRPO"
)

for entry in "${MODELS[@]}"; do
    model="${entry%%|*}"
    label="${entry##*|}"
    if [ ! -d "$model" ]; then
        echo "[MISS] $label: $model"
        continue
    fi
    torchrun --standalone --nproc_per_node=8 \
        experiments/rlvr_geometry_paper/scripts/eval/eval_arith_train_domain.py "$model" "$label" 2>&1 | strings | grep -E "Model:|TRAIN|EVAL "
done
