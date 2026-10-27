#!/bin/bash
# Evaluate all RLVR checkpoints on MMLU and GSM8K
# Run on Green cluster pod: sunguha-rlvr-phase2-ext-otgf-worker-0
# v2: Fixed model paths to point to actual model checkpoints (not eval output dirs)

set -e
export PATH=/dev/shm/.local/bin:$PATH
cd /scratch/sunguha/rlvr_experiments

OUTPUT_DIR="experiments/rlvr_geometry_paper/outputs/mmlu_gsm8k_eval_v2"
mkdir -p $OUTPUT_DIR
LOG_FILE="logs/mmlu_gsm8k_eval_v2.log"
mkdir -p logs

# Tasks: math-related MMLU + GSM8K
TASKS="gsm8k,mmlu_abstract_algebra,mmlu_high_school_mathematics,mmlu_elementary_mathematics"

echo "=============================================="
echo "MMLU/GSM8K Evaluation for RLVR Phases 1-4 (v2)"
echo "Tasks: $TASKS"
echo "Started: $(date)"
echo "=============================================="

run_eval() {
    local name=$1
    local path=$2
    local output_file="$OUTPUT_DIR/${name}.json"

    # Check if already evaluated (look for files matching pattern)
    if ls "$OUTPUT_DIR/${name}"*.json 1>/dev/null 2>&1; then
        echo "  [SKIP] $name - already evaluated"
        return
    fi

    if [ ! -d "$path" ]; then
        echo "  [SKIP] $name - path not found: $path"
        return
    fi

    # Check for config.json
    if [ ! -f "$path/config.json" ]; then
        echo "  [SKIP] $name - no config.json at: $path"
        return
    fi

    echo "  [EVAL] $name..."
    if python -m lm_eval \
        --model hf \
        --model_args pretrained=$path,trust_remote_code=True,dtype=bfloat16 \
        --tasks $TASKS \
        --batch_size 4 \
        --output_path $output_file \
        2>&1 | tee -a $LOG_FILE; then
        echo "  [DONE] $name"
    else
        echo "  [ERROR] $name - evaluation failed"
    fi
}

# ==============================================================================
# SFT Baselines (3 models)
# ==============================================================================
echo ""
echo "=== SFT Baselines ==="
run_eval "sft_3b" "/scratch/sunguha/rlvr_experiments/outputs/sft_qwen25_3b"
run_eval "sft_4b" "/scratch/sunguha/sft_qwen_4.0b_instruct"
run_eval "sft_8b" "/scratch/sunguha/rlvr_experiments/outputs/sft_qwen3_8b"

# ==============================================================================
# Phase 1: 4B GRPO checkpoints (CORRECTED PATHS)
# ==============================================================================
echo ""
echo "=== Phase 1: 4B GRPO checkpoints ==="
run_eval "grpo_4b_arithmetic" "/scratch/sunguha/rlvr_experiments/results/grpo_qwen4b_basic_arithmetic"
run_eval "grpo_4b_countdown" "/scratch/sunguha/rlvr_experiments/results/grpo_qwen_4b_instruct_countdown_4to5"
run_eval "grpo_4b_shortest_path" "/scratch/sunguha/rlvr_experiments/results/grpo_qwen4b_shortest_path"
run_eval "grpo_4b_syllogism" "/scratch/sunguha/rlvr_experiments/results/grpo_qwen4b_syllogism"
run_eval "grpo_4b_maze" "/scratch/sunguha/rlvr_experiments/results/grpo_qwen4b_maze"
run_eval "grpo_4b_knights_knaves" "/scratch/sunguha/rlvr_experiments/results/grpo_qwen4b_knights_knaves"
run_eval "grpo_4b_mini_sudoku" "/scratch/sunguha/rlvr_experiments/results/grpo_qwen4b_mini_sudoku"
run_eval "grpo_4b_leg_counting" "/scratch/sunguha/rlvr_experiments/results/grpo_qwen4b_leg_counting"
run_eval "grpo_4b_caesar_cipher" "/scratch/sunguha/rlvr_experiments/results/grpo_qwen4b_caesar_cipher"
run_eval "grpo_4b_cryptarithm" "/scratch/sunguha/rlvr_experiments/results/grpo_qwen4b_cryptarithm"

# ==============================================================================
# Phase 1: 3B GRPO checkpoints
# ==============================================================================
echo ""
echo "=== Phase 1: 3B GRPO checkpoints ==="
run_eval "grpo_3b_arithmetic" "/scratch/sunguha/rlvr_experiments/results/grpo_qwen25_3b_basic_arithmetic"
run_eval "grpo_3b_countdown" "/scratch/sunguha/rlvr_experiments/results/grpo_qwen25_3b_countdown"
run_eval "grpo_3b_shortest_path" "/scratch/sunguha/rlvr_experiments/results/grpo_qwen25_3b_shortest_path"
run_eval "grpo_3b_syllogism" "/scratch/sunguha/rlvr_experiments/results/grpo_qwen25_3b_syllogism"
run_eval "grpo_3b_maze" "/scratch/sunguha/rlvr_experiments/results/grpo_qwen25_3b_maze"
run_eval "grpo_3b_knights_knaves" "/scratch/sunguha/rlvr_experiments/results/grpo_qwen25_3b_knights_knaves"
run_eval "grpo_3b_mini_sudoku" "/scratch/sunguha/rlvr_experiments/results/grpo_qwen25_3b_mini_sudoku"
run_eval "grpo_3b_leg_counting" "/scratch/sunguha/rlvr_experiments/results/grpo_qwen25_3b_leg_counting"

# ==============================================================================
# Phase 1: 8B GRPO checkpoints
# ==============================================================================
echo ""
echo "=== Phase 1: 8B GRPO checkpoints ==="
run_eval "grpo_8b_arithmetic" "/scratch/sunguha/rlvr_experiments/results/grpo_qwen3_8b_basic_arithmetic"
run_eval "grpo_8b_countdown" "/scratch/sunguha/rlvr_experiments/results/grpo_qwen3_8b_countdown"
run_eval "grpo_8b_shortest_path" "/scratch/sunguha/rlvr_experiments/results/grpo_qwen3_8b_shortest_path"
run_eval "grpo_8b_syllogism" "/scratch/sunguha/rlvr_experiments/results/grpo_qwen3_8b_syllogism"
run_eval "grpo_8b_maze" "/scratch/sunguha/rlvr_experiments/results/grpo_qwen3_8b_maze"
run_eval "grpo_8b_knights_knaves" "/scratch/sunguha/rlvr_experiments/results/grpo_qwen3_8b_knights_knaves"
run_eval "grpo_8b_mini_sudoku" "/scratch/sunguha/rlvr_experiments/results/grpo_qwen3_8b_mini_sudoku"
run_eval "grpo_8b_leg_counting" "/scratch/sunguha/rlvr_experiments/results/grpo_qwen3_8b_leg_counting"

# ==============================================================================
# Phase 2: Atomic curriculum checkpoints (arithmetic, shortest_path)
# ==============================================================================
echo ""
echo "=== Phase 2: Atomic Curriculum ==="
# Arithmetic
run_eval "p2_arith_atomic_only" "/scratch/sunguha/rlvr_experiments/results/phase2_arith/stage2_l2"
run_eval "p2_arith_atomic_warm" "/scratch/sunguha/rlvr_experiments/results/phase2_arith/stage3_full"
# Shortest path
run_eval "p2_sp_atomic_warm" "/scratch/sunguha/rlvr_experiments/results/phase2_sp/atomic_warm_300"
run_eval "p2_sp_rft_base" "/scratch/sunguha/rlvr_experiments/results/phase2_sp/idea_rft_base/grpo"
run_eval "p2_sp_idea_b" "/scratch/sunguha/rlvr_experiments/results/phase2_sp/idea_b/stage2_full"

# ==============================================================================
# Phase 2 Extended: New 4 tasks
# ==============================================================================
echo ""
echo "=== Phase 2 Extended: 4 new tasks ==="
for task in syllogism knights_knaves maze mini_sudoku; do
    run_eval "p2ext_${task}_direct" "/scratch/sunguha/rlvr_experiments/results/phase2_ext/${task}/direct_rlvr"
    run_eval "p2ext_${task}_atomic_only" "/scratch/sunguha/rlvr_experiments/results/phase2_ext/${task}/stage2_l2"
    run_eval "p2ext_${task}_atomic_warm" "/scratch/sunguha/rlvr_experiments/results/phase2_ext/${task}/stage3_full"
done

# ==============================================================================
# Phase 4: Merged models
# ==============================================================================
echo ""
echo "=== Phase 4: Merged Models ==="
# Pair merges
for merge_dir in /scratch/sunguha/rlvr_experiments/results/phase4_pairs/T*; do
    if [ -d "$merge_dir" ]; then
        merge_name=$(basename $merge_dir)
        run_eval "p4_${merge_name}" "$merge_dir"
    fi
done

# Layer-selective merges
for merge_dir in /scratch/sunguha/rlvr_experiments/results/phase4_layersel/*; do
    if [ -d "$merge_dir" ]; then
        merge_name=$(basename $merge_dir)
        run_eval "p4_layersel_${merge_name}" "$merge_dir"
    fi
done

echo ""
echo "=============================================="
echo "Evaluation Complete!"
echo "Finished: $(date)"
echo "=============================================="

# Summarize results
echo ""
echo "=== Results Summary ==="
python3 << 'EOF'
import json
import os
from pathlib import Path

output_dir = Path("/scratch/sunguha/rlvr_experiments/outputs/mmlu_gsm8k_eval_v2")
results = {}

for f in output_dir.glob("*.json"):
    try:
        with open(f) as fp:
            data = json.load(fp)
        # Extract model name (strip timestamp if present)
        model = f.stem.split('_2026-')[0]
        results[model] = {}
        if "results" in data:
            for task, metrics in data["results"].items():
                key = "exact_match,strict-match" if "exact_match,strict-match" in metrics else "acc,none"
                if key in metrics:
                    results[model][task] = metrics[key]
                elif "acc" in metrics:
                    results[model][task] = metrics["acc"]
                elif "exact_match" in metrics:
                    results[model][task] = metrics["exact_match"]
    except Exception as e:
        print(f"Error reading {f}: {e}")

if not results:
    print("No results found.")
else:
    # Print table
    all_tasks = sorted(set(t for r in results.values() for t in r.keys()))
    header = "Model".ljust(35) + " | " + " | ".join(t[:15].ljust(15) for t in all_tasks)
    print(header)
    print("-" * len(header))
    for model in sorted(results.keys()):
        row = model.ljust(35) + " | "
        for task in all_tasks:
            val = results[model].get(task)
            if val is not None:
                row += f"{val*100:5.1f}%".ljust(15) + " | "
            else:
                row += "N/A".ljust(15) + " | "
        print(row)
EOF
