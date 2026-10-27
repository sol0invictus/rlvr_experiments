#!/bin/bash
# ============================================================================
# Run all evals with DDP (torchrun 8 GPUs), 50 samples, 2048 max tokens
# Usage: bash experiments/rlvr_geometry_paper/scripts/run/run_eval_all.sh [3b|4b|7b|8b]
# ============================================================================
cd /scratch/sunguha/rlvr_experiments

SIZE=$1
if [ -z "$SIZE" ]; then
    echo "Usage: bash experiments/rlvr_geometry_paper/scripts/run/run_eval_all.sh [3b|4b|7b|8b]"
    exit 1
fi

EVAL_TASKS="countdown,maze,mini_sudoku,shortest_path,basic_arithmetic,syllogism,knights_knaves,leg_counting,caesar_cipher,cryptarithm"
NUM_SAMPLES=50
MAX_TOKENS=2048

# Set model prefix and eval output dir
case $SIZE in
    3b)
        PREFIX="grpo_qwen25_3b"
        EVAL_DIR="experiments/rlvr_geometry_paper/outputs/eval_3b"
        BASELINE="experiments/rlvr_geometry_paper/outputs/sft_qwen25_3b"
        MODELS=(countdown maze mini_sudoku shortest_path basic_arithmetic syllogism knights_knaves leg_counting caesar_cipher cryptarithm)
        ;;
    4b)
        PREFIX="grpo_qwen4b"
        EVAL_DIR="experiments/rlvr_geometry_paper/outputs/eval_4b"
        BASELINE="/scratch/sunguha/sft_qwen_4.0b_instruct"
        MODELS=(countdown maze mini_sudoku shortest_path basic_arithmetic syllogism knights_knaves leg_counting caesar_cipher cryptarithm)
        ;;
    7b)
        PREFIX="grpo_qwen25_7b"
        EVAL_DIR="experiments/rlvr_geometry_paper/outputs/eval_7b"
        BASELINE="experiments/rlvr_geometry_paper/outputs/sft_qwen25_7b"
        MODELS=(countdown maze mini_sudoku shortest_path basic_arithmetic syllogism knights_knaves leg_counting caesar_cipher cryptarithm)
        ;;
    8b)
        PREFIX="grpo_qwen3_8b"
        EVAL_DIR="experiments/rlvr_geometry_paper/outputs/eval_8b"
        BASELINE="experiments/rlvr_geometry_paper/outputs/sft_qwen3_8b"
        MODELS=(countdown maze mini_sudoku shortest_path basic_arithmetic syllogism knights_knaves leg_counting caesar_cipher cryptarithm)
        ;;
esac

mkdir -p $EVAL_DIR

echo "============================================"
echo "EVAL: $SIZE (DDP 8 GPUs, $NUM_SAMPLES samples, $MAX_TOKENS tokens)"
echo "Started: $(date)"
echo "============================================"

# Eval baseline first
echo ""
echo "Evaluating: baseline ($SIZE)"
BASELINE_DIR="$EVAL_DIR/baseline"
if [ -d "$BASELINE" ]; then
    torchrun --standalone --nproc_per_node=8 evaluate_transfer.py \
        --model "$BASELINE" --tasks "$EVAL_TASKS" \
        --num_samples $NUM_SAMPLES --max_new_tokens $MAX_TOKENS --batch_size 4 \
        --output_dir "$BASELINE_DIR" \
        2>&1 | tee "$EVAL_DIR/baseline_eval.log"
    echo "Baseline done at $(date)"
else
    echo "[MISS] Baseline not found: $BASELINE"
fi

# Eval each GRPO model
for task in "${MODELS[@]}"; do
    # Handle 4B countdown special naming
    if [ "$SIZE" = "4b" ] && [ "$task" = "countdown" ]; then
        MODEL_DIR="experiments/rlvr_geometry_paper/results/grpo_qwen_4b_instruct_countdown_4to5"
        MODEL_NAME="grpo_qwen_4b_instruct_countdown_4to5"
    else
        MODEL_DIR="experiments/rlvr_geometry_paper/results/${PREFIX}_${task}"
        MODEL_NAME="${PREFIX}_${task}"
    fi

    OUT_DIR="$EVAL_DIR/$MODEL_NAME"

    if [ ! -d "$MODEL_DIR" ]; then
        echo "[MISS] $MODEL_DIR not found"
        continue
    fi

    echo ""
    echo "Evaluating: $MODEL_NAME — $(date)"
    torchrun --standalone --nproc_per_node=8 evaluate_transfer.py \
        --model "$MODEL_DIR" --tasks "$EVAL_TASKS" \
        --num_samples $NUM_SAMPLES --max_new_tokens $MAX_TOKENS --batch_size 4 \
        --output_dir "$OUT_DIR" \
        2>&1 | tee "$EVAL_DIR/${task}_eval.log"
    echo "Done: $MODEL_NAME at $(date)"
done

echo ""
echo "============================================"
echo "$SIZE EVAL COMPLETE at $(date)"
echo "============================================"

# Build matrix
python3 << PYEOF
import json
from pathlib import Path

tasks = ["countdown","maze","mini_sudoku","shortest_path","basic_arithmetic",
         "syllogism","knights_knaves","leg_counting","caesar_cipher","cryptarithm"]
eval_dir = Path("$EVAL_DIR")
prefix = "$PREFIX"
size = "$SIZE"

matrix = {}

# Baseline
bd = eval_dir / "baseline"
mf = sorted(bd.glob("metrics_*.json"))
if mf:
    data = json.loads(mf[-1].read_text())
    matrix["baseline"] = {t: data["tasks"].get(t, {}).get("accuracy", 0) for t in tasks}

# GRPO models
model_names = []
for t in tasks:
    if size == "4b" and t == "countdown":
        mn = "grpo_qwen_4b_instruct_countdown_4to5"
    else:
        mn = f"{prefix}_{t}"
    model_names.append(mn)
    md = eval_dir / mn
    mf = sorted(md.glob("metrics_*.json"))
    if mf:
        data = json.loads(mf[-1].read_text())
        matrix[mn] = {tk: data["tasks"].get(tk, {}).get("accuracy", 0) for tk in tasks}

result = {"matrix": matrix, "tasks": tasks, "models": ["baseline"] + model_names}
(eval_dir / "transfer_matrix.json").write_text(json.dumps(result, indent=2))

print(f"\n{size.upper()} Transfer Matrix (50 samples, 2048 tokens)")
print("="*120)
header = f"{'':>22} | " + " | ".join(f"{t[:10]:>10}" for t in tasks)
print(header)
print("-" * len(header))
for mn in ["baseline"] + model_names:
    label = mn.replace(prefix + "_", "").replace("grpo_qwen_4b_instruct_", "")[:20]
    if mn in matrix:
        vals = " | ".join(f"{matrix[mn].get(t,0)*100:>9.1f}%" for t in tasks)
        print(f"{label:>22} | {vals}")
print("="*120)
PYEOF
