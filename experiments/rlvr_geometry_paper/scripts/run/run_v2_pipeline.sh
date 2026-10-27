#!/bin/bash
# ============================================================================
# v2 Pipeline: lr=1e-6, num_generations=16
# Usage: bash experiments/rlvr_geometry_paper/scripts/run/run_v2_pipeline.sh <7b|8b>
# ============================================================================
SIZE=$1
if [ -z "$SIZE" ]; then echo "Usage: bash experiments/rlvr_geometry_paper/scripts/run/run_v2_pipeline.sh <7b|8b>"; exit 1; fi

cd /scratch/sunguha/rlvr_experiments

case $SIZE in
    7b) PREFIX="grpo_qwen25_7b_v2"; CONFIGS="experiments/rlvr_geometry_paper/configs/transfer_7b_v2"; BASELINE="experiments/rlvr_geometry_paper/outputs/sft_qwen25_7b" ;;
    8b) PREFIX="grpo_qwen3_8b_v2"; CONFIGS="experiments/rlvr_geometry_paper/configs/transfer_8b_v2"; BASELINE="experiments/rlvr_geometry_paper/outputs/sft_qwen3_8b" ;;
    *) echo "Unknown size: $SIZE"; exit 1 ;;
esac

TASKS=(countdown maze mini_sudoku shortest_path basic_arithmetic syllogism knights_knaves leg_counting caesar_cipher cryptarithm)
EVAL_TASKS=$(IFS=,; echo "${TASKS[*]}")
EVAL_DIR="experiments/rlvr_geometry_paper/outputs/eval_${SIZE}_v2"

echo "============================================"
echo "${SIZE} V2 PIPELINE (lr=1e-6, gen=16)"
echo "Started: $(date)"
echo "============================================"

# GRPO Training
for task in "${TASKS[@]}"; do
    DIR="experiments/rlvr_geometry_paper/results/${PREFIX}_${task}"
    if [ -d "$DIR" ] && ls $DIR/checkpoint-* 1>/dev/null 2>&1; then
        echo "[SKIP] $task"
        continue
    fi
    echo ""
    echo "Training: $task — $(date)"
    torchrun --standalone --nproc_per_node=8 \
        train_grpo.py ${CONFIGS}/train_${task}.yaml \
        2>&1 | tee ${PREFIX}_${task}_train.log
    if [ -d "$DIR" ] && ls $DIR/checkpoint-* 1>/dev/null 2>&1; then
        echo "Completed: $task at $(date)"
    else
        echo "FAILED: $task at $(date)"
    fi
done

echo ""
echo "ALL TRAINING DONE at $(date)"

# Eval: baseline + 10 models
mkdir -p $EVAL_DIR

echo "Evaluating baseline..."
torchrun --standalone --nproc_per_node=8 evaluate_transfer.py \
    --model "$BASELINE" --tasks "$EVAL_TASKS" \
    --num_samples 50 --max_new_tokens 2048 --batch_size 4 \
    --output_dir "$EVAL_DIR/baseline" \
    2>&1 | tee "$EVAL_DIR/baseline_eval.log"
echo "Baseline done at $(date)"

for task in "${TASKS[@]}"; do
    MODEL_DIR="experiments/rlvr_geometry_paper/results/${PREFIX}_${task}"
    if [ ! -d "$MODEL_DIR" ]; then
        echo "[MISS] $MODEL_DIR"
        continue
    fi
    echo "Evaluating: ${PREFIX}_${task} — $(date)"
    torchrun --standalone --nproc_per_node=8 evaluate_transfer.py \
        --model "$MODEL_DIR" --tasks "$EVAL_TASKS" \
        --num_samples 50 --max_new_tokens 2048 --batch_size 4 \
        --output_dir "$EVAL_DIR/${PREFIX}_${task}" \
        2>&1 | tee "$EVAL_DIR/${task}_eval.log"
    echo "Done: ${PREFIX}_${task} at $(date)"
done

# Build matrix
python3 << PYEOF
import json
from pathlib import Path

tasks = ["countdown","maze","mini_sudoku","shortest_path","basic_arithmetic",
         "syllogism","knights_knaves","leg_counting","caesar_cipher","cryptarithm"]
prefix = "$PREFIX"
eval_dir = Path("$EVAL_DIR")
short = {"countdown":"cntdwn","maze":"maze","mini_sudoku":"sudoku","shortest_path":"path",
         "basic_arithmetic":"arith","syllogism":"syllog","knights_knaves":"knight",
         "leg_counting":"legcnt","caesar_cipher":"caesar","cryptarithm":"crypto"}

matrix = {}
bd = eval_dir / "baseline"
mf = sorted(bd.glob("metrics_*.json"))
if mf:
    data = json.loads(mf[-1].read_text())
    matrix["baseline"] = {t: data["tasks"].get(t, {}).get("accuracy", 0) for t in tasks}

for t in tasks:
    mn = f"{prefix}_{t}"
    md = eval_dir / mn
    mf = sorted(md.glob("metrics_*.json"))
    if mf:
        data = json.loads(mf[-1].read_text())
        matrix[mn] = {tk: data["tasks"].get(tk, {}).get("accuracy", 0) for tk in tasks}

result = {"matrix": matrix, "tasks": tasks}
(eval_dir / "transfer_matrix.json").write_text(json.dumps(result, indent=2))

baseline = matrix.get("baseline", {})
print(f"\n$SIZE v2 DELTA FROM BASELINE (lr=1e-6, gen=16)")
print("=" * 120)
empty = ""
header = f"{empty:>16} | " + " | ".join(f"{short[t]:>8}" for t in tasks)
print(header)
print("-" * len(header))
row = " | ".join(f"{baseline.get(t,0)*100:>7.0f}%" for t in tasks)
print(f"{'[baseline]':>16} | {row}")
print("-" * len(header))
for mn, vals in matrix.items():
    if mn == "baseline": continue
    trained = mn.replace(prefix + "_", "")
    parts = []
    for t in tasks:
        delta = (vals.get(t,0) - baseline.get(t,0)) * 100
        if t == trained:
            parts.append(f"  *{delta:>+5.0f}%*")
        elif delta != 0:
            parts.append(f"  {delta:>+6.0f}%")
        else:
            parts.append(f"      0% ")
    print(f"{trained[:16]:>16} | " + " | ".join(parts))
print("=" * 120)
PYEOF

echo ""
echo "============================================"
echo "${SIZE} V2 PIPELINE COMPLETE at $(date)"
echo "============================================"
