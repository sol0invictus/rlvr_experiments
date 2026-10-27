#!/bin/bash
# Qwen3-8B: SFT + GRPO on 10 tasks + eval
cd /scratch/sunguha/rlvr_experiments

PREFIX=grpo_qwen3_8b
SFT_MODEL=experiments/rlvr_geometry_paper/outputs/sft_qwen3_8b
TASKS=(countdown maze mini_sudoku shortest_path basic_arithmetic syllogism knights_knaves leg_counting caesar_cipher cryptarithm)
EVAL_TASKS="countdown,maze,mini_sudoku,shortest_path,basic_arithmetic,syllogism,knights_knaves,leg_counting,caesar_cipher,cryptarithm"

echo "=== QWEN3-8B: SFT + GRPO ==="
echo "Started: $(date)"

# SFT warmup
if [ -d "$SFT_MODEL" ] && ls $SFT_MODEL/*.safetensors 1>/dev/null 2>&1; then
    echo "SFT model exists at $SFT_MODEL, skipping."
else
    echo "Running SFT warmup — $(date)"
    torchrun --standalone --nproc_per_node=8 \
        train_sft.py experiments/rlvr_geometry_paper/configs/transfer_8b/config_sft_qwen3_8b.yaml \
        2>&1 | tee sft_qwen3_8b_warmup.log
    echo "SFT done at $(date)"
fi

for task in "${TASKS[@]}"; do
    DIR="experiments/rlvr_geometry_paper/results/${PREFIX}_${task}"
    if [ -d "$DIR" ] && ls $DIR/checkpoint-* 1>/dev/null 2>&1; then
        echo "[SKIP] $task"
        continue
    fi
    echo ""
    echo "Training: $task — $(date)"
    torchrun --standalone --nproc_per_node=8 \
        train_grpo.py experiments/rlvr_geometry_paper/configs/transfer_8b/train_${task}.yaml \
        2>&1 | tee ${PREFIX}_${task}_train.log
    # Check if training produced a checkpoint
    if [ -d "$DIR" ] && ls $DIR/checkpoint-* 1>/dev/null 2>&1; then
        echo "Completed: $task at $(date)"
    else
        echo "FAILED: $task at $(date)"
    fi
done

echo ""
echo "ALL TRAINING DONE at $(date)"

# Eval in parallel waves of 3
EVAL_DIR=experiments/rlvr_geometry_paper/outputs/transfer_matrix_8b
mkdir -p $EVAL_DIR

idx=0
while [ $idx -lt ${#TASKS[@]} ]; do
    PIDS=()
    for gpu in "0,1" "2,3" "4,5"; do
        if [ $idx -ge ${#TASKS[@]} ]; then break; fi
        t=${TASKS[$idx]}
        md="experiments/rlvr_geometry_paper/results/${PREFIX}_${t}"
        if [ -d "$md" ]; then
            echo "Eval $t on GPUs $gpu — $(date)"
            CUDA_VISIBLE_DEVICES=$gpu python evaluate_transfer.py \
                --model "$md" --tasks "$EVAL_TASKS" --num_samples 100 \
                --output_dir "$EVAL_DIR/${PREFIX}_${t}" \
                --max_new_tokens 1024 --batch_size 8 \
                > "$EVAL_DIR/${t}_eval.log" 2>&1 &
            PIDS+=($!)
        else
            echo "[MISS] $md not found, skipping eval"
        fi
        idx=$((idx + 1))
    done
    for p in "${PIDS[@]}"; do wait $p; done
done
echo "EVAL DONE at $(date)"

# Build matrix
python3 << 'PYEOF'
import json
from pathlib import Path

tasks = ["countdown","maze","mini_sudoku","shortest_path","basic_arithmetic",
         "syllogism","knights_knaves","leg_counting","caesar_cipher","cryptarithm"]
prefix = "grpo_qwen3_8b"
output_dir = Path("experiments/rlvr_geometry_paper/outputs/transfer_matrix_8b")
matrix = {}
for t in tasks:
    model = f"{prefix}_{t}"
    mf = sorted((output_dir / model).glob("metrics_*.json"))
    if mf:
        data = json.loads(mf[-1].read_text())
        matrix[model] = {tk: data["tasks"].get(tk, {}).get("accuracy", 0) for tk in tasks}

result = {"matrix": matrix, "tasks": tasks, "models": [f"{prefix}_{t}" for t in tasks]}
(output_dir / "transfer_matrix_10x10.json").write_text(json.dumps(result, indent=2))
print("Matrix saved")
for k, v in matrix.items():
    t = k.replace(prefix + "_", "")
    vals = " ".join(f"{v.get(tk,0)*100:5.1f}%" for tk in tasks)
    print(f"{t:>18}: {vals}")
PYEOF

echo "PIPELINE COMPLETE at $(date)"
