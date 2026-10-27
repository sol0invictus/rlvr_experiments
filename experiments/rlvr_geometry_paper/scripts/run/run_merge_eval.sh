#!/bin/bash
# Merge task vectors from shortest_path (RFT-Base) and arithmetic (Atomic-Only)
# then evaluate on both tasks
set -e
cd /scratch/sunguha/rlvr_experiments

BASE="/scratch/sunguha/sft_qwen_4.0b_instruct"
MODEL_SP="experiments/rlvr_geometry_paper/results/phase2_sp/idea_rft_base/grpo"
MODEL_ARITH="experiments/rlvr_geometry_paper/results/phase2_arith/stage2_l2"
MERGE_DIR="experiments/rlvr_geometry_paper/results/phase2_sp/merged"
EVAL="torchrun --nproc_per_node=8 evaluate_transfer.py"
EVAL_ARGS="--tasks shortest_path,basic_arithmetic,countdown --num_samples 200 --seed 9999 --batch_size 8"

echo "========================================"
echo "  Task Vector Merging Experiment"
echo "  A = shortest_path (RFT-Base, 55.5%)"
echo "  B = basic_arithmetic (Atomic-Only, 78%)"
echo "========================================"

echo "[1] Merging with all strategies..."
python3 experiments/rlvr_geometry_paper/scripts/merge/merge_task_vectors.py \
  --base "$BASE" \
  --model_a "$MODEL_SP" \
  --model_b "$MODEL_ARITH" \
  --output_dir "$MERGE_DIR" \
  --strategy all

echo ""
echo "[2] Evaluating all merged models..."
for strategy in basic scaled_08 scaled_05 depth layer_type ties dare rotation; do
    model_path="$MERGE_DIR/$strategy"
    if [ -d "$model_path" ]; then
        echo ""
        echo "--- $strategy ---"
        $EVAL --model "$model_path" $EVAL_ARGS \
          --output_dir "$MERGE_DIR/${strategy}_eval" 2>&1 | grep -E "Accuracy|Task "
    fi
done

echo ""
echo "========================================"
echo "  RESULTS SUMMARY"
echo "========================================"

# Reference evals
echo "Reference (individual models):"
echo "  shortest_path model:    SP=55.5%, Arith=70%, Count=32%"
echo "  arithmetic model:       SP=?%, Arith=78%, Count=?%"
echo ""
echo "Merged models:"
for strategy in basic scaled_08 scaled_05 depth layer_type ties dare rotation; do
    eval_dir="$MERGE_DIR/${strategy}_eval"
    result_file=$(ls $eval_dir/metrics_*.json 2>/dev/null | head -1)
    if [ -n "$result_file" ]; then
        python3 -c "
import json
d = json.load(open('$result_file'))
tasks = d.get('tasks', {})
parts = []
for t in ['shortest_path', 'basic_arithmetic', 'countdown']:
    if t in tasks:
        parts.append('%s=%.0f%%' % (t[:5], tasks[t]['accuracy']*100))
print('  %-15s %s' % ('$strategy', '  '.join(parts)))
"
    fi
done
echo "========================================"
