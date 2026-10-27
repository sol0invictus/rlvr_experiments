#!/bin/bash
# Batch-run ARS probes on a given SFT checkpoint for all relevant tasks.
# Usage: bash run_ars_batch.sh {3b|4b|8b}
set -e
SCALE="${1:-}"
if [ -z "$SCALE" ]; then echo "Usage: $0 3b|4b|8b"; exit 1; fi
cd /code-fsx/sunguha/rlvr_experiments
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

case "$SCALE" in
  3b) SFT=/code-fsx/sunguha/rlvr_experiments/outputs/sft_qwen25_3b ;;
  4b) SFT=/code-fsx/sunguha/sft_qwen_4.0b_instruct ;;
  8b) SFT=/code-fsx/sunguha/rlvr_experiments/outputs/sft_qwen3_8b ;;
  *) echo "unknown scale"; exit 1;;
esac

OUT_DIR=experiments/rlvr_geometry_paper/paper/data/ars/${SCALE}
mkdir -p $OUT_DIR
LOG_DIR=experiments/rlvr_geometry_paper/paper/data/logs/ars_${SCALE}
mkdir -p $LOG_DIR

# 6 tasks with atoms (leg_counting excluded)
TASKS="basic_arithmetic countdown shortest_path maze mini_sudoku syllogism knights_knaves"

echo "=== $(date) ARS batch @ $SCALE ==="
for task in $TASKS; do
  out=$OUT_DIR/ars_${SCALE}_${task}.json
  if [ -f "$out" ]; then echo "  skip $task (exists)"; continue; fi
  echo "  $(date +%H:%M:%S) probing $task on $SCALE"
  python3 experiments/rlvr_geometry_paper/scripts/eval/probe_atomics.py \
    --model_path "$SFT" --task "$task" --num_samples 50 \
    --output "$out" > "$LOG_DIR/${task}.log" 2>&1 || echo "    FAILED: $task"
done
echo "=== $(date) DONE ==="
