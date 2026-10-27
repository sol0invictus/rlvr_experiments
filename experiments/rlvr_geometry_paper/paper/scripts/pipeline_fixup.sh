#!/bin/bash
# Post-hoc eval for TIES merges (built by fix_ties_and_retry.py)
# and the T7 DARE that was missing.
set -e
cd /code-fsx/sunguha/rlvr_experiments
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

TASKS_LIST=(basic_arithmetic countdown shortest_path maze mini_sudoku syllogism knights_knaves leg_counting caesar_cipher cryptarithm)
SEED=9999
N=200

eval_one() {
  local model=$1
  local out=$2
  local seed=${3:-9999}
  if [ ! -d "$model" ]; then echo "  missing model: $model"; return; fi
  mkdir -p "$out"
  local n_done=$(ls "$out"/metrics_*.json 2>/dev/null | wc -l)
  if [ "$n_done" -ge 10 ]; then
    echo "  skip $out (complete, $n_done metrics)"
    return
  fi
  echo "  eval: model=$model out=$out seed=$seed  ($(date +%H:%M:%S))"
  for task in "${TASKS_LIST[@]}"; do
    if grep -lq "\"$task\"" "$out"/metrics_*.json 2>/dev/null; then continue; fi
    echo "    task=$task  ($(date +%H:%M:%S))"
    torchrun --standalone --nproc_per_node=8 evaluate_transfer.py \
      --model "$model" --tasks "$task" \
      --num_samples $N --seed $seed --batch_size 8 \
      --output_dir "$out" 2>&1 >> experiments/rlvr_geometry_paper/paper/data/logs/eval_$(basename "$out").log || echo "    WARN: task $task failed"
  done
}

echo "=== $(date) fixup pipeline ==="

# All 6 baseline evals (re-do the failed ties, add T7 dare)
for pair in T2_sdk_kk T5_sp_syl T7_cnt_ari; do
  for strat in ties dare; do
    eval_one experiments/rlvr_geometry_paper/paper/data/baselines/${pair}/${strat} experiments/rlvr_geometry_paper/outputs/eval10_baseline_${strat}_${pair} $SEED
  done
done

echo "=== $(date) fixup DONE ==="
