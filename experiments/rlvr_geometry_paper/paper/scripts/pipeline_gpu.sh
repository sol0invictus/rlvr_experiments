#!/bin/bash
# GPU-serial pipeline: Streams A + B + C
# Wall clock target ~8h on 8xB200
set -e
cd /code-fsx/sunguha/rlvr_experiments
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

mkdir -p experiments/rlvr_geometry_paper/paper/data/logs

echo "=== $(date) GPU pipeline start ==="

##############################################
# Stream A: build merged models (CPU, fast)
##############################################
echo "--- $(date) Stream A: build linking merges ---"
python3 experiments/rlvr_geometry_paper/paper/scripts/stream_A_linking.py 2>&1 | tee experiments/rlvr_geometry_paper/paper/data/logs/stream_A_build.log

##############################################
# Stream B: build TIES + DARE merges (CPU, fast)
##############################################
echo "--- $(date) Stream B: build TIES/DARE merges ---"
SFT=/code-fsx/sunguha/sft_qwen_4.0b_instruct

declare -A PAIRS
# pair_name -> "A_path B_path"
PAIRS[T2_sdk_kk]="experiments/rlvr_geometry_paper/results/grpo_qwen4b_mini_sudoku experiments/rlvr_geometry_paper/results/grpo_qwen4b_knights_knaves"
PAIRS[T5_sp_syl]="experiments/rlvr_geometry_paper/results/grpo_qwen4b_shortest_path experiments/rlvr_geometry_paper/results/grpo_qwen4b_syllogism"
PAIRS[T7_cnt_ari]="experiments/rlvr_geometry_paper/results/grpo_qwen_4b_instruct_countdown experiments/rlvr_geometry_paper/results/grpo_qwen4b_basic_arithmetic"

for pair in T2_sdk_kk T5_sp_syl T7_cnt_ari; do
  IFS=' ' read -r A B <<< "${PAIRS[$pair]}"
  for strat in ties dare; do
    out=experiments/rlvr_geometry_paper/paper/data/baselines/${pair}_${strat}
    if [ -f "$out/model.safetensors" ]; then
      echo "  exists: $out — skip"
      continue
    fi
    mkdir -p "$out"
    echo "  building $pair ($strat) -> $out"
    python3 experiments/rlvr_geometry_paper/scripts/merge/merge_task_vectors.py \
      --base "$SFT" \
      --model_a "$A" --model_b "$B" \
      --output_dir experiments/rlvr_geometry_paper/paper/data/baselines/${pair} \
      --strategy "$strat" 2>&1 | tee experiments/rlvr_geometry_paper/paper/data/logs/merge_${pair}_${strat}.log
  done
done

##############################################
# Now: evaluate all new checkpoints on 10 tasks
##############################################
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
    # Skip per-task if a metrics file mentioning that task in its content exists
    if grep -lq "\"$task\"" "$out"/metrics_*.json 2>/dev/null; then
      continue
    fi
    echo "    task=$task  ($(date +%H:%M:%S))"
    torchrun --standalone --nproc_per_node=8 evaluate_transfer.py \
      --model "$model" \
      --tasks "$task" \
      --num_samples $N --seed $seed --batch_size 8 \
      --output_dir "$out" 2>&1 >> experiments/rlvr_geometry_paper/paper/data/logs/eval_$(basename "$out").log || echo "    WARN: task $task failed"
  done
}

echo "--- $(date) Stream A eval: 3 linking merges ---"
eval_one experiments/rlvr_geometry_paper/paper/data/linking/M_aa  experiments/rlvr_geometry_paper/outputs/eval10_linking_aa  $SEED
eval_one experiments/rlvr_geometry_paper/paper/data/linking/M_dd  experiments/rlvr_geometry_paper/outputs/eval10_linking_dd  $SEED
eval_one experiments/rlvr_geometry_paper/paper/data/linking/M_mix experiments/rlvr_geometry_paper/outputs/eval10_linking_mix $SEED

echo "--- $(date) Stream B eval: 3 pairs × 2 methods = 6 merges ---"
for pair in T2_sdk_kk T5_sp_syl T7_cnt_ari; do
  for strat in ties dare; do
    eval_one experiments/rlvr_geometry_paper/paper/data/baselines/${pair}/${strat} experiments/rlvr_geometry_paper/outputs/eval10_baseline_${strat}_${pair} $SEED
  done
done

echo "--- $(date) Stream C: 4 headline checkpoints × 1 extra seed (scope cut for 1-day) ---"
declare -A HEADLINE
HEADLINE[p2_arith_atomic]="experiments/rlvr_geometry_paper/results/phase2_arith/stage2_l2"
HEADLINE[p2_sp_rft_base]="experiments/rlvr_geometry_paper/results/phase2_sp/idea_rft_base/grpo"
HEADLINE[merge_2task_basic]="experiments/rlvr_geometry_paper/results/phase2_sp/merged/basic"
HEADLINE[phase4_T5_sp_syl]="experiments/rlvr_geometry_paper/results/phase4_pairs/T5_sp_syl/basic"

for name in "${!HEADLINE[@]}"; do
  for seed in 1234; do
    eval_one "${HEADLINE[$name]}" "experiments/rlvr_geometry_paper/outputs/eval10_seed${seed}_${name}" $seed
  done
done

echo "=== $(date) GPU pipeline DONE ==="
