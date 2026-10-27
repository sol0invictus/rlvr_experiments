#!/bin/bash
# F28 replication at a given scale.
# Usage: bash f28_replicate_pipeline.sh {3b|8b}
#
# Steps:
#   1. Build configs for atomic arith + atomic SP (stage1 + stage2) at scale
#   2. Train them in sequence (4 GRPO runs total)
#   3. Build M_aa, M_dd, M_mix merges at that scale
#   4. Eval all 3 merges on 10 tasks
#
# Runs in a tight 8xB200 loop. Each pod should run this for one scale.
set -e
SCALE="${1:-}"
if [ -z "$SCALE" ]; then echo "Usage: $0 3b|8b"; exit 1; fi
cd /code-fsx/sunguha/rlvr_experiments
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

case "$SCALE" in
  3b)
    SFT=/code-fsx/sunguha/rlvr_experiments/outputs/sft_qwen25_3b
    DIRECT_SP=experiments/rlvr_geometry_paper/results/grpo_qwen25_3b_shortest_path
    DIRECT_ARI=experiments/rlvr_geometry_paper/results/grpo_qwen25_3b_basic_arithmetic
    MAX_STEPS_L1=100; MAX_STEPS_L2=100
    ;;
  8b)
    SFT=/code-fsx/sunguha/rlvr_experiments/outputs/sft_qwen3_8b
    DIRECT_SP=experiments/rlvr_geometry_paper/results/grpo_qwen3_8b_shortest_path
    DIRECT_ARI=experiments/rlvr_geometry_paper/results/grpo_qwen3_8b_basic_arithmetic
    MAX_STEPS_L1=100; MAX_STEPS_L2=100
    ;;
  *) echo "Unknown scale: $SCALE"; exit 1;;
esac

OUT_ROOT=experiments/rlvr_geometry_paper/results/phase5_replicate_${SCALE}
CFG_ROOT=configs/phase5_replicate_${SCALE}
LOG_ROOT=experiments/rlvr_geometry_paper/paper/data/logs/replicate_${SCALE}
mkdir -p "$OUT_ROOT/arith_l1" "$OUT_ROOT/arith_l2" "$OUT_ROOT/sp_l1" "$OUT_ROOT/sp_l2"
mkdir -p "$CFG_ROOT"
mkdir -p "$LOG_ROOT"

echo "=== $(date) F28 replication @ $SCALE — pod=$(hostname) ==="

##############################################
# Write configs
##############################################
write_cfg() {
  local path=$1 base_task=$2 stage=$3 model_init=$4 out_dir=$5 max_steps=$6
  # stage is L1 or L2
  local atom_spec
  if [ "$base_task" = "basic_arithmetic" ]; then
    if [ "$stage" = "L1" ]; then atom_spec='
    - L1_number_extraction
    - L1_term_count
    - L1_operator_identification'
    else atom_spec='
    - L2_order_of_operations
    - L2_single_op_eval'
    fi
    local max_prompt=512 max_comp=512
  else
    # shortest_path — use atom_layer keyword from 4B SP config
    if [ "$stage" = "L1" ]; then atom_spec=''; layer_key='  atom_layer: L1'
    else atom_spec=''; layer_key='  atom_layer: L2'
    fi
    local max_prompt=1024 max_comp=256
  fi
  cat > "$path" <<YAML
environment:
  name: atomic
  base_task: $base_task
YAML
  if [ "$base_task" = "basic_arithmetic" ]; then
    echo "  atoms:$atom_spec" >> "$path"
  else
    echo "  atom_layer: $stage" >> "$path"
  fi
  cat >> "$path" <<YAML
  samples: 5000
  task_params: {}
generation:
  generation_batch_size: 8
  max_completion_length: $max_comp
  max_prompt_length: $max_prompt
  num_generations: 8
  stop_strings:
    - </answer>
  temperature: 1.0
generation_logging:
  enabled: true
  every_n_steps: 20
  log_dir: $out_dir/generations
  print_n_examples: 3
model:
  name_or_path: $model_init
  torch_dtype: bfloat16
seed: 42
system_prompt: >
  You are a helpful assistant. You must always think step-by-step. Put
  your thinking process inside <think> and </think> tags, and put your final answer
  inside <answer> and </answer> tags.
training:
  beta: 0.0
  bf16: true
  epsilon_high: 5.0
  gradient_accumulation_steps: 1
  kl_coeff: 0.0
  learning_rate: 5.0e-06
  logging_steps: 5
  loss_type: dapo
  lr_scheduler_type: constant
  mask_truncated_completions: true
  max_grad_norm: 1.0
  max_steps: $max_steps
  num_train_epochs: 2
  output_dir: $out_dir
  per_device_train_batch_size: 1
  report_to: []
  save_steps: $max_steps
  save_strategy: steps
  save_total_limit: 1
  warmup_ratio: 0.1
YAML
}

write_cfg "$CFG_ROOT/arith_l1.yaml" basic_arithmetic L1 "$SFT"               "$OUT_ROOT/arith_l1" $MAX_STEPS_L1
write_cfg "$CFG_ROOT/arith_l2.yaml" basic_arithmetic L2 "$OUT_ROOT/arith_l1" "$OUT_ROOT/arith_l2" $MAX_STEPS_L2
write_cfg "$CFG_ROOT/sp_l1.yaml"    shortest_path    L1 "$SFT"               "$OUT_ROOT/sp_l1"    $MAX_STEPS_L1
write_cfg "$CFG_ROOT/sp_l2.yaml"    shortest_path    L2 "$OUT_ROOT/sp_l1"    "$OUT_ROOT/sp_l2"    $MAX_STEPS_L2

##############################################
# Run 4 GRPO stages sequentially
##############################################
run_stage() {
  local cfg=$1 logname=$2 outdir=$3
  if [ -f "$outdir/model.safetensors" ] || ls "$outdir"/model-*.safetensors >/dev/null 2>&1; then
    echo "  SKIP $logname (checkpoint exists at $outdir)"
    return
  fi
  echo "  === $(date) RUN $logname ==="
  torchrun --standalone --nproc_per_node=8 train_grpo.py "$cfg" > "$LOG_ROOT/${logname}.log" 2>&1
  if [ $? -ne 0 ]; then
    echo "  WARN: $logname returned non-zero; check log $LOG_ROOT/${logname}.log"
    tail -30 "$LOG_ROOT/${logname}.log"
  fi
}

run_stage "$CFG_ROOT/arith_l1.yaml" arith_l1 "$OUT_ROOT/arith_l1"
run_stage "$CFG_ROOT/arith_l2.yaml" arith_l2 "$OUT_ROOT/arith_l2"
run_stage "$CFG_ROOT/sp_l1.yaml"    sp_l1    "$OUT_ROOT/sp_l1"
run_stage "$CFG_ROOT/sp_l2.yaml"    sp_l2    "$OUT_ROOT/sp_l2"

##############################################
# Build the 3 merges (same as Stream A)
##############################################
echo "=== $(date) building merges @ $SCALE ==="
python3 experiments/rlvr_geometry_paper/paper/scripts/build_replicate_merges.py --scale "$SCALE" \
  --sft "$SFT" \
  --atomic_ari "$OUT_ROOT/arith_l2" --atomic_sp "$OUT_ROOT/sp_l2" \
  --direct_ari "$DIRECT_ARI" --direct_sp "$DIRECT_SP" \
  --out_dir "experiments/rlvr_geometry_paper/paper/data/linking_${SCALE}" 2>&1 | tee "$LOG_ROOT/build_merges.log"

##############################################
# Eval 3 merges on 10 tasks
##############################################
TASKS_LIST=(basic_arithmetic countdown shortest_path maze mini_sudoku syllogism knights_knaves leg_counting caesar_cipher cryptarithm)
SEED=9999
N=200
eval_one() {
  local model=$1 out=$2
  [ ! -d "$model" ] && { echo "  missing: $model"; return; }
  mkdir -p "$out"
  local n_done=$(ls "$out"/metrics_*.json 2>/dev/null | wc -l)
  [ "$n_done" -ge 10 ] && { echo "  skip $out ($n_done/10)"; return; }
  echo "  eval: $model -> $out"
  for task in "${TASKS_LIST[@]}"; do
    if grep -lq "\"$task\"" "$out"/metrics_*.json 2>/dev/null; then continue; fi
    echo "    $(date +%H:%M:%S) task=$task"
    torchrun --standalone --nproc_per_node=8 evaluate_transfer.py \
      --model "$model" --tasks "$task" \
      --num_samples $N --seed $SEED --batch_size 8 \
      --output_dir "$out" >> "$LOG_ROOT/eval_$(basename "$out").log" 2>&1 || echo "    WARN: task $task failed"
  done
}

echo "=== $(date) evaluating merges @ $SCALE ==="
eval_one experiments/rlvr_geometry_paper/paper/data/linking_${SCALE}/M_aa  experiments/rlvr_geometry_paper/outputs/eval10_linking_${SCALE}_aa
eval_one experiments/rlvr_geometry_paper/paper/data/linking_${SCALE}/M_dd  experiments/rlvr_geometry_paper/outputs/eval10_linking_${SCALE}_dd
eval_one experiments/rlvr_geometry_paper/paper/data/linking_${SCALE}/M_mix experiments/rlvr_geometry_paper/outputs/eval10_linking_${SCALE}_mix

echo "=== $(date) F28 replication @ $SCALE DONE ==="
