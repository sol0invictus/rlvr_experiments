#!/bin/bash
# Re-run the specific (model, task) pairs that crashed during main pipeline.
# Runs after the fixup pipeline finishes (waits for free GPU).
set -e
cd /code-fsx/sunguha/rlvr_experiments
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

SEED_A=9999
SEED_C=1234
N=200

rerun() {
  local model=$1
  local task=$2
  local out=$3
  local seed=$4
  if grep -lq "\"$task\"" "$out"/metrics_*.json 2>/dev/null; then
    echo "  skip ($model, $task) already present in $out"
    return
  fi
  echo "  RE-RUN model=$model task=$task out=$out seed=$seed  ($(date +%H:%M:%S))"
  torchrun --standalone --nproc_per_node=8 evaluate_transfer.py \
    --model "$model" --tasks "$task" \
    --num_samples $N --seed $seed --batch_size 8 \
    --output_dir "$out" 2>&1 >> experiments/rlvr_geometry_paper/paper/data/logs/rerun_$(basename "$out").log || echo "  still failed after retry"
}

# waits for fixup to finish (checks for running torchrun)
while pgrep -f "bash experiments/rlvr_geometry_paper/paper/scripts/pipeline_fixup.sh" > /dev/null; do
  sleep 60
done
echo "=== $(date) fixup done; reruns starting ==="

# Stream A linking_mix: maze + knights_knaves
rerun experiments/rlvr_geometry_paper/paper/data/linking/M_mix maze           experiments/rlvr_geometry_paper/outputs/eval10_linking_mix            $SEED_A
rerun experiments/rlvr_geometry_paper/paper/data/linking/M_mix knights_knaves experiments/rlvr_geometry_paper/outputs/eval10_linking_mix            $SEED_A

# Stream C missing tasks
rerun experiments/rlvr_geometry_paper/results/phase2_arith/stage2_l2      countdown         experiments/rlvr_geometry_paper/outputs/eval10_seed1234_p2_arith_atomic $SEED_C
rerun experiments/rlvr_geometry_paper/results/phase2_sp/merged/basic      basic_arithmetic  experiments/rlvr_geometry_paper/outputs/eval10_seed1234_merge_2task_basic $SEED_C
rerun experiments/rlvr_geometry_paper/results/phase4_pairs/T5_sp_syl/basic cryptarithm      experiments/rlvr_geometry_paper/outputs/eval10_seed1234_phase4_T5_sp_syl $SEED_C

echo "=== $(date) reruns DONE ==="
