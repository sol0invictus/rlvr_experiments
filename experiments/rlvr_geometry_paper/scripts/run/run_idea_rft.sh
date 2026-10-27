#!/bin/bash
# Idea RFT: On-Policy Rejection Sampling Fine-Tuning
# 1. Sample K=16 from B's checkpoint, keep correct, difficulty-weight
# 2. Light SFT on model's own correct answers (lr=1e-6, 1 epoch)
# 3. GRPO 100 steps to sharpen
set -e
cd /scratch/sunguha/rlvr_experiments
mkdir -p experiments/rlvr_geometry_paper/results/phase2_sp/idea_rft

EVAL="torchrun --nproc_per_node=8 evaluate_transfer.py"
EVAL_ARGS="--tasks shortest_path --num_samples 200 --seed 9999 --batch_size 8"

echo "========================================"
echo "  Idea RFT: Rejection Sampling FT"
echo "========================================"

echo "[1] Generating RFT data (K=8, 500 problems)..."
torchrun --nproc_per_node=8 experiments/rlvr_geometry_paper/scripts/data/generate_rft_data.py \
  --model experiments/rlvr_geometry_paper/results/phase2_sp/idea_b/stage2_full \
  --num_problems 500 --k 8 --seed 7777 \
  --temperature 0.8 --max_new_tokens 2048 \
  --output experiments/rlvr_geometry_paper/results/phase2_sp/idea_rft/rft_data.jsonl \
  --max_samples 1000

echo "[2] SFT on model's own correct answers (lr=1e-6)..."
torchrun --nproc_per_node=8 train_sft.py experiments/rlvr_geometry_paper/configs/phase2_sp_ideas/idea_rft_sft.yaml

echo "[2] Eval post-SFT..."
$EVAL --model experiments/rlvr_geometry_paper/results/phase2_sp/idea_rft/sft $EVAL_ARGS \
  --output_dir experiments/rlvr_geometry_paper/results/phase2_sp/idea_rft/sft_eval

echo "[3] GRPO 100 steps to sharpen..."
torchrun --nproc_per_node=8 train_grpo.py experiments/rlvr_geometry_paper/configs/phase2_sp_ideas/idea_rft_grpo.yaml

echo "[3] Final eval..."
$EVAL --model experiments/rlvr_geometry_paper/results/phase2_sp/idea_rft/grpo $EVAL_ARGS \
  --output_dir experiments/rlvr_geometry_paper/results/phase2_sp/idea_rft/grpo_eval

echo ""
echo "========================================"
echo "  RFT DONE"
echo "========================================"
for stage in sft grpo; do
    eval_dir="experiments/rlvr_geometry_paper/results/phase2_sp/idea_rft/${stage}_eval"
    result_file=$(ls $eval_dir/metrics_*.json 2>/dev/null | head -1)
    if [ -n "$result_file" ]; then
        acc=$(python3 -c "import json; d=json.load(open('$result_file')); print(d['tasks']['shortest_path']['accuracy'])")
        printf "  After %-6s: %s\n" "$stage" "$acc"
    fi
done
