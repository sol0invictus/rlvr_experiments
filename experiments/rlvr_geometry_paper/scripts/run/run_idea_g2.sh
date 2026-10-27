#!/bin/bash
# Idea G2: Interleaved GRPO + LIGHT SFT cycles
# Changes from G: 500 samples (not 5000), LR 1e-6 (not 5e-6)
# 3 cycles × (GRPO 50 steps + SFT ~30 steps on 500 samples) = 300 total steps
set -e

cd /scratch/sunguha/rlvr_experiments
mkdir -p experiments/rlvr_geometry_paper/results/phase2_sp/idea_g2

TRAIN_GRPO="torchrun --nproc_per_node=8 train_grpo.py"
TRAIN_SFT="torchrun --nproc_per_node=8 train_sft.py"
EVAL="torchrun --nproc_per_node=8 evaluate_transfer.py"
EVAL_ARGS="--tasks shortest_path --num_samples 200 --seed 9999 --batch_size 8"

echo "========================================"
echo "  IDEA G2: Interleaved GRPO + LIGHT SFT"
echo "  500 samples, LR=1e-6"
echo "========================================"

# ---- Cycle 1 (seed=42) ----
echo ""
echo "[Cycle 1] Generating SFT data (seed=42, 500 samples)..."
python3 experiments/rlvr_geometry_paper/scripts/data/generate_sp_sft_bfs.py --seed 42 --num_samples 500 --output experiments/rlvr_geometry_paper/results/phase2_sp/idea_g2/sft_data_42.jsonl

echo "[Cycle 1] GRPO 50 steps..."
$TRAIN_GRPO experiments/rlvr_geometry_paper/configs/phase2_sp_ideas/idea_g/grpo_cycle1.yaml

echo "[Cycle 1] Light SFT on BFS solutions (500 samples, lr=1e-6)..."
$TRAIN_SFT experiments/rlvr_geometry_paper/configs/phase2_sp_ideas/idea_g/sft_cycle1.yaml

echo "[Cycle 1] Eval..."
$EVAL --model experiments/rlvr_geometry_paper/results/phase2_sp/idea_g2/cycle1_sft $EVAL_ARGS \
  --output_dir experiments/rlvr_geometry_paper/results/phase2_sp/idea_g2/cycle1_eval

# ---- Cycle 2 (seed=137) ----
echo ""
echo "[Cycle 2] Generating SFT data (seed=137, 500 samples)..."
python3 experiments/rlvr_geometry_paper/scripts/data/generate_sp_sft_bfs.py --seed 137 --num_samples 500 --output experiments/rlvr_geometry_paper/results/phase2_sp/idea_g2/sft_data_137.jsonl

echo "[Cycle 2] GRPO 50 steps..."
$TRAIN_GRPO experiments/rlvr_geometry_paper/configs/phase2_sp_ideas/idea_g/grpo_cycle2.yaml

echo "[Cycle 2] Light SFT on BFS solutions..."
$TRAIN_SFT experiments/rlvr_geometry_paper/configs/phase2_sp_ideas/idea_g/sft_cycle2.yaml

echo "[Cycle 2] Eval..."
$EVAL --model experiments/rlvr_geometry_paper/results/phase2_sp/idea_g2/cycle2_sft $EVAL_ARGS \
  --output_dir experiments/rlvr_geometry_paper/results/phase2_sp/idea_g2/cycle2_eval

# ---- Cycle 3 (seed=2025) ----
echo ""
echo "[Cycle 3] Generating SFT data (seed=2025, 500 samples)..."
python3 experiments/rlvr_geometry_paper/scripts/data/generate_sp_sft_bfs.py --seed 2025 --num_samples 500 --output experiments/rlvr_geometry_paper/results/phase2_sp/idea_g2/sft_data_2025.jsonl

echo "[Cycle 3] GRPO 50 steps..."
$TRAIN_GRPO experiments/rlvr_geometry_paper/configs/phase2_sp_ideas/idea_g/grpo_cycle3.yaml

echo "[Cycle 3] Light SFT on BFS solutions..."
$TRAIN_SFT experiments/rlvr_geometry_paper/configs/phase2_sp_ideas/idea_g/sft_cycle3.yaml

echo "[Cycle 3] Eval..."
$EVAL --model experiments/rlvr_geometry_paper/results/phase2_sp/idea_g2/cycle3_sft $EVAL_ARGS \
  --output_dir experiments/rlvr_geometry_paper/results/phase2_sp/idea_g2/cycle3_eval

echo ""
echo "========================================"
echo "  IDEA G2 DONE — Per-Cycle Results:"
echo "========================================"
for cycle in 1 2 3; do
    eval_dir="experiments/rlvr_geometry_paper/results/phase2_sp/idea_g2/cycle${cycle}_eval"
    result_file=$(ls $eval_dir/metrics_*.json 2>/dev/null | head -1)
    if [ -n "$result_file" ]; then
        acc=$(python3 -c "import json; d=json.load(open('$result_file')); print(d['tasks']['shortest_path']['accuracy'])")
        printf "  Cycle %d: %s\n" "$cycle" "$acc"
    else
        printf "  Cycle %d: NO RESULT\n" "$cycle"
    fi
done
