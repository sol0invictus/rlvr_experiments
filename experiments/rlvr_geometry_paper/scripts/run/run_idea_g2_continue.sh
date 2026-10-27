#!/bin/bash
# Continue G2 with 2 more GRPO+SFT cycles (cycles 4 and 5)
# Picks up from G2 cycle 3 checkpoint
set -e

cd /scratch/sunguha/rlvr_experiments

TRAIN_GRPO="torchrun --nproc_per_node=8 train_grpo.py"
TRAIN_SFT="torchrun --nproc_per_node=8 train_sft.py"
EVAL="torchrun --nproc_per_node=8 evaluate_transfer.py"
EVAL_ARGS="--tasks shortest_path --num_samples 200 --seed 9999 --batch_size 8"

echo "========================================"
echo "  G2 CONTINUE: Cycles 4-5"
echo "========================================"

# ---- Cycle 4 (seed=314) ----
echo ""
echo "[Cycle 4] Generating SFT data (seed=314, 500 samples)..."
python3 experiments/rlvr_geometry_paper/scripts/data/generate_sp_sft_bfs.py --seed 314 --num_samples 500 --output experiments/rlvr_geometry_paper/results/phase2_sp/idea_g2/sft_data_314.jsonl

echo "[Cycle 4] GRPO 50 steps..."
$TRAIN_GRPO experiments/rlvr_geometry_paper/configs/phase2_sp_ideas/idea_g/grpo_cycle4.yaml

echo "[Cycle 4] Light SFT..."
$TRAIN_SFT experiments/rlvr_geometry_paper/configs/phase2_sp_ideas/idea_g/sft_cycle4.yaml

echo "[Cycle 4] Eval..."
$EVAL --model experiments/rlvr_geometry_paper/results/phase2_sp/idea_g2/cycle4_sft $EVAL_ARGS \
  --output_dir experiments/rlvr_geometry_paper/results/phase2_sp/idea_g2/cycle4_eval

# ---- Cycle 5 (seed=1618) ----
echo ""
echo "[Cycle 5] Generating SFT data (seed=1618, 500 samples)..."
python3 experiments/rlvr_geometry_paper/scripts/data/generate_sp_sft_bfs.py --seed 1618 --num_samples 500 --output experiments/rlvr_geometry_paper/results/phase2_sp/idea_g2/sft_data_1618.jsonl

echo "[Cycle 5] GRPO 50 steps..."
$TRAIN_GRPO experiments/rlvr_geometry_paper/configs/phase2_sp_ideas/idea_g/grpo_cycle5.yaml

echo "[Cycle 5] Light SFT..."
$TRAIN_SFT experiments/rlvr_geometry_paper/configs/phase2_sp_ideas/idea_g/sft_cycle5.yaml

echo "[Cycle 5] Eval..."
$EVAL --model experiments/rlvr_geometry_paper/results/phase2_sp/idea_g2/cycle5_sft $EVAL_ARGS \
  --output_dir experiments/rlvr_geometry_paper/results/phase2_sp/idea_g2/cycle5_eval

echo ""
echo "========================================"
echo "  G2 CONTINUE DONE — All Cycle Results:"
echo "========================================"
for cycle in 1 2 3 4 5; do
    eval_dir="experiments/rlvr_geometry_paper/results/phase2_sp/idea_g2/cycle${cycle}_eval"
    result_file=$(ls $eval_dir/metrics_*.json 2>/dev/null | head -1)
    if [ -n "$result_file" ]; then
        acc=$(python3 -c "import json; d=json.load(open('$result_file')); print(d['tasks']['shortest_path']['accuracy'])")
        printf "  Cycle %d: %s\n" "$cycle" "$acc"
    else
        printf "  Cycle %d: NO RESULT\n" "$cycle"
    fi
done
