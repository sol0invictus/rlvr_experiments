#!/bin/bash
# Run Ideas A, B, C, D sequentially for shortest_path push-higher experiments.
# Each idea trains ≤300 GRPO steps total, then evaluates on full shortest_path.
#
# Usage: bash experiments/rlvr_geometry_paper/scripts/run/run_ideas_abcd.sh
# Expected runtime: ~4-5 hours total on 8x B200

set -e

cd /scratch/sunguha/rlvr_experiments

EVAL_SCRIPT="torchrun --nproc_per_node=8 evaluate_transfer.py"
TRAIN_SCRIPT="torchrun --nproc_per_node=8 train_grpo.py"
BASE_MODEL="/scratch/sunguha/sft_qwen_4.0b_instruct"
EVAL_ARGS="--tasks shortest_path --num_samples 200 --seed 9999 --batch_size 8"

echo "========================================"
echo "  IDEA A: Progressive Full-Task (3x3→4x4→Full)"
echo "  Total: 100+100+100 = 300 GRPO steps"
echo "========================================"

echo "[A-1] Training on 3x3 grids (100 steps)..."
$TRAIN_SCRIPT experiments/rlvr_geometry_paper/configs/phase2_sp_ideas/idea_a_stage1_3x3.yaml
echo "[A-1] Evaluating stage1..."
$EVAL_SCRIPT --model experiments/rlvr_geometry_paper/results/phase2_sp/idea_a/stage1_3x3 $EVAL_ARGS \
  --output_dir experiments/rlvr_geometry_paper/results/phase2_sp/idea_a/stage1_3x3_eval

echo "[A-2] Training on 4x4 grids (100 steps)..."
$TRAIN_SCRIPT experiments/rlvr_geometry_paper/configs/phase2_sp_ideas/idea_a_stage2_4x4.yaml
echo "[A-2] Evaluating stage2..."
$EVAL_SCRIPT --model experiments/rlvr_geometry_paper/results/phase2_sp/idea_a/stage2_4x4 $EVAL_ARGS \
  --output_dir experiments/rlvr_geometry_paper/results/phase2_sp/idea_a/stage2_4x4_eval

echo "[A-3] Training on full grids (100 steps)..."
$TRAIN_SCRIPT experiments/rlvr_geometry_paper/configs/phase2_sp_ideas/idea_a_stage3_full.yaml
echo "[A-3] Evaluating stage3 (final)..."
$EVAL_SCRIPT --model experiments/rlvr_geometry_paper/results/phase2_sp/idea_a/stage3_full $EVAL_ARGS \
  --output_dir experiments/rlvr_geometry_paper/results/phase2_sp/idea_a/stage3_full_eval

echo ""
echo "========================================"
echo "  IDEA B: Path-Tracing L2 Atoms → Full Task"
echo "  Total: 100+200 = 300 GRPO steps"
echo "========================================"

echo "[B-1] Training on L2-long atoms (100 steps)..."
$TRAIN_SCRIPT experiments/rlvr_geometry_paper/configs/phase2_sp_ideas/idea_b_stage1_long_atoms.yaml
echo "[B-1] Evaluating post-atoms..."
$EVAL_SCRIPT --model experiments/rlvr_geometry_paper/results/phase2_sp/idea_b/stage1_long_atoms $EVAL_ARGS \
  --output_dir experiments/rlvr_geometry_paper/results/phase2_sp/idea_b/stage1_long_atoms_eval

echo "[B-2] Training on full task (200 steps)..."
$TRAIN_SCRIPT experiments/rlvr_geometry_paper/configs/phase2_sp_ideas/idea_b_stage2_full.yaml
echo "[B-2] Evaluating final..."
$EVAL_SCRIPT --model experiments/rlvr_geometry_paper/results/phase2_sp/idea_b/stage2_full $EVAL_ARGS \
  --output_dir experiments/rlvr_geometry_paper/results/phase2_sp/idea_b/stage2_full_eval

echo ""
echo "========================================"
echo "  IDEA C: Partial-Credit Reward Shaping"
echo "  Total: 300 GRPO steps (single stage)"
echo "========================================"

echo "[C] Training with partial-credit rewards (300 steps)..."
$TRAIN_SCRIPT experiments/rlvr_geometry_paper/configs/phase2_sp_ideas/idea_c_partial_credit.yaml
echo "[C] Evaluating final..."
$EVAL_SCRIPT --model experiments/rlvr_geometry_paper/results/phase2_sp/idea_c/partial_credit $EVAL_ARGS \
  --output_dir experiments/rlvr_geometry_paper/results/phase2_sp/idea_c/partial_credit_eval

echo ""
echo "========================================"
echo "  IDEA D: Progressive + Partial Credit"
echo "  Total: 100+100+100 = 300 GRPO steps"
echo "========================================"

echo "[D-1] Training 3x3 + partial credit (100 steps)..."
$TRAIN_SCRIPT experiments/rlvr_geometry_paper/configs/phase2_sp_ideas/idea_d_stage1_3x3_pc.yaml
echo "[D-1] Evaluating stage1..."
$EVAL_SCRIPT --model experiments/rlvr_geometry_paper/results/phase2_sp/idea_d/stage1_3x3_pc $EVAL_ARGS \
  --output_dir experiments/rlvr_geometry_paper/results/phase2_sp/idea_d/stage1_3x3_pc_eval

echo "[D-2] Training 4x4 + partial credit (100 steps)..."
$TRAIN_SCRIPT experiments/rlvr_geometry_paper/configs/phase2_sp_ideas/idea_d_stage2_4x4_pc.yaml
echo "[D-2] Evaluating stage2..."
$EVAL_SCRIPT --model experiments/rlvr_geometry_paper/results/phase2_sp/idea_d/stage2_4x4_pc $EVAL_ARGS \
  --output_dir experiments/rlvr_geometry_paper/results/phase2_sp/idea_d/stage2_4x4_pc_eval

echo "[D-3] Training full + partial credit (100 steps)..."
$TRAIN_SCRIPT experiments/rlvr_geometry_paper/configs/phase2_sp_ideas/idea_d_stage3_full_pc.yaml
echo "[D-3] Evaluating final..."
$EVAL_SCRIPT --model experiments/rlvr_geometry_paper/results/phase2_sp/idea_d/stage3_full_pc $EVAL_ARGS \
  --output_dir experiments/rlvr_geometry_paper/results/phase2_sp/idea_d/stage3_full_pc_eval

echo ""
echo "========================================"
echo "  ALL DONE — SUMMARY"
echo "========================================"
echo "Reference: Base=10%, Direct-RLVR=44%, Current-Best(Atomic-Warm-300)=46%"
echo ""

for idea_dir in idea_a/stage3_full idea_b/stage2_full idea_c/partial_credit idea_d/stage3_full_pc; do
    eval_dir="experiments/rlvr_geometry_paper/results/phase2_sp/${idea_dir}_eval"
    name=$(echo $idea_dir | cut -d'/' -f1)
    result_file=$(ls $eval_dir/metrics_*.json 2>/dev/null | head -1)
    if [ -n "$result_file" ]; then
        acc=$(python3 -c "import json; d=json.load(open('$result_file')); print(d['tasks']['shortest_path']['accuracy'])")
        printf "  %-30s %s\n" "$name" "${acc}"
    else
        printf "  %-30s %s\n" "$name" "NO RESULT"
    fi
done
