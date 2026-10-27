#!/bin/bash
# Best-of-N eval — run 8 models in parallel across 8 GPUs (1 model per GPU)
set -e
cd /scratch/sunguha/rlvr_experiments
mkdir -p experiments/rlvr_geometry_paper/results/phase2_sp/bon_eval

N=8
COMMON="--num_candidates $N --num_samples 50 --seed 9999 --temperature 0.7 --max_new_tokens 2048 --output_dir experiments/rlvr_geometry_paper/results/phase2_sp/bon_eval"

echo "========================================"
echo "  Best-of-$N Eval — 8 models in parallel"
echo "========================================"

# Batch 1: 8 models in parallel (1 per GPU)
CUDA_VISIBLE_DEVICES=0 python3 experiments/rlvr_geometry_paper/scripts/eval/eval_best_of_n_sp.py --model /scratch/sunguha/sft_qwen_4.0b_instruct $COMMON > experiments/rlvr_geometry_paper/results/phase2_sp/bon_eval/log_base.txt 2>&1 &
CUDA_VISIBLE_DEVICES=1 python3 experiments/rlvr_geometry_paper/scripts/eval/eval_best_of_n_sp.py --model experiments/rlvr_geometry_paper/results/phase2_sp/atomic_warm_300 $COMMON > experiments/rlvr_geometry_paper/results/phase2_sp/bon_eval/log_atomic_warm.txt 2>&1 &
CUDA_VISIBLE_DEVICES=2 python3 experiments/rlvr_geometry_paper/scripts/eval/eval_best_of_n_sp.py --model experiments/rlvr_geometry_paper/results/phase2_sp/idea_a/stage3_full $COMMON > experiments/rlvr_geometry_paper/results/phase2_sp/bon_eval/log_idea_a.txt 2>&1 &
CUDA_VISIBLE_DEVICES=3 python3 experiments/rlvr_geometry_paper/scripts/eval/eval_best_of_n_sp.py --model experiments/rlvr_geometry_paper/results/phase2_sp/idea_b/stage2_full $COMMON > experiments/rlvr_geometry_paper/results/phase2_sp/bon_eval/log_idea_b.txt 2>&1 &
CUDA_VISIBLE_DEVICES=4 python3 experiments/rlvr_geometry_paper/scripts/eval/eval_best_of_n_sp.py --model experiments/rlvr_geometry_paper/results/phase2_sp/idea_c/partial_credit $COMMON > experiments/rlvr_geometry_paper/results/phase2_sp/bon_eval/log_idea_c.txt 2>&1 &
CUDA_VISIBLE_DEVICES=5 python3 experiments/rlvr_geometry_paper/scripts/eval/eval_best_of_n_sp.py --model experiments/rlvr_geometry_paper/results/phase2_sp/idea_d/stage3_full_pc $COMMON > experiments/rlvr_geometry_paper/results/phase2_sp/bon_eval/log_idea_d.txt 2>&1 &
CUDA_VISIBLE_DEVICES=6 python3 experiments/rlvr_geometry_paper/scripts/eval/eval_best_of_n_sp.py --model experiments/rlvr_geometry_paper/results/phase2_sp/idea_e/stage4_full $COMMON > experiments/rlvr_geometry_paper/results/phase2_sp/bon_eval/log_idea_e.txt 2>&1 &
CUDA_VISIBLE_DEVICES=7 python3 experiments/rlvr_geometry_paper/scripts/eval/eval_best_of_n_sp.py --model experiments/rlvr_geometry_paper/results/phase2_sp/idea_f/stage2_full $COMMON > experiments/rlvr_geometry_paper/results/phase2_sp/bon_eval/log_idea_f.txt 2>&1 &

echo "Batch 1 launched (8 models), waiting..."
wait
echo "Batch 1 done."

# Batch 2: remaining 3 models
CUDA_VISIBLE_DEVICES=0 python3 experiments/rlvr_geometry_paper/scripts/eval/eval_best_of_n_sp.py --model experiments/rlvr_geometry_paper/results/phase2_sp/idea_g2/cycle3_sft $COMMON > experiments/rlvr_geometry_paper/results/phase2_sp/bon_eval/log_idea_g2.txt 2>&1 &
CUDA_VISIBLE_DEVICES=1 python3 experiments/rlvr_geometry_paper/scripts/eval/eval_best_of_n_sp.py --model experiments/rlvr_geometry_paper/results/phase2_sp/idea_g3 $COMMON > experiments/rlvr_geometry_paper/results/phase2_sp/bon_eval/log_idea_g3.txt 2>&1 &
CUDA_VISIBLE_DEVICES=2 python3 experiments/rlvr_geometry_paper/scripts/eval/eval_best_of_n_sp.py --model experiments/rlvr_geometry_paper/results/phase2_sp/idea_b2/stage2_full $COMMON > experiments/rlvr_geometry_paper/results/phase2_sp/bon_eval/log_idea_b2.txt 2>&1 &

echo "Batch 2 launched (3 models), waiting..."
wait
echo "Batch 2 done."

echo ""
echo "========================================"
echo "  RESULTS"
echo "========================================"
for f in experiments/rlvr_geometry_paper/results/phase2_sp/bon_eval/bon_*.json; do
    python3 -c "
import json,sys
d=json.load(open(sys.argv[1]))
m=d['model'].split('/')[-1]
print('  %-25s  pass@1=%.0f%%  BoN=%.0f%%  lift=%+.0f%%' % (m,d['pass_at_1']*100,d['best_of_n']*100,d['lift']*100))
" "$f"
done
echo "========================================"
