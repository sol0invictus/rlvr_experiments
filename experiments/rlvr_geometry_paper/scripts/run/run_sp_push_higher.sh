#!/bin/bash
set -euo pipefail
cd /scratch/sunguha/rlvr_experiments

N=50
S=9999
R=experiments/rlvr_geometry_paper/results/phase2_sp

echo "======== PUSH SP HIGHER: 2 Approaches ========"
echo "Current best: GRPO-Atomic-Warm-300 = 46%"
echo ""

# ============================================================
# APPROACH 1: SFT-L1 at low LR (5e-6) → GRPO-Full 300
# ============================================================
echo ">>> [1/2] SFT-L1 low-LR (5e-6, 3 epochs)"
torchrun --standalone --nproc_per_node=8 train_sft.py experiments/rlvr_geometry_paper/configs/phase2_shortest_path/sft_l1_low_lr.yaml 2>&1 | tail -3

echo ">>> [1/2] ARS probe post-SFT-lowlr"
python3 experiments/rlvr_geometry_paper/scripts/eval/probe_atomics.py --model_path $R/sft_l1_lowlr --task shortest_path \
    --num_samples $N --seed $S --output $R/sft_l1_lowlr_ars.json 2>&1 | tail -15

echo ">>> [1/2] GRPO-Full 300 steps from SFT-lowlr"
torchrun --standalone --nproc_per_node=8 train_grpo.py experiments/rlvr_geometry_paper/configs/phase2_shortest_path/sft_lowlr_full_300.yaml 2>&1 | tail -3

echo ">>> [1/2] Eval"
torchrun --standalone --nproc_per_node=8 evaluate_transfer.py \
    --model $R/sft_lowlr_full_300 --tasks shortest_path \
    --num_samples $N --seed $S --output_dir $R/sft_lowlr_full_300_eval 2>&1 | tail -5

# ============================================================
# APPROACH 2: Progressive grids (3x3 L1 → 4x4 L2 → full)
# ============================================================
echo ">>> [2/2] Progressive: L1 atoms on 3x3 grids (100 steps)"
torchrun --standalone --nproc_per_node=8 train_grpo.py experiments/rlvr_geometry_paper/configs/phase2_shortest_path/prog_l1_3x3.yaml 2>&1 | tail -3

echo ">>> [2/2] Progressive: L2 atoms on 4x4 grids (100 steps)"
torchrun --standalone --nproc_per_node=8 train_grpo.py experiments/rlvr_geometry_paper/configs/phase2_shortest_path/prog_l2_4x4.yaml 2>&1 | tail -3

echo ">>> [2/2] Progressive: Full task 300 steps"
torchrun --standalone --nproc_per_node=8 train_grpo.py experiments/rlvr_geometry_paper/configs/phase2_shortest_path/prog_full_300.yaml 2>&1 | tail -3

echo ">>> [2/2] Eval"
torchrun --standalone --nproc_per_node=8 evaluate_transfer.py \
    --model $R/prog_full_300 --tasks shortest_path \
    --num_samples $N --seed $S --output_dir $R/prog_full_300_eval 2>&1 | tail -5

# ============================================================
# SUMMARY
# ============================================================
echo ""
echo "======== RESULTS ========"
echo "Reference: Base=14%, Direct-RLVR=44%, Current-Best=46%"
echo ""
for d in sft_lowlr_full_300_eval prog_full_300_eval; do
    f=$(ls -t $R/$d/metrics_*.json 2>/dev/null | head -1)
    if [ -n "$f" ]; then
        acc=$(python3 -c "import json; d=json.load(open('$f')); print(f\"{d['tasks']['shortest_path']['accuracy']*100:.0f}%\")")
        printf '  %-30s %s\n' "$d" "$acc"
    else
        printf '  %-30s %s\n' "$d" "NO METRICS"
    fi
done
echo "=========================="
