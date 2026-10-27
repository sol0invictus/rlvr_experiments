#!/bin/bash
set -euo pipefail
cd /scratch/sunguha/rlvr_experiments

N=50
S=9999
R=experiments/rlvr_geometry_paper/results/phase2_sp

echo "======== SHORTEST PATH: ARS + 3 Stages ========"

# --- ARS PROBE (base) ---
echo ">>> ARS PROBE (base)"
python3 experiments/rlvr_geometry_paper/scripts/eval/probe_atomics.py --model_path /scratch/sunguha/sft_qwen_4.0b_instruct --task shortest_path --num_samples $N --seed $S --output $R/ars_base.json 2>&1 | tail -15

# --- BASE EVAL ---
echo ">>> BASE EVAL"
torchrun --standalone --nproc_per_node=8 evaluate_transfer.py --model /scratch/sunguha/sft_qwen_4.0b_instruct --tasks shortest_path --num_samples $N --seed $S --output_dir $R/base_eval 2>&1 | tail -5

# --- STAGE 1 ---
echo ">>> STAGE 1: L1 atoms (100 steps)"
torchrun --standalone --nproc_per_node=8 train_grpo.py experiments/rlvr_geometry_paper/configs/phase2_shortest_path/stage1_l1_atoms.yaml 2>&1 | tail -3
echo ">>> STAGE 1 EVAL"
torchrun --standalone --nproc_per_node=8 evaluate_transfer.py --model $R/stage1_l1 --tasks shortest_path --num_samples $N --seed $S --output_dir $R/stage1_eval 2>&1 | tail -5

# --- STAGE 2 ---
echo ">>> STAGE 2: L2 atoms (100 steps)"
torchrun --standalone --nproc_per_node=8 train_grpo.py experiments/rlvr_geometry_paper/configs/phase2_shortest_path/stage2_l2_atoms.yaml 2>&1 | tail -3
echo ">>> STAGE 2 EVAL (= Atomic-Only)"
torchrun --standalone --nproc_per_node=8 evaluate_transfer.py --model $R/stage2_l2 --tasks shortest_path --num_samples $N --seed $S --output_dir $R/stage2_eval 2>&1 | tail -5

# --- STAGE 3 ---
echo ">>> STAGE 3: Full task (100 steps)"
torchrun --standalone --nproc_per_node=8 train_grpo.py experiments/rlvr_geometry_paper/configs/phase2_shortest_path/stage3_full_task.yaml 2>&1 | tail -3
echo ">>> STAGE 3 EVAL (= Atomic-Warm)"
torchrun --standalone --nproc_per_node=8 evaluate_transfer.py --model $R/stage3_full --tasks shortest_path --num_samples $N --seed $S --output_dir $R/stage3_eval 2>&1 | tail -5

# --- SUMMARY ---
echo ""
echo "======== SHORTEST PATH RESULTS ========"
for d in base_eval stage1_eval stage2_eval stage3_eval; do
    f=$(ls -t $R/$d/metrics_*.json 2>/dev/null | head -1)
    if [ -n "$f" ]; then
        acc=$(python3 -c "import json; d=json.load(open('$f')); print(f\"{d['tasks']['shortest_path']['accuracy']*100:.0f}%\")")
        printf '  %-25s %s\n' "$d" "$acc"
    else
        printf '  %-25s %s\n' "$d" "NO METRICS"
    fi
done
echo "========================================"
