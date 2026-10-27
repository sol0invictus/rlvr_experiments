#!/bin/bash
set -euo pipefail
cd /scratch/sunguha/rlvr_experiments

N=50
S=9999
R=experiments/rlvr_geometry_paper/results/phase2_sp

echo "======== SHORTEST PATH v2: SFT-L1 → GRPO-L2 → GRPO-Full ========"

# --- Generate SFT data for L1 atoms ---
echo ">>> Generating L1 SFT data (2000 samples with reasoning traces)"
python3 experiments/rlvr_geometry_paper/scripts/data/generate_sp_l1_sft_data.py

# --- SFT on L1 atoms ---
echo ">>> SFT on L1 atoms"
torchrun --standalone --nproc_per_node=8 train_sft.py experiments/rlvr_geometry_paper/configs/phase2_shortest_path/sft_l1_atoms.yaml 2>&1 | tail -5

# --- Eval post-SFT ---
echo ">>> Post-SFT eval on full shortest_path"
torchrun --standalone --nproc_per_node=8 evaluate_transfer.py \
    --model $R/sft_l1 --tasks shortest_path \
    --num_samples $N --seed $S --output_dir $R/sft_l1_eval 2>&1 | tail -5

# --- ARS probe post-SFT ---
echo ">>> ARS probe post-SFT"
python3 experiments/rlvr_geometry_paper/scripts/eval/probe_atomics.py --model_path $R/sft_l1 --task shortest_path \
    --num_samples $N --seed $S --output $R/sft_l1_ars.json 2>&1 | tail -15

# --- GRPO on L2 atoms (init from SFT L1) ---
echo ">>> GRPO on L2 atoms (100 steps, init from SFT-L1)"
torchrun --standalone --nproc_per_node=8 train_grpo.py experiments/rlvr_geometry_paper/configs/phase2_shortest_path/sft_then_l2_grpo.yaml 2>&1 | tail -3

# --- Eval post-L2 (= SFT-Atomic-Only) ---
echo ">>> Post-L2 eval (SFT-Atomic-Only)"
torchrun --standalone --nproc_per_node=8 evaluate_transfer.py \
    --model $R/sft_then_l2 --tasks shortest_path \
    --num_samples $N --seed $S --output_dir $R/sft_then_l2_eval 2>&1 | tail -5

# --- GRPO on full task (300 steps, init from SFT-L1→GRPO-L2) ---
echo ">>> GRPO on full task (300 steps, init from SFT→L2)"
torchrun --standalone --nproc_per_node=8 train_grpo.py experiments/rlvr_geometry_paper/configs/phase2_shortest_path/sft_then_full_300.yaml 2>&1 | tail -3

# --- Final eval ---
echo ">>> Final eval (SFT-Warm-300)"
torchrun --standalone --nproc_per_node=8 evaluate_transfer.py \
    --model $R/sft_warm_300 --tasks shortest_path \
    --num_samples $N --seed $S --output_dir $R/sft_warm_300_eval 2>&1 | tail -5

# --- Summary ---
echo ""
echo "======== RESULTS ========"
echo "Reference: Base=14%, Direct-RLVR=44%, GRPO-Atomic-Warm-300=46%"
echo ""
for d in base_eval sft_l1_eval sft_then_l2_eval sft_warm_300_eval; do
    f=$(ls -t $R/$d/metrics_*.json 2>/dev/null | head -1)
    if [ -n "$f" ]; then
        acc=$(python3 -c "import json; d=json.load(open('$f')); print(f\"{d['tasks']['shortest_path']['accuracy']*100:.0f}%\")")
        printf '  %-25s %s\n' "$d" "$acc"
    else
        printf '  %-25s %s\n' "$d" "NO METRICS"
    fi
done
echo "=========================="
