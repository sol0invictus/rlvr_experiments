#!/usr/bin/env bash
# Phase B3a: which "instruct" base was behind sft_qwen_4.0b_instruct?
# SFT both candidates with the original warm-up recipe, then score each W_sft on the
# paper's SFT row (Tab. 2 / App. F): arith 72, syllogism 79.5, kk 78, countdown 26,
# maze 21.5, shortest_path 10, mini_sudoku 4, leg_counting 52 (200 samples, seed 9999).
# Run inside tmux:  tmux new -d -s sftcmp 'bash scripts/vm/sft_compare.sh'
set -euo pipefail
source /data/env.sh
J="bash scripts/vm/run_job.sh"
R=/data/rlvr
TASKS=basic_arithmetic,syllogism,knights_knaves,countdown,maze,shortest_path,mini_sudoku,leg_counting

python data_gen/generate_sft_warmup.py --num_samples 1000 --seed 42
sha256sum data_gen/data/sft_warmup.jsonl | tee $R/sft_warmup.sha256

for cand in sft_qwen3_4b sft_qwen3_4b_instruct2507; do
    $J $cand 60m python train_sft.py configs/replication/$cand.yaml
    $J eval_${cand}_vllm 60m python evaluate_transfer.py --model $R/$cand --tasks $TASKS \
        --num_samples 200 --backend vllm --output_dir $R/eval/$cand
done

# HF-generate reference on the arithmetic row, to check vLLM eval against it.
for cand in sft_qwen3_4b sft_qwen3_4b_instruct2507; do
    $J eval_${cand}_hf 60m python evaluate_transfer.py --model $R/$cand --tasks basic_arithmetic \
        --num_samples 200 --backend hf --output_dir $R/eval/$cand
done

echo "SFT COMPARE ALL DONE"
