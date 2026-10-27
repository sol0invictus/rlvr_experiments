#!/bin/bash
# Run custom GSM8K evaluation on all RLVR checkpoints
# Handles transformers 5.6.2 tokenizer issue

set -e
export PATH=/dev/shm/.local/bin:$PATH
cd /scratch/sunguha/rlvr_experiments

echo "=============================================="
echo "GSM8K Evaluation for RLVR Phases 1-4"
echo "Started: $(date)"
echo "=============================================="

python experiments/rlvr_geometry_paper/scripts/eval/eval_gsm8k_custom.py \
    --output_dir experiments/rlvr_geometry_paper/outputs/gsm8k_eval \
    2>&1 | tee logs/gsm8k_eval.log

echo ""
echo "=============================================="
echo "Evaluation Complete!"
echo "Finished: $(date)"
echo "=============================================="
