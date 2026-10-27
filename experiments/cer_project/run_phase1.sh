#!/bin/bash
# Run Phase 1: Instrumented GRPO on green cluster P6 pod
# Usage: bash run_phase1.sh

set -e

cd /scratch/sunguha/cer_project

echo "=== Phase 1: Instrumented GRPO ==="
echo "Model: Qwen/Qwen2.5-1.5B-Instruct"
echo "Task: GSM8K"
echo "Goal: Validate inverted-U hypothesis"
echo ""

# Install dependencies
pip install -q trl>=0.27.0 transformers>=4.45.0 accelerate>=0.25.0 \
    datasets math-verify pyyaml tensorboard flash-attn --no-build-isolation 2>/dev/null || true

# Run instrumented GRPO with 8 GPUs
torchrun --standalone --nproc_per_node=8 \
    phase1_instrumented_grpo.py \
    configs/phase1_qwen15b_gsm8k.yaml

echo ""
echo "=== Phase 1 Training Complete ==="
echo "Running analysis..."

# Analyze results
python analyze_phase1.py \
    --diagnostics_dir /scratch/sunguha/cer_project/phase1_instrumented_grpo/token_diagnostics \
    --output_plot /scratch/sunguha/cer_project/phase1_instrumented_grpo/phase1_inverted_u.png

echo ""
echo "=== Done! Check: ==="
echo "  Plot: /scratch/sunguha/cer_project/phase1_instrumented_grpo/phase1_inverted_u.png"
echo "  Diagnostics: /scratch/sunguha/cer_project/phase1_instrumented_grpo/token_diagnostics/"
