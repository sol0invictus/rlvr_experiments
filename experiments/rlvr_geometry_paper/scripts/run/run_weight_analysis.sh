#!/bin/bash
# Run RLVR weight change analysis for all model sizes
# Usage: bash experiments/rlvr_geometry_paper/scripts/run/run_weight_analysis.sh

set -e

OUTPUT_BASE="/code-fsx/sunguha/rlvr_experiments/results/weight_analysis"
mkdir -p $OUTPUT_BASE

echo "=========================================="
echo "RLVR Weight Change Analysis"
echo "=========================================="
echo "Output directory: $OUTPUT_BASE"
echo ""

# Install dependencies if needed
pip install matplotlib scipy --quiet 2>/dev/null || true

# Run analysis for 4B model (has transfer matrix for correlation)
echo ""
echo "=========================================="
echo "Analyzing 4B model (Qwen3-4B)"
echo "=========================================="
python /code-fsx/sunguha/rlvr_experiments/experiments/rlvr_geometry_paper/scripts/analysis/analyze_rlvr_weight_changes.py \
    --model_size 4b \
    --output_dir $OUTPUT_BASE \
    2>&1 | tee $OUTPUT_BASE/4b_analysis.log

# Run analysis for 3B model
echo ""
echo "=========================================="
echo "Analyzing 3B model (Qwen2.5-3B)"
echo "=========================================="
python /code-fsx/sunguha/rlvr_experiments/experiments/rlvr_geometry_paper/scripts/analysis/analyze_rlvr_weight_changes.py \
    --model_size 3b \
    --output_dir $OUTPUT_BASE \
    2>&1 | tee $OUTPUT_BASE/3b_analysis.log

# Run analysis for 7B model
echo ""
echo "=========================================="
echo "Analyzing 7B model (Qwen2.5-7B)"
echo "=========================================="
python /code-fsx/sunguha/rlvr_experiments/experiments/rlvr_geometry_paper/scripts/analysis/analyze_rlvr_weight_changes.py \
    --model_size 7b \
    --output_dir $OUTPUT_BASE \
    2>&1 | tee $OUTPUT_BASE/7b_analysis.log

# Run analysis for 8B model
echo ""
echo "=========================================="
echo "Analyzing 8B model (Qwen3-8B)"
echo "=========================================="
python /code-fsx/sunguha/rlvr_experiments/experiments/rlvr_geometry_paper/scripts/analysis/analyze_rlvr_weight_changes.py \
    --model_size 8b \
    --output_dir $OUTPUT_BASE \
    2>&1 | tee $OUTPUT_BASE/8b_analysis.log

echo ""
echo "=========================================="
echo "Analysis Complete!"
echo "=========================================="
echo "Results saved to: $OUTPUT_BASE"
echo ""
echo "Generated files:"
ls -la $OUTPUT_BASE/*/

echo ""
echo "Done!"
