#!/usr/bin/env bash
# Phase B3-B5: 300-step GRPO replication from the chosen W_sft, then eval + geometry.
#   SFT_DIR=/data/rlvr/sft_qwen3_4b CONFIG=configs/replication/grpo_basic_arithmetic_4b_vllm.yaml \
#     tmux new -d -s repl 'bash scripts/vm/replication.sh'
# The GRPO config's model.name_or_path must point at SFT_DIR.
set -euo pipefail
source /data/env.sh
J="bash scripts/vm/run_job.sh"
R=/data/rlvr
SFT_DIR=${SFT_DIR:-$R/sft_qwen3_4b}
CONFIG=${CONFIG:-configs/replication/grpo_basic_arithmetic_4b_vllm.yaml}
OUT=$(python -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['training']['output_dir'])" "$CONFIG")

grep -q "name_or_path: $SFT_DIR" "$CONFIG" || { echo "config does not load $SFT_DIR"; exit 1; }

$J grpo_main 4h python train_grpo.py "$CONFIG"

# Eval: HF reference on SFT + final; vLLM on every checkpoint (accuracy + mean length).
$J eval_final_hf 60m python evaluate_transfer.py --model "$OUT" --tasks basic_arithmetic \
    --num_samples 200 --backend hf --output_dir "$OUT/eval/final_hf"
for ck in $(ls -d "$OUT"/checkpoint-* | sort -t- -k2 -n); do
    $J eval_$(basename "$ck")_vllm 30m python evaluate_transfer.py --model "$ck" --tasks basic_arithmetic \
        --num_samples 200 --backend vllm --output_dir "$OUT/eval/$(basename "$ck")"
done

# Geometry: full SVD metrics on the final tau, cheap metrics on every checkpoint.
$J geometry_final 60m python analyze_task_vector.py --sft "$SFT_DIR" --grpo "$OUT" --svd \
    --out "$OUT/geometry/final.json"
$J geometry_ckpts 60m python analyze_task_vector.py --sft "$SFT_DIR" \
    --grpo $(ls -d "$OUT"/checkpoint-* | sort -t- -k2 -n) --out_dir "$OUT/geometry/checkpoints"

echo "REPLICATION ALL DONE"
