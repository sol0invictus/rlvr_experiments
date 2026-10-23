#!/usr/bin/env bash
# Phase B1 smoke test on a small GPU (A6000): the whole pipeline on Qwen2.5-0.5B.
#   SFT (20 steps) -> GRPO 10 steps with HF generate -> GRPO 10 steps with vLLM
#   -> eval (HF and vLLM, 32 samples) -> task-vector geometry.
# Run inside tmux:  tmux new -d -s smoke 'bash scripts/vm/smoke.sh'
set -euo pipefail
source /data/env.sh
J="bash scripts/vm/run_job.sh"
R=/data/rlvr

python data_gen/generate_sft_warmup.py --num_samples 1000 --seed 42
sha256sum data_gen/data/sft_warmup.jsonl | tee $R/sft_warmup.sha256

$J smoke_sft      20m python train_sft.py configs/replication/smoke_sft_qwen25_0.5b.yaml
mkdir -p $R/equiv
$J smoke_equiv    30m python scripts/equivalence_rollouts.py --model $R/smoke_sft_0.5b \
    --config configs/replication/smoke_grpo_qwen25_0.5b.yaml --prompts 64 --out $R/equiv/smoke_sft_0.5b.json
$J smoke_grpo_hf  30m python train_grpo.py configs/replication/smoke_grpo_qwen25_0.5b.yaml --max_steps 10
$J smoke_grpo_vllm 30m python train_grpo.py configs/replication/smoke_grpo_qwen25_0.5b_vllm.yaml --max_steps 10

for m in smoke_sft_0.5b smoke_grpo_0.5b smoke_grpo_0.5b_vllm; do
    for b in hf vllm; do
        $J eval_${m}_$b 20m python evaluate_transfer.py --model $R/$m --tasks basic_arithmetic \
            --num_samples 32 --backend $b --output_dir $R/eval/$m
    done
done

$J smoke_geometry 20m python analyze_task_vector.py --sft $R/smoke_sft_0.5b \
    --grpo $R/smoke_grpo_0.5b $R/smoke_grpo_0.5b_vllm --svd --out_dir $R/geometry/smoke

echo "SMOKE ALL DONE"
