#!/usr/bin/env bash
# Unattended baseline replication (HF generate, original recipe) on a 1x A100 80GB VM.
#   SFT -> eval W_sft -> HF-vs-vLLM rollout check -> 300-step GRPO -> evals -> geometry
#   -> self-delete the VM.
# Everything lands on the /data volume, which outlives the VM.
#
#   RUN=qwen3_4b tmux new -d -s overnight 'bash scripts/vm/overnight.sh'
#
# RUN picks the SFT base (configs/replication/sft_<RUN>.yaml):
#   qwen3_4b_instruct2507 (default, first run) | qwen3_4b
#
# Steps after SFT/GRPO are best-effort: a failed eval or geometry step is recorded
# and the pipeline moves on, so the VM is always deleted at the end.
set -uo pipefail
source /data/env.sh
J="bash scripts/vm/run_job.sh"
R=/data/rlvr
RUN=${RUN:-qwen3_4b_instruct2507}
SFT_CONFIG=configs/replication/sft_$RUN.yaml
SFT=$R/sft_$RUN
if [ "$RUN" = qwen3_4b ]; then
    CONFIG=configs/replication/grpo_basic_arithmetic_4b.yaml
    OUT=$R/grpo_basic_arithmetic_4b
else
    CONFIG=configs/replication/grpo_basic_arithmetic_4b_$RUN.yaml
    OUT=$R/grpo_basic_arithmetic_4b_$RUN
fi
grep -q "name_or_path: $SFT\$" "$CONFIG" || { echo "$CONFIG does not load $SFT"; exit 1; }
TASKS=basic_arithmetic,syllogism,knights_knaves,countdown,maze,shortest_path,mini_sudoku,leg_counting
STATUS=$R/status/overnight.done
failed=()

finish() {
    printf 'failed=%s\nend=%s\n' "${failed[*]:-none}" "$(date -u +%FT%TZ)" > "$STATUS"
    nvidia-smi > /data/logs/nvidia_smi_end.txt 2>&1 || true
    sync
    cd "$HOME"   # leave /data so it can be unmounted
    bash "$HOME/self_delete.sh"
}
trap finish EXIT

python data_gen/generate_sft_warmup.py --num_samples 1000 --seed 42
sha256sum data_gen/data/sft_warmup.jsonl | tee $R/sft_warmup.sha256

# --- SFT warm-up -> W_sft --------------------------------------------------
if [ ! -f "$SFT/config.json" ]; then
    $J sft 90m python train_sft.py $SFT_CONFIG || { failed+=(sft); exit 1; }
fi
$J eval_sft_vllm 90m python evaluate_replication.py --model $SFT --tasks $TASKS \
    --num_samples 200 --backend vllm --output_dir $R/eval/sft_$RUN || failed+=(eval_sft_vllm)
$J eval_sft_hf 60m python evaluate_replication.py --model $SFT --tasks basic_arithmetic \
    --num_samples 200 --backend hf --output_dir $R/eval/sft_$RUN || failed+=(eval_sft_hf)

# --- Is vLLM generation a recipe-preserving swap at 4B? (HF vs vLLM rollouts) -------
mkdir -p $R/equiv
$J equiv_4b 60m python scripts/equivalence_rollouts.py --model $SFT --config $CONFIG \
    --prompts 64 --batch_size 64 --out $R/equiv/sft_$RUN.json || failed+=(equiv_4b)

# --- GRPO, 300 steps, HF generate (the original generation path) ---------------
$J grpo 5h python train_grpo.py $CONFIG || failed+=(grpo)

# --- Evals ------------------------------------------------------------------------
if [ -f "$OUT/config.json" ]; then
    $J eval_final_hf 60m python evaluate_replication.py --model $OUT --tasks basic_arithmetic \
        --num_samples 200 --backend hf --output_dir $OUT/eval/final_hf || failed+=(eval_final_hf)
fi
for ck in $(ls -d $OUT/checkpoint-* 2>/dev/null | sort -t- -k2 -n); do
    n=$(basename $ck)
    $J eval_${n}_vllm 30m python evaluate_replication.py --model $ck --tasks basic_arithmetic \
        --num_samples 200 --backend vllm --output_dir $OUT/eval/$n || failed+=(eval_$n)
done

# --- Geometry -----------------------------------------------------------------------
# Final tau: all metrics, the fp32-GPU-SVD variant, and the controls (random matched-norm
# null, fp32 CPU SVD, spectral gap) on every 4th layer -- the rotation-artifact check.
if [ -f "$OUT/config.json" ]; then
    $J geometry_final 2h python analyze_task_vector.py --sft $SFT --grpo $OUT --svd \
        --fp32_svd_variant --controls_every 4 --out $OUT/geometry/final.json || failed+=(geometry_final)
fi
# Rotation over training (SVD metrics) at steps 100 and 200.
for n in 100 200; do
    [ -d $OUT/checkpoint-$n ] || continue
    $J geometry_ckpt$n 1h python analyze_task_vector.py --sft $SFT --grpo $OUT/checkpoint-$n --svd \
        --fp32_svd_variant --out $OUT/geometry/checkpoint-$n.json || failed+=(geometry_ckpt$n)
done
# Cheap metrics (norms, depth profile) on every checkpoint.
ck_list=$(ls -d $OUT/checkpoint-* 2>/dev/null | sort -t- -k2 -n)
if [ -n "$ck_list" ]; then
    $J geometry_ckpts 1h python analyze_task_vector.py --sft $SFT --grpo $ck_list \
        --out_dir $OUT/geometry/checkpoints || failed+=(geometry_ckpts)
fi

echo "OVERNIGHT ALL DONE (failed: ${failed[*]:-none})"
