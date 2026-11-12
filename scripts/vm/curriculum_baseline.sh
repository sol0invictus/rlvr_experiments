#!/usr/bin/env bash
# Curriculum study, arm A (full task only, the fixed baseline) on a 1x A100 80GB VM with the data
# volume at /data (scripts/vm/setup.sh). Ends by deleting the VM; the volume keeps everything.
#
#   tmux new -d -s curriculum 'bash scripts/vm/curriculum_baseline.sh'
#
# 1. eval the SFT start once (500 puzzles, eval seed 9999: 16 samples at T=1.0 + 1 at T=0.1)
# 2. 2-step smoke run of the config (catches vLLM/memory problems before the long runs)
# 3. per seed: 300 GRPO steps (experiments/atom_decomposition/configs/baseline_arith_4b.yaml),
#    then the same eval on checkpoints 100, 200, 300
# 4. curriculum_report.py over everything
# Reruns skip finished training runs and resume unfinished evals.
#
# DRY=1 R=<dir> prints the training commands and runs the evals with the fake backend (no GPU,
# no self-delete), to check paths and the report.
set -uo pipefail
DRY=${DRY:-0}
SEEDS=${SEEDS:-"1"}   # one seed while the setup is being fixed; SEEDS="1 2 3" for the final runs
ARM=${ARM:-A}   # run names arith_<ARM>_seed<k>; e.g. ARM=Anolp CONFIG=.../baseline_arith_4b_nolp.yaml
CONFIG=${CONFIG:-experiments/atom_decomposition/configs/baseline_arith_4b.yaml}
if [ "$DRY" = 1 ]; then
    R=${R:?set R=<scratch dir> for a dry run}
    SFT=none; BACKEND="--backend fake --fake_p_correct 0.6"; N=20
    J() { local name=$1; shift 2; echo "[$name] $*"; case "$1" in python) [ "$2" = train_grpo.py ] || "$@";; *) "$@";; esac; }
else
    source /data/env.sh
    R=/data/rlvr/curriculum
    SFT=/data/rlvr/sft_qwen3_4b; BACKEND="--backend vllm"; N=500
    J() { bash scripts/vm/run_job.sh "$@"; }
fi
E=$R/eval
PY="python -m experiments.atom_decomposition"
EVAL_ARGS="full --tasks basic_arithmetic --n $N --seed 9999 --k 16 --t01 $BACKEND"
mkdir -p "$R" "$E"
DONE=$R/baseline_$ARM.done
failed=()

finish() {
    printf 'failed=%s\nend=%s\n' "${failed[*]:-none}" "$(date -u +%FT%TZ)" > "$DONE"
    [ "$DRY" = 1 ] && return
    nvidia-smi > /data/logs/nvidia_smi_end_curriculum.txt 2>&1 || true
    sync
    cd "$HOME"   # leave /data so it can be unmounted
    bash "$HOME/self_delete.sh"
}
trap finish EXIT
rm -f "$DONE"   # a marker left by an earlier run would tell the local watcher we are done

J eval_sft 40m $PY.eval_atoms $EVAL_ARGS --model $SFT --out_dir $E/sft/step000 || failed+=(eval_sft)

if [ "$DRY" != 1 ] && [ ! -f $R/smoke.ok ]; then
    J smoke 30m python train_grpo.py $CONFIG --max_steps 2 --output_dir $R/smoke \
        && touch $R/smoke.ok && rm -rf $R/smoke || { failed+=(smoke); exit 1; }
fi

for s in $SEEDS; do
    run=arith_${ARM}_seed$s
    if [ ! -f $R/$run/checkpoint-300/config.json ]; then
        J train_$run 4h python train_grpo.py $CONFIG --seed $s --output_dir $R/$run || { failed+=(train_$run); continue; }
    fi
    for n in 100 200 300; do
        ck=$R/$run/checkpoint-$n
        [ "$DRY" = 1 ] || [ -d $ck ] || { failed+=(missing_$run-$n); continue; }
        J eval_${run}_$n 40m $PY.eval_atoms $EVAL_ARGS --model $ck --out_dir $E/$run/step$n \
            || failed+=(eval_${run}_$n)
    done
done

$PY.curriculum_report --eval_root $E --run_root $R --eval_n $N --out $R/report_$ARM.json | tee $R/report_$ARM.txt \
    || failed+=(report)
echo "CURRICULUM BASELINE DONE (failed: ${failed[*]:-none})"
