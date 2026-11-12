#!/usr/bin/env bash
# Curriculum study, mild length penalty pair, on a 1x A100 80GB VM with the data volume at /data
# (scripts/vm/setup.sh) and local disk at /ephemeral. Ends by deleting the VM.
#
#   tmux new -d -s curriculum 'bash scripts/vm/curriculum_lp_pair.sh'
#
# For each run in RUNS (default: the arm, then its control):
#   cur_mixed_lp   experiments/atom_decomposition/configs/cur_mixed_lp.yaml  (atoms + mild penalty)
#   full_lp        experiments/atom_decomposition/configs/full_lp.yaml       (mild penalty, no atoms)
# 1. 300 GRPO steps; checkpoints 100/200/300 on /ephemeral (lost with the VM: the volume is nearly full)
# 2. full-task eval (same as the baselines) of checkpoints 100, 200, 300; atom eval (96-token cap) of 300
# 3. trainer_state.json and generation logs copied to /data/rlvr/curriculum/<run>_seed1/
# Waits HOLD seconds (default 300) before deleting the VM, so results can be pulled off /data.
#
# DRY=1 R=<dir> prints the training commands and runs the evals with the fake backend.
set -uo pipefail
DRY=${DRY:-0}
RUNS=${RUNS:-"cur_mixed_lp full_lp"}
if [ "$DRY" = 1 ]; then
    R=${R:?set R=<scratch dir> for a dry run}; CK=$R/ephemeral
    BACKEND="--backend fake --fake_p_correct 0.6"; N=20
    J() { local name=$1; shift 2; echo "[$name] $*"; case "$1" in python) [ "$2" = train_grpo.py ] || "$@";; *) "$@";; esac; }
else
    source /data/env.sh
    R=/data/rlvr/curriculum; CK=/ephemeral/curriculum
    sudo mkdir -p $CK && sudo chown "$USER" $CK
    BACKEND="--backend vllm"; N=500
    J() { bash scripts/vm/run_job.sh "$@"; }
fi
E=$R/eval
PY="python -m experiments.atom_decomposition"
MUL='{"basic_arithmetic": {"min_terms": 2, "max_terms": 2, "min_digits": 2, "max_digits": 4, "operators": ["*"]}}'
ATOM_ARGS="full --tasks basic_arithmetic --n $N --seed 9999 --k 8 --t01 --max_new_tokens_full 96 $BACKEND"
FULL_ARGS="full --tasks basic_arithmetic --n $N --seed 9999 --k 16 --t01 $BACKEND"
mkdir -p "$R" "$E" "$CK"
DONE=$R/lp_pair.done
failed=()

finish() {
    printf 'failed=%s\nend=%s\n' "${failed[*]:-none}" "$(date -u +%FT%TZ)" > "$DONE"
    [ "$DRY" = 1 ] && return
    nvidia-smi > /data/logs/nvidia_smi_end_lp_pair.txt 2>&1 || true
    sync
    sleep "${HOLD:-300}"
    cd "$HOME"   # leave /data so it can be unmounted
    bash "$HOME/self_delete.sh"
}
trap finish EXIT
rm -f "$DONE"   # a marker left by an earlier run would tell the local watcher we are done

for name in $RUNS; do
    run=${name}_seed1
    cfg=experiments/atom_decomposition/configs/$name.yaml
    if [ ! -f $CK/$run/checkpoint-300/config.json ]; then
        J train_$run 3h python train_grpo.py $cfg --output_dir $CK/$run || failed+=(train_$run)
    fi
    mkdir -p $R/$run
    cp $CK/$run/trainer_state.json $R/$run/ 2>/dev/null || cp $CK/$run/checkpoint-*/trainer_state.json $R/$run/ 2>/dev/null || true
    cp -r $CK/$run/generations $R/$run/ 2>/dev/null || true
    for n in 100 200 300; do
        ck=$CK/$run/checkpoint-$n
        [ "$DRY" = 1 ] || [ -d $ck ] || { failed+=(missing_$run-$n); continue; }
        [ -f $E/$run/step$n/manifest.json ] || \
            J eval_${run}_$n 40m $PY.eval_atoms $FULL_ARGS --model $ck --out_dir $E/$run/step$n || failed+=(eval_${run}_$n)
    done
    ck=$CK/$run/checkpoint-300
    if [ "$DRY" = 1 ] || [ -d $ck ]; then
        [ -f $E/atom_mul/${run}_300/manifest.json ] || \
            J atom_eval_${run}_300 20m $PY.eval_atoms $ATOM_ARGS --task_configs "$MUL" --model $ck \
                --out_dir $E/atom_mul/${run}_300 || failed+=(atom_eval_${run}_300)
    fi
done
echo "LP PAIR DONE (failed: ${failed[*]:-none})"
