#!/usr/bin/env bash
# Curriculum study, arm "mul100 -> T200", on a 1x A100 80GB VM with the data volume at /data
# (scripts/vm/setup.sh). Needs the atom pilot's checkpoint-100 (scripts/vm/curriculum_atom_pilot.sh).
# Ends by deleting the VM; the volume keeps everything.
#
#   tmux new -d -s curriculum 'bash scripts/vm/curriculum_mul_then_T.sh'
#
# 1. 200 full-task GRPO steps from the atom checkpoint (experiments/atom_decomposition/configs/cur_mul100_T200.yaml)
# 2. full-task eval (same as the baselines) of its checkpoints 100 and 200 (= 200 and 300 steps in all),
#    and the atom eval (96-token cap) of checkpoint 200, to see whether the atom skill survives
#
# DRY=1 R=<dir> prints the training command and runs the evals with the fake backend.
set -uo pipefail
DRY=${DRY:-0}
CONFIG=experiments/atom_decomposition/configs/cur_mul100_T200.yaml
run=cur_mul100_T200_seed1
if [ "$DRY" = 1 ]; then
    R=${R:?set R=<scratch dir> for a dry run}
    BACKEND="--backend fake --fake_p_correct 0.6"; N=20
    J() { local name=$1; shift 2; echo "[$name] $*"; case "$1" in python) [ "$2" = train_grpo.py ] || "$@";; *) "$@";; esac; }
else
    source /data/env.sh
    R=/data/rlvr/curriculum
    BACKEND="--backend vllm"; N=500
    J() { bash scripts/vm/run_job.sh "$@"; }
    [ -f $R/atom_mul_seed1/checkpoint-100/config.json ] || { echo "missing atom checkpoint-100"; exit 1; }
fi
E=$R/eval
PY="python -m experiments.atom_decomposition"
MUL='{"basic_arithmetic": {"min_terms": 2, "max_terms": 2, "min_digits": 2, "max_digits": 4, "operators": ["*"]}}'
ATOM_ARGS="full --tasks basic_arithmetic --n $N --seed 9999 --k 8 --t01 --max_new_tokens_full 96 $BACKEND"
FULL_ARGS="full --tasks basic_arithmetic --n $N --seed 9999 --k 16 --t01 $BACKEND"
mkdir -p "$R" "$E"
DONE=$R/$run.done
failed=()

finish() {
    printf 'failed=%s\nend=%s\n' "${failed[*]:-none}" "$(date -u +%FT%TZ)" > "$DONE"
    [ "$DRY" = 1 ] && return
    nvidia-smi > /data/logs/nvidia_smi_end_$run.txt 2>&1 || true
    sync
    cd "$HOME"   # leave /data so it can be unmounted
    bash "$HOME/self_delete.sh"
}
trap finish EXIT
rm -f "$DONE"   # a marker left by an earlier run would tell the local watcher we are done

if [ ! -f $R/$run/checkpoint-200/config.json ]; then
    J train_$run 4h python train_grpo.py $CONFIG --output_dir $R/$run || failed+=(train_$run)
fi
for n in 100 200; do
    ck=$R/$run/checkpoint-$n
    [ "$DRY" = 1 ] || [ -d $ck ] || { failed+=(missing_$run-$n); continue; }
    [ -f $E/$run/step$n/manifest.json ] && continue
    J eval_${run}_$n 40m $PY.eval_atoms $FULL_ARGS --model $ck --out_dir $E/$run/step$n || failed+=(eval_${run}_$n)
done
ck=$R/$run/checkpoint-200
if [ "$DRY" = 1 ] || [ -d $ck ]; then
    [ -f $E/atom_mul/${run}_200/manifest.json ] || \
        J atom_eval_${run}_200 20m $PY.eval_atoms $ATOM_ARGS --task_configs "$MUL" --model $ck \
            --out_dir $E/atom_mul/${run}_200 || failed+=(atom_eval_${run}_200)
fi
du -sh $R/$run || true
echo "CURRICULUM $run DONE (failed: ${failed[*]:-none})"
