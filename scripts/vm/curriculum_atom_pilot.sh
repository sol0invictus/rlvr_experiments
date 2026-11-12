#!/usr/bin/env bash
# Curriculum study, atom pilot, on a 1x A100 80GB VM with the data volume at /data (scripts/vm/setup.sh).
# Can the L2 multiplication atom, trained with answers capped at 96 tokens, make full-task traces shorter?
# Ends by deleting the VM; the volume keeps everything.
#
#   tmux new -d -s curriculum 'bash scripts/vm/curriculum_atom_pilot.sh'
#
# 1. atom eval (one-step multiplication, 96-token cap, 500 prompts, eval seed 9999) of the SFT start and
#    of the no-penalty full-task run's checkpoint-300
# 2. 300 GRPO steps on the atom (experiments/atom_decomposition/configs/atom_mul_4b.yaml)
# 3. atom eval of checkpoints 100, 200, 300; full-task eval (same as the baselines) of 100 and 300
#
# DRY=1 R=<dir> prints the training commands and runs the evals with the fake backend (no GPU,
# no self-delete), to check paths.
set -uo pipefail
DRY=${DRY:-0}
CONFIG=experiments/atom_decomposition/configs/atom_mul_4b.yaml
run=atom_mul_seed1
if [ "$DRY" = 1 ]; then
    R=${R:?set R=<scratch dir> for a dry run}
    SFT=none; NOLP=none; BACKEND="--backend fake --fake_p_correct 0.6"; N=20
    J() { local name=$1; shift 2; echo "[$name] $*"; case "$1" in python) [ "$2" = train_grpo.py ] || "$@";; *) "$@";; esac; }
else
    source /data/env.sh
    R=/data/rlvr/curriculum
    SFT=/data/rlvr/sft_qwen3_4b; NOLP=$R/arith_Anolp_seed1/checkpoint-300; BACKEND="--backend vllm"; N=500
    J() { bash scripts/vm/run_job.sh "$@"; }
fi
E=$R/eval
PY="python -m experiments.atom_decomposition"
MUL='{"basic_arithmetic": {"min_terms": 2, "max_terms": 2, "min_digits": 2, "max_digits": 4, "operators": ["*"]}}'
ATOM_ARGS="full --tasks basic_arithmetic --n $N --seed 9999 --k 8 --t01 --max_new_tokens_full 96 $BACKEND"
FULL_ARGS="full --tasks basic_arithmetic --n $N --seed 9999 --k 16 --t01 $BACKEND"
mkdir -p "$R" "$E"
DONE=$R/atom_pilot.done
failed=()

finish() {
    printf 'failed=%s\nend=%s\n' "${failed[*]:-none}" "$(date -u +%FT%TZ)" > "$DONE"
    [ "$DRY" = 1 ] && return
    nvidia-smi > /data/logs/nvidia_smi_end_atom_pilot.txt 2>&1 || true
    sync
    cd "$HOME"   # leave /data so it can be unmounted
    bash "$HOME/self_delete.sh"
}
trap finish EXIT
rm -f "$DONE"   # a marker left by an earlier run would tell the local watcher we are done

atom_eval() {   # name model out_dir
    [ -f "$3/manifest.json" ] && return 0
    J "$1" 20m $PY.eval_atoms $ATOM_ARGS --task_configs "$MUL" --model "$2" --out_dir "$3" || failed+=("$1")
}

atom_eval atom_eval_sft "$SFT" $E/atom_mul/sft
atom_eval atom_eval_nolp300 "$NOLP" $E/atom_mul/arith_Anolp_seed1_300

if [ ! -f $R/$run/checkpoint-300/config.json ]; then
    J train_$run 2h python train_grpo.py $CONFIG --output_dir $R/$run || failed+=(train_$run)
fi
for n in 100 200 300; do
    ck=$R/$run/checkpoint-$n
    [ "$DRY" = 1 ] || [ -d $ck ] || { failed+=(missing_$run-$n); continue; }
    atom_eval atom_eval_${run}_$n $ck $E/atom_mul/${run}_$n
done
for n in 100 300; do
    ck=$R/$run/checkpoint-$n
    [ "$DRY" = 1 ] || [ -d $ck ] || continue
    [ -f $E/$run/step$n/manifest.json ] && continue
    J eval_${run}_$n 40m $PY.eval_atoms $FULL_ARGS --model $ck --out_dir $E/$run/step$n || failed+=(eval_${run}_$n)
done
du -sh $R/$run || true
echo "ATOM PILOT DONE (failed: ${failed[*]:-none})"
