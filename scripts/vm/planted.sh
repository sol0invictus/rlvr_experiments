#!/usr/bin/env bash
# Planted shortcut + frontier ladder on arithmetic (brainstorm sections 3 and 8), on a 1x A100 80GB VM with the
# data volume at /data (scripts/vm/setup.sh) and local disk at /ephemeral. Ends by deleting the VM.
#
#   tmux new -d -s planted 'bash scripts/vm/planted.sh'
#
# Runs (experiments/atom_decomposition/configs/arith_planted_q*.yaml, from the 4B SFT start, 300 steps each):
#   q00 = no planting (the 94.2% no-penalty baseline; seed 1 repeats that run), q03 / q06 = 30% / 60% of the
#   training puzzles answer -1. Order: seed 1 of q00, q06, q03, then seed 2 of each, so a dead VM still
#   leaves one full seed.
# Evals:
#   std     500 puzzles (seed 9999), 8 samples at T=1.0, at steps 25 50 100 150 200 300;
#           at step 300 16 samples + 1 at T=0.1 (as the earlier baseline evals)
#   ladder  100 puzzles per digit level 1..6 (seed 4242), 8 samples at T=1.0; every eval step for q00,
#           step 300 only for q03/q06; also the SFT start
# Checkpoints (save_at_steps in the configs) stay on /ephemeral and are deleted after each run's evals unless
# KEEP_CKPT=1 (they are 8 GB each and the volume has no room). trainer_state.json, effective_config.json and the
# generation logs go to /data/rlvr/planted/<run>/.
# Waits HOLD seconds before deleting the VM.
#
# DRY=1 R=<dir> prints the training commands and runs the evals with the fake backend.
set -uo pipefail
DRY=${DRY:-0}
RUNS=${RUNS:-"q00_seed1 q06_seed1 q03_seed1 q00_seed2 q06_seed2 q03_seed2"}
STEPS="25 50 100 150 200 300"
KEEP_CKPT=${KEEP_CKPT:-0}
PY="python -m experiments.atom_decomposition"
LEVELS='{"d1": {"min_digits": 1, "max_digits": 1}, "d2": {"min_digits": 2, "max_digits": 2}, "d3": {"min_digits": 3, "max_digits": 3}, "d4": {"min_digits": 4, "max_digits": 4}, "d5": {"min_digits": 5, "max_digits": 5}, "d6": {"min_digits": 6, "max_digits": 6}}'
if [ "$DRY" = 1 ]; then
    R=${R:?set R=<scratch dir> for a dry run}; CK=$R/ephemeral; SFT=none
    BACKEND="--backend fake --fake_p_correct 0.6"; N=20; NL=5
    J() { local name=$1; shift 2; echo "[$name] $*"; case "$1" in python) [ "$2" = train_grpo.py ] || "$@";; *) "$@";; esac; }
else
    source /data/env.sh
    R=/data/rlvr/planted; CK=/ephemeral/planted; SFT=/data/rlvr/sft_qwen3_4b
    sudo mkdir -p $CK && sudo chown "$USER" $CK
    BACKEND="--backend vllm"; N=500; NL=100
    J() { bash scripts/vm/run_job.sh "$@"; }
fi
E=$R/eval
STD="full --tasks basic_arithmetic --n $N --seed 9999 --k 8 $BACKEND"
STD_LAST="full --tasks basic_arithmetic --n $N --seed 9999 --k 16 --t01 $BACKEND"
LADDER="full --tasks basic_arithmetic --n $NL --seed 4242 --k 8 $BACKEND"
mkdir -p "$R" "$E" "$CK"
DONE=$R/planted.done
failed=()

finish() {
    printf 'failed=%s\nend=%s\n' "${failed[*]:-none}" "$(date -u +%FT%TZ)" > "$DONE"
    [ "$DRY" = 1 ] && return
    nvidia-smi > /data/logs/nvidia_smi_end_planted.txt 2>&1 || true
    sync
    sleep "${HOLD:-300}"
    cd "$HOME"   # leave /data so it can be unmounted
    bash "$HOME/self_delete.sh"
}
trap finish EXIT
rm -f "$DONE"   # a marker left by an earlier run would tell the local watcher we are done

if [ "$DRY" != 1 ]; then
    free_kb=$(df --output=avail /data | tail -1)
    [ "$free_kb" -gt 1500000 ] || { echo "only $((free_kb / 1024)) MB free on /data"; failed+=(no_space); exit 1; }
fi

# --- SFT start ---
[ -f $E/sft/step000/std/manifest.json ] || J eval_sft_std 30m $PY.eval_atoms $STD_LAST --model $SFT \
    --out_dir $E/sft/step000/std || failed+=(eval_sft_std)
[ -f $E/sft/step000/ladder/manifest.json ] || J eval_sft_ladder 40m $PY.eval_atoms $LADDER --levels "$LEVELS" \
    --model $SFT --out_dir $E/sft/step000/ladder || failed+=(eval_sft_ladder)

# --- smoke: 6 steps of the q06 config (builds the planted set; save_at_steps must write checkpoint-5), then check
#     what the trainer received ---
if [ "$DRY" != 1 ] && [ ! -f $R/smoke.ok ]; then
    J smoke 30m python train_grpo.py experiments/atom_decomposition/configs/arith_planted_q06.yaml --max_steps 6 \
        --output_dir $CK/smoke || { failed+=(smoke); exit 1; }
    [ -f $CK/smoke/checkpoint-5/config.json ] || { echo "save_at_steps wrote no checkpoint-5"; failed+=(smoke_save); exit 1; }
    python scripts/vm/check_effective_config.py $CK/smoke/effective_config.json > $R/smoke_config_check.txt 2>&1 \
        || { cat $R/smoke_config_check.txt; failed+=(config_check); exit 1; }
    grep -h "Planted shortcut" /data/logs/smoke.log > $R/smoke_planted_stats.txt || true
    touch $R/smoke.ok && rm -rf $CK/smoke
fi

# --- runs ---
for run in $RUNS; do
    q=${run%_seed*}; seed=${run#*_seed}
    name=arith_$run
    cfg=experiments/atom_decomposition/configs/arith_planted_$q.yaml
    if [ ! -f $CK/$name/checkpoint-300/config.json ]; then
        J train_$name 3h python train_grpo.py $cfg --seed $seed --output_dir $CK/$name || failed+=(train_$name)
    fi
    mkdir -p $R/$name
    cp $CK/$name/trainer_state.json $CK/$name/effective_config.json $R/$name/ 2>/dev/null || \
        cp $CK/$name/checkpoint-300/trainer_state.json $R/$name/ 2>/dev/null || true
    cp -r $CK/$name/generations $R/$name/ 2>/dev/null || true
    for n in $STEPS; do
        ck=$CK/$name/checkpoint-$n
        [ "$DRY" = 1 ] || [ -d $ck ] || { failed+=(missing_$name-$n); continue; }
        args=$STD; [ $n = 300 ] && args=$STD_LAST
        [ -f $E/$name/step$n/std/manifest.json ] || J eval_${name}_std_$n 40m $PY.eval_atoms $args --model $ck \
            --out_dir $E/$name/step$n/std || failed+=(eval_${name}_std_$n)
        if [ $q = q00 ] || [ $n = 300 ]; then
            [ -f $E/$name/step$n/ladder/manifest.json ] || J eval_${name}_ladder_$n 40m $PY.eval_atoms $LADDER \
                --levels "$LEVELS" --model $ck --out_dir $E/$name/step$n/ladder || failed+=(eval_${name}_ladder_$n)
        fi
    done
    [ "$DRY" = 1 ] || [ "$KEEP_CKPT" = 1 ] || rm -rf $CK/$name/checkpoint-*
    $PY.planted_report --eval_root $E --out $R/report.json > $R/report.txt 2>&1 || true
done
$PY.planted_report --eval_root $E --out $R/report.json | tee $R/report.txt || failed+=(report)
echo "PLANTED DONE (failed: ${failed[*]:-none})"
