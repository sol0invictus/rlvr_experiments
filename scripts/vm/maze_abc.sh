#!/usr/bin/env bash
# Maze A/B/C test, on a 1x A100 80GB VM with the data volume at /data (scripts/vm/setup.sh) and local
# disk at /ephemeral. Ends by deleting the VM.
#
#   tmux new -d -s maze 'bash scripts/vm/maze_abc.sh'
#
# Arms (experiments/atom_decomposition/configs/maze_{A,B,C}.yaml), 300 GRPO steps each, seed 1:
#   A  hard mazes only (path 10-15)
#   B  hard + easier whole mazes (path 1-9), half the prompts
#   C  hard + solver-traced atoms of those easier mazes, half the prompts
# Evals (hard = path 10-15 on 10-14 grids, 200 prompts, eval seed 9999, 8 samples at T=1.0 + 1 at T=0.1):
#   SFT start on hard; each arm's checkpoints 100/200/300 on hard; checkpoint-300 on default-size mazes and
#   on the maze probe's hard atom build (atoms + full task, /data/rlvr/maze_probe/hard/build).
# Checkpoints stay on /ephemeral (lost with the VM: the volume is nearly full); trainer_state.json and
# generation logs are copied to /data/rlvr/maze_abc/<run>/. Waits HOLD seconds before deleting the VM.
#
# DRY=1 R=<dir> prints the training commands and runs the evals with the fake backend.
set -uo pipefail
DRY=${DRY:-0}
RUNS=${RUNS:-"maze_A maze_B maze_C"}
HARD='{"maze": {"min_dist": 10, "max_dist": 15, "min_grid_size": 10, "max_grid_size": 14}}'
if [ "$DRY" = 1 ]; then
    R=${R:?set R=<scratch dir> for a dry run}; CK=$R/ephemeral; SFT=none
    BACKEND="--backend fake --fake_p_correct 0.3"; N=10
    ATOM_BUILD=$R/hard_build
    [ -f $ATOM_BUILD/manifest.json ] || python -m experiments.atom_decomposition.build_atoms --tasks maze --n 3 \
        --seed 2028 --max_l2 12 --task_configs "$HARD" --out_dir $ATOM_BUILD > /dev/null
    J() { local name=$1; shift 2; echo "[$name] $*"; case "$1" in python) [ "$2" = train_grpo.py ] || "$@";; *) "$@";; esac; }
else
    source /data/env.sh
    R=/data/rlvr/maze_abc; CK=/ephemeral/maze; SFT=/data/rlvr/sft_qwen3_4b
    sudo mkdir -p $CK && sudo chown "$USER" $CK
    BACKEND="--backend vllm"; N=200
    ATOM_BUILD=/data/rlvr/maze_probe/hard/build
    J() { bash scripts/vm/run_job.sh "$@"; }
fi
E=$R/eval
PY="python -m experiments.atom_decomposition"
FULL="full --tasks maze --n $N --seed 9999 --k 8 --t01 --max_new_tokens_full 2048 --max_model_len 4096 $BACKEND"
mkdir -p "$R" "$E" "$CK"
DONE=$R/maze_abc.done
failed=()

finish() {
    printf 'failed=%s\nend=%s\n' "${failed[*]:-none}" "$(date -u +%FT%TZ)" > "$DONE"
    [ "$DRY" = 1 ] && return
    nvidia-smi > /data/logs/nvidia_smi_end_maze_abc.txt 2>&1 || true
    sync
    sleep "${HOLD:-300}"
    cd "$HOME"   # leave /data so it can be unmounted
    bash "$HOME/self_delete.sh"
}
trap finish EXIT
rm -f "$DONE"   # a marker left by an earlier run would tell the local watcher we are done

[ -f $E/sft/hard/manifest.json ] || \
    J eval_maze_sft_hard 30m $PY.eval_atoms $FULL --task_configs "$HARD" --model $SFT --out_dir $E/sft/hard \
    || failed+=(eval_sft_hard)

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
        [ -f $E/$run/hard_step$n/manifest.json ] || \
            J eval_${run}_hard_$n 30m $PY.eval_atoms $FULL --task_configs "$HARD" --model $ck \
                --out_dir $E/$run/hard_step$n || failed+=(eval_${run}_hard_$n)
    done
    ck=$CK/$run/checkpoint-300
    if [ "$DRY" = 1 ] || [ -d $ck ]; then
        [ -f $E/$run/default_step300/manifest.json ] || \
            J eval_${run}_default_300 30m $PY.eval_atoms $FULL --model $ck --out_dir $E/$run/default_step300 \
            || failed+=(eval_${run}_default_300)
        [ -f $E/$run/atoms_step300/manifest.json ] || \
            J atoms_${run}_300 60m $PY.eval_atoms atoms --model $ck $BACKEND --build_dir $ATOM_BUILD \
                --out_dir $E/$run/atoms_step300 --t01 --k_atom 4 --k_full 8 --max_new_tokens_full 2048 \
                --max_model_len 4096 || failed+=(atoms_${run}_300)
    fi
done
echo "MAZE ABC DONE (failed: ${failed[*]:-none})"
