#!/usr/bin/env bash
# Maze follow-up to the A/B/C test, on a 1x A100 80GB VM with the data volume at /data (scripts/vm/setup.sh)
# and local disk at /ephemeral. Ends by deleting the VM.
#
#   tmux new -d -s maze 'bash scripts/vm/maze_next.sh'
#
# Arms (experiments/atom_decomposition/configs/), seed 1:
#   maze_B600   arm B (hard + easier whole mazes) for 600 steps: does its easy-maze skill reach hard mazes?
#   maze_Cread  hard + grid-reading atoms of hard-size mazes, 300 steps: atoms aimed at the failing sub-skill
# Evals as in maze_abc.sh (hard = path 10-15 on 10-14 grids, 200 prompts, eval seed 9999, 8 samples at T=1.0 +
# 1 at T=0.1), plus a grid-reading atom build of the maze probe's 150 hard instances (seed 2028), scored for the
# SFT start and each arm's last checkpoint. Checkpoints stay on /ephemeral; trainer_state.json and generation
# logs go to /data/rlvr/maze_next/<run>/. Waits HOLD seconds before deleting the VM.
#
# DRY=1 R=<dir> prints the training commands and runs the evals with the fake backend.
set -uo pipefail
DRY=${DRY:-0}
RUNS=${RUNS:-"maze_B600 maze_Cread"}
HARD='{"maze": {"min_dist": 10, "max_dist": 15, "min_grid_size": 10, "max_grid_size": 14}}'
PY="python -m experiments.atom_decomposition"
if [ "$DRY" = 1 ]; then
    R=${R:?set R=<scratch dir> for a dry run}; CK=$R/ephemeral; SFT=none
    BACKEND="--backend fake --fake_p_correct 0.3"; N=10; NB=3
    BFS_BUILD=$R/hard_build
    [ -f $BFS_BUILD/manifest.json ] || $PY.build_atoms --tasks maze --n 3 --seed 2028 --max_l2 12 \
        --task_configs "$HARD" --out_dir $BFS_BUILD > /dev/null
    J() { local name=$1; shift 2; echo "[$name] $*"; case "$1" in python) [ "$2" = train_grpo.py ] || "$@";; *) "$@";; esac; }
else
    source /data/env.sh
    R=/data/rlvr/maze_next; CK=/ephemeral/maze; SFT=/data/rlvr/sft_qwen3_4b
    sudo mkdir -p $CK && sudo chown "$USER" $CK
    BACKEND="--backend vllm"; N=200; NB=150
    BFS_BUILD=/data/rlvr/maze_probe/hard/build
    J() { bash scripts/vm/run_job.sh "$@"; }
fi
E=$R/eval
READ_BUILD=$R/reading_build
FULL="full --tasks maze --n $N --seed 9999 --k 8 --t01 --max_new_tokens_full 2048 --max_model_len 4096 $BACKEND"
# reading atoms only: 4 samples per atom; the full-task part of atoms mode is kept to 1 sample (already measured)
READ="atoms $BACKEND --build_dir $READ_BUILD --k_atom 4 --k_full 1 --max_new_tokens_full 2048 --max_model_len 4096"
mkdir -p "$R" "$E" "$CK"
DONE=$R/maze_next.done
failed=()

finish() {
    printf 'failed=%s\nend=%s\n' "${failed[*]:-none}" "$(date -u +%FT%TZ)" > "$DONE"
    [ "$DRY" = 1 ] && return
    nvidia-smi > /data/logs/nvidia_smi_end_maze_next.txt 2>&1 || true
    sync
    sleep "${HOLD:-300}"
    cd "$HOME"   # leave /data so it can be unmounted
    bash "$HOME/self_delete.sh"
}
trap finish EXIT
rm -f "$DONE"   # a marker left by an earlier run would tell the local watcher we are done

steps() { case $1 in maze_B600) echo "300 600" ;; *) echo "100 200 300" ;; esac; }
last()  { case $1 in maze_B600) echo 600 ;; *) echo 300 ;; esac; }
limit() { case $1 in maze_B600) echo 420m ;; *) echo 270m ;; esac; }

[ -f $READ_BUILD/manifest.json ] || J build_reading 15m $PY.build_atoms --tasks maze --n $NB --seed 2028 \
    --max_l2 0 --graphs reading --task_configs "$HARD" --out_dir $READ_BUILD || failed+=(build_reading)
[ -f $E/sft/reading/manifest.json ] || \
    J reading_sft 30m $PY.eval_atoms $READ --model $SFT --out_dir $E/sft/reading || failed+=(reading_sft)

for name in $RUNS; do
    run=${name}_seed1
    cfg=experiments/atom_decomposition/configs/$name.yaml
    L=$(last $name)
    if [ ! -f $CK/$run/checkpoint-$L/config.json ]; then
        J train_$run $(limit $name) python train_grpo.py $cfg --output_dir $CK/$run || failed+=(train_$run)
    fi
    mkdir -p $R/$run
    cp $CK/$run/trainer_state.json $R/$run/ 2>/dev/null || cp $CK/$run/checkpoint-$L/trainer_state.json $R/$run/ 2>/dev/null || true
    cp -r $CK/$run/generations $R/$run/ 2>/dev/null || true
    for n in $(steps $name); do
        ck=$CK/$run/checkpoint-$n
        [ "$DRY" = 1 ] || [ -d $ck ] || { failed+=(missing_$run-$n); continue; }
        [ -f $E/$run/hard_step$n/manifest.json ] || \
            J eval_${run}_hard_$n 30m $PY.eval_atoms $FULL --task_configs "$HARD" --model $ck \
                --out_dir $E/$run/hard_step$n || failed+=(eval_${run}_hard_$n)
    done
    ck=$CK/$run/checkpoint-$L
    if [ "$DRY" = 1 ] || [ -d $ck ]; then
        [ -f $E/$run/default_step$L/manifest.json ] || \
            J eval_${run}_default_$L 30m $PY.eval_atoms $FULL --model $ck --out_dir $E/$run/default_step$L \
            || failed+=(eval_${run}_default_$L)
        [ -f $E/$run/reading_step$L/manifest.json ] || \
            J reading_${run}_$L 30m $PY.eval_atoms $READ --model $ck --out_dir $E/$run/reading_step$L \
            || failed+=(reading_${run}_$L)
        [ -f $E/$run/atoms_step$L/manifest.json ] || \
            J atoms_${run}_$L 60m $PY.eval_atoms atoms --model $ck $BACKEND --build_dir $BFS_BUILD \
                --out_dir $E/$run/atoms_step$L --t01 --k_atom 4 --k_full 8 --max_new_tokens_full 2048 \
                --max_model_len 4096 || failed+=(atoms_${run}_$L)
    fi
done
echo "MAZE NEXT DONE (failed: ${failed[*]:-none})"
