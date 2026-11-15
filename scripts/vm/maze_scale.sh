#!/usr/bin/env bash
# Maze scale-up, on a 1x A100 80GB VM with the data volume at /data (scripts/vm/setup.sh) and local disk at
# /ephemeral. Ends by deleting the VM.
#
#   tmux new -d -s maze 'bash scripts/vm/maze_scale.sh'
#
# 1. Model-size probe (eval only): raw Qwen/Qwen3-8B and, as the same-family control, raw Qwen/Qwen3-4B (neither
#    has our SFT). Default and hard mazes (200 prompts, eval seed 9999, 4 samples at T=1.0, 4096 tokens, since
#    raw Qwen3 thinks at length) and the grid-reading atoms (maze_next's reading build, 2 samples per atom).
#    Models download to /ephemeral/hf (the volume has ~4 GB free). run_job.sh re-sources /data/env.sh, which
#    sets HF_HOME=/data/hf, so the probe commands set HF_HOME themselves through env.
# 2. Arms (experiments/atom_decomposition/configs/), seed 1, from the 4B SFT start:
#      maze_Prefix     300 steps  hard + hard mazes with the first k moves of a shortest path in the prompt
#      maze_Prog       300 steps  hard only, plus a 0.5-weight legal-progress reward on the reasoning
#      maze_B1000     1000 steps  hard + easier whole mazes
#      maze_Cread1000 1000 steps  hard + grid-reading atoms of hard-size mazes
#    Evals as in maze_next.sh: hard at 100/200/300 (300-step arms) or 300/600/1000 (8 samples at T=1.0 + 1 at
#    T=0.1, 2048 tokens); at the last step also default mazes, reading atoms and the BFS atom build.
# Checkpoints stay on /ephemeral; trainer_state.json and generation logs go to /data/rlvr/maze_scale/<run>/.
# Waits HOLD seconds before deleting the VM.
#
# DRY=1 R=<dir> prints the training commands and runs the evals with the fake backend.
set -uo pipefail
DRY=${DRY:-0}
RUNS=${RUNS:-"maze_Prefix maze_Prog maze_B1000 maze_Cread1000"}
PROBE_MODELS=${PROBE_MODELS:-"Qwen/Qwen3-8B Qwen/Qwen3-4B"}
HARD='{"maze": {"min_dist": 10, "max_dist": 15, "min_grid_size": 10, "max_grid_size": 14}}'
PY="python -m experiments.atom_decomposition"
PYP="env HF_HOME=/ephemeral/hf $PY"   # probe jobs: raw models download to local disk, not the volume
if [ "$DRY" = 1 ]; then
    R=${R:?set R=<scratch dir> for a dry run}; CK=$R/ephemeral; SFT=none
    BACKEND="--backend fake --fake_p_correct 0.3"; N=10; NB=3
    BFS_BUILD=$R/hard_build; READ_BUILD=$R/reading_build
    [ -f $BFS_BUILD/manifest.json ] || $PY.build_atoms --tasks maze --n 3 --seed 2028 --max_l2 12 \
        --task_configs "$HARD" --out_dir $BFS_BUILD > /dev/null
    [ -f $READ_BUILD/manifest.json ] || $PY.build_atoms --tasks maze --n $NB --seed 2028 --max_l2 0 \
        --graphs reading --task_configs "$HARD" --out_dir $READ_BUILD > /dev/null
    J() { local name=$1; shift 2; echo "[$name] $*"; case "$1" in python) [ "$2" = train_grpo.py ] || "$@";; *) "$@";; esac; }
else
    source /data/env.sh
    R=/data/rlvr/maze_scale; CK=/ephemeral/maze; SFT=/data/rlvr/sft_qwen3_4b
    sudo mkdir -p $CK /ephemeral/hf && sudo chown "$USER" $CK /ephemeral/hf
    BACKEND="--backend vllm"; N=200
    BFS_BUILD=/data/rlvr/maze_probe/hard/build
    READ_BUILD=/data/rlvr/maze_next/reading_build
    J() { bash scripts/vm/run_job.sh "$@"; }
fi
E=$R/eval
FULL="full --tasks maze --n $N --seed 9999 --k 8 --t01 --max_new_tokens_full 2048 --max_model_len 4096 $BACKEND"
READ="atoms $BACKEND --build_dir $READ_BUILD --k_atom 4 --k_full 1 --max_new_tokens_full 2048 --max_model_len 4096"
PFULL="full --tasks maze --n $N --seed 9999 --k 4 --max_new_tokens_full 4096 --max_model_len 6144 $BACKEND"
PREAD="atoms $BACKEND --build_dir $READ_BUILD --k_atom 2 --k_full 1 --max_new_tokens_atom 2048 --max_new_tokens_full 4096 --max_model_len 6144"
mkdir -p "$R" "$E" "$CK"
DONE=$R/maze_scale.done
failed=()

finish() {
    printf 'failed=%s\nend=%s\n' "${failed[*]:-none}" "$(date -u +%FT%TZ)" > "$DONE"
    [ "$DRY" = 1 ] && return
    nvidia-smi > /data/logs/nvidia_smi_end_maze_scale.txt 2>&1 || true
    sync
    sleep "${HOLD:-300}"
    cd "$HOME"   # leave /data so it can be unmounted
    bash "$HOME/self_delete.sh"
}
trap finish EXIT
rm -f "$DONE"   # a marker left by an earlier run would tell the local watcher we are done

# --- 1. model-size probe ---
for m in $PROBE_MODELS; do
    tag=probe_$(basename $m | tr 'A-Z-' 'a-z_')     # probe_qwen3_8b
    model=$m; [ "$DRY" = 1 ] && model=none
    [ -f $E/$tag/default/manifest.json ] || J ${tag}_default 60m $PYP.eval_atoms $PFULL --model $model \
        --out_dir $E/$tag/default || failed+=(${tag}_default)
    [ -f $E/$tag/hard/manifest.json ] || J ${tag}_hard 75m $PYP.eval_atoms $PFULL --task_configs "$HARD" \
        --model $model --out_dir $E/$tag/hard || failed+=(${tag}_hard)
    [ -f $E/$tag/reading/manifest.json ] || J ${tag}_reading 45m $PYP.eval_atoms $PREAD --model $model \
        --out_dir $E/$tag/reading || failed+=(${tag}_reading)
done
[ "$DRY" = 1 ] || sudo rm -rf /ephemeral/hf   # free the disk before training

# --- 2. arms ---
steps() { case $1 in *1000) echo "300 600 1000" ;; *) echo "100 200 300" ;; esac; }
last()  { case $1 in *1000) echo 1000 ;; *) echo 300 ;; esac; }
limit() { case $1 in *1000) echo 380m ;; *) echo 200m ;; esac; }
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
    # free the disk: keep only the checkpoints the evals used
    [ "$DRY" = 1 ] || find $CK/$run -maxdepth 1 -name 'checkpoint-*' \
        $(for n in $(steps $name); do printf -- "! -name checkpoint-%s " $n; done) -exec rm -rf {} +
done
echo "MAZE SCALE DONE (failed: ${failed[*]:-none})"
