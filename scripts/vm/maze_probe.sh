#!/usr/bin/env bash
# Maze difficulty probe, on a 1-GPU VM with the data volume at /data (scripts/vm/setup.sh). Eval only.
# Ends by deleting the VM.
#
#   tmux new -d -s probe 'bash scripts/vm/maze_probe.sh'
#
# Where does full-task RL have signal on maze, and how good are the atoms there? For four difficulty
# levels, 150 instances each (build seed 2028): the SFT start's full-task accuracy (8 samples at T=1.0 +
# 1 at T=0.1, 2048 tokens) and its atom accuracy (L1 locate start/goal, L2 one BFS layer; 4 samples each,
# first 12 layers per instance).
#   easy     dist 1-4,   grid 5-7
#   default  dist 5-10,  grid 5-10   (reasoning-gym default)
#   hard     dist 10-15, grid 10-14
#   vhard    dist 16-25, grid 14-20
#
# DRY=1 R=<dir> runs everything with the fake backend.
set -uo pipefail
DRY=${DRY:-0}
if [ "$DRY" = 1 ]; then
    R=${R:?set R=<scratch dir> for a dry run}
    SFT=none; BACKEND="--backend fake --fake_p_correct 0.6"; N=5
    J() { local name=$1; shift 2; echo "[$name] $*"; "$@"; }
else
    source /data/env.sh
    R=/data/rlvr/maze_probe
    SFT=/data/rlvr/sft_qwen3_4b; BACKEND="--backend vllm"; N=150
    J() { bash scripts/vm/run_job.sh "$@"; }
fi
PY="python -m experiments.atom_decomposition"
mkdir -p "$R"
DONE=$R/maze_probe.done
failed=()

finish() {
    printf 'failed=%s\nend=%s\n' "${failed[*]:-none}" "$(date -u +%FT%TZ)" > "$DONE"
    [ "$DRY" = 1 ] && return
    sync
    sleep "${HOLD:-300}"
    cd "$HOME"   # leave /data so it can be unmounted
    bash "$HOME/self_delete.sh"
}
trap finish EXIT
rm -f "$DONE"

cfg() {
    case $1 in
        easy)    echo '{"maze": {"min_dist": 1, "max_dist": 4, "min_grid_size": 5, "max_grid_size": 7}}' ;;
        default) echo '{"maze": {}}' ;;
        hard)    echo '{"maze": {"min_dist": 10, "max_dist": 15, "min_grid_size": 10, "max_grid_size": 14}}' ;;
        vhard)   echo '{"maze": {"min_dist": 16, "max_dist": 25, "min_grid_size": 14, "max_grid_size": 20}}' ;;
    esac
}
for lvl in easy default hard vhard; do
    b=$R/$lvl/build; e=$R/$lvl/eval
    [ -f $b/manifest.json ] || J maze_build_$lvl 15m $PY.build_atoms --tasks maze --n $N --seed 2028 --max_l2 12 \
        --task_configs "$(cfg $lvl)" --out_dir $b || { failed+=(build_$lvl); continue; }
    [ -f $e/manifest.json ] || J maze_eval_$lvl 60m $PY.eval_atoms atoms --model $SFT $BACKEND --build_dir $b \
        --out_dir $e --t01 --k_atom 4 --k_full 8 --max_new_tokens_full 2048 --max_model_len 4096 \
        || failed+=(eval_$lvl)
done
echo "MAZE PROBE DONE (failed: ${failed[*]:-none})"
