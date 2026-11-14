#!/usr/bin/env bash
# Task probe for the next atom study, on a 1x A100 80GB VM with the data volume at /data
# (scripts/vm/setup.sh). Eval only, no training. Ends by deleting the VM.
#
#   tmux new -d -s probe 'bash scripts/vm/task_probe.sh'
#
# The SFT start (the RL starting point) on candidate tasks, 200 prompts each, 8 samples at T=1.0 plus
# one at T=0.1, up to 2048 new tokens: accuracy and trace length, to find a task where the model is
# mid-range and the traces are long for reasons other than raw computation.
#   default: knights_knaves (2 people), maze, shortest_path, zebra_puzzles (4x4), countdown
#   variant: knights_knaves with 4 people, zebra_puzzles 3x3
#
# DRY=1 R=<dir> runs the evals with the fake backend.
set -uo pipefail
DRY=${DRY:-0}
if [ "$DRY" = 1 ]; then
    R=${R:?set R=<scratch dir> for a dry run}
    SFT=none; BACKEND="--backend fake --fake_p_correct 0.6"; N=10
    J() { local name=$1; shift 2; echo "[$name] $*"; "$@"; }
else
    source /data/env.sh
    R=/data/rlvr/probe
    SFT=/data/rlvr/sft_qwen3_4b; BACKEND="--backend vllm"; N=200
    J() { bash scripts/vm/run_job.sh "$@"; }
fi
PY="python -m experiments.atom_decomposition"
ARGS="full --n $N --seed 9999 --k 8 --t01 --max_new_tokens_full 2048 --max_model_len 4096 $BACKEND --model $SFT"
mkdir -p "$R"
DONE=$R/probe.done
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

[ -f $R/default/manifest.json ] || \
    J probe_default 60m $PY.eval_atoms $ARGS --tasks knights_knaves,maze,shortest_path,zebra_puzzles,countdown \
        --out_dir $R/default || failed+=(probe_default)
[ -f $R/variant/manifest.json ] || \
    J probe_variant 45m $PY.eval_atoms $ARGS --tasks knights_knaves,zebra_puzzles \
        --task_configs '{"knights_knaves": {"n_people": 4}, "zebra_puzzles": {"num_people": 3, "num_characteristics": 3}}' \
        --out_dir $R/variant || failed+=(probe_variant)
echo "PROBE DONE (failed: ${failed[*]:-none})"
