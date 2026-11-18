#!/usr/bin/env bash
# Queue 2: runs after scripts/vm/planted.sh on the same VM (~/queue.sh fetches the branch and calls this, then
# deletes the VM). The rest of the brainstorm's plan (sections 4-7, 11, 12), from the 4B SFT start:
#   raw         raw Qwen/Qwen3-4B on arithmetic, eval only: standard eval + ladder at the SFT model's 2048-token
#               budget, standard eval at 4096 (the notes had no raw-model arithmetic number)
#   atoms_ctrl  atoms closing control: half L2-multiplication atoms, no token cap, 600 steps
#   maze_ProgA  Prog repair: progress reward only when an answer is given (hard mazes, 300 steps)
#   zebra_plain zebra 4x4, correctness only, under its 2048-token cap (300 steps)
#   arith_hard  planted shortcut on big-product puzzles only, q = 0 and 0.3 (300 steps)
# Order: raw, seed 1 of atoms_ctrl / maze_ProgA / zebra_plain / arith_hard_q00 / arith_hard_q03, then seed 2 of
# atoms_ctrl and maze_ProgA. Arithmetic evals go to $R/eval (planted_report layout), maze and zebra to $R/eval_other.
# Checkpoints stay on /ephemeral and are deleted after each run's evals.
#
# DRY=1 R=<dir> prints the training commands and runs the evals with the fake backend.
set -uo pipefail
DRY=${DRY:-0}
RUNS=${RUNS:-"atoms_ctrl_seed1 maze_ProgA_seed1 zebra_plain_seed1 arith_hard_q00_seed1 arith_hard_q03_seed1 atoms_ctrl_seed2 maze_ProgA_seed2"}
PY="python -m experiments.atom_decomposition"
LEVELS='{"d1": {"min_digits": 1, "max_digits": 1}, "d2": {"min_digits": 2, "max_digits": 2}, "d3": {"min_digits": 3, "max_digits": 3}, "d4": {"min_digits": 4, "max_digits": 4}, "d5": {"min_digits": 5, "max_digits": 5}, "d6": {"min_digits": 6, "max_digits": 6}}'
HARD='{"maze": {"min_dist": 10, "max_dist": 15, "min_grid_size": 10, "max_grid_size": 14}}'
if [ "$DRY" = 1 ]; then
    R=${R:?set R=<scratch dir> for a dry run}; CK=$R/ephemeral; SFT=none; RAW=none
    BACKEND="--backend fake --fake_p_correct 0.5"; N=20; NL=5; NM=10; NZ=10
    J() { local name=$1; shift 2; echo "[$name] $*"; case "$1" in python) [ "$2" = train_grpo.py ] || "$@";; env) "$@";; *) "$@";; esac; }
else
    source /data/env.sh
    R=/data/rlvr/queue2; CK=/ephemeral/queue2; SFT=/data/rlvr/sft_qwen3_4b; RAW=Qwen/Qwen3-4B
    sudo mkdir -p $CK /ephemeral/hf && sudo chown "$USER" $CK /ephemeral/hf
    BACKEND="--backend vllm"; N=500; NL=100; NM=200; NZ=200
    J() { bash scripts/vm/run_job.sh "$@"; }
fi
E=$R/eval; EO=$R/eval_other
STD="full --tasks basic_arithmetic --n $N --seed 9999 --k 8 $BACKEND"
STD_LAST="full --tasks basic_arithmetic --n $N --seed 9999 --k 16 --t01 $BACKEND"
LADDER="full --tasks basic_arithmetic --n $NL --seed 4242 --k 8 --levels"
MAZE="full --tasks maze --n $NM --seed 9999 --k 8 --t01 --max_new_tokens_full 2048 --max_model_len 4096 $BACKEND"
ZEBRA="full --tasks zebra_puzzles --n $NZ --seed 9999 --k 8 --t01 --max_new_tokens_full 2048 --max_model_len 4096 $BACKEND"
mkdir -p "$R" "$E" "$EO" "$CK"
DONE=$R/queue2.done
failed=()
trap 'printf "failed=%s\nend=%s\n" "${failed[*]:-none}" "$(date -u +%FT%TZ)" > "$DONE"' EXIT
rm -f "$DONE"
ev() {   # ev <name> <limit> <out_dir> <args...>: run an eval unless its manifest exists
    local name=$1 limit=$2 out=$3; shift 3
    [ -f $out/manifest.json ] || J $name $limit "$@" --out_dir $out || failed+=($name)
}

# --- raw Qwen3-4B (downloads to local disk; run_job.sh re-sources env.sh, so HF_HOME is set in the command) ---
PYR="env HF_HOME=/ephemeral/hf $PY"
ev raw_std 40m $E/raw_qwen3_4b/step000/std $PYR.eval_atoms $STD_LAST --model $RAW
ev raw_ladder 60m $E/raw_qwen3_4b/step000/ladder $PYR.eval_atoms $LADDER "$LEVELS" $BACKEND --model $RAW
ev raw_std4096 60m $E/raw_qwen3_4b_4096/step000/std $PYR.eval_atoms $STD --max_new_tokens_full 4096 \
    --max_model_len 6144 --model $RAW
[ "$DRY" = 1 ] || sudo rm -rf /ephemeral/hf
# SFT start for the maze-repair and zebra runs (same eval settings as their checkpoints)
ev sft_zebra 60m $EO/sft/zebra $PY.eval_atoms $ZEBRA --model $SFT

for run in $RUNS; do
    name=${run%_seed*}; seed=${run#*_seed}
    cfg=experiments/atom_decomposition/configs/$name.yaml
    case $name in
        atoms_ctrl) steps="150 300 450 600"; limit=5h ;;
        zebra_plain) steps="100 200 300"; limit=5h ;;
        *) steps="100 200 300"; limit=3h ;;
    esac
    last=${steps##* }
    if [ ! -f $CK/$run/checkpoint-$last/config.json ]; then
        J train_$run $limit python train_grpo.py $cfg --seed $seed --output_dir $CK/$run || failed+=(train_$run)
    fi
    mkdir -p $R/$run
    cp $CK/$run/trainer_state.json $CK/$run/effective_config.json $R/$run/ 2>/dev/null || true
    cp -r $CK/$run/generations $R/$run/ 2>/dev/null || true
    for n in $steps; do
        ck=$CK/$run/checkpoint-$n
        [ "$DRY" = 1 ] || [ -d $ck ] || { failed+=(missing_$run-$n); continue; }
        case $name in
            maze_ProgA)
                ev eval_${run}_hard_$n 30m $EO/$run/hard_step$n $PY.eval_atoms $MAZE --task_configs "$HARD" --model $ck
                [ $n = $last ] && ev eval_${run}_default_$n 30m $EO/$run/default_step$n $PY.eval_atoms $MAZE --model $ck ;;
            zebra_plain)
                ev eval_${run}_zebra_$n 60m $EO/$run/zebra_step$n $PY.eval_atoms $ZEBRA --model $ck ;;
            *)
                args=$STD; [ $n = $last ] && args=$STD_LAST
                ev eval_${run}_std_$n 40m $E/$run/step$n/std $PY.eval_atoms $args --model $ck
                [ $n = $last ] && ev eval_${run}_ladder_$n 40m $E/$run/step$n/ladder $PY.eval_atoms $LADDER "$LEVELS" \
                    $BACKEND --model $ck ;;
        esac
    done
    [ "$DRY" = 1 ] || rm -rf $CK/$run/checkpoint-*
    $PY.planted_report --eval_root $E --out $R/report.json > $R/report.txt 2>&1 || true
done
$PY.planted_report --eval_root $E --out $R/report.json | tee $R/report.txt || failed+=(report)
echo "QUEUE2 DONE (failed: ${failed[*]:-none})"
