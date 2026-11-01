#!/usr/bin/env bash
# Phase C runs on a 1x A100 80GB VM with the data volume mounted at /data (scripts/vm/setup.sh,
# BRANCH=main). Each STAGE is a separate paid run, started only after an explicit OK, and ends by
# deleting the VM (the volume keeps everything).
#
#   STAGE=pilot tmux new -d -s phase_c 'bash scripts/vm/phase_c.sh'
#   STAGE=gate  tmux new -d -s phase_c 'bash scripts/vm/phase_c.sh'
#
# pilot  full tasks only, 15 tasks x 200 instances (seed 2026): 4 samples at T=1.0 + 1 at T=0.1.
#        Then the difficulty rule (pilot_decide.py, PHASE_C.md §4.2) re-pilots out-of-band gate tasks
#        (up to 2 more rounds) and freezes their configs. Then a small atom pilot (25 instances per
#        gate task, seed 2026) to calibrate the power simulation and check the atom prompts.
# gate   the gate run: rescore the pilot and recompute the frozen configs (freeze), build seed 2027 x 500
#        per gate task with them, then every atom
#        (8 samples) and every full task (8 samples + 1 at T=0.1). fit_gates.py runs on the Mac.
#
# DRY=1 R=<dir> runs the whole thing with the fake backend and tiny sizes, no GPU and no self-delete
# (FAKE_P sets the fake model's accuracy; 0.1 exercises the difficulty ladder).
set -uo pipefail
STAGE=${STAGE:?set STAGE=pilot or STAGE=gate}
DRY=${DRY:-0}
if [ "$DRY" = 1 ]; then
    R=${R:?set R=<scratch dir> for a dry run}
    MODEL=none; BACKEND="--backend fake --fake_p_correct ${FAKE_P:-0.6}"; N_PILOT=12; N_ATOM_PILOT=3; N_GATE=20
    J() { local name=$1; shift 2; echo "[$name]"; "$@"; }
else
    source /data/env.sh
    R=/data/rlvr/phase_c
    MODEL=/data/rlvr/sft_qwen3_4b; BACKEND="--backend vllm"; N_PILOT=200; N_ATOM_PILOT=25; N_GATE=500
    J() { bash scripts/vm/run_job.sh "$@"; }
fi
PY="python -m experiments.atom_decomposition"
PILOT_TASKS=basic_arithmetic,countdown,leg_counting,gcd,simple_equations,syllogism,knights_knaves,circuit_logic,maze,mini_sudoku,shortest_path,tower_of_hanoi,jugs,caesar_cipher,cryptarithm
GATE_TASKS=basic_arithmetic,mini_sudoku,maze,gcd
COUNTDOWN='{"countdown": {"min_numbers": 4, "max_numbers": 5, "min_target": 50, "max_target": 200}}'
mkdir -p "$R"
failed=()

# Rescore the pilot rounds with the current scoring rules, rerun the difficulty rule, write frozen configs.
freeze() {
    local rounds
    for d in $(ls -d $R/pilot/r[0-9]* | sort); do $PY.rescore --results_dir $d >/dev/null || return 1; done
    rounds=$(ls -d $R/pilot/r[0-9]* | sort | paste -sd, -)
    $PY.pilot_decide --round_dirs $rounds --out $R/pilot/decision.json >/dev/null || return 1
    python - "$R/pilot/decision.json" "$R/frozen_task_configs.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
json.dump({t: s["config"] for t, s in d["state"].items()}, open(sys.argv[2], "w"), indent=2)
print("frozen:", json.dumps({t: s["config"] for t, s in d["state"].items()}))
PY
}

finish() {
    printf 'stage=%s\nfailed=%s\nend=%s\n' "$STAGE" "${failed[*]:-none}" "$(date -u +%FT%TZ)" > "$R/$STAGE.done"
    [ "$DRY" = 1 ] && return
    nvidia-smi > /data/logs/nvidia_smi_end_phase_c.txt 2>&1 || true
    sync
    cd "$HOME"   # leave /data so it can be unmounted
    bash "$HOME/self_delete.sh"
}
trap finish EXIT

if [ "$STAGE" = pilot ]; then
    # --- full-task pilot, round 1: all 15 tasks, the paper's eval settings ---------------
    J pilot_r1 3h $PY.eval_atoms full --model $MODEL $BACKEND --tasks $PILOT_TASKS --n $N_PILOT \
        --seed 2026 --k 4 --t01 --task_configs "$COUNTDOWN" --out_dir $R/pilot/r1 || { failed+=(pilot_r1); exit 1; }
    rounds=$R/pilot/r1
    for n in 2 3; do
        next=$($PY.pilot_decide --round_dirs $rounds --out $R/pilot/decision.json) || { failed+=(decide); exit 1; }
        [ "$next" = "{}" ] && break
        tasks=$(python -c "import json,sys; print(','.join(json.loads(sys.argv[1])))" "$next")
        J pilot_r$n 1h $PY.eval_atoms full --model $MODEL $BACKEND --tasks $tasks --n $N_PILOT \
            --seed 2026 --k 4 --task_configs "$next" --out_dir $R/pilot/r$n || { failed+=(pilot_r$n); break; }
        rounds=$rounds,$R/pilot/r$n
    done
    # Freeze: tasks still pending after the last round keep their last config (flagged in decision.json).
    freeze || { failed+=(freeze); exit 1; }

    # --- atom pilot: calibrate the power simulation, check atom prompts on the real model --
    frozen=$(cat $R/frozen_task_configs.json)
    J atom_pilot_build 10m $PY.build_atoms --tasks $GATE_TASKS --n $N_ATOM_PILOT --seed 2026 \
        --task_configs "$frozen" --out_dir $R/pilot/atoms_build || { failed+=(atom_pilot_build); exit 1; }
    J atom_pilot 1h $PY.eval_atoms atoms --model $MODEL $BACKEND --build_dir $R/pilot/atoms_build \
        --out_dir $R/pilot/atoms_eval || failed+=(atom_pilot)

elif [ "$STAGE" = gate ]; then
    [ -d $R/pilot/r1 ] || { echo "no pilot results: run STAGE=pilot first"; failed+=(no_pilot); exit 1; }
    # Recompute the freeze from the saved pilot responses, so a scoring change since the pilot applies.
    freeze || { failed+=(freeze); exit 1; }
    frozen=$(cat $R/frozen_task_configs.json)
    J gate_build 20m $PY.build_atoms --tasks $GATE_TASKS --n $N_GATE --seed 2027 \
        --task_configs "$frozen" --out_dir $R/atoms_seed2027 || { failed+=(gate_build); exit 1; }
    J gate_eval 9h $PY.eval_atoms atoms --model $MODEL $BACKEND --build_dir $R/atoms_seed2027 \
        --out_dir $R/eval_seed2027 --t01 || failed+=(gate_eval)
else
    echo "unknown STAGE=$STAGE"; failed+=(bad_stage); exit 1
fi
echo "PHASE C $STAGE DONE (failed: ${failed[*]:-none})"
