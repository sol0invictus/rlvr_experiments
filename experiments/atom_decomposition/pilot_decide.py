"""
Apply the pilot's difficulty rule (PHASE_C.md §4.2) and freeze the gate tasks' reasoning-gym configs.

After each pilot round: read the round's T = 1.0 full-task accuracy per gate task. A task inside
[20%, 80%] is frozen at its current config. A task outside moves to the next pre-set config in its
ladder (toward easier settings below 20%, harder above 80%) and is re-piloted; a task with no step
left is frozen where it is and flagged. No other changes are allowed.

Usage (from the repo root):
    python -m experiments.atom_decomposition.pilot_decide --round_dirs r1[,r2,...] --out state.json
Prints the tasks and configs to pilot next (empty when everything is frozen).
"""

import argparse
import json
import os

BAND = (0.20, 0.80)
GATE_TASKS = ("basic_arithmetic", "mini_sudoku", "maze", "gcd")
# Pilot settings for the paper's baselines: rg defaults except countdown, as in evaluate_transfer.py.
PILOT_CONFIGS = {"countdown": {"min_numbers": 4, "max_numbers": 5, "min_target": 50, "max_target": 200}}
# Config ladders per gate task, tried in order. maze and mini_sudoku are the ones fixed in §4.2;
# basic_arithmetic and gcd were added with gcd (2026-11-01).
LADDERS = {
    "maze": {"low": [{"min_dist": 1}, {"min_dist": 1, "max_grid_size": 7}], "high": []},
    "mini_sudoku": {"low": [{"min_empty": 4}], "high": []},
    "basic_arithmetic": {"low": [{"max_digits": 2}], "high": []},
    "gcd": {"low": [{"max_value": 100}], "high": [{"max_value": 10000}]},
}


def round_accuracy(round_dir, temperature=1.0):
    acc = {}
    for l in open(os.path.join(round_dir, "full_results.jsonl")):
        r = json.loads(l)
        if r["temperature"] == temperature:
            c = acc.setdefault(r["task"], [0, 0])
            c[0] += sum(r["correct"])
            c[1] += len(r["correct"])
    return {t: s / n for t, (s, n) in acc.items()}


def decide(round_dirs):
    """Walk the rounds in order. Returns (state per gate task, next round {task: config})."""
    state = {t: {"config": PILOT_CONFIGS.get(t, {}), "step": None, "history": []} for t in GATE_TASKS}
    pending = dict.fromkeys(GATE_TASKS)
    for rd in round_dirs:
        acc = round_accuracy(rd)
        for t in list(pending):
            if t not in acc:
                continue
            s = state[t]
            s["history"].append({"round": os.path.basename(rd.rstrip("/")), "config": s["config"],
                                 "accuracy": acc[t]})
            side = "low" if acc[t] < BAND[0] else "high" if acc[t] > BAND[1] else None
            if side is None:
                s["frozen"], s["reason"] = True, "in band"
                del pending[t]
                continue
            if s.get("side") and s["side"] != side:   # crossed the band: keep the last step
                s["frozen"], s["reason"] = True, f"crossed the band ({s['side']} -> {side}); frozen as is"
                del pending[t]
                continue
            step = 0 if s["step"] is None else s["step"] + 1
            ladder = LADDERS[t][side]
            if step >= len(ladder):
                s["frozen"], s["reason"] = True, f"accuracy {side} and no step left; frozen as is"
                del pending[t]
                continue
            s["config"], s["step"], s["side"] = ladder[step], step, side
    next_round = {t: state[t]["config"] for t in pending}
    for t in pending:
        state[t]["frozen"] = False
    return state, next_round


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--round_dirs", required=True, help="comma-separated pilot round dirs, in order")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    state, next_round = decide(a.round_dirs.split(","))
    frozen = {t: s["config"] for t, s in state.items() if s["frozen"]}
    with open(a.out, "w") as f:
        json.dump({"band": BAND, "state": state, "next_round": next_round,
                   "frozen_task_configs": frozen if not next_round else None}, f, indent=2)
    print(json.dumps(next_round))


if __name__ == "__main__":
    main()
