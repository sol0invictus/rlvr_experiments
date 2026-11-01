"""The gate test recovers planted truths, refuses difficulty-only data, and reads result files.

Run from the repo root:  python -m pytest experiments/atom_decomposition/tests -q
"""

import json

import numpy as np
import pytest

from experiments.atom_decomposition.build_atoms import build
from experiments.atom_decomposition.fit_gates import (compare, evaluate_task, load_build,
                                                     load_results, simulate)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    d = tmp_path_factory.mktemp("atoms")
    build(["basic_arithmetic", "mini_sudoku", "maze", "gcd"], 300, 11, 40, str(d))
    return str(d), *load_build(str(d))


def _run(built, task, truth, w, base_acc=0.4, seed=1):
    _, inst, atoms = built
    atom_obs, full_obs = simulate(task, inst, atoms, truth, w, base_acc, np.random.default_rng(seed))
    return evaluate_task(task, inst, atoms, atom_obs, full_obs, n_repeats=2, n_boot=500, seed=seed)


def test_strong_atom_truth_passes_both_gates(built):
    r = _run(built, "mini_sudoku", "propagation", w=0.9)
    assert r["evaluable"]
    assert r["gate_b_verdict"] == "pass"
    assert r["gate_a_verdict"] == "propagation"


def test_planted_alternative_is_picked(built):
    r = _run(built, "mini_sudoku", "backtracking", w=0.9)
    assert r["gate_a_verdict"] == "backtracking"
    assert r["gate_b"]["backtracking"]["ci_level"] == 0.975   # Bonferroni over two graphs


@pytest.mark.parametrize("task", ["basic_arithmetic", "mini_sudoku", "maze", "gcd"])
def test_difficulty_only_does_not_pass(built, task):
    r = _run(built, task, "difficulty", w=0.0)
    assert r["gate_b_verdict"] == "fail"
    if "gate_a_verdict" in r:
        assert r["gate_a_verdict"] == "tie"


def test_floor_is_not_evaluable(built):
    r = _run(built, "maze", "bfs", w=0.9, base_acc=0.01)
    assert not r["evaluable"] and r["gate_b_verdict"] == "not evaluable"


def test_compare_identical_losses():
    l = np.random.default_rng(0).random(200)
    c = compare(l, l.copy(), 200, np.random.default_rng(0))
    assert c["delta"] == 0 and c["ci"] == [0.0, 0.0] and c["ci_level"] == 0.95


def test_result_files_round_trip(built, tmp_path):
    build_dir, inst, atoms = built
    atom_obs, full_obs = simulate("maze", inst, atoms, "bfs", 0.5, 0.4, np.random.default_rng(2))
    with open(tmp_path / "atom_results.jsonl", "w") as f:
        for key, (s, k, _) in atom_obs.items():
            f.write(json.dumps({"key": key, "correct": [1] * s + [0] * (k - s), "has_answer_tag": [1] * k}) + "\n")
    with open(tmp_path / "full_results.jsonl", "w") as f:
        for (task, idx), (s, k) in full_obs.items():
            f.write(json.dumps({"task": task, "idx": idx, "correct": [1] * s + [0] * (k - s),
                                "has_answer_tag": [1] * k}) + "\n")
    a2, f2 = load_results(str(tmp_path))
    assert a2 == atom_obs and f2 == full_obs
