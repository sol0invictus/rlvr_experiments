"""The pilot difficulty rule: freeze in band, climb the ladder out of band, stop when it runs out
or the band is crossed."""

import json

from experiments.atom_decomposition.pilot_decide import decide


def _round(tmp_path, name, acc):
    d = tmp_path / name
    d.mkdir()
    with open(d / "full_results.jsonl", "w") as f:
        for task, a in acc.items():
            k = round(a * 100)
            f.write(json.dumps({"task": task, "idx": 0, "temperature": 1.0,
                                "correct": [1] * k + [0] * (100 - k)}) + "\n")
            f.write(json.dumps({"task": task, "idx": 0, "temperature": 0.1, "correct": [1]}) + "\n")
    return str(d)


def test_ladder(tmp_path):
    r1 = _round(tmp_path, "r1", {"basic_arithmetic": 0.6, "mini_sudoku": 0.1, "maze": 0.02, "gcd": 0.9,
                                 "countdown": 0.3})
    state, nxt = decide([r1])
    assert state["basic_arithmetic"]["frozen"] and state["basic_arithmetic"]["config"] == {}
    assert nxt == {"mini_sudoku": {"min_empty": 4}, "maze": {"min_dist": 1}, "gcd": {"max_value": 10000}}

    r2 = _round(tmp_path, "r2", {"mini_sudoku": 0.3, "maze": 0.1, "gcd": 0.1})
    state, nxt = decide([r1, r2])
    assert state["mini_sudoku"]["frozen"] and state["mini_sudoku"]["config"] == {"min_empty": 4}
    assert state["gcd"]["frozen"] and "crossed" in state["gcd"]["reason"]
    assert nxt == {"maze": {"min_dist": 1, "max_grid_size": 7}}

    r3 = _round(tmp_path, "r3", {"maze": 0.15})
    state, nxt = decide([r1, r2, r3])
    assert nxt == {} and state["maze"]["frozen"] and "no step left" in state["maze"]["reason"]
    assert state["maze"]["config"] == {"min_dist": 1, "max_grid_size": 7}
