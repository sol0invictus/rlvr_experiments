"""eval_atoms with a fake model: every sample is scored right, reruns resume, results feed fit_gates.

Run from the repo root:  python -m pytest experiments/atom_decomposition/tests -q
(needs torch + transformers installed, since evaluate_replication imports them)
"""

import json

from experiments.atom_decomposition.build_atoms import build
from experiments.atom_decomposition.eval_atoms import (DEFAULT_SYSTEM_PROMPT, FakeBackend, run_atoms,
                                                      run_full)
from experiments.atom_decomposition.fit_gates import load_build, load_results


def _fake(build_dir, rows):
    gold = {r["question"]: r["answer"] for r in rows}
    for l in open(f"{build_dir}/atoms.jsonl"):
        a = json.loads(l)
        gold[a["question"]] = a["answer"]
    return FakeBackend({f"[system] {DEFAULT_SYSTEM_PROMPT}\n[user] {q}": g for q, g in gold.items()}, 0.5, 0)


def test_fake_eval_scores_and_resumes(tmp_path):
    build_dir, out = str(tmp_path / "build"), str(tmp_path / "eval")
    build(["basic_arithmetic", "mini_sudoku", "maze", "gcd"], 15, 3, 40, build_dir)
    rows = [json.loads(l) for l in open(f"{build_dir}/instances.jsonl")]
    backend = _fake(build_dir, rows)
    import os
    os.makedirs(out)
    n = run_atoms(backend, build_dir, 4, 64, 1.0, out, chunk=500)
    run_full(backend, rows, [(1.0, 4), (0.1, 1)], 64, out, chunk=500)

    for name in ("atom_results.jsonl", "full_results.jsonl"):
        for l in open(f"{out}/{name}"):
            r = json.loads(l)
            for text, c in zip(r["responses"], r["correct"]):
                assert c == int("wrong" not in text), (name, r.get("check", r["task"]), text, r["answer"])

    sizes = {f: sum(1 for _ in open(f"{out}/{f}")) for f in ("atom_results.jsonl", "full_results.jsonl")}
    assert sizes["atom_results.jsonl"] == n and sizes["full_results.jsonl"] == 2 * len(rows)
    run_atoms(backend, build_dir, 4, 64, 1.0, out, chunk=500)          # rerun: nothing new
    run_full(backend, rows, [(1.0, 4), (0.1, 1)], 64, out, chunk=500)
    assert sizes == {f: sum(1 for _ in open(f"{out}/{f}")) for f in sizes}

    # fit_gates reads the T = 1.0 rows; the T = 0.1 rows are a separate accuracy number
    atom_obs, full_obs = load_results(out)
    inst, atoms = load_build(build_dir)
    assert set(full_obs) == {(r["task"], r["idx"]) for r in inst}
    assert all(k == 4 for _, k in full_obs.values())
    assert all(any(a["key"] in atom_obs for a in v) for v in atoms.values())
