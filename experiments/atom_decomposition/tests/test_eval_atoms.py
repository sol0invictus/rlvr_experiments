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


def test_post_think_extraction():
    import reasoning_gym as rg
    from experiments.atom_decomposition.eval_atoms import FullScorer, extract_post_think, score_atom
    entry = rg.create_dataset("mini_sudoku", size=1, seed=5)[0]
    grid = entry["answer"]
    text = "<think>work it out</think>\n\n" + "\n".join(row + "  " for row in grid.splitlines())
    assert extract_post_think(text).split() == grid.split()
    assert extract_post_think("<think>x</think><answer>7</answer> trailing") == "7"      # tags win
    assert extract_post_think("<think>\nnever closed\n5") == "5"      # no </think>: the paper's last line
    s = FullScorer()("mini_sudoku", entry, text)
    assert s["correct"] == 1 and s["correct_paper"] == 0 and s["correct_strict"] == 0
    assert score_atom("int", "5", "<think>2 + 3</think>\nThe answer is 5.") == \
        {"correct": 1, "correct_paper": 1, "correct_strict": 0}


def test_rescore_rewrites_fields(tmp_path):
    from experiments.atom_decomposition.rescore import rescore
    build_dir, out = str(tmp_path / "build"), tmp_path / "eval"
    build(["basic_arithmetic", "mini_sudoku"], 6, 4, 40, build_dir)
    rows = [json.loads(l) for l in open(f"{build_dir}/instances.jsonl")]
    out.mkdir()
    # tagless answers after </think>: the old rule misses sudoku grids, the new one reads them
    with open(out / "full_results.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps({"task": r["task"], "idx": r["idx"], "temperature": 1.0, "answer": r["answer"],
                                "correct": [0], "responses": [f"<think>t</think>\n{r['answer']}"],
                                "has_answer_tag": [0], "n_tokens": [5]}) + "\n")
    json.dump({"args": {"mode": "atoms", "build_dir": build_dir}}, open(out / "manifest.json", "w"))
    assert rescore(str(out)) == (len(rows), 0)
    new = [json.loads(l) for l in open(out / "full_results.jsonl")]
    assert all(r["correct"] == [1] for r in new)
    assert {r["task"]: r["correct_paper"][0] for r in new if r["task"] == "mini_sudoku"} == {"mini_sudoku": 0}
    assert (out / "full_results.jsonl.orig").exists()


def test_degenerate_responses_do_not_crash_scoring():
    """Gate attempt 1 died on a response of ~1000 nested parentheses (RecursionError in parse_strict)."""
    import reasoning_gym as rg
    from experiments.atom_decomposition.eval_atoms import FullScorer, score_atom
    from experiments.atom_decomposition.tasks.arithmetic import parse_strict
    junk = ["(" * 5000, "-" * 5000, "(-" * 3000, "9" * 5000, "(" * 5000 + "1" + ")" * 5000]
    for j in junk:
        assert parse_strict(j) is None
        text = f"<think>x</think><answer>{j}</answer>"
        for check, gold in (("paren_tree", "(1 + 2)"), ("int", "5"), ("int_set", "1, 2"),
                            ("cell", "(1, 1)"), ("cell_set", "(1, 1)")):
            assert score_atom(check, gold, text)["correct"] == 0
        for task in ("basic_arithmetic", "mini_sudoku", "gcd", "maze", "tower_of_hanoi"):
            entry = rg.create_dataset(task, size=1, seed=2)[0]
            assert FullScorer()(task, entry, text)["correct"] == 0


def test_scoring_exception_counts_as_wrong_and_is_tallied():
    from experiments.atom_decomposition.common import CHECKERS
    from experiments.atom_decomposition.eval_atoms import SCORING_ERRORS, score_atom

    def boom(extracted, gold):
        raise ValueError("bad")
    CHECKERS["boom"] = boom
    try:
        before = sum(SCORING_ERRORS.values())
        assert score_atom("boom", "1", "<answer>1</answer>") == \
            {"correct": 0, "correct_paper": 0, "correct_strict": 0}
        assert sum(SCORING_ERRORS.values()) == before + 3
    finally:
        del CHECKERS["boom"]


def test_resume_drops_a_cut_off_last_line(tmp_path):
    from experiments.atom_decomposition.eval_atoms import _done_keys
    p = tmp_path / "r.jsonl"
    p.write_text('{"key": "a"}\n{"key": "b"}\n{"key": "c', encoding="utf-8")
    assert _done_keys(str(p), lambda r: r["key"]) == {"a", "b"}
    assert p.read_text(encoding="utf-8") == '{"key": "a"}\n{"key": "b"}\n'
