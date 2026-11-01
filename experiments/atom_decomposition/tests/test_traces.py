"""Solvers reproduce reasoning-gym's answers, and every gold atom answer passes its own checker.

Run from the repo root:  python -m pytest experiments/atom_decomposition/tests -q
"""

import json

import pytest
import reasoning_gym as rg

from experiments.atom_decomposition.build_atoms import build, check_trace
from experiments.atom_decomposition.common import CHECKERS, parse_cells
from experiments.atom_decomposition.tasks import TASKS
from experiments.atom_decomposition.tasks.arithmetic import parse_strict, render, tree_of
from experiments.atom_decomposition.tasks.mini_sudoku import trace_backtracking, trace_propagation

N = 1000


@pytest.mark.parametrize("task", list(TASKS))
def test_solver_matches_reasoning_gym(task):
    ds = rg.create_dataset(task, size=N, seed=1)
    for i in range(N):
        check_trace(task, ds[i])


def test_sudoku_both_graphs_reach_solution():
    ds = rg.create_dataset("mini_sudoku", size=N, seed=1)
    for i in range(N):
        m = ds[i]["metadata"]
        for solve in (trace_propagation, trace_backtracking):
            _, grid, solved = solve(m["puzzle"])
            assert solved and grid == m["solution"], solve.__name__


def _wrong(check, gold):
    """A near miss for each checker's gold format."""
    if check == "int":
        return str(int(gold) + 1)
    if check == "int_set":
        return "none" if gold != "none" else "1"
    if check in ("cell", "cell_set"):
        cells = sorted(parse_cells(gold))
        if not cells:
            return "(1, 1)"
        r, c = cells[0]
        return ", ".join(f"({a}, {b})" for a, b in [(r, c + 1)] + cells[1:])
    if check == "paren_tree":
        t = tree_of(gold.replace("(", " ").replace(")", " ").strip())  # precedence-only reading
        return render(t) if render(t) != gold else "(0 + 0)"
    raise ValueError(check)


@pytest.mark.parametrize("task", list(TASKS))
def test_gold_answers_pass_and_near_misses_fail(task):
    mod = TASKS[task]
    ds = rg.create_dataset(task, size=200, seed=3)
    for i in range(200):
        for graph in mod.GRAPHS:
            for node in mod.trace(ds[i], graph):
                chk = CHECKERS[node.check]
                assert chk(node.answer, node.answer), (task, node.id, node.answer)
                wrong = _wrong(node.check, node.answer)
                if wrong != node.answer and not (node.check == "paren_tree"
                                                 and parse_strict(wrong) == parse_strict(node.answer)):
                    assert not chk(wrong, node.answer), (task, node.id, node.answer, wrong)


def test_paren_parser():
    expr = "2 - 3 * 4 + 5"
    gold = render(tree_of(expr))
    assert gold == "((2 - (3 * 4)) + 5)"
    ok = CHECKERS["paren_tree"]
    assert ok("((2 - (3 * 4)) + 5)", gold)
    assert ok("(2 - (3 * 4)) + 5", gold)              # outermost pair may be left off
    assert ok("((2 - ((3) * 4)) + 5).", gold)         # redundant parentheses are fine
    assert not ok("2 - 3 * 4 + 5", gold)              # copying the expression is not a parse
    assert not ok("(2 - ((3 * 4) + 5))", gold)        # wrong grouping
    assert not ok(None, gold)
    neg = render(tree_of("-(3 + 4) * -5"))
    assert neg == "(-(3 + 4) * -5)"
    assert ok("((-(3 + 4)) * (-5))", neg)
    assert not ok("(-((3 + 4) * -5))", neg)


def test_checkers():
    assert CHECKERS["int"]("-4.", "-4") and CHECKERS["int"]("= -4", "-4")
    assert not CHECKERS["int"]("-4 or 4", "-4")
    assert CHECKERS["int_set"]("4, 1", "1, 4") and CHECKERS["int_set"]("none", "none")
    assert not CHECKERS["int_set"]("1", "1, 4")
    assert not CHECKERS["int_set"]("wrong", "none") and not CHECKERS["int_set"]("", "none")
    assert not CHECKERS["cell_set"]("wrong", "none") and not CHECKERS["int_or_none"]("wrong", "none")
    assert CHECKERS["cell_set"]("(2,3), (1, 1)", "(1, 1), (2, 3)")
    assert not CHECKERS["cell_set"]("(1, 1)", "(1, 1), (2, 3)")
    assert not CHECKERS["cell_set"]("I don't know", "(1, 1)")
    assert CHECKERS["cell"]("(3, 4)", "(3, 4)") and not CHECKERS["cell"]("(3, 4), (1, 1)", "(3, 4)")


def test_build_is_deterministic(tmp_path):
    a = build(list(TASKS), 20, 7, 40, str(tmp_path / "a"))
    b = build(list(TASKS), 20, 7, 40, str(tmp_path / "b"))
    assert a == b
    assert (tmp_path / "a" / "atoms.jsonl").read_text() == (tmp_path / "b" / "atoms.jsonl").read_text()
    rows = [json.loads(l) for l in open(tmp_path / "a" / "instances.jsonl")]
    assert len(rows) == 20 * len(TASKS)
