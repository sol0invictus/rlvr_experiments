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


def test_maze_reading_atoms_match_grid():
    """Grid-reading atoms (graph "reading", not in GRAPHS): answers match the raw grid, pass their checker,
    and the choice of rows and cells is the same on every call."""
    from experiments.atom_decomposition.tasks import maze
    assert "reading" not in maze.GRAPHS
    ds = rg.create_dataset("maze", size=200, seed=3, min_dist=10, max_dist=15, min_grid_size=10, max_grid_size=14)
    for i in range(200):
        s = ds[i]
        g, w = s["metadata"]["grid"], s["metadata"]["wall"]
        nodes = maze.trace(s, "reading")
        assert [x.question for x in nodes] == [x.question for x in maze.trace(s, "reading")]
        assert {x.type for x in nodes} == {"locate", "read_row", "read_neighbours"}
        for x in nodes:
            assert CHECKERS[x.check](x.answer, x.answer)
            got = parse_cells(x.answer)
            if x.type == "read_row":
                r = int(x.id[3:])
                assert got == {(r, c + 1) for c, ch in enumerate(g[r - 1]) if ch != w}
            elif x.type == "read_neighbours":
                r, c = map(int, x.id[3:].split("_"))
                exp = {(r + a, c + b) for a, b in ((-1, 0), (1, 0), (0, -1), (0, 1)) if g[r + a - 1][c + b - 1] != w}
                assert got == exp and exp


def test_maze_prefix_prompts_give_a_legal_shortest_prefix():
    """Graph "prefix": the hint is the first k moves of a real shortest path (legal, 1-based, k < D), the rest of
    the prompt is the instance's own question, and the answer is the full path length."""
    import re
    from experiments.atom_decomposition.tasks import maze
    assert "prefix" not in maze.GRAPHS
    ds = rg.create_dataset("maze", size=200, seed=3, min_dist=10, max_dist=15, min_grid_size=10, max_grid_size=14)
    for i in range(200):
        s = ds[i]
        m = s["metadata"]
        g, w = m["grid"], m["wall"]
        goal = maze._find(g, m["goal"])
        dist = maze.dist_to(g, w, goal)
        nodes = maze.trace(s, "prefix")
        assert nodes and [x.question for x in nodes] == [x.question for x in maze.trace(s, "prefix")]
        for x in nodes:
            assert x.answer == s["answer"] and CHECKERS[x.check](x.answer, x.answer)
            head, hint = x.question.split("begins with these ")
            assert s["question"].split("What is the minimum")[0] in head
            k = int(hint.split(" moves")[0])
            cells = [(int(a) - 1, int(b) - 1) for a, b in re.findall(r"\((\d+), (\d+)\)", hint.split("Continue")[0])]
            assert len(cells) == k + 1 and 1 <= k < int(s["answer"])
            assert cells[0] == maze._find(g, m["start"])
            for j, (a, b) in enumerate(zip(cells, cells[1:])):
                assert abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1 and g[b[0]][b[1]] != w
                assert dist[b] == dist[a] - 1   # every move gets one step closer: a shortest-path prefix
            assert x.question.rstrip().endswith(s["question"].rstrip().split("\n")[-1])


def test_maze_walk_progress():
    """Legal progress: 1 for a legal walk to the goal (either coordinate base), partial for a legal walk that stops
    short, 0 for wandering away or for a walk that does not start at the start, and a wall cut stops the walk."""
    from experiments.atom_decomposition.tasks import maze
    ds = rg.create_dataset("maze", size=50, seed=5, min_dist=10, max_dist=15, min_grid_size=10, max_grid_size=14)
    for i in range(50):
        m = ds[i]["metadata"]
        g, w = m["grid"], m["wall"]
        s, goal = maze._find(g, m["start"]), maze._find(g, m["goal"])
        path = maze.shortest_path(g, w, s, goal)
        D = len(path) - 1
        fmt = lambda cells, off=1: " -> ".join(f"({r + off}, {c + off})" for r, c in cells)
        assert maze.walk_progress(f"<think>{fmt(path)}</think><answer>{D}</answer>", m) == 1.0
        assert maze.walk_progress(f"<think>{fmt(path, 0)}</think>", m) == 1.0
        assert maze.walk_progress(f"<think>{fmt(path[:D // 2 + 1])}</think>", m) == pytest.approx((D // 2) / D)
        assert maze.walk_progress(f"<think>{fmt(path[1:])}</think>", m) == 0.0          # does not start at start
        assert maze.walk_progress("<think>no coordinates</think><answer>12</answer>", m) == 0.0
        assert maze.walk_progress(f"<think>x</think> {fmt(path)}", m) == 0.0            # only the reasoning counts
        wall = next(((r, c) for r in range(len(g)) for c in range(len(g[0]))
                     if g[r][c] == w and abs(r - s[0]) + abs(c - s[1]) == 1), None)
        if wall:   # a step into a wall ends the walk there
            assert maze.walk_progress(f"<think>{fmt([s, wall] + path[1:])}</think>", m) == 0.0
