"""
maze: BFS frontier expansion.

Graph "bfs": locate the start and the goal (L1), then expand the frontier one distance
layer at a time (L2): given the cells at distance d and those reached earlier, list the
cells first reached at distance d+1. The goal first appears in layer D = the answer.
Moves are up/down/left/right, as in reasoning-gym's own BFS. Coordinates are
(row, column), numbered from 1 at the top-left.

Graph "reading" (not in GRAPHS, so gate fits and default builds leave it out): grid-reading atoms, the
skill every model fails at on big grids. L1 locate start and goal; L0 list the open cells of a row
(3 interior rows) and the open neighbours of a cell (3 open cells), chosen by a seed taken from the grid.

Graph "prefix" (also extra): the full task with a partial solution in context. The instance's own question,
with the first k moves of a shortest path given (k from 1 to D-1, up to 4 values per maze, seeded by the
grid); the answer is still the full shortest-path length D. Atoms inside the task, not beside it.

walk_progress(): how far a completion's reasoning walks legally from the start toward the goal, in [0, 1]
(1 = a legal walk reaches the goal); the dense "legal progress" reward.
"""

import random
import re
import zlib
from collections import deque
from typing import Dict, List, Optional, Set, Tuple

from ..common import Node, fmt_cells

GRAPHS = ("bfs",)
EXTRA_GRAPHS = ("reading", "prefix")
COORDS = "Rows and columns are numbered from 1, starting at the top-left corner."
Cell = Tuple[int, int]


def _find(grid: List[str], ch: str) -> Cell:
    for r, row in enumerate(grid):
        c = row.find(ch)
        if c >= 0:
            return r, c
    raise ValueError(ch)


def _open(grid: List[str], wall: str, r: int, c: int) -> bool:
    return 0 <= r < len(grid) and 0 <= c < len(grid[r]) and grid[r][c] != wall


def _one_based(cells) -> Set[Cell]:
    return {(r + 1, c + 1) for r, c in cells}


def bfs_layers(meta: dict) -> List[Set[Cell]]:
    """Distance layers from the start, stopping at the layer that contains the goal (0-based cells)."""
    grid, wall = meta["grid"], meta["wall"]
    start, goal = _find(grid, meta["start"]), _find(grid, meta["goal"])
    layers, seen = [{start}], {start}
    while goal not in layers[-1]:
        nxt = set()
        for r, c in layers[-1]:
            for dr, dc in ((-1, 0), (0, 1), (1, 0), (0, -1)):
                n = (r + dr, c + dc)
                if n not in seen and _open(grid, wall, *n):
                    nxt.add(n)
        if not nxt:
            raise ValueError("goal unreachable")
        seen |= nxt
        layers.append(nxt)
    return layers


def trace(sample: dict, graph: str = "bfs") -> List[Node]:
    assert graph in GRAPHS + EXTRA_GRAPHS, graph
    if graph == "reading":
        return _reading(sample)
    if graph == "prefix":
        return _prefix(sample)
    meta = sample["metadata"]
    grid, wall, path = meta["grid"], meta["wall"], meta["path"]
    maze_txt = "```\n" + "\n".join(grid) + "\n```"
    legend = (f"Legend: '{wall}' = Wall, '{path}' = Passage, '{meta['start']}' = start, "
              f"'{meta['goal']}' = goal.")
    head = f"Here is a maze:\n\n{maze_txt}\n{legend}\n{COORDS}\n"
    start, goal = _find(grid, meta["start"]), _find(grid, meta["goal"])

    nodes = [
        Node("start", "L1", "locate", head + f"What are the coordinates of the start '{meta['start']}'? "
             "Answer as (row, column).", fmt_cells(_one_based([start])), "cell"),
        Node("goal", "L1", "locate", head + f"What are the coordinates of the goal '{meta['goal']}'? "
             "Answer as (row, column).", fmt_cells(_one_based([goal])), "cell"),
    ]
    layers = bfs_layers(meta)
    reached = set()
    for d in range(len(layers) - 1):
        reached |= layers[d]
        earlier = reached - layers[d]
        q = head + (
            f"Moves go one cell up, down, left or right, never into a wall. The cells at distance "
            f"{d} from the start are: {fmt_cells(_one_based(layers[d]))}."
        )
        if earlier:
            q += f" The cells reached earlier are: {fmt_cells(_one_based(earlier))}."
        q += (f" Which cells are at distance {d + 1}, i.e. one move from a distance-{d} cell and "
              "not reached before? List them as (row, column) pairs separated by commas.")
        deps = ["start"] + ([f"layer{d}"] if d else []) + (["goal"] if d == len(layers) - 2 else [])
        nodes.append(Node(f"layer{d + 1}", "L2", "bfs_layer", q,
                          fmt_cells(_one_based(layers[d + 1])), "cell_set", deps))
    return nodes


def _reading(sample: dict, n_rows: int = 3, n_cells: int = 3) -> List[Node]:
    meta = sample["metadata"]
    grid, wall, path = meta["grid"], meta["wall"], meta["path"]
    head, start, goal = _head(meta)
    rng = random.Random(zlib.crc32("\n".join(grid).encode()))
    nodes = [
        Node("start", "L1", "locate", head + f"What are the coordinates of the start '{meta['start']}'? "
             "Answer as (row, column).", fmt_cells(_one_based([start])), "cell"),
        Node("goal", "L1", "locate", head + f"What are the coordinates of the goal '{meta['goal']}'? "
             "Answer as (row, column).", fmt_cells(_one_based([goal])), "cell"),
    ]
    rows = [r for r in range(1, len(grid) - 1) if any(ch != wall for ch in grid[r])]
    for r in sorted(rng.sample(rows, min(n_rows, len(rows)))):
        open_cells = [(r, c) for c in range(len(grid[r])) if grid[r][c] != wall]
        nodes.append(Node(f"row{r + 1}", "L0", "read_row", head + (
            f"Which cells in row {r + 1} are open (not walls)? The start and the goal count as open. "
            "List them as (row, column) pairs separated by commas."),
            fmt_cells(_one_based(open_cells)), "cell_set"))
    cells = [(r, c) for r in range(len(grid)) for c in range(len(grid[r]))
             if grid[r][c] != wall and any(_open(grid, wall, r + dr, c + dc)
                                           for dr, dc in ((-1, 0), (0, 1), (1, 0), (0, -1)))]
    for r, c in sorted(rng.sample(cells, min(n_cells, len(cells)))):
        nbrs = [(r + dr, c + dc) for dr, dc in ((-1, 0), (0, 1), (1, 0), (0, -1)) if _open(grid, wall, r + dr, c + dc)]
        nodes.append(Node(f"nbr{r + 1}_{c + 1}", "L0", "read_neighbours", head + (
            f"Which of the four cells next to ({r + 1}, {c + 1}) (up, down, left, right) are open (not walls)? "
            "List them as (row, column) pairs separated by commas."),
            fmt_cells(_one_based(nbrs)), "cell_set"))
    return nodes


def dist_to(grid: List[str], wall: str, target: Cell) -> Dict[Cell, int]:
    """BFS distance from every reachable open cell to target (0-based cells)."""
    D, Q = {target: 0}, deque([target])
    while Q:
        r, c = Q.popleft()
        for dr, dc in ((-1, 0), (0, 1), (1, 0), (0, -1)):
            n = (r + dr, c + dc)
            if n not in D and _open(grid, wall, *n):
                D[n] = D[(r, c)] + 1
                Q.append(n)
    return D


def shortest_path(grid: List[str], wall: str, start: Cell, goal: Cell) -> List[Cell]:
    """One shortest path start..goal (0-based cells), walking down the distance-to-goal field."""
    D = dist_to(grid, wall, goal)
    path = [start]
    while path[-1] != goal:
        r, c = path[-1]
        path.append(next(n for n in ((r - 1, c), (r, c + 1), (r + 1, c), (r, c - 1))
                         if D.get(n, -1) == D[(r, c)] - 1))
    return path


def _prefix(sample: dict, n_k: int = 4) -> List[Node]:
    meta, q = sample["metadata"], sample["question"]
    grid, wall = meta["grid"], meta["wall"]
    start, goal = _find(grid, meta["start"]), _find(grid, meta["goal"])
    path = shortest_path(grid, wall, start, goal)
    D = len(path) - 1
    cut = q.find("What is the minimum number of steps")
    assert cut > 0, q
    rng = random.Random(zlib.crc32(("prefix\n" + "\n".join(grid)).encode()))
    nodes = []
    for k in sorted(rng.sample(range(1, D), min(n_k, D - 1))):
        moves = " -> ".join(f"({r + 1}, {c + 1})" for r, c in path[:k + 1])
        hint = (f"{COORDS}\nA shortest path from the start begins with these {k} moves: {moves}. "
                "Continue it to the goal.\n\n")
        nodes.append(Node(f"prefix{k}", "P", "prefix", q[:cut] + hint + q[cut:], str(D), "int"))
    return nodes


COORD_RE = re.compile(r'\(\s*(?:row\s*)?(\d+)\s*,\s*(?:col(?:umn)?\s*)?(\d+)\s*\)')


def walk_progress(text: str, meta: dict, dist: Optional[Dict[Cell, int]] = None) -> float:
    """Legal progress of the reasoning toward the goal, in [0, 1]. A walk is a run of consecutive coordinates
    in the reasoning (before </think>) that starts at the start cell, where every step moves one cell
    up/down/left/right onto an open cell. Progress = (d(start) - closest d reached) / d(start), d = BFS distance
    to the goal, maximised over walks and over 0- and 1-based coordinates. Wandering earns nothing; a legal
    walk that reaches the goal earns 1."""
    grid, wall = meta["grid"], meta["wall"]
    start, goal = _find(grid, meta["start"]), _find(grid, meta["goal"])
    D = dist if dist is not None else dist_to(grid, wall, goal)
    d0 = D.get(start)
    if not d0:
        return 0.0
    raw = [(int(a), int(b)) for a, b in COORD_RE.findall(text.split("</think>")[0])]
    best = 0.0
    for off in (0, 1):
        cs = [(r - off, c - off) for r, c in raw]
        for i, p in enumerate(cs):
            if p != start:
                continue
            low, j = d0, i
            while (j + 1 < len(cs) and abs(cs[j + 1][0] - cs[j][0]) + abs(cs[j + 1][1] - cs[j][1]) == 1
                   and _open(grid, wall, *cs[j + 1])):
                j += 1
                low = min(low, D.get(cs[j], d0))
                if low == 0:
                    break
            best = max(best, (d0 - low) / d0)
            if best == 1.0:
                return 1.0
    return best


def _head(meta: dict):
    grid, wall, path = meta["grid"], meta["wall"], meta["path"]
    maze_txt = "```\n" + "\n".join(grid) + "\n```"
    legend = (f"Legend: '{wall}' = Wall, '{path}' = Passage, '{meta['start']}' = start, "
              f"'{meta['goal']}' = goal.")
    return (f"Here is a maze:\n\n{maze_txt}\n{legend}\n{COORDS}\n",
            _find(grid, meta["start"]), _find(grid, meta["goal"]))


def difficulty_features(sample: dict) -> dict:
    m = sample["metadata"]
    n_open = sum(ch != m["wall"] for row in m["grid"] for ch in row)
    return {"grid_size": m["grid_size"], "n_open": n_open,
            "shortest_path_length": m["shortest_path_length"]}
