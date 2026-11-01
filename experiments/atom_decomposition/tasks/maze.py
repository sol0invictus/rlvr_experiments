"""
maze: BFS frontier expansion.

Graph "bfs": locate the start and the goal (L1), then expand the frontier one distance
layer at a time (L2): given the cells at distance d and those reached earlier, list the
cells first reached at distance d+1. The goal first appears in layer D = the answer.
Moves are up/down/left/right, as in reasoning-gym's own BFS. Coordinates are
(row, column), numbered from 1 at the top-left.
"""

from typing import List, Set, Tuple

from ..common import Node, fmt_cells

GRAPHS = ("bfs",)
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
    assert graph in GRAPHS, graph
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


def difficulty_features(sample: dict) -> dict:
    m = sample["metadata"]
    n_open = sum(ch != m["wall"] for row in m["grid"] for ch in row)
    return {"grid_size": m["grid_size"], "n_open": n_open,
            "shortest_path_length": m["shortest_path_length"]}
