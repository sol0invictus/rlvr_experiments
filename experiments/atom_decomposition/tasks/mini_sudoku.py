"""
mini_sudoku (4x4, 2x2 boxes): two candidate graphs.

Both share the same L1 reads (what is given in each row, column and box). They differ
in the L2 steps:

  "propagation"  repeatedly fill a forced cell: a naked single (the cell has one candidate
                 left), else a hidden single (a number has one place left in a row, column
                 or box). Each fill is one node, asked on the grid as it stands then.
                 On 4x4 puzzles naked singles alone solve every instance seen so far.
  "backtracking" visit empty cells in row-major order and try numbers in increasing order;
                 a visit with no number left sends the search back a cell. Each visit is one
                 node, dead ends included.

L2 nodes of both graphs ask the same question (which numbers can go in this cell, on this
grid), so a difference between the graphs comes from the states they visit, not the wording.
Hidden singles keep their own question.

The answer is the solved grid in reasoning-gym's format; full-task scoring is reasoning-gym's
exact match (the paper's rule), see PHASE_C.md.
"""

from typing import List, Optional, Tuple

from ..common import Node, fmt_ints

GRAPHS = ("propagation", "backtracking")
N, B = 4, 2
RULES = ("In 4x4 Mini Sudoku, each row, each column and each 2x2 box must contain each "
         "number from 1 to 4 exactly once.")
BOX_NAMES = {(0, 0): "top-left", (0, 1): "top-right", (1, 0): "bottom-left", (1, 1): "bottom-right"}

Grid = List[List[int]]


def fmt_grid(g: Grid) -> str:
    return "\n".join(" ".join(str(v) if v else "_" for v in row) for row in g)


def box_of(r: int, c: int) -> Tuple[int, int]:
    return r // B, c // B


def box_label(br: int, bc: int) -> str:
    return (f"the {BOX_NAMES[(br, bc)]} 2x2 box (rows {br * B + 1}-{br * B + B}, "
            f"columns {bc * B + 1}-{bc * B + B})")


def box_cells(br: int, bc: int):
    return [(r, c) for r in range(br * B, br * B + B) for c in range(bc * B, bc * B + B)]


def candidates(g: Grid, r: int, c: int) -> List[int]:
    used = set(g[r]) | {g[i][c] for i in range(N)} | {g[i][j] for i, j in box_cells(*box_of(r, c))}
    return [v for v in range(1, N + 1) if v not in used]


def _units():
    for r in range(N):
        yield ("row", r), [(r, c) for c in range(N)]
    for c in range(N):
        yield ("col", c), [(r, c) for r in range(N)]
    for br in range(N // B):
        for bc in range(N // B):
            yield ("box", (br, bc)), box_cells(br, bc)


def _unit_text(unit) -> str:
    kind, k = unit
    if kind == "row":
        return f"row {k + 1}"
    if kind == "col":
        return f"column {k + 1}"
    return box_label(*k)


def _given_question(puzzle: Grid, unit) -> str:
    return (f"{RULES}\nHere is the puzzle (_ marks an empty cell):\n{fmt_grid(puzzle)}\n"
            f"Which numbers are already given in {_unit_text(unit)}? List them separated by "
            "commas, or answer none.")


def l1_nodes(puzzle: Grid) -> List[Node]:
    nodes = []
    for unit, cells in _units():
        kind, k = unit
        nid = f"given_{kind}{k if kind != 'box' else f'{k[0]}{k[1]}'}"
        nodes.append(Node(
            id=nid, layer="L1", type=f"given_{kind}",
            question=_given_question(puzzle, unit),
            answer=fmt_ints(puzzle[r][c] for r, c in cells if puzzle[r][c]),
            check="int_set",
        ))
    return nodes


def _unit_ids(r: int, c: int) -> List[str]:
    br, bc = box_of(r, c)
    return [f"given_row{r}", f"given_col{c}", f"given_box{br}{bc}"]


def _grid_preamble(g: Grid) -> str:
    return f"{RULES}\nHere is a partly filled puzzle (_ marks an empty cell):\n{fmt_grid(g)}\n"


def _candidates_question(g: Grid, r: int, c: int) -> str:
    """The one L2 question both graphs ask, so they differ only in which grid states they visit."""
    return (_grid_preamble(g) + f"Which numbers from 1 to 4 can go in row {r + 1}, column {c + 1} "
            "without repeating a number in its row, column or 2x2 box? List all of them separated "
            "by commas, or answer none if no number fits.")


# ----------------------------------------------------------------------
# Constraint propagation
# ----------------------------------------------------------------------

def _next_forced(g: Grid):
    """First naked single in row-major order, else first hidden single by unit order."""
    for r in range(N):
        for c in range(N):
            if not g[r][c]:
                cand = candidates(g, r, c)
                if len(cand) == 1:
                    return "naked_single", (r, c), cand[0], None
    for unit, cells in _units():
        empties = [(r, c) for r, c in cells if not g[r][c]]
        for v in range(1, N + 1):
            if any(g[r][c] == v for r, c in cells):
                continue
            spots = [(r, c) for r, c in empties if v in candidates(g, r, c)]
            if len(spots) == 1:
                return "hidden_single", spots[0], v, unit
    return None


def trace_propagation(puzzle: Grid) -> Tuple[List[Node], Grid, bool]:
    g = [row[:] for row in puzzle]
    nodes = l1_nodes(puzzle)
    last_fill = {}   # cell -> node id that filled it
    while True:
        step = _next_forced(g)
        if step is None:
            break
        kind, (r, c), v, unit = step
        related = {(i, j) for _, cells in _units() if (r, c) in cells for i, j in cells}
        deps = _unit_ids(r, c) + sorted({last_fill[x] for x in related if x in last_fill})
        nid = f"fill{len(last_fill)}"
        if kind == "naked_single":
            nodes.append(Node(nid, "L2", kind, _candidates_question(g, r, c),
                              fmt_ints(candidates(g, r, c)), "int_set", deps))
        else:
            ukind, _ = unit
            where = {"row": "column", "col": "row"}.get(ukind)
            q = (_grid_preamble(g) + f"In {_unit_text(unit)}, the number {v} fits in only one "
                 "of the empty cells. ")
            if where:
                q += f"Which {where} is that cell in? Give only the {where} number."
                ans = str(c + 1 if ukind == "row" else r + 1)
                nodes.append(Node(nid, "L2", kind, q, ans, "int", deps))
            else:
                q += "Which cell is it? Answer as (row, column)."
                nodes.append(Node(nid, "L2", kind, q, f"({r + 1}, {c + 1})", "cell", deps))
        g[r][c] = v
        last_fill[(r, c)] = nid
    solved = all(all(row) for row in g)
    return nodes, g, solved


# ----------------------------------------------------------------------
# Backtracking
# ----------------------------------------------------------------------

def trace_backtracking(puzzle: Grid, max_nodes: Optional[int] = None) -> Tuple[List[Node], Grid, bool]:
    g = [row[:] for row in puzzle]
    nodes = l1_nodes(puzzle)
    empties = [(r, c) for r in range(N) for c in range(N) if not g[r][c]]
    tried = [0] * len(empties)       # last number tried at each empty cell
    visit_id = [None] * len(empties)  # node that set each cell's current value
    i = 0
    n_visits = 0
    while 0 <= i < len(empties):
        r, c = empties[i]
        g[r][c] = 0
        lo = tried[i]
        fits = [v for v in candidates(g, r, c) if v > lo]
        v = fits[0] if fits else None
        prev = [visit_id[j] for j in range(i) if visit_id[j]]
        nid = f"visit{n_visits}"
        nodes.append(Node(nid, "L2", "bt_visit", _candidates_question(g, r, c),
                          fmt_ints(candidates(g, r, c)), "int_set", _unit_ids(r, c) + prev))
        n_visits += 1
        if max_nodes and n_visits >= max_nodes:
            break
        if v:
            g[r][c] = v
            tried[i] = v
            visit_id[i] = nid
            i += 1
        else:
            tried[i] = 0
            visit_id[i] = None
            i -= 1
    solved = all(all(row) for row in g)
    return nodes, g, solved


def trace(sample: dict, graph: str) -> List[Node]:
    puzzle = sample["metadata"]["puzzle"]
    if graph == "propagation":
        nodes, _, _ = trace_propagation(puzzle)
    elif graph == "backtracking":
        nodes, _, _ = trace_backtracking(puzzle)
    else:
        raise ValueError(graph)
    return nodes


def difficulty_features(sample: dict) -> dict:
    return {"num_empty": sample["metadata"]["num_empty"]}
