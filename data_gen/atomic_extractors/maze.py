"""Maze atomic skill extractor (M.L1.1-M.L1.3, M.L2.1-M.L2.3)."""

import random


def extract_atoms(instance: dict, rng: random.Random) -> list[dict]:
    meta = instance["metadata"]
    grid = meta["grid"]           # list of strings, e.g. ['>>>>', '>ee>', ...]
    wall = meta["wall"]           # wall character, e.g. '>'
    path = meta["path"]           # path character, e.g. 'e'
    start = meta["start"]         # start character
    goal = meta["goal"]           # goal character
    grid_size = meta["grid_size"]
    idx = meta.get("source_index", 0)
    atoms = []

    # Find start and goal positions
    start_pos = goal_pos = None
    for r, row in enumerate(grid):
        for c, ch in enumerate(row):
            if ch == start:
                start_pos = (r, c)
            if ch == goal:
                goal_pos = (r, c)

    # ── L1.1 Grid reading ─────────────────────────────────────────
    for _ in range(5):
        r = rng.randint(0, len(grid) - 1)
        c = rng.randint(0, len(grid[r]) - 1)
        ch = grid[r][c]
        cell_type = "wall" if ch == wall else "passage" if ch == path else f"'{ch}'"
        atoms.append({
            "question": f"In this {grid_size}x{grid_size} grid, what is at row {r}, column {c}?",
            "answer": cell_type,
            "skill": "grid_reading", "layer": "L1", "source_index": idx,
        })

    # ── L1.2 Coordinate parsing ───────────────────────────────────
    if start_pos:
        atoms.append({
            "question": f"The start is marked '{start}'. What are its (row, col) coordinates?",
            "answer": f"({start_pos[0]}, {start_pos[1]})",
            "skill": "coordinate_parsing", "layer": "L1", "source_index": idx,
        })
    if goal_pos:
        atoms.append({
            "question": f"The goal is marked '{goal}'. What are its (row, col) coordinates?",
            "answer": f"({goal_pos[0]}, {goal_pos[1]})",
            "skill": "coordinate_parsing", "layer": "L1", "source_index": idx,
        })

    # ── L1.3 Start/goal identification ────────────────────────────
    atoms.append({
        "question": f"Which character marks the start in this maze?",
        "answer": str(start),
        "skill": "start_goal_identification", "layer": "L1", "source_index": idx,
    })
    atoms.append({
        "question": f"Which character marks the goal in this maze?",
        "answer": str(goal),
        "skill": "start_goal_identification", "layer": "L1", "source_index": idx,
    })

    # ── L2.1 Neighbor enumeration ─────────────────────────────────
    # Pick random passage cells and list non-wall neighbors
    passage_cells = [(r, c) for r in range(len(grid)) for c in range(len(grid[r]))
                     if grid[r][c] != wall]
    for pos in rng.sample(passage_cells, min(5, len(passage_cells))):
        r, c = pos
        neighbors = []
        for dr, dc, direction in [(-1, 0, "up"), (1, 0, "down"), (0, -1, "left"), (0, 1, "right")]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < len(grid) and 0 <= nc < len(grid[nr]) and grid[nr][nc] != wall:
                neighbors.append(direction)
        atoms.append({
            "question": f"At position ({r}, {c}), which directions lead to non-wall cells?",
            "answer": ", ".join(neighbors) if neighbors else "none",
            "skill": "neighbor_enumeration", "layer": "L2", "source_index": idx,
        })

    # ── L2.2 Move legality ────────────────────────────────────────
    for _ in range(4):
        r = rng.randint(0, len(grid) - 1)
        c = rng.randint(0, len(grid[r]) - 1)
        direction = rng.choice(["up", "down", "left", "right"])
        dr, dc = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}[direction]
        nr, nc = r + dr, c + dc
        if 0 <= nr < len(grid) and 0 <= nc < len(grid[nr]):
            legal = grid[nr][nc] != wall
            atoms.append({
                "question": f"From ({r}, {c}), is moving {direction} legal (not into a wall)?",
                "answer": "Yes" if legal else "No, that cell is a wall",
                "skill": "move_legality", "layer": "L2", "source_index": idx,
            })
        else:
            atoms.append({
                "question": f"From ({r}, {c}), is moving {direction} legal (not into a wall)?",
                "answer": "No, that is out of bounds",
                "skill": "move_legality", "layer": "L2", "source_index": idx,
            })

    # ── L2.3 Progress judgment ────────────────────────────────────
    if start_pos and goal_pos:
        for _ in range(3):
            pos_a = rng.choice(passage_cells)
            pos_b = rng.choice(passage_cells)
            dist_a = abs(pos_a[0] - goal_pos[0]) + abs(pos_a[1] - goal_pos[1])
            dist_b = abs(pos_b[0] - goal_pos[0]) + abs(pos_b[1] - goal_pos[1])
            closer = "A" if dist_a < dist_b else "B" if dist_b < dist_a else "equal"
            atoms.append({
                "question": f"Goal is at {goal_pos}. Position A is {pos_a}, position B is {pos_b}. "
                            f"Which is closer to the goal (Manhattan distance)?",
                "answer": f"{closer} (distance A={dist_a}, B={dist_b})",
                "skill": "progress_judgment", "layer": "L2", "source_index": idx,
            })

    return atoms
