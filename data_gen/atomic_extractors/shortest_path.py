"""Shortest path atomic skill extractor (P.L1.1-P.L1.3, P.L2.1-P.L2.3)."""

import random


def extract_atoms(instance: dict, rng: random.Random) -> list[dict]:
    meta = instance["metadata"]
    matrix = meta["matrix"]
    solution = meta["solution"]
    idx = meta.get("source_index", 0)
    atoms = []
    rows = len(matrix)
    cols = len(matrix[0]) if rows > 0 else 0

    start_pos = goal_pos = None
    for r in range(rows):
        for c in range(cols):
            if matrix[r][c] == '*':
                start_pos = (r, c)
            elif matrix[r][c] == '#':
                goal_pos = (r, c)

    # L1.1 Cell reading
    for _ in range(5):
        r, c = rng.randint(0, rows-1), rng.randint(0, cols-1)
        label = {"O": "open", "X": "blocked", "*": "start", "#": "goal"}.get(matrix[r][c], matrix[r][c])
        atoms.append({"question": f"In this {rows}x{cols} grid, what is at row {r}, column {c}?",
                       "answer": label, "skill": "cell_reading", "layer": "L1", "source_index": idx})

    # L1.2 Adjacency reading
    open_cells = [(r, c) for r in range(rows) for c in range(cols) if matrix[r][c] != 'X']
    for pos in rng.sample(open_cells, min(4, len(open_cells))):
        r, c = pos
        neighbors = []
        for dr, dc, d in [(-1,0,"up"),(1,0,"down"),(0,-1,"left"),(0,1,"right")]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < rows and 0 <= nc < cols and matrix[nr][nc] != 'X':
                neighbors.append(d)
        atoms.append({"question": f"From ({r},{c}), which directions lead to non-blocked cells?",
                       "answer": ", ".join(neighbors) if neighbors else "none",
                       "skill": "adjacency_reading", "layer": "L1", "source_index": idx})

    # L1.3 Source/destination identification
    if start_pos:
        atoms.append({"question": "Where is the start position (*)?",
                       "answer": f"({start_pos[0]}, {start_pos[1]})",
                       "skill": "source_dest_identification", "layer": "L1", "source_index": idx})
    if goal_pos:
        atoms.append({"question": "Where is the goal position (#)?",
                       "answer": f"({goal_pos[0]}, {goal_pos[1]})",
                       "skill": "source_dest_identification", "layer": "L1", "source_index": idx})

    # L2.1 Path cost computation
    atoms.append({"question": f"A path takes these steps: {', '.join(solution)}. How many steps?",
                   "answer": str(len(solution)),
                   "skill": "path_cost_computation", "layer": "L2", "source_index": idx})

    # L2.2 Path comparison
    if start_pos and goal_pos:
        real_len = len(solution)
        fake_len = real_len + rng.randint(1, 3)
        atoms.append({"question": f"Path A has {real_len} steps, Path B has {fake_len} steps. Which is shorter?",
                       "answer": f"Path A ({real_len} steps)",
                       "skill": "path_comparison", "layer": "L2", "source_index": idx})
        manhattan = abs(start_pos[0]-goal_pos[0]) + abs(start_pos[1]-goal_pos[1])
        atoms.append({"question": f"Start at {start_pos}, goal at {goal_pos}. Manhattan distance?",
                       "answer": str(manhattan),
                       "skill": "path_comparison", "layer": "L2", "source_index": idx})

    # L2.3 Relaxation step
    if start_pos and len(solution) >= 2:
        r, c = start_pos
        dirs = {"up": (-1,0), "down": (1,0), "left": (0,-1), "right": (0,1)}
        mid = len(solution) // 2
        for step in solution[:mid]:
            dr, dc = dirs[step]
            r, c = r+dr, c+dc
        remaining = len(solution) - mid
        alt = remaining + rng.randint(0, 2)
        atoms.append({
            "question": f"At ({r},{c}), {mid} steps from start. Best remaining: {remaining} steps. "
                        f"Alternate route: {alt} steps. Is alternate shorter?",
            "answer": "Yes" if alt < remaining else "No" if alt > remaining else "Equal",
            "skill": "relaxation_step", "layer": "L2", "source_index": idx})

    return atoms
