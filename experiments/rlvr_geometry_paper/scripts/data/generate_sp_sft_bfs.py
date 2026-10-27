"""
Generate SFT training data for shortest_path with BFS-computed solutions.

For each problem, runs BFS to find the shortest path, then generates a
step-by-step reasoning trace in <think>/<answer> format.

Usage:
    python experiments/rlvr_geometry_paper/scripts/data/generate_sp_sft_bfs.py --seed 42 --num_samples 5000 --output sft_data_42.jsonl
"""

import argparse
import json
import random
import re
from collections import deque

try:
    import reasoning_gym
except ImportError:
    raise ImportError("reasoning-gym required")


def bfs_solve(matrix):
    """BFS to find shortest path. Returns (path_directions, path_coords) or (None, None)."""
    rows, cols = len(matrix), len(matrix[0])
    start = goal = None
    for r in range(rows):
        for c in range(cols):
            if matrix[r][c] == '*':
                start = (r, c)
            elif matrix[r][c] == '#':
                goal = (r, c)

    if not start or not goal:
        return None, None

    queue = deque([(start[0], start[1], [], [])])
    visited = {start}
    dir_map = [('up', -1, 0), ('down', 1, 0), ('left', 0, -1), ('right', 0, 1)]

    while queue:
        r, c, dirs, coords = queue.popleft()
        if (r, c) == goal:
            return dirs, coords
        for name, dr, dc in dir_map:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited and matrix[nr][nc] != 'X':
                visited.add((nr, nc))
                queue.append((nr, nc, dirs + [name], coords + [(nr, nc)]))

    return None, None


def format_grid(matrix):
    return "\n".join(" ".join(row) for row in matrix)


def find_char(matrix, char):
    for r, row in enumerate(matrix):
        for c, cell in enumerate(row):
            if cell == char:
                return r, c
    return None, None


def get_neighbors_description(matrix, r, c):
    """Describe what's around a cell."""
    rows, cols = len(matrix), len(matrix[0])
    dirs = [('up', -1, 0), ('down', 1, 0), ('left', 0, -1), ('right', 0, 1)]
    parts = []
    for name, dr, dc in dirs:
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            cell = matrix[nr][nc]
            if cell == 'X':
                parts.append(f"{name} is blocked (X)")
            else:
                parts.append(f"{name} leads to ({nr},{nc}) which is {cell}")
        else:
            parts.append(f"{name} is out of bounds")
    return ". ".join(parts)


def make_reasoning_trace(matrix, path_dirs, path_coords, is_feasible):
    """Generate a step-by-step reasoning trace."""
    sr, sc = find_char(matrix, '*')
    gr, gc = find_char(matrix, '#')
    rows, cols = len(matrix), len(matrix[0])

    lines = []
    lines.append(f"The grid is {rows}x{cols}. Start (*) is at ({sr},{sc}). Goal (#) is at ({gr},{gc}).")
    lines.append("")

    if not is_feasible:
        lines.append(f"Let me check if a path exists from ({sr},{sc}) to ({gr},{gc}).")
        nbr = get_neighbors_description(matrix, sr, sc)
        lines.append(f"From start ({sr},{sc}): {nbr}.")
        lines.append("")
        lines.append("After exploring all reachable cells, the goal is not reachable. No path exists.")
        trace = "\n".join(lines)
        return f"<think>\n{trace}\n</think>\n<answer>infeasible</answer>"

    lines.append(f"I need to find the shortest path from ({sr},{sc}) to ({gr},{gc}).")
    lines.append("")

    # Show first few neighbors from start
    nbr = get_neighbors_description(matrix, sr, sc)
    lines.append(f"From start ({sr},{sc}): {nbr}.")
    lines.append("")

    # Trace the path
    lines.append("Tracing the shortest path:")
    cr, cc = sr, sc
    for i, (d, (nr, nc)) in enumerate(zip(path_dirs, path_coords)):
        cell = matrix[nr][nc]
        cell_desc = "goal #" if cell == '#' else cell
        lines.append(f"  Step {i+1}: {d} from ({cr},{cc}) to ({nr},{nc}) — {cell_desc}")
        cr, cc = nr, nc

    lines.append("")
    lines.append(f"Path found with {len(path_dirs)} moves: {' '.join(path_dirs)}")

    # Verify
    lines.append("")
    coord_chain = f"({sr},{sc})" + "".join(f"→({r},{c})" for r, c in path_coords)
    lines.append(f"Verify: {coord_chain}. Valid shortest path.")

    trace = "\n".join(lines)
    answer = " ".join(path_dirs)
    return f"<think>\n{trace}\n</think>\n<answer>{answer}</answer>"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_samples", type=int, default=5000)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    print(f"Generating {args.num_samples} shortest_path SFT samples (seed={args.seed})...")

    dataset = reasoning_gym.create_dataset(
        'shortest_path',
        size=args.num_samples,
        seed=args.seed,
    )

    samples = []
    feasible_count = 0
    infeasible_count = 0

    for item in dataset:
        matrix = item['metadata']['matrix']
        question = item['question']
        gt_answer = str(item['answer'])

        path_dirs, path_coords = bfs_solve(matrix)
        is_feasible = path_dirs is not None

        if is_feasible:
            feasible_count += 1
        else:
            infeasible_count += 1

        response = make_reasoning_trace(matrix, path_dirs, path_coords, is_feasible)

        samples.append({
            "messages": [
                {"role": "system", "content": "You are a helpful assistant. You must always think step-by-step. Put your thinking process inside <think> and </think> tags, and put your final answer inside <answer> and </answer> tags."},
                {"role": "user", "content": question},
                {"role": "assistant", "content": response},
            ]
        })

    with open(args.output, 'w') as f:
        for s in samples:
            f.write(json.dumps(s) + '\n')

    print(f"Wrote {len(samples)} samples to {args.output}")
    print(f"  Feasible: {feasible_count}, Infeasible: {infeasible_count}")


if __name__ == "__main__":
    main()
