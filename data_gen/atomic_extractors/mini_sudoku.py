"""Mini Sudoku atomic skill extractor (S.L1.1-S.L1.3, S.L2.1-S.L2.4)."""

import random


def extract_atoms(instance: dict, rng: random.Random) -> list[dict]:
    meta = instance["metadata"]
    puzzle = meta["puzzle"]       # 4x4 list of ints, 0 = empty
    solution = meta["solution"]   # 4x4 list of ints, complete
    idx = meta.get("source_index", 0)
    atoms = []
    size = len(puzzle)

    # ── L1.1 Grid reading ─────────────────────────────────────────
    for _ in range(4):
        r = rng.randint(0, size - 1)
        c = rng.randint(0, size - 1)
        val = puzzle[r][c]
        atoms.append({
            "question": f"In this {size}x{size} Sudoku puzzle, what value is at row {r+1}, column {c+1}?",
            "answer": str(val) if val != 0 else "empty",
            "skill": "grid_reading", "layer": "L1", "source_index": idx,
        })

    # ── L1.2 Empty cell identification ────────────────────────────
    empties = [(r, c) for r in range(size) for c in range(size) if puzzle[r][c] == 0]
    atoms.append({
        "question": f"How many empty cells are in this puzzle?",
        "answer": str(len(empties)),
        "skill": "empty_cell_identification", "layer": "L1", "source_index": idx,
    })
    if empties:
        sample_empty = rng.choice(empties)
        atoms.append({
            "question": f"Is the cell at row {sample_empty[0]+1}, column {sample_empty[1]+1} empty?",
            "answer": "Yes",
            "skill": "empty_cell_identification", "layer": "L1", "source_index": idx,
        })

    # ── L1.3 Region boundary recognition ──────────────────────────
    # For 4x4 mini sudoku, boxes are 2x2
    box_size = 2
    for _ in range(2):
        r = rng.randint(0, size - 1)
        c = rng.randint(0, size - 1)
        box_r = r // box_size
        box_c = c // box_size
        atoms.append({
            "question": f"In a {size}x{size} Mini Sudoku with {box_size}x{box_size} boxes, "
                        f"which box does cell (row {r+1}, col {c+1}) belong to?",
            "answer": f"Box ({box_r+1}, {box_c+1})",
            "skill": "region_boundary", "layer": "L1", "source_index": idx,
        })

    # ── L2.1 Row constraint check ─────────────────────────────────
    for r in range(size):
        vals_in_row = [puzzle[r][c] for c in range(size) if puzzle[r][c] != 0]
        atoms.append({
            "question": f"What values are already placed in row {r+1}?",
            "answer": str(sorted(vals_in_row)) if vals_in_row else "none",
            "skill": "row_constraint_check", "layer": "L2", "source_index": idx,
        })

    # ── L2.2 Column constraint check ──────────────────────────────
    for c in range(size):
        vals_in_col = [puzzle[r][c] for r in range(size) if puzzle[r][c] != 0]
        atoms.append({
            "question": f"What values are already placed in column {c+1}?",
            "answer": str(sorted(vals_in_col)) if vals_in_col else "none",
            "skill": "col_constraint_check", "layer": "L2", "source_index": idx,
        })

    # ── L2.3 Box constraint check ─────────────────────────────────
    for br in range(size // box_size):
        for bc in range(size // box_size):
            vals = []
            for r in range(br * box_size, (br + 1) * box_size):
                for c in range(bc * box_size, (bc + 1) * box_size):
                    if puzzle[r][c] != 0:
                        vals.append(puzzle[r][c])
            atoms.append({
                "question": f"What values are already in box ({br+1}, {bc+1})?",
                "answer": str(sorted(vals)) if vals else "none",
                "skill": "box_constraint_check", "layer": "L2", "source_index": idx,
            })

    # ── L2.4 Candidate computation ────────────────────────────────
    all_vals = set(range(1, size + 1))
    for r, c in rng.sample(empties, min(4, len(empties))):
        row_vals = {puzzle[r][cc] for cc in range(size) if puzzle[r][cc] != 0}
        col_vals = {puzzle[rr][c] for rr in range(size) if puzzle[rr][c] != 0}
        br, bc = r // box_size, c // box_size
        box_vals = set()
        for rr in range(br * box_size, (br + 1) * box_size):
            for cc in range(bc * box_size, (bc + 1) * box_size):
                if puzzle[rr][cc] != 0:
                    box_vals.add(puzzle[rr][cc])
        candidates = sorted(all_vals - row_vals - col_vals - box_vals)
        atoms.append({
            "question": f"What values can go in the empty cell at row {r+1}, column {c+1}? "
                        f"(Consider row, column, and box constraints.)",
            "answer": str(candidates),
            "skill": "candidate_computation", "layer": "L2", "source_index": idx,
        })

    return atoms
