"""Tower of Hanoi atomic skill extractor (T.L1.1-T.L1.2, T.L2.1-T.L2.3)."""

import random


def extract_atoms(instance: dict, rng: random.Random) -> list[dict]:
    meta = instance["metadata"]
    num_disks = meta["num_disks"]
    num_pegs = meta["num_pegs"]
    start_peg = meta["start_peg"]
    target_peg = meta["target_peg"]
    idx = meta.get("source_index", 0)
    atoms = []

    # Initial state: all disks on start_peg, largest at bottom
    disks = list(range(1, num_disks + 1))  # 1=smallest, N=largest

    # ── L1.1 State reading ────────────────────────────────────────
    atoms.append({
        "question": f"In this Tower of Hanoi puzzle with {num_disks} disks and {num_pegs} pegs, "
                    f"which peg are all disks initially on?",
        "answer": f"Peg {start_peg}",
        "skill": "state_reading", "layer": "L1", "source_index": idx,
    })
    atoms.append({
        "question": f"Which peg should all disks be moved to?",
        "answer": f"Peg {target_peg}",
        "skill": "state_reading", "layer": "L1", "source_index": idx,
    })
    atoms.append({
        "question": f"How many disks are there?",
        "answer": str(num_disks),
        "skill": "state_reading", "layer": "L1", "source_index": idx,
    })

    # ── L1.2 Disk size parsing ────────────────────────────────────
    atoms.append({
        "question": f"There are {num_disks} disks numbered 1 through {num_disks}. "
                    f"Which is the smallest disk?",
        "answer": "Disk 1",
        "skill": "disk_size_parsing", "layer": "L1", "source_index": idx,
    })
    atoms.append({
        "question": f"Is disk {num_disks} larger or smaller than disk 1?",
        "answer": "Larger",
        "skill": "disk_size_parsing", "layer": "L1", "source_index": idx,
    })
    if num_disks >= 3:
        a, b = rng.sample(disks, 2)
        atoms.append({
            "question": f"Which is larger, disk {a} or disk {b}?",
            "answer": f"Disk {max(a, b)}",
            "skill": "disk_size_parsing", "layer": "L1", "source_index": idx,
        })

    # ── L2.1 Move legality ────────────────────────────────────────
    # Simulate some states and ask about move legality
    # Initial state: ask if moving disk 1 (top) is legal
    atoms.append({
        "question": f"All {num_disks} disks are on Peg {start_peg} (disk 1 on top). "
                    f"Can you move disk 1 to Peg {target_peg}?",
        "answer": "Yes, disk 1 is the top disk and Peg " + str(target_peg) + " is empty",
        "skill": "move_legality", "layer": "L2", "source_index": idx,
    })
    atoms.append({
        "question": f"All {num_disks} disks are on Peg {start_peg}. "
                    f"Can you move disk {num_disks} (bottom) directly?",
        "answer": "No, disk " + str(num_disks) + " is not the top disk",
        "skill": "move_legality", "layer": "L2", "source_index": idx,
    })
    # Legality: can a larger disk go on a smaller?
    if num_disks >= 2:
        atoms.append({
            "question": f"Disk 1 is on Peg {target_peg}. Can you place disk 2 on Peg {target_peg}?",
            "answer": "No, disk 2 is larger than disk 1",
            "skill": "move_legality", "layer": "L2", "source_index": idx,
        })

    # ── L2.2 Top-disc identification ──────────────────────────────
    atoms.append({
        "question": f"All disks are on Peg {start_peg}, stacked 1 (top) to {num_disks} (bottom). "
                    f"Which disk is on top?",
        "answer": "Disk 1",
        "skill": "top_disc_identification", "layer": "L2", "source_index": idx,
    })
    # Partial state: remove top disk
    if num_disks >= 2:
        atoms.append({
            "question": f"Disk 1 was moved away from Peg {start_peg}. "
                        f"Disks 2 through {num_disks} remain. Which disk is now on top?",
            "answer": "Disk 2",
            "skill": "top_disc_identification", "layer": "L2", "source_index": idx,
        })

    # ── L2.3 Goal distance ────────────────────────────────────────
    atoms.append({
        "question": f"All {num_disks} disks are on Peg {start_peg}. Target is Peg {target_peg}. "
                    f"How many disks are misplaced (not on the target peg)?",
        "answer": str(num_disks),
        "skill": "goal_distance", "layer": "L2", "source_index": idx,
    })
    # Optimal solution length for Tower of Hanoi is 2^n - 1
    optimal = 2**num_disks - 1
    atoms.append({
        "question": f"The minimum number of moves to solve Tower of Hanoi with {num_disks} disks "
                    f"is 2^{num_disks} - 1. What is that?",
        "answer": str(optimal),
        "skill": "goal_distance", "layer": "L2", "source_index": idx,
    })

    return atoms
