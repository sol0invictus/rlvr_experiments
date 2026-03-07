"""Graph coloring atomic skill extractor (G.L1.1-G.L1.2, G.L2.1-G.L2.3)."""

import random


def extract_atoms(instance: dict, rng: random.Random) -> list[dict]:
    meta = instance["metadata"]
    puzzle = meta["puzzle"]
    vertices = puzzle["vertices"]             # e.g. [0, 1, 2, ...]
    edges = puzzle["edges"]                   # e.g. [(0,1), (0,7), ...]
    num_colors = puzzle["num_colors"]
    possible_answer = meta.get("possible_answer", {})  # {0:1, 1:2, ...}
    idx = meta.get("source_index", 0)
    atoms = []

    # Build adjacency list
    adj = {v: [] for v in vertices}
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    # ── L1.1 Graph reading ────────────────────────────────────────
    for node in rng.sample(vertices, min(5, len(vertices))):
        neighbors = sorted(adj[node])
        atoms.append({
            "question": f"In this graph, what are the neighbors of node {node}?",
            "answer": str(neighbors) if neighbors else "none",
            "skill": "graph_reading", "layer": "L1", "source_index": idx,
        })

    atoms.append({
        "question": f"How many vertices does this graph have?",
        "answer": str(len(vertices)),
        "skill": "graph_reading", "layer": "L1", "source_index": idx,
    })
    atoms.append({
        "question": f"How many edges does this graph have?",
        "answer": str(len(edges)),
        "skill": "graph_reading", "layer": "L1", "source_index": idx,
    })

    # ── L1.2 Coloring reading (from possible_answer) ─────────────
    if possible_answer:
        for node in rng.sample(list(possible_answer.keys()), min(4, len(possible_answer))):
            atoms.append({
                "question": f"In the proposed coloring, what color is assigned to node {node}?",
                "answer": str(possible_answer[node]),
                "skill": "coloring_reading", "layer": "L1", "source_index": idx,
            })

    # ── L2.1 Edge validity ────────────────────────────────────────
    edge_set = {(min(u, v), max(u, v)) for u, v in edges}
    # Sample some existing edges and non-edges
    for _ in range(4):
        u, v = rng.sample(vertices, 2)
        key = (min(u, v), max(u, v))
        has_edge = key in edge_set
        atoms.append({
            "question": f"Is there an edge between node {u} and node {v}?",
            "answer": "Yes" if has_edge else "No",
            "skill": "edge_validity", "layer": "L2", "source_index": idx,
        })

    # ── L2.2 Local constraint check ───────────────────────────────
    if possible_answer:
        colors = list(range(1, num_colors + 1))
        for node in rng.sample(vertices, min(4, len(vertices))):
            neighbor_colors = {possible_answer.get(n) for n in adj[node]
                               if n in possible_answer}
            test_color = rng.choice(colors)
            ok = test_color not in neighbor_colors
            atoms.append({
                "question": f"Node {node}'s neighbors have colors {sorted(neighbor_colors)}. "
                            f"Can node {node} be assigned color {test_color}?",
                "answer": "Yes" if ok else f"No, a neighbor already has color {test_color}",
                "skill": "local_constraint_check", "layer": "L2", "source_index": idx,
            })

    # ── L2.3 Conflict detection ───────────────────────────────────
    if possible_answer:
        conflicts = []
        for u, v in edges:
            if u in possible_answer and v in possible_answer:
                if possible_answer[u] == possible_answer[v]:
                    conflicts.append((u, v))
        atoms.append({
            "question": f"In the proposed coloring, do any adjacent nodes share the same color?",
            "answer": f"No conflicts" if not conflicts else f"Yes: {conflicts}",
            "skill": "conflict_detection", "layer": "L2", "source_index": idx,
        })
        # Check specific edges
        for u, v in rng.sample(edges, min(3, len(edges))):
            if u in possible_answer and v in possible_answer:
                same = possible_answer[u] == possible_answer[v]
                atoms.append({
                    "question": f"Edge ({u}, {v}): node {u} has color {possible_answer[u]}, "
                                f"node {v} has color {possible_answer[v]}. Is this a conflict?",
                    "answer": "Yes, they share a color" if same else "No, different colors",
                    "skill": "conflict_detection", "layer": "L2", "source_index": idx,
                })

    return atoms
