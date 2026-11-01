"""
Build the Phase C atom set: reasoning-gym instances, their full-task prompts, and one atom per
solver step for every candidate graph.

Every trace is checked against reasoning-gym's own answer before it is written, so a solver bug
fails the build instead of producing wrong gold answers.

Usage (from the repo root):
    python -m experiments.atom_decomposition.build_atoms --n 500 --seed 2027 \
        --out_dir experiments/atom_decomposition/results/atoms_seed2027

Writes:
    instances.jsonl  one row per instance: task, idx, question, answer, difficulty features,
                     and per graph the node counts (n_L2_total before any subsampling)
    atoms.jsonl      one row per (instance, graph, node): question, gold answer, checker, deps;
                     `key` is shared by identical questions across graphs, so each is asked once
    manifest.json    arguments, reasoning-gym version, counts
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
import random

import reasoning_gym as rg

from .tasks import TASKS
from .tasks.arithmetic import final_value
from .tasks.gcd import final_value as gcd_value
from .tasks.maze import bfs_layers
from .tasks.mini_sudoku import trace_backtracking, trace_propagation


def check_trace(task: str, sample: dict) -> None:
    """Raise if a solver disagrees with reasoning-gym's answer for this instance."""
    m = sample["metadata"]
    if task == "basic_arithmetic":
        assert final_value(sample) == int(sample["answer"]), (m["expression"], sample["answer"])
    elif task == "mini_sudoku":
        for solve in (trace_propagation, trace_backtracking):
            _, grid, solved = solve(m["puzzle"])
            if solve is trace_backtracking or solved:
                assert solved and grid == m["solution"], (solve.__name__, m["puzzle"])
    elif task == "maze":
        assert len(bfs_layers(m)) - 1 == m["shortest_path_length"]
    elif task == "gcd":
        for graph in ("euclid", "factorisation"):
            assert gcd_value(sample, graph) == int(sample["answer"]), (graph, m["numbers"])


def build(tasks, n, seed, max_l2, out_dir, task_configs=None):
    task_configs = task_configs or {}
    os.makedirs(out_dir, exist_ok=True)
    rng = random.Random(seed)
    counts = {}
    with open(os.path.join(out_dir, "instances.jsonl"), "w") as fi, \
         open(os.path.join(out_dir, "atoms.jsonl"), "w") as fa:
        for task in tasks:
            mod = TASKS[task]
            ds = rg.create_dataset(task, size=n, seed=seed, **task_configs.get(task, {}))
            c = counts.setdefault(task, {"instances": 0, "atoms": 0, "unique_questions": set()})
            for idx in range(n):
                sample = ds[idx]
                check_trace(task, sample)
                row = {"task": task, "idx": idx, "question": sample["question"],
                       "answer": sample["answer"], "metadata": sample["metadata"],
                       "difficulty": mod.difficulty_features(sample), "graphs": {}}
                for graph in mod.GRAPHS:
                    nodes = mod.trace(sample, graph)
                    l1 = [x for x in nodes if x.layer == "L1"]
                    l2 = [x for x in nodes if x.layer == "L2"]
                    info = {"n_L1": len(l1), "n_L2_total": len(l2)}
                    if task == "mini_sudoku" and graph == "propagation":
                        info["solved_by_singles"] = trace_propagation(sample["metadata"]["puzzle"])[2]
                    if max_l2 and len(l2) > max_l2:
                        keep = set(rng.sample(range(len(l2)), max_l2))
                        l2 = [x for i, x in enumerate(l2) if i in keep]
                    info["n_L2_asked"] = len(l2)
                    row["graphs"][graph] = info
                    for x in l1 + l2:
                        key = hashlib.sha1(f"{task}|{idx}|{x.question}".encode()).hexdigest()[:16]
                        fa.write(json.dumps({
                            "task": task, "idx": idx, "graph": graph, "node": x.id,
                            "layer": x.layer, "type": x.type, "deps": x.deps,
                            "question": x.question, "answer": x.answer, "check": x.check,
                            "key": key,
                        }) + "\n")
                        c["atoms"] += 1
                        c["unique_questions"].add(key)
                fi.write(json.dumps(row) + "\n")
                c["instances"] += 1
    for c in counts.values():
        c["unique_questions"] = len(c["unique_questions"])
    manifest = {"tasks": tasks, "n": n, "seed": seed, "max_l2": max_l2, "task_configs": task_configs,
                "reasoning_gym": importlib.metadata.version("reasoning-gym"), "counts": counts}
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", default=",".join(TASKS))
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--seed", type=int, default=2027)
    p.add_argument("--max_l2", type=int, default=40,
                   help="cap on L2 atoms asked per (instance, graph); a seeded random subset")
    p.add_argument("--task_configs", default="{}",
                   help='JSON of reasoning-gym config overrides per task, e.g. '
                        '\'{"maze": {"min_dist": 1, "max_dist": 6}}\'; default = the training configs')
    p.add_argument("--out_dir", required=True)
    a = p.parse_args()
    m = build(a.tasks.split(","), a.n, a.seed, a.max_l2, a.out_dir, json.loads(a.task_configs))
    print(json.dumps(m["counts"], indent=2))


if __name__ == "__main__":
    main()
