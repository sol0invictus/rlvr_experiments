"""
Rescore saved eval_atoms responses with the current scoring rules (no generation).

Reads <results_dir>/manifest.json to rebuild each instance (full mode: reasoning-gym with the run's
seed, n and task configs; atoms mode: the build's instances.jsonl), then rewrites full_results.jsonl and
atom_results.jsonl in place with every field in eval_atoms.RULES plus rg_score. The previous files are
kept as *.jsonl.orig the first time.

Usage (from the repo root):
    python -m experiments.atom_decomposition.rescore --results_dir R [--build_dir B]
"""

import argparse
import json
import os
import shutil

import reasoning_gym as rg

from .eval_atoms import RULES, FullScorer, score_atom


def _entries(results_dir, build_dir=None):
    args = json.load(open(os.path.join(results_dir, "manifest.json")))["args"]
    if args["mode"] == "atoms":
        path = os.path.join(build_dir or args["build_dir"], "instances.jsonl")
        return {(r["task"], r["idx"]): r for r in map(json.loads, open(path))}
    cfgs = json.loads(args["task_configs"])
    out = {}
    for task in args["tasks"].split(","):
        ds = rg.create_dataset(task, size=args["n"], seed=args["seed"], **cfgs.get(task, {}))
        out.update({(task, i): ds[i] for i in range(args["n"])})
    return out


def _rewrite(path, fn):
    if not os.path.exists(path):
        return 0
    if not os.path.exists(path + ".orig"):
        shutil.copy(path, path + ".orig")
    rows = [fn(json.loads(l)) for l in open(path + ".orig")]
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return len(rows)


def rescore(results_dir, build_dir=None):
    entries = _entries(results_dir, build_dir)
    scorer = FullScorer()

    def full(r):
        e = entries[(r["task"], r["idx"])]
        scored = [scorer(r["task"], e, t) for t in r["responses"]]
        r.update({f: [x[f] for x in scored] for f in list(RULES) + ["rg_score"]})
        return r

    def atom(r):
        scored = [score_atom(r["check"], r["answer"], t) for t in r["responses"]]
        r.update({f: [x[f] for x in scored] for f in RULES})
        return r

    n_full = _rewrite(os.path.join(results_dir, "full_results.jsonl"), full)
    n_atom = _rewrite(os.path.join(results_dir, "atom_results.jsonl"), atom)
    return n_full, n_atom


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", required=True)
    p.add_argument("--build_dir", default=None, help="override the build path recorded in the manifest")
    a = p.parse_args()
    print(dict(zip(("full_rows", "atom_rows"), rescore(a.results_dir, a.build_dir))))


if __name__ == "__main__":
    main()
