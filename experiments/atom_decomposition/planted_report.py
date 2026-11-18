"""
Report for the planted-shortcut and frontier-ladder runs (scripts/vm/planted.sh).

    python -m experiments.atom_decomposition.planted_report --eval_root <R>/eval [--out report.json]

Per run and checkpoint, on the standard eval (500 puzzles, seed 9999, where "-1" is right ~2% of the time):
  acc        accuracy at T=1.0 (mean over samples); acc01 at T=0.1 where the eval has it
  planted    % of samples answering the planted constant, on puzzles whose answer is not that constant
             (shortcut adoption; ~0 if the model ignores the shortcut)
  expand     % of samples on big-product puzzles that break a product into partial products (heuristic regex,
             not yet checked against hand labels)
  tokens     median completion tokens
On the ladder (100 puzzles per digit level, 1-6 digits): accuracy per level and the digit level at which
accuracy crosses 50% (linear interpolation between levels; the "frontier").
"""

import argparse
import glob
import json
import os
import re
import statistics
from collections import defaultdict

from .eval_atoms import extract_post_think
from .planted import needs_bigmul

BREAK = re.compile(r"break(?:ing)? (?:it )?down|\(\s*\d+0+\s*[+\-]|\d+0{2,}\s*[+\-]\s*\d", re.I)


def norm_int(s):
    if s is None:
        return None
    m = re.findall(r"-?\d+", s.replace(",", ""))
    return m[-1] if m else None


def std_stats(path, constant):
    rows = [json.loads(l) for l in open(path)]
    out = {}
    for T in sorted({r["temperature"] for r in rows}):
        R = [r for r in rows if r["temperature"] == T]
        acc = [c for r in R for c in r["correct"]]
        planted, expand, toks = [], [], []
        for r in R:
            for text in r["responses"]:
                if str(r["answer"]) != constant:
                    planted.append(int(norm_int(extract_post_think(text)) == constant))
            toks += r["n_tokens"]
        out[f"T{T}"] = {"n": len(acc), "acc": 100 * sum(acc) / len(acc), "planted": 100 * statistics.mean(planted),
                        "tokens": statistics.median(toks)}
    return out


def expansion(path, questions):
    """% of T=1.0 samples on big-product puzzles whose reasoning breaks a product down."""
    hits = []
    for l in open(path):
        r = json.loads(l)
        if r["temperature"] != 1.0 or not needs_bigmul(questions.get(r["idx"], "")):
            continue
        for text in r["responses"]:
            think = text.split("</think>")[0]
            hits.append(int(bool(BREAK.search(think))))
    return 100 * statistics.mean(hits) if hits else None


def ladder_stats(path):
    by = defaultdict(list)
    for l in open(path):
        r = json.loads(l)
        if r["temperature"] == 1.0:
            by[r["level"]] += r["correct"]
    levels = sorted(by, key=lambda s: int(re.sub(r"\D", "", s) or 0))
    acc = {lv: 100 * sum(by[lv]) / len(by[lv]) for lv in levels}
    frontier = None
    xs = [int(re.sub(r"\D", "", lv)) for lv in levels]
    for (x0, l0), (x1, l1) in zip(zip(xs, levels), zip(xs[1:], levels[1:])):
        a0, a1 = acc[l0], acc[l1]
        if a0 >= 50 > a1:
            frontier = x0 + (a0 - 50) / (a0 - a1) * (x1 - x0)
            break
    if frontier is None and levels:
        frontier = xs[-1] if acc[levels[-1]] >= 50 else (xs[0] if acc[levels[0]] < 50 else None)
    return {"acc": acc, "frontier_digits": frontier}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--eval_root", required=True)
    p.add_argument("--constant", default="-1")
    p.add_argument("--out", default=None)
    a = p.parse_args()

    import reasoning_gym as rg
    ds = rg.create_dataset("basic_arithmetic", size=500, seed=9999)
    questions = {i: ds[i]["metadata"]["expression"] for i in range(500)}

    report = {}
    for path in sorted(glob.glob(os.path.join(a.eval_root, "*", "step*", "*", "full_results.jsonl"))):
        kind = os.path.basename(os.path.dirname(path))            # std | ladder
        step_dir = os.path.dirname(os.path.dirname(path))
        run, step = os.path.basename(os.path.dirname(step_dir)), int(os.path.basename(step_dir)[4:])
        d = report.setdefault(run, {}).setdefault(step, {})
        if kind == "std":
            d.update(std_stats(path, a.constant))
            d["expand"] = expansion(path, questions)
        elif kind == "ladder":
            d["ladder"] = ladder_stats(path)

    print(f"{'run':18s} {'step':>4s} {'acc':>6s} {'acc01':>6s} {'-1 %':>6s} {'expand':>6s} {'tokens':>6s}  "
          f"ladder acc by digits (frontier)")
    for run in sorted(report):
        for step in sorted(report[run]):
            d = report[run][step]
            t1, t01 = d.get("T1.0", {}), d.get("T0.1", {})
            f = lambda v: f"{v:6.1f}" if isinstance(v, (int, float)) else f"{'-':>6s}"
            lad = d.get("ladder")
            lad_s = (" ".join(f"{v:3.0f}" for v in lad["acc"].values()) +
                     f" ({lad['frontier_digits']:.2f})" if lad and lad["frontier_digits"] is not None else
                     (" ".join(f"{v:3.0f}" for v in lad["acc"].values()) if lad else ""))
            print(f"{run:18s} {step:4d} {f(t1.get('acc'))} {f(t01.get('acc'))} {f(t1.get('planted'))} "
                  f"{f(d.get('expand'))} {f(t1.get('tokens'))}  {lad_s}")
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(report, fh, indent=2)


if __name__ == "__main__":
    main()
