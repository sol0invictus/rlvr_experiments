"""
Summarise curriculum-study evals: one row per (run, checkpoint), then mean +- sd over seeds.

Expects eval_atoms.py full-mode output (k samples at T = 1.0 plus 1 at T = 0.1 per puzzle):
    <eval_root>/<run>/<step>/full_results.jsonl      e.g. eval/arith_A_seed1/step100/
    <eval_root>/sft/step000/full_results.jsonl       the shared starting point
A run named <arm>_seed<k> is grouped with the other seeds of <arm>.

Per row: accuracy at T = 0.1 with a 95% Wilson interval, mean accuracy at T = 1.0, unbiased pass@k,
T = 1.0 accuracy by number of terms in the expression, and median response tokens.
If <run_root>/<run>/checkpoint-*/trainer_state.json exists, also the share of GRPO groups with
zero reward spread (all 8 samples right or all wrong), which give no gradient.

Usage (from the repo root):
    python -m experiments.atom_decomposition.curriculum_report --eval_root E [--run_root R] [--out report.json]
"""

import argparse
import glob
import json
import math
import os
import re
import statistics
from collections import defaultdict

PASS_K = (1, 4, 16)


def pass_at_k(n, c, k):
    """Unbiased pass@k from n samples with c correct (Chen et al. 2021)."""
    if k > n:
        return None
    if n - c < k:
        return 1.0
    return 1.0 - math.prod((n - c - i) / (n - i) for i in range(k))


def wilson(successes, n, z=1.96):
    if n == 0:
        return (None, None)
    p = successes / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(c - h, 4), round(c + h, 4))


def n_terms(question):
    return len(re.findall(r"\d+", question))


def summarise(path, questions=None):
    t01, t1 = [], []
    for l in open(path):
        r = json.loads(l)
        (t01 if r["temperature"] == 0.1 else t1).append(r)
    out = {}
    if t01:
        s = sum(r["correct"][0] for r in t01)
        out["acc_T0.1"] = round(s / len(t01), 4)
        out["acc_T0.1_ci95"] = wilson(s, len(t01))
        out["n"] = len(t01)
    if t1:
        out["acc_T1.0"] = round(statistics.mean(c for r in t1 for c in r["correct"]), 4)
        n = min(len(r["correct"]) for r in t1)
        for k in PASS_K:
            v = [pass_at_k(len(r["correct"]), sum(r["correct"]), k) for r in t1]
            if None not in v:
                out[f"pass@{k}"] = round(statistics.mean(v), 4)
        out["k"] = n
        out["median_tokens"] = statistics.median(t for r in t1 for t in r["n_tokens"])
        if questions:
            by = defaultdict(list)
            for r in t1:
                by[n_terms(questions[r["idx"]])] += r["correct"]
            out["acc_T1.0_by_terms"] = {t: round(statistics.mean(v), 3) for t, v in sorted(by.items())}
    return out


def zero_spread(run_dir):
    """Share of logged steps' groups with zero reward spread, from the last checkpoint's trainer_state."""
    states = sorted(glob.glob(os.path.join(run_dir, "checkpoint-*", "trainer_state.json")),
                    key=lambda p: int(p.split("checkpoint-")[1].split("/")[0]))
    if not states:
        return None
    hist = json.load(open(states[-1]))["log_history"]
    v = [h["frac_reward_zero_std"] for h in hist if "frac_reward_zero_std" in h]
    return round(statistics.mean(v), 3) if v else None


def load_questions(seed, n):
    """Puzzle texts for the by-length table (needs reasoning-gym); None if unavailable."""
    try:
        import reasoning_gym as rg
    except ImportError:
        return None
    ds = rg.create_dataset("basic_arithmetic", size=n, seed=seed)
    return [ds[i]["question"] for i in range(n)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--eval_root", required=True)
    p.add_argument("--run_root", default=None, help="training output dirs, for the zero-spread share")
    p.add_argument("--eval_seed", type=int, default=9999)
    p.add_argument("--eval_n", type=int, default=500)
    p.add_argument("--out", default=None)
    a = p.parse_args()
    questions = load_questions(a.eval_seed, a.eval_n)
    rows = {}
    for path in sorted(glob.glob(os.path.join(a.eval_root, "*", "step*", "full_results.jsonl"))):
        run, step = path.split(os.sep)[-3:-1]
        rows[(run, step)] = summarise(path, questions)
        if a.run_root and run != "sft":
            rows[(run, step)]["zero_spread_groups"] = zero_spread(os.path.join(a.run_root, run))
    arms = defaultdict(lambda: defaultdict(list))
    for (run, step), r in rows.items():
        arms[re.sub(r"_seed\d+$", "", run)][step].append(r)
    agg = {}
    for arm, steps in arms.items():
        for step, rs in sorted(steps.items()):
            m = {}
            for key in ("acc_T0.1", "acc_T1.0") + tuple(f"pass@{k}" for k in PASS_K) + ("median_tokens",):
                v = [r[key] for r in rs if r.get(key) is not None]
                if v:
                    m[key] = [round(statistics.mean(v), 4), round(statistics.stdev(v), 4) if len(v) > 1 else None]
            m["seeds"] = len(rs)
            agg[f"{arm}/{step}"] = m
    print(f"{'run/step':30s} {'T0.1 (95% CI)':22s} {'T1.0':>6s} " + " ".join(f"{'p@' + str(k):>6s}" for k in PASS_K)
          + f" {'tok':>5s} {'zero':>5s}")
    for (run, step), r in sorted(rows.items()):
        ci = r.get("acc_T0.1_ci95", (None, None))
        print(f"{run + '/' + step:30s} {r.get('acc_T0.1', float('nan')):.3f} ({ci[0]}-{ci[1]})".ljust(53)
              + f" {r.get('acc_T1.0', float('nan')):6.3f} "
              + " ".join(f"{r.get(f'pass@{k}', float('nan')):6.3f}" for k in PASS_K)
              + f" {r.get('median_tokens', float('nan')):5.0f} {r.get('zero_spread_groups') or float('nan'):5.2f}")
    print("\nmean +- sd over seeds:")
    for k, m in agg.items():
        print(f"  {k:28s} " + ", ".join(f"{x}={v[0]}" + (f"+-{v[1]}" if v[1] is not None else "") for x, v in m.items()
                                        if x != "seeds") + f"  (seeds={m['seeds']})")
    if a.out:
        json.dump({"rows": {f"{r}/{s}": v for (r, s), v in rows.items()}, "by_arm": agg}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
