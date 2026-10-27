"""
Does ARS predict merge compatibility?

For each of the 7 Phase-4 pair-merges, compute ARS-based features of the
task pair (task_A, task_B) and correlate with the 4B pair-merge damage.

Feature candidates (all computable at any probe scale):
  - min(ARS_A, ARS_B)                           "weakest link"
  - ARS_A * ARS_B                                "both tasks well-grounded"
  - |ARS_A - ARS_B|                              "asymmetry"
  - mean(L1_A, L1_B)                             "grounding level"
  - mean(L2_A, L2_B)                             "reasoning level"

Phase-4 damage (from experiments/rlvr_geometry_paper/reports/report_phase4_merge_scaling.md, measured at 4B):
  T1 sdk+leg: 18, T2 sdk+kk: 22, T3 ari+leg: 12.5, T4 ari+syl: 4.5,
  T5 sp+syl: 7, T6 sp+kk: 18, T7 cnt+ari: 24

Note: T1, T3 involve leg_counting (no atoms). These pairs will be excluded
unless we use a surrogate. Report with and without those pairs.
"""

import json, glob
from pathlib import Path
import numpy as np

BASE = "/code-fsx/sunguha/rlvr_experiments/experiments/rlvr_geometry_paper"

# Phase 4 Part 2 damage numbers at 4B:
DAMAGE_4B = {
    ("mini_sudoku", "leg_counting"):    18.0,  # T1
    ("mini_sudoku", "knights_knaves"):  22.0,  # T2
    ("basic_arithmetic", "leg_counting"): 12.5,  # T3
    ("basic_arithmetic", "syllogism"):   4.5,  # T4
    ("shortest_path", "syllogism"):      7.0,  # T5
    ("shortest_path", "knights_knaves"): 18.0, # T6
    ("countdown", "basic_arithmetic"):   24.0, # T7
}
PAIR_NAMES = {
    ("mini_sudoku", "leg_counting"): "T1_sdk_leg",
    ("mini_sudoku", "knights_knaves"): "T2_sdk_kk",
    ("basic_arithmetic", "leg_counting"): "T3_ari_leg",
    ("basic_arithmetic", "syllogism"): "T4_ari_syl",
    ("shortest_path", "syllogism"): "T5_sp_syl",
    ("shortest_path", "knights_knaves"): "T6_sp_kk",
    ("countdown", "basic_arithmetic"): "T7_cnt_ari",
}


def load_ars(scale):
    """Load all ARS probe results for a scale. Returns {task: {"L1": x, "L2": y, "all": z}}."""
    # First check experiments/rlvr_geometry_paper/paper/data/ars/{scale}/, then fall back to experiments/rlvr_geometry_paper/outputs/ars_{scale}_*.json
    out = {}
    for pattern in [f"{BASE}/paper/data/ars/{scale}/ars_{scale}_*.json",
                    f"{BASE}/outputs/ars_{scale}_*.json"]:
        for f in sorted(glob.glob(pattern)):
            d = json.load(open(f))
            # get task name from filename
            name = Path(f).stem.replace(f"ars_{scale}_", "").replace("_v2", "")
            if name == "arithmetic":
                name = "basic_arithmetic"
            per_atom = d.get("per_atom", {})
            l1 = [a["accuracy"] for n, a in per_atom.items() if n.startswith("L1")]
            l2 = [a["accuracy"] for n, a in per_atom.items() if n.startswith("L2")]
            all_ = [a["accuracy"] for a in per_atom.values()]
            out[name] = {
                "L1": float(np.mean(l1)) if l1 else None,
                "L2": float(np.mean(l2)) if l2 else None,
                "all": float(np.mean(all_)) if all_ else None,
                "n_atoms": len(per_atom),
            }
    # leg_counting has no atom defs — use base-task accuracy as surrogate
    leg_dir = f"{BASE}/paper/data/ars/{scale}/sft_base_leg_counting"
    for mfile in glob.glob(f"{leg_dir}/metrics_*.json"):
        m = json.load(open(mfile))
        for t, s in m.get("tasks", {}).items():
            if t == "leg_counting":
                acc = s.get("accuracy", 0)
                out["leg_counting"] = {
                    "L1": acc, "L2": acc, "all": acc, "n_atoms": 0,
                    "surrogate": "base_task_acc",
                }
    return out


def build_features(ars):
    """For each pair, produce feature dict. Returns rows = [(pair_name, features, damage), ...]."""
    rows = []
    for (ta, tb), dmg in DAMAGE_4B.items():
        pname = PAIR_NAMES[(ta, tb)]
        if ta not in ars or tb not in ars:
            rows.append((pname, {"missing": f"{ta if ta not in ars else tb}"}, dmg))
            continue
        a = ars[ta]; b = ars[tb]
        feats = {
            "ars_a": a["all"], "ars_b": b["all"],
            "min_ars": min(a["all"], b["all"]),
            "product_ars": a["all"] * b["all"],
            "abs_diff_ars": abs(a["all"] - b["all"]),
            "L1_mean": (a["L1"] + b["L1"]) / 2 if (a["L1"] is not None and b["L1"] is not None) else None,
            "L2_mean": (a["L2"] + b["L2"]) / 2 if (a["L2"] is not None and b["L2"] is not None) else None,
            "L1_min": min(a["L1"], b["L1"]) if (a["L1"] is not None and b["L1"] is not None) else None,
            "L2_min": min(a["L2"], b["L2"]) if (a["L2"] is not None and b["L2"] is not None) else None,
        }
        rows.append((pname, feats, dmg))
    return rows


def corr_report(rows, feature_keys, label=""):
    damages = np.array([d for _, _, d in rows])
    results = {}
    print(f"\n=== Correlations {label} (n={len(rows)}) ===")
    for key in feature_keys:
        vals = []
        keep_dmg = []
        for _, f, d in rows:
            v = f.get(key) if isinstance(f, dict) else None
            if v is not None:
                vals.append(v); keep_dmg.append(d)
        if len(vals) < 3:
            print(f"  {key:15s}: only n={len(vals)}, skip")
            continue
        r = float(np.corrcoef(vals, keep_dmg)[0, 1])
        results[key] = {"n": len(vals), "r": r}
        print(f"  {key:15s}  r={r:+.3f}  n={len(vals)}")
    return results


def main():
    all_results = {}
    for scale in ["3b", "4b", "8b"]:
        ars = load_ars(scale)
        if not ars:
            print(f"=== {scale}: no ARS data ===")
            continue
        print(f"\n=== {scale} ARS coverage ===")
        for task, a in sorted(ars.items()):
            print(f"  {task:20s}  ARS={a['all']:.3f}  (L1={a['L1']:.3f} L2={a['L2']:.3f} n={a['n_atoms']})")
        rows = build_features(ars)
        print(f"\n--- {scale} pair features ---")
        print(f"{'pair':15s} {'min_ars':>8s} {'prod':>8s} {'abs_diff':>8s} {'L1_mean':>8s} {'L2_mean':>8s} {'damage':>7s}")
        for pname, f, d in rows:
            if isinstance(f, dict) and "missing" in f:
                print(f"{pname:15s}  (missing ARS for: {f['missing']})  damage={d}")
                continue
            def fmt(v): return f'{v:.3f}' if v is not None else '  -  '
            print(f"{pname:15s} {fmt(f['min_ars']):>8s} {fmt(f['product_ars']):>8s} "
                  f"{fmt(f['abs_diff_ars']):>8s} {fmt(f['L1_mean']):>8s} {fmt(f['L2_mean']):>8s} {d:>7.1f}")
        keys = ["ars_a","ars_b","min_ars","product_ars","abs_diff_ars","L1_mean","L2_mean","L1_min","L2_min"]
        all_results[scale] = corr_report(rows, keys, label=f"{scale}")

    # Save
    out = Path(f"{BASE}/paper/data/analysis/ars_vs_damage.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
