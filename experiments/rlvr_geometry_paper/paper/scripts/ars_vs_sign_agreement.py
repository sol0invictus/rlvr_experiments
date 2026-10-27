"""
Does 3B L1_mean add information beyond 3B sign_agreement?
  - Correlation between L1_mean and sign_agreement (within 3B)
  - Partial correlation of each with damage controlling for the other
  - Multiple regression: damage ~ L1_mean + sign_agreement
"""
import json, glob
import numpy as np
from pathlib import Path

BASE = "/code-fsx/sunguha/rlvr_experiments/experiments/rlvr_geometry_paper"

# 3B sign-agreement from existing data
sa_3b_list = json.load(open(f"{BASE}/results/weight_analysis/3b/sign_agreement.json"))
sa_3b = {row["pair"]: row for row in sa_3b_list}

# L1_mean (just computed) — re-derive from ARS data
def load_ars(scale):
    out = {}
    for pattern in [f"{BASE}/paper/data/ars/{scale}/ars_{scale}_*.json",
                    f"{BASE}/outputs/ars_{scale}_*.json"]:
        for f in sorted(glob.glob(pattern)):
            d = json.load(open(f))
            name = Path(f).stem.replace(f"ars_{scale}_", "").replace("_v2","")
            if name == "arithmetic": name = "basic_arithmetic"
            per_atom = d.get("per_atom", {})
            l1 = [a["accuracy"] for n, a in per_atom.items() if n.startswith("L1")]
            l2 = [a["accuracy"] for n, a in per_atom.items() if n.startswith("L2")]
            out[name] = {"L1": float(np.mean(l1)) if l1 else None,
                         "L2": float(np.mean(l2)) if l2 else None,
                         "all": float(np.mean(l1+l2)) if (l1 or l2) else None}
    # leg_counting surrogate
    for mfile in glob.glob(f"{BASE}/paper/data/ars/{scale}/sft_base_leg_counting/metrics_*.json"):
        m = json.load(open(mfile))
        for t, s in m.get("tasks", {}).items():
            if t == "leg_counting":
                acc = s.get("accuracy", 0)
                out["leg_counting"] = {"L1": acc, "L2": acc, "all": acc}
    return out

PAIRS = [
    ("T1_sdk_leg", "mini_sudoku",      "leg_counting",    18.0),
    ("T2_sdk_kk",  "mini_sudoku",      "knights_knaves",  22.0),
    ("T3_ari_leg", "basic_arithmetic", "leg_counting",    12.5),
    ("T4_ari_syl", "basic_arithmetic", "syllogism",        4.5),
    ("T5_sp_syl",  "shortest_path",    "syllogism",        7.0),
    ("T6_sp_kk",   "shortest_path",    "knights_knaves",  18.0),
    ("T7_cnt_ari", "countdown",        "basic_arithmetic", 24.0),
]

ars_3b = load_ars("3b")
ars_4b = load_ars("4b")

rows = []
for pname, ta, tb, dmg in PAIRS:
    r3 = sa_3b.get(pname, {})
    a3a = ars_3b.get(ta, {}); a3b = ars_3b.get(tb, {})
    a4a = ars_4b.get(ta, {}); a4b = ars_4b.get(tb, {})
    rows.append({
        "pair": pname,
        "damage_4b": dmg,
        "sign_agree_3b": r3.get("mag_agree"),
        "L1_mean_3b": (a3a["L1"] + a3b["L1"]) / 2 if (a3a.get("L1") is not None and a3b.get("L1") is not None) else None,
        "L1_mean_4b": (a4a["L1"] + a4b["L1"]) / 2 if (a4a.get("L1") is not None and a4b.get("L1") is not None) else None,
        "ARS_mean_3b": (a3a["all"] + a3b["all"]) / 2 if (a3a.get("all") is not None and a3b.get("all") is not None) else None,
        "ARS_mean_4b": (a4a["all"] + a4b["all"]) / 2 if (a4a.get("all") is not None and a4b.get("all") is not None) else None,
    })

print(f"{'pair':15s} {'dmg':>5s} {'sign_agr_3b':>12s} {'L1_mean_3b':>11s} {'L1_mean_4b':>11s} {'ARS_mean_3b':>12s}")
for r in rows:
    print(f"{r['pair']:15s} {r['damage_4b']:5.1f} "
          f"{r['sign_agree_3b']:12.4f} "
          f"{r['L1_mean_3b']:11.4f} "
          f"{r['L1_mean_4b']:11.4f} "
          f"{r['ARS_mean_3b']:12.4f}")

y = np.array([r["damage_4b"] for r in rows])

# Individual Pearson r
preds = ["sign_agree_3b", "L1_mean_3b", "L1_mean_4b", "ARS_mean_3b", "ARS_mean_4b"]
print("\n--- Pairwise r with 4B damage ---")
for p in preds:
    x = np.array([r[p] for r in rows])
    print(f"  {p:15s}  r={np.corrcoef(x,y)[0,1]:+.3f}")

# r(L1_mean_3b, sign_agree_3b) — are they redundant?
print("\n--- Are 3B L1_mean and 3B sign_agreement redundant? ---")
x1 = np.array([r["L1_mean_3b"] for r in rows])
x2 = np.array([r["sign_agree_3b"] for r in rows])
print(f"  r(L1_mean_3b, sign_agree_3b) = {np.corrcoef(x1,x2)[0,1]:+.3f}")

# Multi-regression: damage ~ a*L1_mean_3b + b*sign_agree_3b + c
print("\n--- Multi-regression: damage = a·L1_mean_3b + b·sign_agree_3b + c ---")
X = np.column_stack([x1, x2, np.ones_like(x1)])
coef, res, rank, sv = np.linalg.lstsq(X, y, rcond=None)
yhat = X @ coef
r2 = 1 - np.sum((y-yhat)**2) / np.sum((y-y.mean())**2)
print(f"  coef L1_mean_3b = {coef[0]:+.2f}")
print(f"  coef sign_agree_3b = {coef[1]:+.2f}")
print(f"  intercept = {coef[2]:+.2f}")
print(f"  R² = {r2:.3f}  (vs single-predictor R²: L1_mean_3b={np.corrcoef(x1,y)[0,1]**2:.3f}, sign_agree_3b={np.corrcoef(x2,y)[0,1]**2:.3f})")

# Partial correlations
def partial_corr(x, y, z):
    # residuals of x ~ z and y ~ z, then corr
    def residuals(a, b):
        B = np.column_stack([b, np.ones_like(b)])
        coef, *_ = np.linalg.lstsq(B, a, rcond=None)
        return a - B @ coef
    rx = residuals(x, z); ry = residuals(y, z)
    return float(np.corrcoef(rx, ry)[0,1])

pc1 = partial_corr(x1, y, x2)
pc2 = partial_corr(x2, y, x1)
print(f"\n--- Partial correlations ---")
print(f"  r(L1_mean_3b, damage | sign_agree_3b) = {pc1:+.3f}")
print(f"  r(sign_agree_3b, damage | L1_mean_3b) = {pc2:+.3f}")

# Save everything
out = Path(f"{BASE}/paper/data/analysis/ars_vs_sign_agreement.json")
with open(out, "w") as f:
    json.dump({
        "rows": rows,
        "individual_r": {p: float(np.corrcoef(np.array([r[p] for r in rows]), y)[0,1]) for p in preds},
        "r_L1_sign_agree": float(np.corrcoef(x1, x2)[0,1]),
        "multi_regression": {"coef_L1": float(coef[0]), "coef_sign_agree": float(coef[1]),
                             "intercept": float(coef[2]), "R2": float(r2)},
        "partial_r": {"L1_given_sign_agree": pc1, "sign_agree_given_L1": pc2},
    }, f, indent=2)
print(f"\nwrote {out}")
