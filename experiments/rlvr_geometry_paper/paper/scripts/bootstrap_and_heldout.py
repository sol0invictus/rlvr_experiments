"""Fixed version — loads sign_agreement.json as a list of dicts."""
import json
import numpy as np
from pathlib import Path

BASE = "/code-fsx/sunguha/rlvr_experiments/experiments/rlvr_geometry_paper"
OUT = Path(f"{BASE}/paper/data/analysis")
OUT.mkdir(parents=True, exist_ok=True)

PAIR_NAMES = ["T1_sdk_leg", "T2_sdk_kk", "T3_ari_leg", "T4_ari_syl",
              "T5_sp_syl", "T6_sp_kk", "T7_cnt_ari"]
DAMAGE_4B = {
    "T1_sdk_leg": 18.0, "T2_sdk_kk": 22.0, "T3_ari_leg": 12.5,
    "T4_ari_syl": 4.5, "T5_sp_syl": 7.0, "T6_sp_kk": 18.0, "T7_cnt_ari": 24.0,
}


def load_pairs(scale):
    lst = json.load(open(f"{BASE}/results/weight_analysis/{scale}/sign_agreement.json"))
    by_name = {row["pair"]: row for row in lst}
    return by_name


def bootstrap():
    print("=== bootstrap ===", flush=True)
    sa = load_pairs("3b")
    rows = []
    for p in PAIR_NAMES:
        r = sa.get(p)
        if r is None: continue
        rows.append((p, r["mag_agree"], DAMAGE_4B[p]))
    print(f"  n={len(rows)}")
    for r in rows:
        print(f"    {r[0]:15s}  mag_agree_3B={r[1]:.4f}  damage_4B={r[2]}")
    x = np.array([r[1] for r in rows])
    y = np.array([r[2] for r in rows])
    r_point = float(np.corrcoef(x, y)[0, 1])
    rng = np.random.default_rng(42)
    rs = []
    for _ in range(10000):
        idx = rng.integers(0, len(rows), size=len(rows))
        if len(np.unique(x[idx])) < 2: continue
        if len(np.unique(y[idx])) < 2: continue
        xi, yi = x[idx], y[idx]
        if xi.std() == 0 or yi.std() == 0: continue
        rs.append(float(np.corrcoef(xi, yi)[0, 1]))
    rs = np.array([r for r in rs if not np.isnan(r)])
    ci_lo, ci_hi = np.percentile(rs, [2.5, 97.5])
    perm_rs = []
    for _ in range(10000):
        perm_y = rng.permutation(y)
        perm_rs.append(np.corrcoef(x, perm_y)[0, 1])
    perm_rs = np.array(perm_rs)
    p_two = float((np.abs(perm_rs) >= abs(r_point)).mean())
    out = {
        "n": len(rows),
        "rows": [{"pair": p[0], "mag_agree_3b": p[1], "damage_4b": p[2]} for p in rows],
        "r_point": r_point,
        "r_ci95": [float(ci_lo), float(ci_hi)],
        "p_two_sided_perm": p_two,
        "bootstrap_n": 10000,
    }
    with open(OUT / "bootstrap_r.json", "w") as f: json.dump(out, f, indent=2)
    print(f"  r = {r_point:.3f}, 95% CI [{ci_lo:.3f}, {ci_hi:.3f}], p={p_two:.4f}")


def held_out():
    print("=== held_out ===", flush=True)
    sa = load_pairs("3b")
    rows_np = []
    for p in PAIR_NAMES:
        r = sa.get(p)
        if r is None: continue
        rows_np.append((p, float(r["mag_agree"]), float(DAMAGE_4B[p])))

    # LOO
    preds, truths = [], []
    for i, (n, m, d) in enumerate(rows_np):
        train = [(mm, dd) for j, (_, mm, dd) in enumerate(rows_np) if j != i]
        tx = np.array([t[0] for t in train]); ty = np.array([t[1] for t in train])
        slope, intercept = np.polyfit(tx, ty, 1)
        preds.append(slope * m + intercept); truths.append(d)
    preds = np.array(preds); truths = np.array(truths)
    loo_r = float(np.corrcoef(preds, truths)[0, 1])
    loo_mae = float(np.mean(np.abs(preds - truths)))
    loo_rmse = float(np.sqrt(np.mean((preds - truths) ** 2)))

    def split_r(tr_names, te_names):
        tr = [(m, d) for (n, m, d) in rows_np if n in tr_names]
        te = [(n, m, d) for (n, m, d) in rows_np if n in te_names]
        tx = np.array([t[0] for t in tr]); ty = np.array([t[1] for t in tr])
        slope, intercept = np.polyfit(tx, ty, 1)
        preds = [slope * m + intercept for (_, m, _) in te]
        truths = [d for (_, _, d) in te]
        return {
            "train": list(tr_names), "test": list(te_names),
            "pred": preds, "truth": truths,
            "r": float(np.corrcoef(preds, truths)[0, 1]) if len(preds) > 1 else None,
            "mae": float(np.mean(np.abs(np.array(preds) - np.array(truths)))),
            "slope": float(slope), "intercept": float(intercept),
        }

    sA = split_r(["T1_sdk_leg", "T2_sdk_kk", "T3_ari_leg", "T4_ari_syl"],
                 ["T5_sp_syl", "T6_sp_kk", "T7_cnt_ari"])
    sB = split_r(["T5_sp_syl", "T6_sp_kk", "T7_cnt_ari"],
                 ["T1_sdk_leg", "T2_sdk_kk", "T3_ari_leg", "T4_ari_syl"])

    out = {
        "rows": [{"pair": n, "mag_agree_3b": m, "damage_4b": d} for (n, m, d) in rows_np],
        "leave_one_out": {"r_pred_vs_truth": loo_r, "mae": loo_mae, "rmse": loo_rmse},
        "split_A_first4_train": sA,
        "split_B_last3_train": sB,
    }
    with open(OUT / "held_out_predictor.json", "w") as f: json.dump(out, f, indent=2)
    print(f"  LOO pred vs truth r={loo_r:.3f} mae={loo_mae:.2f}")
    print(f"  split A r={sA['r']:.3f} mae={sA['mae']:.2f}")
    print(f"  split B r={sB['r']:.3f} mae={sB['mae']:.2f}")


if __name__ == "__main__":
    bootstrap()
    held_out()
    print("DONE")
