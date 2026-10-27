"""
Stream D — CPU analysis (runs in parallel with GPU streams).

Three sub-analyses:
  (i)  attn.{k,v,o} novel-subspace rotation at 4B + 8B for 7 Phase-4 pairs.
  (ii) Held-out test of the r=-0.81 sign-agreement → merge damage predictor.
  (iii) Bootstrap CI on the Pearson r between 3B mag_agree and 4B damage.
"""
import json, os, sys, time
from pathlib import Path
import numpy as np
from safetensors.torch import load_file

BASE = "/code-fsx/sunguha/rlvr_experiments/experiments/rlvr_geometry_paper"
OUT = Path(f"{BASE}/paper/data/analysis")
OUT.mkdir(parents=True, exist_ok=True)

PAIRS = [
    ("T1_sdk_leg",  "mini_sudoku",      "leg_counting"),
    ("T2_sdk_kk",   "mini_sudoku",      "knights_knaves"),
    ("T3_ari_leg",  "basic_arithmetic", "leg_counting"),
    ("T4_ari_syl",  "basic_arithmetic", "syllogism"),
    ("T5_sp_syl",   "shortest_path",    "syllogism"),
    ("T6_sp_kk",    "shortest_path",    "knights_knaves"),
    ("T7_cnt_ari",  "countdown",        "basic_arithmetic"),
]
DAMAGE_4B = {
    "T1_sdk_leg": 18.0, "T2_sdk_kk": 22.0, "T3_ari_leg": 12.5,
    "T4_ari_syl": 4.5, "T5_sp_syl": 7.0, "T6_sp_kk": 18.0, "T7_cnt_ari": 24.0,
}

# ---------------- (iii) Bootstrap CI on r=-0.81 (3B -> 4B) ----------------
def bootstrap_r():
    print("=== (iii) bootstrap CI on r(3B mag_agree, 4B damage) ===", flush=True)
    sa_3b_path = f"{BASE}/results/weight_analysis/3b/sign_agreement.json"
    if not Path(sa_3b_path).exists():
        print(f"  missing: {sa_3b_path}")
        return
    sa = json.load(open(sa_3b_path))
    rows = []
    for tname, _, _ in PAIRS:
        s = sa.get(tname) or sa.get("pairs", {}).get(tname)
        if s is None:
            print(f"  WARN: {tname} not in 3B sign_agreement.json"); continue
        mag = s.get("mag_agree") or s.get("weighted_agree") or s.get("magnitude_agree")
        rows.append((tname, mag, DAMAGE_4B[tname]))
    print(f"  n={len(rows)}")
    for r in rows:
        print(f"    {r[0]}  mag_agree_3B={r[1]:.4f}  damage_4B={r[2]}")
    x = np.array([r[1] for r in rows])
    y = np.array([r[2] for r in rows])
    # point estimate
    r_point = float(np.corrcoef(x, y)[0, 1])
    # bootstrap
    rng = np.random.default_rng(42)
    rs = []
    for _ in range(10000):
        idx = rng.integers(0, len(rows), size=len(rows))
        if len(np.unique(x[idx])) < 2: continue
        rs.append(np.corrcoef(x[idx], y[idx])[0, 1])
    rs = np.array(rs)
    ci_lo, ci_hi = np.percentile(rs, [2.5, 97.5])
    # permutation p-value (two-sided)
    perm_rs = []
    for _ in range(10000):
        perm_y = rng.permutation(y)
        perm_rs.append(np.corrcoef(x, perm_y)[0, 1])
    perm_rs = np.array(perm_rs)
    p_two_sided = float((np.abs(perm_rs) >= abs(r_point)).mean())
    out = {
        "n": len(rows),
        "rows": [{"pair": p[0], "mag_agree_3b": p[1], "damage_4b": p[2]} for p in rows],
        "r_point": r_point,
        "r_ci95": [float(ci_lo), float(ci_hi)],
        "p_two_sided_perm": p_two_sided,
        "bootstrap_n": 10000,
    }
    with open(OUT / "bootstrap_r.json", "w") as f: json.dump(out, f, indent=2)
    print(f"  r = {r_point:.3f}, 95% CI [{ci_lo:.3f}, {ci_hi:.3f}], p={p_two_sided:.4f}")


# ---------------- (ii) Held-out predictor test ----------------
def held_out_predictor():
    print("=== (ii) held-out sign-agreement -> damage ===", flush=True)
    sa_3b = json.load(open(f"{BASE}/results/weight_analysis/3b/sign_agreement.json"))
    rows = []
    for tname, _, _ in PAIRS:
        s = sa_3b.get(tname) or sa_3b.get("pairs", {}).get(tname)
        if s is None: continue
        mag = s.get("mag_agree") or s.get("weighted_agree")
        rows.append((tname, mag, DAMAGE_4B[tname]))
    rows_np = [(n, float(m), float(d)) for n, m, d in rows]

    # Leave-one-out: fit on 6, predict 1, then correlate predictions with truths
    preds, truths = [], []
    for i, (n, m, d) in enumerate(rows_np):
        train = [(mm, dd) for j, (_, mm, dd) in enumerate(rows_np) if j != i]
        tx = np.array([t[0] for t in train]); ty = np.array([t[1] for t in train])
        # linear fit
        slope, intercept = np.polyfit(tx, ty, 1)
        pred = slope * m + intercept
        preds.append(pred); truths.append(d)
    preds = np.array(preds); truths = np.array(truths)
    loo_r = float(np.corrcoef(preds, truths)[0, 1])
    loo_mae = float(np.mean(np.abs(preds - truths)))
    loo_rmse = float(np.sqrt(np.mean((preds - truths) ** 2)))

    # Fixed split: {T1..T4} train, {T5..T7} test (and reverse)
    def split_r(train_names, test_names):
        tr = [(m, d) for (n, m, d) in rows_np if n in train_names]
        te = [(n, m, d) for (n, m, d) in rows_np if n in test_names]
        tx = np.array([t[0] for t in tr]); ty = np.array([t[1] for t in tr])
        slope, intercept = np.polyfit(tx, ty, 1)
        preds = [slope * m + intercept for (_, m, _) in te]
        truths = [d for (_, _, d) in te]
        return {
            "train": list(train_names), "test": list(test_names),
            "pred": preds, "truth": truths,
            "r": float(np.corrcoef(preds, truths)[0, 1]) if len(preds) > 1 else None,
            "mae": float(np.mean(np.abs(np.array(preds) - np.array(truths)))),
            "slope": float(slope), "intercept": float(intercept),
        }

    split_A = split_r(["T1_sdk_leg", "T2_sdk_kk", "T3_ari_leg", "T4_ari_syl"],
                      ["T5_sp_syl", "T6_sp_kk", "T7_cnt_ari"])
    split_B = split_r(["T5_sp_syl", "T6_sp_kk", "T7_cnt_ari"],
                      ["T1_sdk_leg", "T2_sdk_kk", "T3_ari_leg", "T4_ari_syl"])

    out = {
        "rows": [{"pair": n, "mag_agree_3b": m, "damage_4b": d} for (n, m, d) in rows_np],
        "leave_one_out": {"r_pred_vs_truth": loo_r, "mae": loo_mae, "rmse": loo_rmse},
        "split_A_first4_train": split_A,
        "split_B_last3_train": split_B,
    }
    with open(OUT / "held_out_predictor.json", "w") as f: json.dump(out, f, indent=2)
    print(f"  LOO pred vs truth r={loo_r:.3f} mae={loo_mae:.2f}")
    print(f"  Split A (train T1-T4): r={split_A['r']:.3f} mae={split_A['mae']:.2f}")
    print(f"  Split B (train T5-T7): r={split_B['r']:.3f} mae={split_B['mae']:.2f}")


# ---------------- (i) attn.{k,v,o} novel-subspace rotation ----------------
def load_sd(p):
    files = sorted(Path(p).glob("model*.safetensors"))
    sd = {}
    for f in files: sd.update(load_file(str(f)))
    return sd


def novel_subspace_angle_and_flat_cos(W_sft, W_a, W_b, k=32):
    """
    Phase-3-style novel-subspace analysis for one layer, one pair.
    - τ_a = W_a - W_sft; τ_b = W_b - W_sft
    - Get top-k SVD of each τ; compute principal angle between the two novel subspaces
    - "flat_cos" = cos between the dominant singular direction of τ_a and τ_b
    Returns (principal_angle_deg, flat_cos, ‖τa‖, ‖τb‖).
    """
    import torch
    sft = W_sft.float()
    a = (W_a.float() - sft).cpu()
    b = (W_b.float() - sft).cpu()
    # SVD — reshape 2D
    Ua, _, _ = torch.linalg.svd(a, full_matrices=False)
    Ub, _, _ = torch.linalg.svd(b, full_matrices=False)
    Ua = Ua[:, :k].numpy(); Ub = Ub[:, :k].numpy()
    # principal angles between subspaces (smallest angle)
    _, sv, _ = np.linalg.svd(Ua.T @ Ub)
    sv = np.clip(sv, -1, 1)
    min_angle_deg = float(np.degrees(np.arccos(sv[0])))  # smallest principal angle
    # flat cos: dominant singular direction cosine
    flat = float(np.dot(Ua[:, 0], Ub[:, 0]))
    return min_angle_deg, flat, float(a.norm().item()), float(b.norm().item())


def attn_kvo_rotation(scale):
    print(f"=== (i) attn.{{k,v,o}} rotation at {scale} ===", flush=True)
    if scale == "4b":
        cfg = {
            "base": f"/code-fsx/sunguha/sft_qwen_4.0b_instruct",
            "ckpts": {
                "basic_arithmetic": f"{BASE}/results/grpo_qwen4b_basic_arithmetic",
                "countdown": f"{BASE}/results/grpo_qwen_4b_instruct_countdown",
                "shortest_path": f"{BASE}/results/grpo_qwen4b_shortest_path",
                "mini_sudoku": f"{BASE}/results/grpo_qwen4b_mini_sudoku",
                "syllogism": f"{BASE}/results/grpo_qwen4b_syllogism",
                "knights_knaves": f"{BASE}/results/grpo_qwen4b_knights_knaves",
                "leg_counting": f"{BASE}/results/grpo_qwen4b_leg_counting",
            },
            "num_layers": 36,
        }
    elif scale == "8b":
        cfg = {
            "base": f"{BASE}/outputs/sft_qwen3_8b",
            "ckpts": {
                "basic_arithmetic": f"{BASE}/results/grpo_qwen3_8b_basic_arithmetic",
                "countdown": f"{BASE}/results/grpo_qwen3_8b_countdown",
                "shortest_path": f"{BASE}/results/grpo_qwen3_8b_shortest_path",
                "mini_sudoku": f"{BASE}/results/grpo_qwen3_8b_mini_sudoku",
                "syllogism": f"{BASE}/results/grpo_qwen3_8b_syllogism",
                "knights_knaves": f"{BASE}/results/grpo_qwen3_8b_knights_knaves",
                "leg_counting": f"{BASE}/results/grpo_qwen3_8b_leg_counting",
            },
            "num_layers": 36,
        }
    else:
        return

    # probe layers spread across depth
    num_layers = cfg["num_layers"]
    probe_layer_idxs = sorted(set([num_layers//8, num_layers//4, num_layers//2,
                                    3*num_layers//4, num_layers-2]))
    probe_keys = []
    for idx in probe_layer_idxs:
        for lt in ["k_proj", "v_proj", "o_proj"]:
            probe_keys.append(f"model.layers.{idx}.self_attn.{lt}.weight")

    print(f"  probe_keys: {len(probe_keys)} layers × types")
    print(f"  loading sft base {cfg['base']}", flush=True)
    t0 = time.time()
    sft_sd = load_sd(cfg["base"])
    print(f"    done ({time.time()-t0:.1f}s)")

    # load only the ckpts we need, one at a time, memory permitting
    task_sds = {}
    for tname, p in cfg["ckpts"].items():
        if not Path(p).exists():
            print(f"  WARN missing {tname}: {p}")
            continue
        print(f"  loading {tname}", flush=True)
        # only keep probe keys to save memory
        full = load_sd(p)
        task_sds[tname] = {k: full[k] for k in probe_keys if k in full}
        del full

    # analyze each pair
    result = {}
    for tname, ta, tb in PAIRS:
        if ta not in task_sds or tb not in task_sds:
            print(f"  skip {tname}: missing task")
            continue
        per_layer = []
        for key in probe_keys:
            if key not in sft_sd or key not in task_sds[ta] or key not in task_sds[tb]:
                continue
            try:
                ang, flat, na, nb = novel_subspace_angle_and_flat_cos(
                    sft_sd[key], task_sds[ta][key], task_sds[tb][key]
                )
                # strip prefix for readability
                short = key.replace("model.layers.", "L").replace(".self_attn.", ".").replace(".weight", "")
                per_layer.append({
                    "layer": short,
                    "angle_deg": ang,
                    "flat_cos": flat,
                    "norm_a": na,
                    "norm_b": nb,
                })
            except Exception as e:
                print(f"    layer {key} failed: {e}")
        # aggregate by layer_type (k, v, o)
        agg = {}
        for row in per_layer:
            lt = row["layer"].split(".")[-1]
            d = agg.setdefault(lt, {"angles": [], "flats": []})
            d["angles"].append(row["angle_deg"])
            d["flats"].append(row["flat_cos"])
        summary = {lt: {"mean_angle": float(np.mean(v["angles"])),
                        "mean_flat": float(np.mean(v["flats"])),
                        "n": len(v["angles"])} for lt, v in agg.items()}
        result[tname] = {"per_layer": per_layer, "per_type_mean": summary}
        print(f"  {tname}: " + " ".join(f"{lt}_ang={v['mean_angle']:.1f}°" for lt, v in summary.items()))

    with open(OUT / f"rotation_attn_kvo_{scale}.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"  wrote rotation_attn_kvo_{scale}.json")


if __name__ == "__main__":
    # quick ones first
    try:
        bootstrap_r()
    except Exception as e:
        print(f"BOOTSTRAP FAILED: {e}")
    try:
        held_out_predictor()
    except Exception as e:
        print(f"HELD_OUT FAILED: {e}")
    # long ones
    try:
        attn_kvo_rotation("4b")
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"ATTN_KVO 4B FAILED: {e}")
    try:
        attn_kvo_rotation("8b")
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"ATTN_KVO 8B FAILED: {e}")
    print("STREAM D COMPLETE")
