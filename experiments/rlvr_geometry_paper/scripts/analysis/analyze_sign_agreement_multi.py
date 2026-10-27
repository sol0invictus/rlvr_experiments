"""
Sign-agreement analysis, parameterized by scale.

Usage:
  python3 experiments/rlvr_geometry_paper/scripts/analysis/analyze_sign_agreement_multi.py --scale 4b
  python3 experiments/rlvr_geometry_paper/scripts/analysis/analyze_sign_agreement_multi.py --scale 3b
  python3 experiments/rlvr_geometry_paper/scripts/analysis/analyze_sign_agreement_multi.py --scale 8b

For each scale, computes fraction / magnitude-weighted / conflict-mass sign
agreement stats across the 7 Phase-4 pair definitions.

4B damage column is the one we measured; 3B/8B damages are None (no pair merges
run at those scales). Correlations are still informative for 4B; for 3B/8B the
statistics become a transferability check: are the mag_agree / conflict_mass
numbers similar across scales for the same task pairs?
"""

import argparse, json, os
from pathlib import Path
import numpy as np
from safetensors.torch import load_file


def load_sd(p):
    files = sorted(Path(p).glob("model*.safetensors"))
    sd = {}
    for f in files:
        sd.update(load_file(str(f)))
    return sd


PAIRS = [
    ("T1_sdk_leg",  "mini_sudoku",      "leg_counting"),
    ("T2_sdk_kk",   "mini_sudoku",      "knights_knaves"),
    ("T3_ari_leg",  "basic_arithmetic", "leg_counting"),
    ("T4_ari_syl",  "basic_arithmetic", "syllogism"),
    ("T5_sp_syl",   "shortest_path",    "syllogism"),
    ("T6_sp_kk",    "shortest_path",    "knights_knaves"),
    ("T7_cnt_ari",  "countdown",        "basic_arithmetic"),
]

DAMAGE_4B = {  # from Part 2; None at other scales
    "T1_sdk_leg": 18.0, "T2_sdk_kk": 22.0, "T3_ari_leg": 12.5,
    "T4_ari_syl": 4.5, "T5_sp_syl": 7.0, "T6_sp_kk": 18.0, "T7_cnt_ari": 24.0,
}


def cfg_for_scale(scale):
    scale = scale.lower()
    if scale == "4b":
        return {
            "base": "/scratch/sunguha/sft_qwen_4.0b_instruct",
            "ckpts": {
                "basic_arithmetic": "experiments/rlvr_geometry_paper/results/grpo_qwen4b_basic_arithmetic",
                "countdown": "experiments/rlvr_geometry_paper/results/grpo_qwen_4b_instruct_countdown_4to5",
                "shortest_path": "experiments/rlvr_geometry_paper/results/grpo_qwen4b_shortest_path",
                "mini_sudoku": "experiments/rlvr_geometry_paper/results/grpo_qwen4b_mini_sudoku",
                "syllogism": "experiments/rlvr_geometry_paper/results/grpo_qwen4b_syllogism",
                "knights_knaves": "experiments/rlvr_geometry_paper/results/grpo_qwen4b_knights_knaves",
                "leg_counting": "experiments/rlvr_geometry_paper/results/grpo_qwen4b_leg_counting",
            },
        }
    if scale == "3b":
        return {
            "base": "experiments/rlvr_geometry_paper/outputs/sft_qwen25_3b",
            "ckpts": {
                "basic_arithmetic": "experiments/rlvr_geometry_paper/results/grpo_qwen25_3b_basic_arithmetic",
                "countdown": "experiments/rlvr_geometry_paper/results/grpo_qwen25_3b_countdown",
                "shortest_path": "experiments/rlvr_geometry_paper/results/grpo_qwen25_3b_shortest_path",
                "mini_sudoku": "experiments/rlvr_geometry_paper/results/grpo_qwen25_3b_mini_sudoku",
                "syllogism": "experiments/rlvr_geometry_paper/results/grpo_qwen25_3b_syllogism",
                "knights_knaves": "experiments/rlvr_geometry_paper/results/grpo_qwen25_3b_knights_knaves",
                "leg_counting": "experiments/rlvr_geometry_paper/results/grpo_qwen25_3b_leg_counting",
            },
        }
    if scale == "8b":
        return {
            "base": "experiments/rlvr_geometry_paper/outputs/sft_qwen3_8b",
            "ckpts": {
                "basic_arithmetic": "experiments/rlvr_geometry_paper/results/grpo_qwen3_8b_basic_arithmetic",
                "countdown": "experiments/rlvr_geometry_paper/results/grpo_qwen3_8b_countdown",
                "shortest_path": "experiments/rlvr_geometry_paper/results/grpo_qwen3_8b_shortest_path",
                "mini_sudoku": "experiments/rlvr_geometry_paper/results/grpo_qwen3_8b_mini_sudoku",
                "syllogism": "experiments/rlvr_geometry_paper/results/grpo_qwen3_8b_syllogism",
                "knights_knaves": "experiments/rlvr_geometry_paper/results/grpo_qwen3_8b_knights_knaves",
                "leg_counting": "experiments/rlvr_geometry_paper/results/grpo_qwen3_8b_leg_counting",
            },
        }
    raise ValueError(scale)


def pearson(x, y):
    x = np.array(x); y = np.array(y)
    if x.std() == 0 or y.std() == 0: return 0
    return float(((x - x.mean()) * (y - y.mean())).mean() / (x.std() * y.std()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", required=True, choices=["3b", "4b", "8b"])
    args = parser.parse_args()
    cfg = cfg_for_scale(args.scale)

    print(f"Loading SFT base ({args.scale}): {cfg['base']}")
    base = load_sd(cfg["base"])

    taus = {}
    for task, path in cfg["ckpts"].items():
        if not os.path.exists(path):
            print(f"SKIP {task}: {path}"); continue
        print(f"  loading {task}")
        sd = load_sd(path)
        t = {}
        for k in base:
            if k in sd and sd[k].shape == base[k].shape:
                t[k] = (sd[k].float() - base[k].float())
        taus[task] = t
        del sd

    print("\n" + "=" * 110)
    print(f"SIGN-AGREEMENT PER PAIR ({args.scale})")
    print("=" * 110)
    print(f"{'pair':15} {'damage':>8} {'frac_agree':>11} {'mag_agree':>10} {'conflict_mass':>14} {'up_conflict':>12} {'q_conflict':>11}")

    rows = []
    for pair_id, ta, tb in PAIRS:
        if ta not in taus or tb not in taus:
            print(f"{pair_id}: SKIP (missing tau)"); continue
        total_params = 0; n_agree = 0
        mag_num = 0.0; mag_den = 0.0
        conf_num = 0.0; conf_den = 0.0
        up_num = 0.0; up_den = 0.0
        q_num = 0.0; q_den = 0.0
        for k in taus[ta]:
            if k not in taus[tb]: continue
            a = taus[ta][k]; b = taus[tb][k]
            if a.shape != b.shape: continue
            sa = a.sign(); sb = b.sign()
            agree = (sa == sb) & (sa != 0)
            n_agree += agree.sum().item()
            total_params += a.numel()
            w = a.abs() * b.abs()
            mag_num += (w * agree.float()).sum().item()
            mag_den += w.sum().item()
            disagree = (sa != sb) & (sa != 0) & (sb != 0)
            conf_num += (w * disagree.float()).sum().item()
            conf_den += w.sum().item()
            if 'up_proj' in k or 'gate_proj' in k:
                up_num += (w * disagree.float()).sum().item()
                up_den += w.sum().item()
            elif 'q_proj' in k:
                q_num += (w * disagree.float()).sum().item()
                q_den += w.sum().item()

        frac_agree = n_agree / total_params if total_params else 0
        mag_agree = mag_num / mag_den if mag_den else 0
        conf = conf_num / conf_den if conf_den else 0
        upc = up_num / up_den if up_den else 0
        qc = q_num / q_den if q_den else 0
        dmg = DAMAGE_4B.get(pair_id) if args.scale == "4b" else None
        dmg_s = f"{dmg:8.1f}" if dmg is not None else f"{'--':>8}"
        rows.append({"pair": pair_id, "damage": dmg, "frac_agree": frac_agree,
                     "mag_agree": mag_agree, "conflict_mass": conf,
                     "up_conflict": upc, "q_conflict": qc})
        print(f"{pair_id:15} {dmg_s} {frac_agree:11.4f} {mag_agree:10.4f} {conf:14.4f} {upc:12.4f} {qc:11.4f}")

    # Correlate with 4B damage if we have it
    if args.scale == "4b":
        damages = [r["damage"] for r in rows]
        print("\nCORRELATIONS (Pearson r vs 4B merge damage, n={})".format(len(rows)))
        for key in ["frac_agree", "mag_agree", "conflict_mass", "up_conflict", "q_conflict"]:
            vals = [r[key] for r in rows]
            print(f"  {key:20}: r = {pearson(damages, vals):+.3f}")
    else:
        # Use 4B damage as ground truth and correlate 3B/8B sign stats with it
        damages = [DAMAGE_4B[r["pair"]] for r in rows]
        print(f"\nCORRELATIONS — {args.scale} sign stats vs 4B damage (transferability check)")
        for key in ["frac_agree", "mag_agree", "conflict_mass", "up_conflict", "q_conflict"]:
            vals = [r[key] for r in rows]
            print(f"  {key:20}: r = {pearson(damages, vals):+.3f}")

    out = f"experiments/rlvr_geometry_paper/results/weight_analysis/{args.scale}/sign_agreement.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
