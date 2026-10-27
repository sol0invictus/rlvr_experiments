"""
Sign-agreement analysis for Phase 4 pair merges.

For each pair (tau_A, tau_B):
  - per-parameter sign agreement: sign(tau_A) == sign(tau_B)
  - magnitude-weighted sign agreement: weight each position by |tau_A| * |tau_B|
  - fraction of "high-conflict" positions: sign disagrees AND both magnitudes above median

Then correlate these metrics with the known merge damages from Phase 4 Part 2.
"""

import json, os
from pathlib import Path
import torch
import numpy as np
from safetensors.torch import load_file


def load_sd(p):
    files = sorted(Path(p).glob("model*.safetensors"))
    sd = {}
    for f in files:
        sd.update(load_file(str(f)))
    return sd


BASE = "/scratch/sunguha/sft_qwen_4.0b_instruct"

TASK_CKPT = {
    "basic_arithmetic": "experiments/rlvr_geometry_paper/results/grpo_qwen4b_basic_arithmetic",
    "countdown": "experiments/rlvr_geometry_paper/results/grpo_qwen_4b_instruct_countdown_4to5",
    "shortest_path": "experiments/rlvr_geometry_paper/results/grpo_qwen4b_shortest_path",
    "mini_sudoku": "experiments/rlvr_geometry_paper/results/grpo_qwen4b_mini_sudoku",
    "syllogism": "experiments/rlvr_geometry_paper/results/grpo_qwen4b_syllogism",
    "knights_knaves": "experiments/rlvr_geometry_paper/results/grpo_qwen4b_knights_knaves",
    "leg_counting": "experiments/rlvr_geometry_paper/results/grpo_qwen4b_leg_counting",
}

PAIRS = [
    ("T1_sdk_leg",  "mini_sudoku",      "leg_counting",     18.0),
    ("T2_sdk_kk",   "mini_sudoku",      "knights_knaves",   22.0),
    ("T3_ari_leg",  "basic_arithmetic", "leg_counting",     12.5),
    ("T4_ari_syl",  "basic_arithmetic", "syllogism",         4.5),
    ("T5_sp_syl",   "shortest_path",    "syllogism",         7.0),
    ("T6_sp_kk",    "shortest_path",    "knights_knaves",   18.0),
    ("T7_cnt_ari",  "countdown",        "basic_arithmetic", 24.0),
]


def pearson(x, y):
    x = np.array(x); y = np.array(y)
    if x.std() == 0 or y.std() == 0: return 0
    return float(((x - x.mean()) * (y - y.mean())).mean() / (x.std() * y.std()))


def main():
    print("Loading base SFT...")
    base = load_sd(BASE)

    print("Computing task vectors (CPU)...")
    taus = {}
    for task, path in TASK_CKPT.items():
        print(f"  {task}")
        sd = load_sd(path)
        t = {}
        for k in base:
            if k in sd and sd[k].shape == base[k].shape:
                t[k] = (sd[k].float() - base[k].float())
        taus[task] = t
        del sd

    # Compute stats per pair
    print("\n" + "=" * 110)
    print("SIGN-AGREEMENT PER PAIR")
    print("=" * 110)
    print(f"{'pair':15} {'damage':>7} {'frac_agree':>11} {'mag_agree':>10} {'conflict_mass':>14} {'up_conflict':>12} {'q_conflict':>11}")

    rows = []
    for pair_id, ta, tb, damage in PAIRS:
        total_params = 0
        n_agree = 0
        mag_agree_num = 0.0
        mag_agree_den = 0.0
        # conflict = sign disagree AND both magnitudes > per-layer median
        conflict_mass_num = 0.0
        conflict_mass_den = 0.0
        up_conflict_num = 0.0; up_conflict_den = 0.0
        q_conflict_num = 0.0; q_conflict_den = 0.0

        for k in taus[ta]:
            if k not in taus[tb]:
                continue
            a = taus[ta][k]; b = taus[tb][k]
            if a.shape != b.shape: continue
            sign_a = a.sign(); sign_b = b.sign()
            agree = (sign_a == sign_b) & (sign_a != 0)

            n_agree += agree.sum().item()
            total_params += a.numel()

            # magnitude-weighted agreement
            w = a.abs() * b.abs()
            mag_agree_num += (w * agree.float()).sum().item()
            mag_agree_den += w.sum().item()

            # conflict mass: sign disagrees, weighted by |a|*|b|
            disagree = (sign_a != sign_b) & (sign_a != 0) & (sign_b != 0)
            conflict_mass_num += (w * disagree.float()).sum().item()
            conflict_mass_den += w.sum().item()

            if 'up_proj' in k or 'gate_proj' in k:
                up_conflict_num += (w * disagree.float()).sum().item()
                up_conflict_den += w.sum().item()
            elif 'q_proj' in k:
                q_conflict_num += (w * disagree.float()).sum().item()
                q_conflict_den += w.sum().item()

        frac_agree = n_agree / total_params if total_params else 0
        mag_agree = mag_agree_num / mag_agree_den if mag_agree_den else 0
        conflict_mass = conflict_mass_num / conflict_mass_den if conflict_mass_den else 0
        up_conflict = up_conflict_num / up_conflict_den if up_conflict_den else 0
        q_conflict = q_conflict_num / q_conflict_den if q_conflict_den else 0

        rows.append({
            "pair": pair_id, "damage": damage,
            "frac_agree": frac_agree, "mag_agree": mag_agree,
            "conflict_mass": conflict_mass, "up_conflict": up_conflict, "q_conflict": q_conflict,
        })
        print(f"{pair_id:15} {damage:7.1f} {frac_agree:11.4f} {mag_agree:10.4f} {conflict_mass:14.4f} {up_conflict:12.4f} {q_conflict:11.4f}")

    damages = [r["damage"] for r in rows]
    print("\n" + "=" * 80)
    print("CORRELATIONS (Pearson r vs total merge damage, n=7)")
    print("=" * 80)
    for key in ["frac_agree", "mag_agree", "conflict_mass", "up_conflict", "q_conflict"]:
        vals = [r[key] for r in rows]
        print(f"  {key:20}: r = {pearson(damages, vals):+.3f}")

    out_path = "experiments/rlvr_geometry_paper/results/weight_analysis/4b/sign_agreement.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
