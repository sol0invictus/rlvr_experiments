"""
Stream D.1 — Extend Phase-3 §5.3 novel-subspace rotation analysis from
{mlp.up, mlp.down, attn.q} to also include {attn.k, attn.v, attn.o} at 4B
and 8B.

Same methodology as experiments/rlvr_geometry_paper/scripts/analysis/analyze_rotation_direction_4b.py:
  - For each layer, compute U_sk = top-k left singular basis of W_sft and
    U_gk = top-k left singular basis of W_grpo.
  - Novel subspace basis Q = QR(U_gk - U_sk @ U_sk^T @ U_gk).
  - Pairwise principal angles + flat cosine between the flattened
    U_gk - U_sk @ U_sk^T @ U_gk vectors.

Probes 8 layers across depth (spread uniformly over model), 4 attn types.

Output:
  experiments/rlvr_geometry_paper/results/weight_analysis/<scale>/rotation_direction_attn_kvo.json
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open


def load_sd(p):
    p = Path(p)
    sd = {}
    for f in sorted(p.glob("model*.safetensors")):
        with safe_open(f, framework="pt") as sf:
            for k in sf.keys():
                sd[k] = sf.get_tensor(k)
    return sd


CONFIGS = {
    "4b": {
        "sft": "/scratch/sunguha/sft_qwen_4.0b_instruct",
        "n_layers": 36,
        "tasks": {
            "basic_arithmetic": "experiments/rlvr_geometry_paper/results/grpo_qwen4b_basic_arithmetic",
            "countdown": "experiments/rlvr_geometry_paper/results/grpo_qwen_4b_instruct_countdown_4to5",
            "shortest_path": "experiments/rlvr_geometry_paper/results/grpo_qwen4b_shortest_path",
            "mini_sudoku": "experiments/rlvr_geometry_paper/results/grpo_qwen4b_mini_sudoku",
            "syllogism": "experiments/rlvr_geometry_paper/results/grpo_qwen4b_syllogism",
            "knights_knaves": "experiments/rlvr_geometry_paper/results/grpo_qwen4b_knights_knaves",
            "leg_counting": "experiments/rlvr_geometry_paper/results/grpo_qwen4b_leg_counting",
        },
    },
    "8b": {
        "sft": "experiments/rlvr_geometry_paper/outputs/sft_qwen3_8b",
        "n_layers": 36,
        "tasks": {
            "basic_arithmetic": "experiments/rlvr_geometry_paper/results/grpo_qwen3_8b_basic_arithmetic",
            "countdown": "experiments/rlvr_geometry_paper/results/grpo_qwen3_8b_countdown",
            "shortest_path": "experiments/rlvr_geometry_paper/results/grpo_qwen3_8b_shortest_path",
            "mini_sudoku": "experiments/rlvr_geometry_paper/results/grpo_qwen3_8b_mini_sudoku",
            "syllogism": "experiments/rlvr_geometry_paper/results/grpo_qwen3_8b_syllogism",
            "knights_knaves": "experiments/rlvr_geometry_paper/results/grpo_qwen3_8b_knights_knaves",
            "leg_counting": "experiments/rlvr_geometry_paper/results/grpo_qwen3_8b_leg_counting",
        },
    },
}

PAIRS = [
    ("T1_sdk_leg",  "mini_sudoku",      "leg_counting"),
    ("T2_sdk_kk",   "mini_sudoku",      "knights_knaves"),
    ("T3_ari_leg",  "basic_arithmetic", "leg_counting"),
    ("T4_ari_syl",  "basic_arithmetic", "syllogism"),
    ("T5_sp_syl",   "shortest_path",    "syllogism"),
    ("T6_sp_kk",    "shortest_path",    "knights_knaves"),
    ("T7_cnt_ari",  "countdown",        "basic_arithmetic"),
]

K = 100


def novel_stuff(W_sft, W_grpo, k):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    W_sft = W_sft.float().to(device)
    W_grpo = W_grpo.float().to(device)
    U_s, _, _ = torch.linalg.svd(W_sft, full_matrices=False)
    U_g, _, _ = torch.linalg.svd(W_grpo, full_matrices=False)
    U_sk = U_s[:, :k]
    U_gk = U_g[:, :k]
    novel = U_gk - U_sk @ (U_sk.T @ U_gk)
    Q, R = torch.linalg.qr(novel)
    novel_norm = torch.linalg.norm(novel).item()
    flat = novel.flatten().cpu().numpy()
    Q_cpu = Q.cpu().numpy()
    del W_sft, W_grpo, U_s, U_g, U_sk, U_gk, novel, Q, R
    if device == "cuda":
        torch.cuda.empty_cache()
    return {"Q": Q_cpu, "novel_norm": novel_norm, "flat": flat}


def principal_angles(Q_a, Q_b):
    s = np.linalg.svd(Q_a.T @ Q_b, compute_uv=False)
    s = np.clip(s, -1.0, 1.0)
    deg = np.arccos(s) * 180 / np.pi
    return {"mean_deg": float(deg.mean()), "median_deg": float(np.median(deg))}


def cos_flat(a, b):
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return float((a @ b) / (na * nb))


def build_probe_layers(n_layers):
    # 8 layers spread across depth
    idxs = [int(round(i)) for i in np.linspace(0, n_layers - 1, 8)]
    heads = ["q_proj", "k_proj", "v_proj", "o_proj"]
    layers = []
    for li in idxs:
        for h in heads:
            layers.append(f"model.layers.{li}.self_attn.{h}.weight")
    return layers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", required=True, choices=["4b", "8b"])
    args = parser.parse_args()
    cfg = CONFIGS[args.scale]
    layers = build_probe_layers(cfg["n_layers"])

    print(f"Loading SFT ({args.scale}): {cfg['sft']}")
    sft_sd = load_sd(cfg["sft"])

    per_task = {}
    for task, path in cfg["tasks"].items():
        if not os.path.exists(path):
            print(f"  SKIP {task}: {path}")
            continue
        print(f"  Loading {task}")
        grpo_sd = load_sd(path)
        per_task[task] = {}
        for lname in layers:
            if lname not in sft_sd or lname not in grpo_sd:
                continue
            try:
                per_task[task][lname] = novel_stuff(sft_sd[lname], grpo_sd[lname], K)
            except Exception as e:
                print(f"    FAILED {task} {lname}: {e}")
        del grpo_sd

    # Aggregated per-pair × layer-type means
    results = {}
    for lname in layers:
        if not any(lname in per_task[t] for t in per_task):
            continue
        entry = {"novel_norms": {}, "pair_angles": {}, "pair_cos": {}}
        for t, d in per_task.items():
            if lname in d:
                entry["novel_norms"][t] = d[lname]["novel_norm"]
        for pid, ta, tb in PAIRS:
            if ta in per_task and tb in per_task:
                if lname in per_task[ta] and lname in per_task[tb]:
                    entry["pair_angles"][pid] = principal_angles(
                        per_task[ta][lname]["Q"], per_task[tb][lname]["Q"]
                    )
                    entry["pair_cos"][pid] = cos_flat(
                        per_task[ta][lname]["flat"], per_task[tb][lname]["flat"]
                    )
        results[lname] = entry

    # Aggregate per-pair across layer type
    by_pair_angle = {pid: {"attn_q": [], "attn_k": [], "attn_v": [], "attn_o": []} for pid, _, _ in PAIRS}
    by_pair_cos = {pid: {"attn_q": [], "attn_k": [], "attn_v": [], "attn_o": []} for pid, _, _ in PAIRS}
    for lname, d in results.items():
        if "q_proj" in lname:
            lt = "attn_q"
        elif "k_proj" in lname:
            lt = "attn_k"
        elif "v_proj" in lname:
            lt = "attn_v"
        elif "o_proj" in lname:
            lt = "attn_o"
        else:
            lt = None
        if lt is None:
            continue
        for pid, _, _ in PAIRS:
            if pid in d["pair_angles"]:
                by_pair_angle[pid][lt].append(d["pair_angles"][pid]["mean_deg"])
                by_pair_cos[pid][lt].append(d["pair_cos"][pid])

    print("\nPER-PAIR × ATTN-SUBTYPE mean principal angle (deg) | mean flat cosine")
    for pid, ta, tb in PAIRS:
        row = f"{pid:18}"
        for lt in ["attn_q", "attn_k", "attn_v", "attn_o"]:
            angs = by_pair_angle[pid][lt]
            coss = by_pair_cos[pid][lt]
            if angs:
                row += f"  {lt:6} ang={np.mean(angs):.2f}  cos={np.mean(coss):+.4f}"
            else:
                row += f"  {lt:6} --"
        print(row)

    summary_agg = {
        pid: {lt: {"mean_angle_deg": float(np.mean(by_pair_angle[pid][lt])) if by_pair_angle[pid][lt] else None,
                   "mean_flat_cos":  float(np.mean(by_pair_cos[pid][lt]))   if by_pair_cos[pid][lt]   else None,
                   "n_layers": len(by_pair_angle[pid][lt])}
              for lt in ["attn_q", "attn_k", "attn_v", "attn_o"]}
        for pid, _, _ in PAIRS
    }

    out = f"experiments/rlvr_geometry_paper/results/weight_analysis/{args.scale}/rotation_direction_attn_kvo.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    to_save = {
        "scale": args.scale,
        "k": K,
        "layers_probed": layers,
        "per_layer": {ln: {"novel_norms": d["novel_norms"],
                           "pair_angles": d["pair_angles"],
                           "pair_cos":    d["pair_cos"]} for ln, d in results.items()},
        "aggregated_by_pair_and_attn_subtype": summary_agg,
    }
    with open(out, "w") as f:
        json.dump(to_save, f, indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
