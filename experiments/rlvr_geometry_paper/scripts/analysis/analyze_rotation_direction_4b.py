"""
4B Rotation-Direction Analysis — extends experiments/rlvr_geometry_paper/scripts/analysis/analyze_rotation_direction.py (originally 8B).

Covers the 7 tasks involved in the Phase 4 pair tests:
  basic_arithmetic, countdown, shortest_path, mini_sudoku, syllogism,
  knights_knaves, leg_counting

For each task pair tested in Phase 4 merges, we compute:
  - Novel subspace at each probe layer (mlp.up_proj, attn.q_proj in mid-range layers)
  - Principal angle between novel subspaces (Phase 3's §5.3 measure)
  - Flat cosine between novel subspace vectors (Phase 3's §5.3 scalar view)

Output: experiments/rlvr_geometry_paper/results/weight_analysis/4b/rotation_direction.json (+ stdout summary)
"""

import json
import os
from pathlib import Path

import torch
import numpy as np
from safetensors import safe_open


def load_state_dict_from_safetensors(model_path: str) -> dict:
    model_path = Path(model_path)
    state_dict = {}
    safetensor_files = sorted(model_path.glob("model*.safetensors"))
    for f in safetensor_files:
        with safe_open(f, framework="pt") as sf:
            for key in sf.keys():
                state_dict[key] = sf.get_tensor(key)
    return state_dict


SFT_PATH = "/scratch/sunguha/sft_qwen_4.0b_instruct"

TASKS_4B = {
    "basic_arithmetic": "experiments/rlvr_geometry_paper/results/grpo_qwen4b_basic_arithmetic",
    "countdown": "experiments/rlvr_geometry_paper/results/grpo_qwen_4b_instruct_countdown_4to5",
    "shortest_path": "experiments/rlvr_geometry_paper/results/grpo_qwen4b_shortest_path",
    "mini_sudoku": "experiments/rlvr_geometry_paper/results/grpo_qwen4b_mini_sudoku",
    "syllogism": "experiments/rlvr_geometry_paper/results/grpo_qwen4b_syllogism",
    "knights_knaves": "experiments/rlvr_geometry_paper/results/grpo_qwen4b_knights_knaves",
    "leg_counting": "experiments/rlvr_geometry_paper/results/grpo_qwen4b_leg_counting",
}

# Phase 4 pair matrix (matches the 7 merges)
PAIRS = [
    ("T1_sdk_leg",  "mini_sudoku",      "leg_counting"),
    ("T2_sdk_kk",   "mini_sudoku",      "knights_knaves"),
    ("T3_ari_leg",  "basic_arithmetic", "leg_counting"),
    ("T4_ari_syl",  "basic_arithmetic", "syllogism"),
    ("T5_sp_syl",   "shortest_path",    "syllogism"),
    ("T6_sp_kk",    "shortest_path",    "knights_knaves"),
    ("T7_cnt_ari",  "countdown",        "basic_arithmetic"),
]

# Probe layers — 4B has 36 transformer blocks. Choose same index set as 8B analysis
LAYERS = [
    "model.layers.0.mlp.down_proj.weight",
    "model.layers.0.self_attn.q_proj.weight",
    "model.layers.2.mlp.down_proj.weight",
    "model.layers.6.mlp.up_proj.weight",
    "model.layers.9.mlp.up_proj.weight",
    "model.layers.14.mlp.up_proj.weight",
    "model.layers.15.mlp.up_proj.weight",
    "model.layers.16.mlp.up_proj.weight",
    "model.layers.17.mlp.up_proj.weight",
    "model.layers.17.mlp.down_proj.weight",
    "model.layers.18.mlp.down_proj.weight",
    "model.layers.19.mlp.up_proj.weight",
    "model.layers.21.mlp.up_proj.weight",
    "model.layers.27.mlp.down_proj.weight",
    "model.layers.35.mlp.down_proj.weight",
    "model.layers.35.self_attn.q_proj.weight",
    "model.layers.15.self_attn.q_proj.weight",
    "model.layers.17.self_attn.q_proj.weight",
]

K = 100


def compute_novel_subspace(W_sft, W_grpo, k):
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

    r_vals = torch.diag(R).abs()
    effective_rank_novel = (r_vals > 1e-6).sum().item()

    Q_cpu = Q.cpu().numpy()
    del W_sft, W_grpo, U_s, U_g, U_sk, U_gk, novel, Q, R
    if device == "cuda":
        torch.cuda.empty_cache()
    return {
        "Q": Q_cpu,
        "novel_norm": novel_norm,
        "flat": flat,
        "effective_rank": effective_rank_novel,
    }


def principal_angles_between_spans(Q_a, Q_b):
    M = Q_a.T @ Q_b
    s = np.linalg.svd(M, compute_uv=False)
    s = np.clip(s, -1.0, 1.0)
    angles = np.arccos(s) * 180 / np.pi
    return {
        "mean_deg": float(angles.mean()),
        "max_deg": float(angles.max()),
        "min_deg": float(angles.min()),
        "median_deg": float(np.median(angles)),
    }


def cos_flat(v_a, v_b):
    na = np.linalg.norm(v_a); nb = np.linalg.norm(v_b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return float((v_a @ v_b) / (na * nb))


def main():
    print(f"Loading SFT: {SFT_PATH}")
    sft_sd = load_state_dict_from_safetensors(SFT_PATH)

    per_task_novel = {}
    for task, path in TASKS_4B.items():
        if not os.path.exists(path):
            print(f"SKIP {task}: {path}")
            continue
        print(f"\n=== {task} ===")
        grpo_sd = load_state_dict_from_safetensors(path)
        per_task_novel[task] = {}
        for lname in LAYERS:
            if lname not in sft_sd or lname not in grpo_sd:
                continue
            print(f"  {lname}")
            out = compute_novel_subspace(sft_sd[lname], grpo_sd[lname], K)
            per_task_novel[task][lname] = out
        del grpo_sd

    # Pairwise for Phase 4 pairs
    results = {}
    for lname in LAYERS:
        if not all(lname in per_task_novel[t] for t in TASKS_4B if t in per_task_novel):
            continue
        results[lname] = {
            "novel_norms": {t: per_task_novel[t][lname]["novel_norm"]
                            for t in TASKS_4B if t in per_task_novel},
            "pair_angles": {},
            "pair_cos": {},
        }
        for pair_id, ta, tb in PAIRS:
            if ta not in per_task_novel or tb not in per_task_novel:
                continue
            Q_a = per_task_novel[ta][lname]["Q"]
            Q_b = per_task_novel[tb][lname]["Q"]
            ang = principal_angles_between_spans(Q_a, Q_b)
            results[lname]["pair_angles"][pair_id] = ang
            v_a = per_task_novel[ta][lname]["flat"]
            v_b = per_task_novel[tb][lname]["flat"]
            results[lname]["pair_cos"][pair_id] = cos_flat(v_a, v_b)

    # Summary: per-pair aggregates across layer types
    print("\n" + "=" * 100)
    print("PHASE 4 PAIR × ROTATION DIRECTION SUMMARY (4B, k=100)")
    print("=" * 100)
    print(f'{"Layer":45}' + ''.join(f' {p:>12}' for p, _, _ in PAIRS))
    print("-- mean principal angle (°) --")
    for lname, d in results.items():
        sn = lname.replace("model.layers.", "L").replace(".weight", "")
        row = f'{sn:45}'
        for pair_id, _, _ in PAIRS:
            if pair_id in d["pair_angles"]:
                row += f' {d["pair_angles"][pair_id]["mean_deg"]:11.2f}°'
            else:
                row += f' {"":>12}'
        print(row)

    print("\n-- flat cosine --")
    print(f'{"Layer":45}' + ''.join(f' {p:>12}' for p, _, _ in PAIRS))
    for lname, d in results.items():
        sn = lname.replace("model.layers.", "L").replace(".weight", "")
        row = f'{sn:45}'
        for pair_id, _, _ in PAIRS:
            if pair_id in d["pair_cos"]:
                row += f' {d["pair_cos"][pair_id]:+12.4f}'
            else:
                row += f' {"":>12}'
        print(row)

    print("\n-- aggregated (by layer type, across probe layers) --")
    by_pair_angle = {pid: {"mlp_up": [], "mlp_down": [], "attn_q": []} for pid, _, _ in PAIRS}
    by_pair_cos   = {pid: {"mlp_up": [], "mlp_down": [], "attn_q": []} for pid, _, _ in PAIRS}
    for lname, d in results.items():
        if "mlp.up_proj" in lname: lt = "mlp_up"
        elif "mlp.down_proj" in lname: lt = "mlp_down"
        elif "q_proj" in lname: lt = "attn_q"
        else: lt = None
        if lt is None: continue
        for pid, _, _ in PAIRS:
            if pid in d["pair_angles"]:
                by_pair_angle[pid][lt].append(d["pair_angles"][pid]["mean_deg"])
                by_pair_cos[pid][lt].append(d["pair_cos"][pid])

    print(f'{"Pair":20} {"mlp_up_ang":>11} {"mlp_up_cos":>11} {"mlp_down_ang":>13} {"mlp_down_cos":>13} {"attn_q_ang":>11} {"attn_q_cos":>11}')
    for pid, ta, tb in PAIRS:
        row = f'{pid:20}'
        for lt in ["mlp_up", "mlp_down", "attn_q"]:
            angs = by_pair_angle[pid][lt]
            coss = by_pair_cos[pid][lt]
            if angs:
                row += f' {np.mean(angs):10.2f}° {np.mean(coss):+10.4f}'.ljust(26 if lt != "mlp_down" else 30)
            else:
                row += f' {"--":>11} {"--":>11}'
        print(row)

    out_path = "experiments/rlvr_geometry_paper/results/weight_analysis/4b/rotation_direction.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    to_save = {}
    for lname, d in results.items():
        to_save[lname] = {k: v for k, v in d.items()}
    with open(out_path, "w") as f:
        json.dump(to_save, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
