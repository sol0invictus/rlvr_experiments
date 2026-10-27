"""
Measure the DIRECTION of basis rotation per task.

For each task A and layer L:
  - W_sft has top-k singular subspace S_sft = span(U_sft[:, :k])
  - W_grpo(A) has top-k subspace S_A = span(U_grpo(A)[:, :k])
  - The "rotation direction" for task A is the novel subspace it added:
      novel_A = (I - U_sft U_sft^T) U_grpo(A)[:, :k]
    (= the component of U_grpo(A) orthogonal to U_sft's top-k)
  - QR-decompose to get an orthonormal basis for novel_A's span.

Between tasks A and B:
  - Principal angles between novel_A span and novel_B span.
  - 0°: tasks rotated in the same direction (added same novel dirs)
  - 90°: tasks rotated in perpendicular directions
  - >90°: this shouldn't happen; span angles are in [0, 90°]

Also compute the cosine between the "flat" rotation vectors for a scalar measure:
  v_A = vec((I - U_sft U_sft^T) U_grpo(A)[:, :k])
  cos(v_A, v_B) per task pair.
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
    safetensor_files = sorted(model_path.glob("*.safetensors"))
    if not safetensor_files:
        ckpt_dirs = sorted(model_path.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
        if ckpt_dirs:
            safetensor_files = sorted(ckpt_dirs[-1].glob("*.safetensors"))
    for f in safetensor_files:
        with safe_open(f, framework="pt") as sf:
            for key in sf.keys():
                state_dict[key] = sf.get_tensor(key)
    return state_dict


TASKS_8B = {
    "countdown": "/code-fsx/sunguha/rlvr_experiments/results/grpo_qwen3_8b_countdown",
    "basic_arithmetic": "/code-fsx/sunguha/rlvr_experiments/results/grpo_qwen3_8b_basic_arithmetic",
    "caesar_cipher": "/code-fsx/sunguha/rlvr_experiments/results/grpo_qwen3_8b_caesar_cipher",
    "syllogism": "/code-fsx/sunguha/rlvr_experiments/results/grpo_qwen3_8b_syllogism",
}
SFT_PATH = "/code-fsx/sunguha/rlvr_experiments/outputs/sft_qwen3_8b"

LAYERS = [
    "model.layers.0.mlp.down_proj.weight",
    "model.layers.0.self_attn.q_proj.weight",
    "model.layers.2.mlp.down_proj.weight",
    "model.layers.6.mlp.up_proj.weight",
    "model.layers.9.mlp.up_proj.weight",
    "model.layers.17.mlp.up_proj.weight",
    "model.layers.17.mlp.down_proj.weight",
    "model.layers.18.mlp.down_proj.weight",
    "model.layers.19.mlp.up_proj.weight",
    "model.layers.27.mlp.down_proj.weight",
    "model.layers.35.mlp.down_proj.weight",
    "model.layers.35.self_attn.q_proj.weight",
]

K = 100


def compute_novel_subspace(W_sft, W_grpo, k):
    """Compute the novel top-k left-singular subspace added by GRPO.

    novel = (I - U_sft U_sft^T) U_grpo[:, :k]
    Returns an orthonormal basis Q for the column span of novel, and the flat vec.
    """
    W_sft = W_sft.float().cuda()
    W_grpo = W_grpo.float().cuda()

    U_s, _, _ = torch.linalg.svd(W_sft, full_matrices=False)
    U_g, _, _ = torch.linalg.svd(W_grpo, full_matrices=False)

    U_sk = U_s[:, :k]
    U_gk = U_g[:, :k]

    # Novel component of U_gk orthogonal to U_sk's span
    novel = U_gk - U_sk @ (U_sk.T @ U_gk)
    # QR to get orthonormal basis for the column span of novel
    Q, R = torch.linalg.qr(novel)
    novel_norm = torch.linalg.norm(novel).item()

    # Flat vector for cosine similarity
    flat = novel.flatten().cpu().numpy()

    # Also orthogonalize the U_gk to get a clean "rotation direction" if novel is full-rank
    # We'll keep rank-k Q (columns of Q span novel's column space)
    # Rank may be less than k if rotation was tiny; capture that
    r_vals = torch.diag(R).abs()
    effective_rank_novel = (r_vals > 1e-6).sum().item()

    Q_cpu = Q.cpu().numpy()
    del W_sft, W_grpo, U_s, U_g, U_sk, U_gk, novel, Q, R
    torch.cuda.empty_cache()
    return {
        "Q": Q_cpu,  # (n, k) orthonormal basis of novel column span
        "novel_norm": novel_norm,
        "flat": flat,
        "effective_rank": effective_rank_novel,
    }


def principal_angles_between_spans(Q_a: np.ndarray, Q_b: np.ndarray) -> dict:
    """Principal angles between two orthonormal bases (column span)."""
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


def cos_flat(v_a: np.ndarray, v_b: np.ndarray) -> float:
    na = np.linalg.norm(v_a)
    nb = np.linalg.norm(v_b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return float((v_a @ v_b) / (na * nb))


def main():
    print(f"Loading SFT: {SFT_PATH}")
    sft_sd = load_state_dict_from_safetensors(SFT_PATH)

    # For each task, for each layer, compute novel subspace
    per_task_novel = {}
    for task, path in TASKS_8B.items():
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

    # Pairwise analysis
    tasks = list(TASKS_8B.keys())
    results = {}
    for lname in LAYERS:
        if not all(lname in per_task_novel[t] for t in tasks):
            continue
        results[lname] = {"pair_angles": {}, "pair_cos": {}, "novel_norms": {}, "effective_ranks": {}}
        for t in tasks:
            results[lname]["novel_norms"][t] = per_task_novel[t][lname]["novel_norm"]
            results[lname]["effective_ranks"][t] = per_task_novel[t][lname]["effective_rank"]
        for i, ta in enumerate(tasks):
            for j, tb in enumerate(tasks):
                if i >= j:
                    continue
                pair = f"{ta}__{tb}"
                Q_a = per_task_novel[ta][lname]["Q"]
                Q_b = per_task_novel[tb][lname]["Q"]
                ang = principal_angles_between_spans(Q_a, Q_b)
                results[lname]["pair_angles"][pair] = ang

                v_a = per_task_novel[ta][lname]["flat"]
                v_b = per_task_novel[tb][lname]["flat"]
                results[lname]["pair_cos"][pair] = cos_flat(v_a, v_b)

    # Print summary
    print("\n" + "=" * 80)
    print("NOVEL-SUBSPACE DIRECTION COMPARISON (8B, k=100)")
    print("=" * 80)

    print("\n=== Novel subspace norm per task per layer (how much did each task rotate) ===")
    print(f'{"Layer":40}' + ''.join(f' {t[:10]:>10}' for t in tasks))
    for lname, d in results.items():
        sn = lname.replace("model.layers.", "L").replace(".weight", "")
        row = f'{sn:40}'
        for t in tasks:
            row += f' {d["novel_norms"][t]:10.3f}'
        print(row)

    print("\n=== Effective rank of novel subspace (how many new directions each task added) ===")
    print(f'{"Layer":40}' + ''.join(f' {t[:10]:>10}' for t in tasks))
    for lname, d in results.items():
        sn = lname.replace("model.layers.", "L").replace(".weight", "")
        row = f'{sn:40}'
        for t in tasks:
            row += f' {d["effective_ranks"][t]:10d}'
        print(row)

    print("\n=== Mean principal angle between tasks' novel subspaces (SAME basis = 0°, PERP = 90°) ===")
    pair_list = list(next(iter(results.values()))["pair_angles"].keys())
    print(f'{"Layer":40}' + ''.join(f' {p[:10]:>10}' for p in pair_list))
    for lname, d in results.items():
        sn = lname.replace("model.layers.", "L").replace(".weight", "")
        row = f'{sn:40}'
        for pair in pair_list:
            a = d["pair_angles"][pair]["mean_deg"]
            row += f' {a:9.2f}°'
        print(row)

    print("\n=== Cosine between flat rotation vectors (+1 = same direction, 0 = orthogonal) ===")
    print(f'{"Layer":40}' + ''.join(f' {p[:10]:>10}' for p in pair_list))
    for lname, d in results.items():
        sn = lname.replace("model.layers.", "L").replace(".weight", "")
        row = f'{sn:40}'
        for pair in pair_list:
            row += f' {d["pair_cos"][pair]:10.4f}'
        print(row)

    # Aggregate
    print("\n=== AGGREGATE (mean across layer type, 8B) ===")
    print("\nMean PAIRWISE angle between novel subspaces (deg), by layer type:")
    by_type_angle = {}
    by_type_cos = {}
    for lname, d in results.items():
        if 'mlp.down' in lname: lt = 'mlp_down'
        elif 'mlp.up' in lname: lt = 'mlp_up'
        elif 'self_attn.q' in lname or 'attn.q' in lname: lt = 'attn_q'
        else: lt = 'other'
        for pair in pair_list:
            by_type_angle.setdefault(lt, []).append(d["pair_angles"][pair]["mean_deg"])
            by_type_cos.setdefault(lt, []).append(d["pair_cos"][pair])
    for lt in sorted(by_type_angle.keys()):
        a = np.array(by_type_angle[lt])
        c = np.array(by_type_cos[lt])
        print(f'  {lt:10}: angle mean={a.mean():5.2f}° (std={a.std():.2f}, min={a.min():.2f}, max={a.max():.2f})  '
              f'flat_cos mean={c.mean():+.4f}')

    # Save
    out_path = "experiments/rlvr_geometry_paper/results/weight_analysis/8b/rotation_direction.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    # Strip numpy arrays (Q not JSON-serializable) before saving
    to_save = {}
    for lname, d in results.items():
        to_save[lname] = {
            "pair_angles": d["pair_angles"],
            "pair_cos": d["pair_cos"],
            "novel_norms": d["novel_norms"],
            "effective_ranks": d["effective_ranks"],
        }
    with open(out_path, "w") as f:
        json.dump(to_save, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
