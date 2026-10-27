"""
Measure two things the singular-value comparison can't reveal:

(1) Subspace projection ratio of τ onto W_sft's principal subspace:
      ρ_k = ‖U_k U_k^T τ V_k V_k^T‖_F / ‖τ‖_F
    If ρ_k → 1 quickly with small k, τ lives inside W's dominant subspace
    (i.e. it rescales existing directions). If ρ_k stays small for k up
    to the matrix rank, τ spans a complementary subspace.

(2) Principal angles between U_sft[:, :k] and U_grpo[:, :k] (left) and V (right):
    arccos of singular values of U_sft[:, :k]^T U_grpo[:, :k].
    Non-zero angles ⇒ the basis rotated even when σ values didn't move.
"""

import argparse
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


MODEL_CONFIGS = {
    "3b": {
        "sft": "/code-fsx/sunguha/rlvr_experiments/outputs/sft_qwen25_3b",
        "prefix": "/code-fsx/sunguha/rlvr_experiments/results/grpo_qwen25_3b_",
        "tasks": ["countdown", "basic_arithmetic", "caesar_cipher", "syllogism", "cryptarithm"],
    },
    "4b": {
        "sft": "/code-fsx/sunguha/sft_qwen_4.0b_instruct",
        "prefix": "/code-fsx/sunguha/rlvr_experiments/results/grpo_qwen4b_",
        "task_overrides": {
            "countdown": "/code-fsx/sunguha/rlvr_experiments/results/phase2_countdown/direct_rlvr",
        },
        "tasks": ["countdown", "basic_arithmetic", "caesar_cipher", "syllogism", "cryptarithm"],
    },
    "7b": {
        "sft": "/code-fsx/sunguha/rlvr_experiments/outputs/sft_qwen25_7b",
        "prefix": "/code-fsx/sunguha/rlvr_experiments/results/grpo_qwen25_7b_",
        "tasks": ["countdown", "basic_arithmetic", "caesar_cipher", "syllogism", "cryptarithm"],
    },
    "8b": {
        "sft": "/code-fsx/sunguha/rlvr_experiments/outputs/sft_qwen3_8b",
        "prefix": "/code-fsx/sunguha/rlvr_experiments/results/grpo_qwen3_8b_",
        "tasks": ["countdown", "basic_arithmetic", "caesar_cipher", "syllogism", "cryptarithm"],
    },
}


def get_layer_list(n_layers: int) -> list:
    """Return a representative subset of layers spanning model depth."""
    # Sample 12 layers across depth
    anchors = [0, 0, 2, 6, 9, 17, 17, 18, 19, 27, n_layers - 1, n_layers - 1]
    # Scale anchors if n_layers < 36
    if n_layers < 36:
        anchors = [min(a, n_layers - 1) for a in anchors]
    layer_types = [
        "mlp.down_proj", "self_attn.q_proj", "mlp.down_proj", "mlp.up_proj",
        "mlp.up_proj", "mlp.up_proj", "mlp.down_proj", "mlp.down_proj",
        "mlp.up_proj", "mlp.down_proj", "mlp.down_proj", "self_attn.q_proj",
    ]
    return [f"model.layers.{i}.{t}.weight" for i, t in zip(anchors, layer_types)]

K_VALUES = [1, 10, 100, 500, 1000]


def analyze_layer(W_sft: torch.Tensor, W_grpo: torch.Tensor, k_values) -> dict:
    W_sft = W_sft.float().cuda()
    W_grpo = W_grpo.float().cuda()

    tau = W_grpo - W_sft
    tau_norm = torch.linalg.norm(tau).item()
    if tau_norm < 1e-10:
        del W_sft, W_grpo, tau
        torch.cuda.empty_cache()
        return {"tau_norm": 0.0, "zero_update": True}

    # SVDs
    U_s, S_s, Vh_s = torch.linalg.svd(W_sft, full_matrices=False)
    U_g, S_g, Vh_g = torch.linalg.svd(W_grpo, full_matrices=False)
    V_s = Vh_s.T
    V_g = Vh_g.T

    r = min(W_sft.shape)  # full rank cap

    # Projection ratios: fraction of tau captured by top-k subspace of W_sft
    # We compute ‖U_k U_k^T τ V_k V_k^T‖_F / ‖τ‖_F
    projection_ratios = {}
    # Also one-sided projections to see row vs column subspace contribution
    left_proj_ratios = {}
    right_proj_ratios = {}
    for k in k_values:
        if k > r:
            continue
        U_k = U_s[:, :k]
        V_k = V_s[:, :k]
        # Left projection only
        tau_left = U_k @ (U_k.T @ tau)
        left_proj_ratios[k] = (torch.linalg.norm(tau_left).item() / tau_norm) ** 2
        # Right projection only
        tau_right = (tau @ V_k) @ V_k.T
        right_proj_ratios[k] = (torch.linalg.norm(tau_right).item() / tau_norm) ** 2
        # Both
        tau_both = U_k @ (U_k.T @ tau @ V_k) @ V_k.T
        projection_ratios[k] = (torch.linalg.norm(tau_both).item() / tau_norm) ** 2

    # Principal angles between top-k left singular vectors
    # arccos(singular values of U_s[:, :k]^T U_g[:, :k])
    left_angles = {}
    right_angles = {}
    for k in k_values:
        if k > r:
            continue
        cross = U_s[:, :k].T @ U_g[:, :k]
        sv = torch.linalg.svdvals(cross).clamp(-1.0, 1.0)
        angles = torch.acos(sv)  # radians
        left_angles[k] = {
            "mean_rad": angles.mean().item(),
            "max_rad": angles.max().item(),
            "mean_deg": (angles.mean() * 180 / np.pi).item(),
            "max_deg": (angles.max() * 180 / np.pi).item(),
            # Grassmann distance = sqrt(sum sin^2(theta_i))
            "grassmann": torch.sqrt((angles.sin() ** 2).sum()).item(),
        }

        cross_v = V_s[:, :k].T @ V_g[:, :k]
        sv_v = torch.linalg.svdvals(cross_v).clamp(-1.0, 1.0)
        angles_v = torch.acos(sv_v)
        right_angles[k] = {
            "mean_rad": angles_v.mean().item(),
            "max_rad": angles_v.max().item(),
            "mean_deg": (angles_v.mean() * 180 / np.pi).item(),
            "max_deg": (angles_v.max() * 180 / np.pi).item(),
            "grassmann": torch.sqrt((angles_v.sin() ** 2).sum()).item(),
        }

    out = {
        "tau_norm": tau_norm,
        "w_sft_norm": torch.linalg.norm(W_sft).item(),
        "tau_rel_norm": tau_norm / torch.linalg.norm(W_sft).item(),
        "rank": r,
        "shape": list(W_sft.shape),
        "projection_ratios_both": projection_ratios,   # two-sided
        "projection_ratios_left": left_proj_ratios,    # row-space only
        "projection_ratios_right": right_proj_ratios,  # column-space only
        "left_principal_angles": left_angles,
        "right_principal_angles": right_angles,
        "zero_update": False,
    }

    del W_sft, W_grpo, tau, U_s, U_g, V_s, V_g, S_s, S_g, Vh_s, Vh_g
    torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_size", required=True, choices=["3b", "4b", "7b", "8b"])
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    cfg = MODEL_CONFIGS[args.model_size]
    output = args.output or f"experiments/rlvr_geometry_paper/results/weight_analysis/{args.model_size}/subspace_rotation.json"

    SFT_PATH = cfg["sft"]
    print(f"Loading SFT: {SFT_PATH}")
    sft_sd = load_state_dict_from_safetensors(SFT_PATH)
    print(f"  {len(sft_sd)} params")

    # Determine number of layers
    n_layers = max(
        int(k.split("layers.")[1].split(".")[0])
        for k in sft_sd if "layers." in k
    ) + 1
    print(f"  model has {n_layers} layers")

    LAYERS_TO_ANALYZE = get_layer_list(n_layers)
    print(f"  analyzing {len(LAYERS_TO_ANALYZE)} representative layers")

    # Build task paths
    task_paths = {}
    overrides = cfg.get("task_overrides", {})
    for task in cfg["tasks"]:
        if task in overrides:
            task_paths[task] = overrides[task]
        else:
            task_paths[task] = cfg["prefix"] + task

    results = {}
    for task, path in task_paths.items():
        print(f"\n=== {task} ===")
        grpo_sd = load_state_dict_from_safetensors(path)
        task_results = {}
        for lname in LAYERS_TO_ANALYZE:
            if lname not in sft_sd or lname not in grpo_sd:
                continue
            print(f"  {lname}")
            out = analyze_layer(sft_sd[lname], grpo_sd[lname], K_VALUES)
            task_results[lname] = out
        results[task] = task_results
        del grpo_sd

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for task, lr in results.items():
        print(f"\n--- {task} ---")
        for lname, d in lr.items():
            if d.get("zero_update"):
                continue
            sn = lname.replace("model.layers.", "L").replace(".weight", "")
            print(f"  {sn:35} shape={d['shape']} rank={d['rank']}  "
                  f"‖τ‖/‖W‖={d['tau_rel_norm']*100:.3f}%")
            # Projection ratios (fraction of ‖τ‖² captured)
            pr_both = d["projection_ratios_both"]
            pr_left = d["projection_ratios_left"]
            print(f"    Two-sided proj onto top-k subspace of W_sft (fraction of ‖τ‖²):")
            for k in K_VALUES:
                if k in pr_both:
                    print(f"      k={k:5d}: two={pr_both[k]*100:5.2f}%  left={pr_left[k]*100:5.2f}%  "
                          f"right={d['projection_ratios_right'][k]*100:5.2f}%")
            # Principal angles
            la = d["left_principal_angles"]
            ra = d["right_principal_angles"]
            print(f"    Principal angles (top-k left, top-k right):")
            for k in K_VALUES:
                if k in la:
                    print(f"      k={k:5d}: left mean={la[k]['mean_deg']:5.2f}° max={la[k]['max_deg']:5.2f}°  "
                          f"right mean={ra[k]['mean_deg']:5.2f}° max={ra[k]['max_deg']:5.2f}°")

    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()
