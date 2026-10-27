"""
Per-layer cross-task comparison:

(a) For each selected layer, compute cosine similarity of τ_A vs τ_B for every task pair.
    Are tasks moving in the same direction layer-by-layer?

(b) For each selected layer, compute principal angles between U_grpo(A) and U_grpo(B).
    Are different tasks rotating the basis the same way?

(c) Aggregate: which layers are most shared (tasks rotate together) vs task-specific.
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
}  # cryptarithm excluded: no update
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

K = 100  # use k=100 for principal angles; balance between signal and cost


def flat_cos(a: torch.Tensor, b: torch.Tensor) -> float:
    a_f = a.flatten().float()
    b_f = b.flatten().float()
    return (a_f @ b_f / (a_f.norm() * b_f.norm() + 1e-12)).item()


def principal_angles_deg(U_a: torch.Tensor, U_b: torch.Tensor) -> dict:
    cross = U_a.T @ U_b
    sv = torch.linalg.svdvals(cross).clamp(-1.0, 1.0)
    ang = torch.acos(sv) * 180.0 / np.pi
    return {"mean": ang.mean().item(), "max": ang.max().item(), "median": ang.median().item()}


def main():
    print(f"Loading SFT: {SFT_PATH}")
    sft_sd = load_state_dict_from_safetensors(SFT_PATH)

    # Preload per-task tau and U_grpo for each selected layer
    per_layer = {}  # per_layer[lname][task] = {"tau": tensor on cpu, "U": Uk tensor on cpu, "V": Vk tensor on cpu}

    for task, path in TASKS_8B.items():
        print(f"\n=== Loading {task} ===")
        grpo_sd = load_state_dict_from_safetensors(path)
        for lname in LAYERS:
            if lname not in sft_sd or lname not in grpo_sd:
                continue
            W_sft = sft_sd[lname].float().cuda()
            W_grpo = grpo_sd[lname].float().cuda()
            tau = (W_grpo - W_sft).cpu()  # keep on CPU to save GPU memory
            # top-k SVD of W_grpo
            U_g, _, Vh_g = torch.linalg.svd(W_grpo, full_matrices=False)
            U_k = U_g[:, :K].cpu()
            V_k = Vh_g[:K, :].T.cpu()
            per_layer.setdefault(lname, {})[task] = {"tau": tau, "U": U_k, "V": V_k}
            del W_sft, W_grpo, U_g, Vh_g
            torch.cuda.empty_cache()
        del grpo_sd

    # Also compute U_sft top-k once per layer (we don't strictly need it)
    tasks = list(TASKS_8B.keys())

    # Build pairwise comparisons
    results = {}
    for lname, task_data in per_layer.items():
        results[lname] = {"tau_cos": {}, "u_angles": {}, "v_angles": {}}
        for i, t_a in enumerate(tasks):
            for j, t_b in enumerate(tasks):
                if i >= j:
                    continue
                pair = f"{t_a}__{t_b}"
                # Flat cosine between deltas
                c = flat_cos(task_data[t_a]["tau"], task_data[t_b]["tau"])
                results[lname]["tau_cos"][pair] = c

                # Principal angles between task-specific GRPO bases (top-k)
                U_a = task_data[t_a]["U"].cuda()
                U_b = task_data[t_b]["U"].cuda()
                a_ang = principal_angles_deg(U_a, U_b)
                results[lname]["u_angles"][pair] = a_ang

                V_a = task_data[t_a]["V"].cuda()
                V_b = task_data[t_b]["V"].cuda()
                v_ang = principal_angles_deg(V_a, V_b)
                results[lname]["v_angles"][pair] = v_ang

                del U_a, U_b, V_a, V_b
                torch.cuda.empty_cache()

    # Print summary
    print("\n" + "=" * 80)
    print("PER-LAYER TASK COMPARISON (8B)")
    print("=" * 80)

    print("\n=== Cosine similarity of τ (per layer, per task pair) ===")
    print(f'{"Layer":40} ' + " ".join(f'{p[:3]}x{p.split("__")[1][:3]:<3}' for p in results[LAYERS[0]]["tau_cos"]))
    for lname in LAYERS:
        if lname not in results:
            continue
        sn = lname.replace("model.layers.", "L").replace(".weight", "")
        row = f'{sn:40} '
        for pair, c in results[lname]["tau_cos"].items():
            row += f'{c:6.3f}  '
        print(row)

    print("\n=== Mean cosine-of-τ per layer (averaged over task pairs) ===")
    for lname in LAYERS:
        if lname not in results:
            continue
        sn = lname.replace("model.layers.", "L").replace(".weight", "")
        vals = list(results[lname]["tau_cos"].values())
        print(f'  {sn:40}  mean={np.mean(vals):.4f}  max={np.max(vals):.4f}  min={np.min(vals):.4f}')

    print("\n=== Principal angles between U_grpo(A) and U_grpo(B) at k=100 (mean/max deg) ===")
    print(f'{"Layer":40} ' + " ".join(f'{p[:4]}x{p.split("__")[1][:4]:<4}' for p in results[LAYERS[0]]["u_angles"]))
    for lname in LAYERS:
        if lname not in results:
            continue
        sn = lname.replace("model.layers.", "L").replace(".weight", "")
        row = f'{sn:40} '
        for pair, d in results[lname]["u_angles"].items():
            row += f'{d["mean"]:5.2f} '
        print(row)

    # Save
    out_path = "experiments/rlvr_geometry_paper/results/weight_analysis/8b/per_layer_task_diff.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
