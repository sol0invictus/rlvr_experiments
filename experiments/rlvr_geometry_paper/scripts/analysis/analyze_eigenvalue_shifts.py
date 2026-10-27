"""
Compare full singular value spectra of W_sft vs W_grpo, layer by layer.

For a set of representative layers across model depth, compute:
  - Full SVD of W_sft and W_grpo
  - Element-wise (σ_i^grpo - σ_i^sft) differences
  - Relative shift (σ_i^grpo / σ_i^sft - 1)
  - Whether the spectrum shape changes or merely translates
"""

import argparse
import json
import os
from pathlib import Path

import torch
import numpy as np
from safetensors import safe_open
from tqdm import tqdm


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
    "cryptarithm": "/code-fsx/sunguha/rlvr_experiments/results/grpo_qwen3_8b_cryptarithm",
}
SFT_PATH = "/code-fsx/sunguha/rlvr_experiments/outputs/sft_qwen3_8b"

# Pick a representative subset of layers spanning the depth
LAYERS_TO_ANALYZE = [
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


def svd_compare(W_sft: torch.Tensor, W_grpo: torch.Tensor) -> dict:
    """Compute full SVD of both and return per-index stats."""
    W_sft = W_sft.float().cuda()
    W_grpo = W_grpo.float().cuda()

    U_s, S_s, _ = torch.linalg.svd(W_sft, full_matrices=False)
    S_s = S_s.cpu().numpy()

    U_g, S_g, _ = torch.linalg.svd(W_grpo, full_matrices=False)
    S_g = S_g.cpu().numpy()

    # Also SVD of delta
    delta = W_grpo - W_sft
    _, S_d, _ = torch.linalg.svd(delta, full_matrices=False)
    S_d = S_d.cpu().numpy()

    del W_sft, W_grpo, U_s, U_g
    torch.cuda.empty_cache()

    return {
        "n": len(S_s),
        "s_sft": S_s,
        "s_grpo": S_g,
        "s_delta": S_d,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="experiments/rlvr_geometry_paper/results/weight_analysis/8b/eigenvalue_shifts.json")
    args = ap.parse_args()

    print(f"Loading SFT: {SFT_PATH}")
    sft_sd = load_state_dict_from_safetensors(SFT_PATH)
    print(f"  {len(sft_sd)} params")

    results = {}

    for task, path in TASKS_8B.items():
        print(f"\n=== {task} ===")
        grpo_sd = load_state_dict_from_safetensors(path)
        task_results = {}
        for lname in LAYERS_TO_ANALYZE:
            if lname not in sft_sd or lname not in grpo_sd:
                print(f"  skip {lname} (missing)")
                continue
            print(f"  {lname}")
            out = svd_compare(sft_sd[lname], grpo_sd[lname])
            task_results[lname] = {
                "n": out["n"],
                "s_sft": out["s_sft"].tolist(),
                "s_grpo": out["s_grpo"].tolist(),
                "s_delta": out["s_delta"].tolist(),
            }
        results[task] = task_results
        del grpo_sd

    # Summary stats
    print("\n" + "=" * 80)
    print("Summary: relative shift per spectral band")
    print("=" * 80)
    for task, lr in results.items():
        print(f"\n--- {task} ---")
        for lname, d in lr.items():
            s_sft = np.array(d["s_sft"])
            s_grpo = np.array(d["s_grpo"])
            s_delta = np.array(d["s_delta"])
            rel = s_grpo / np.clip(s_sft, 1e-8, None) - 1.0
            n = len(s_sft)
            # bucket by rank band
            bands = [(0, 0.01), (0.01, 0.1), (0.1, 0.5), (0.5, 1.0)]
            sn = lname.replace("model.layers.", "L").replace(".weight", "")
            parts = []
            for lo, hi in bands:
                i0, i1 = int(lo * n), int(hi * n)
                if i1 <= i0:
                    continue
                band_rel = rel[i0:i1]
                parts.append(f"[{lo:.2f}-{hi:.2f}]: {band_rel.mean()*100:+.3f}% (|{np.abs(band_rel).max()*100:.2f}%|)")
            # Relative delta magnitude
            delta_frac = s_delta.mean() / s_sft.mean()
            print(f"  {sn:40} | delta/sft mean = {delta_frac*100:.3f}%")
            for p in parts:
                print(f"    {p}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f)
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
