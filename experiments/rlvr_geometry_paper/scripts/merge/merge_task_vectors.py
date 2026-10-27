"""
Task Vector Merging for Multi-Task RLVR Models.

Implements multiple merging strategies informed by Phase 3 weight analysis:
1. Basic addition: W_sft + τ_A + τ_B
2. Scaled addition: W_sft + α·τ_A + β·τ_B
3. Depth-selective: apply τ_A at A's active depths, τ_B at B's
4. Layer-type selective: only merge up_proj/q_proj (where rotations live)
5. TIES: Trim low-magnitude, elect sign, merge
6. DARE: Drop and rescale

Usage:
    python experiments/rlvr_geometry_paper/scripts/merge/merge_task_vectors.py \
        --base /path/to/sft_model \
        --model_a /path/to/grpo_task_a \
        --model_b /path/to/grpo_task_b \
        --output_dir /path/to/merged \
        --strategy all
"""

import argparse
import json
import os
import shutil
import torch
import numpy as np
from pathlib import Path
from safetensors.torch import load_file, save_file


def load_state_dict(model_path):
    """Load state dict from safetensors or pytorch format."""
    safetensor_files = list(Path(model_path).glob("model*.safetensors"))
    if safetensor_files:
        state_dict = {}
        for f in sorted(safetensor_files):
            state_dict.update(load_file(str(f)))
        return state_dict
    pt_file = Path(model_path) / "pytorch_model.bin"
    if pt_file.exists():
        return torch.load(str(pt_file), map_location="cpu")
    raise FileNotFoundError(f"No model files found in {model_path}")


def compute_task_vector(base_sd, task_sd):
    """Compute τ = W_task - W_base for all shared keys."""
    tau = {}
    for key in base_sd:
        if key in task_sd and base_sd[key].shape == task_sd[key].shape:
            tau[key] = task_sd[key].float() - base_sd[key].float()
    return tau


def get_layer_info(key):
    """Extract layer index and type from parameter name."""
    import re
    layer_match = re.search(r'layers\.(\d+)', key)
    layer_idx = int(layer_match.group(1)) if layer_match else -1

    if 'down_proj' in key:
        layer_type = 'down_proj'
    elif 'up_proj' in key or 'gate_proj' in key:
        layer_type = 'up_proj'
    elif 'q_proj' in key:
        layer_type = 'q_proj'
    elif 'k_proj' in key:
        layer_type = 'k_proj'
    elif 'v_proj' in key:
        layer_type = 'v_proj'
    elif 'o_proj' in key:
        layer_type = 'o_proj'
    else:
        layer_type = 'other'

    return layer_idx, layer_type


def compute_depth_signature(tau):
    """Compute per-layer change mass (Frobenius norm)."""
    layer_norms = {}
    for key, val in tau.items():
        layer_idx, _ = get_layer_info(key)
        if layer_idx >= 0:
            if layer_idx not in layer_norms:
                layer_norms[layer_idx] = 0.0
            layer_norms[layer_idx] += val.norm().item() ** 2
    total = sum(layer_norms.values())
    if total > 0:
        for k in layer_norms:
            layer_norms[k] = layer_norms[k] / total
    return layer_norms


def merge_basic_addition(base_sd, tau_a, tau_b, alpha=1.0, beta=1.0):
    """Strategy 1/2: W = W_sft + α·τ_A + β·τ_B"""
    merged = {}
    for key in base_sd:
        w = base_sd[key].float()
        if key in tau_a:
            w = w + alpha * tau_a[key]
        if key in tau_b:
            w = w + beta * tau_b[key]
        merged[key] = w.to(base_sd[key].dtype)
    return merged


def merge_depth_selective(base_sd, tau_a, tau_b, depth_a, depth_b, num_layers):
    """Strategy 3: apply τ at each task's dominant depths, average at overlap."""
    merged = {}
    for key in base_sd:
        w = base_sd[key].float()
        layer_idx, _ = get_layer_info(key)

        if layer_idx < 0 or (key not in tau_a and key not in tau_b):
            merged[key] = base_sd[key]
            continue

        mass_a = depth_a.get(layer_idx, 0.0)
        mass_b = depth_b.get(layer_idx, 0.0)

        if mass_a + mass_b == 0:
            merged[key] = base_sd[key]
            continue

        # Weight by relative change mass at this depth
        weight_a = mass_a / (mass_a + mass_b)
        weight_b = mass_b / (mass_a + mass_b)

        if key in tau_a:
            w = w + weight_a * tau_a[key]
        if key in tau_b:
            w = w + weight_b * tau_b[key]

        merged[key] = w.to(base_sd[key].dtype)
    return merged


def merge_layer_type_selective(base_sd, tau_a, tau_b):
    """Strategy 4: only merge up_proj and q_proj (where rotations live per Phase 3)."""
    merged = {}
    rotation_types = {'up_proj', 'q_proj', 'gate_proj'}

    for key in base_sd:
        w = base_sd[key].float()
        _, layer_type = get_layer_info(key)

        if layer_type in rotation_types:
            if key in tau_a:
                w = w + tau_a[key]
            if key in tau_b:
                w = w + tau_b[key]

        merged[key] = w.to(base_sd[key].dtype)
    return merged


def merge_ties(base_sd, tau_a, tau_b, density=0.5):
    """Strategy 5: TIES - Trim, Elect Sign, Merge."""
    merged = {}
    for key in base_sd:
        w = base_sd[key].float()

        vecs = []
        for tau in [tau_a, tau_b]:
            if key in tau:
                v = tau[key].clone()
                # Trim: zero out bottom (1-density) by magnitude
                threshold = v.abs().quantile(1 - density)
                v[v.abs() < threshold] = 0
                vecs.append(v)

        if not vecs:
            merged[key] = base_sd[key]
            continue

        # Stack and elect sign (majority vote)
        stacked = torch.stack(vecs)
        sign_sum = stacked.sign().sum(dim=0)
        elected_sign = sign_sum.sign()
        elected_sign[elected_sign == 0] = 1

        # Merge: average only values that agree with elected sign
        result = torch.zeros_like(w)
        count = torch.zeros_like(w)
        for v in vecs:
            mask = (v.sign() == elected_sign) & (v != 0)
            result[mask] += v[mask]
            count[mask] += 1
        count[count == 0] = 1
        result = result / count

        merged[key] = (w + result).to(base_sd[key].dtype)
    return merged


def merge_dare(base_sd, tau_a, tau_b, drop_rate=0.5):
    """Strategy 6: DARE - Drop And REscale."""
    merged = {}
    for key in base_sd:
        w = base_sd[key].float()

        for tau in [tau_a, tau_b]:
            if key in tau:
                v = tau[key].clone()
                # Random drop
                mask = torch.bernoulli(torch.ones_like(v) * (1 - drop_rate))
                # Rescale to preserve expected magnitude
                v = v * mask / (1 - drop_rate)
                w = w + v

        merged[key] = w.to(base_sd[key].dtype)
    return merged


def merge_rotation_compose(base_sd, tau_a, tau_b):
    """Strategy 7: Compose rotations for up_proj/q_proj, add for others.

    For layers where Phase 3 found rotation is the mechanism:
    Compute R_A = (W_sft + τ_A) @ W_sft^+  (rotation from SFT to task A)
    Compute R_B = (W_sft + τ_B) @ W_sft^+  (rotation from SFT to task B)
    Apply W_merged = R_B @ R_A @ W_sft  (compose rotations)
    """
    merged = {}
    rotation_types = {'up_proj', 'q_proj'}

    for key in base_sd:
        w_base = base_sd[key].float()
        _, layer_type = get_layer_info(key)

        has_a = key in tau_a
        has_b = key in tau_b

        if layer_type in rotation_types and has_a and has_b:
            w_a = w_base + tau_a[key]
            w_b = w_base + tau_b[key]

            # Compute pseudo-inverse of W_base (truncated for stability)
            try:
                U, S, Vh = torch.linalg.svd(w_base, full_matrices=False)
                # Truncate small singular values
                k = min(100, (S > S[0] * 1e-4).sum().item())
                S_inv = torch.zeros_like(S)
                S_inv[:k] = 1.0 / S[:k]
                W_pinv = Vh[:k].T @ torch.diag(S_inv[:k]) @ U[:, :k].T

                R_a = w_a @ W_pinv
                R_b = w_b @ W_pinv
                w_merged = R_b @ R_a @ w_base
                merged[key] = w_merged.to(base_sd[key].dtype)
            except Exception:
                # Fallback to simple addition
                w = w_base
                if has_a:
                    w = w + tau_a[key]
                if has_b:
                    w = w + tau_b[key]
                merged[key] = w.to(base_sd[key].dtype)
        else:
            w = w_base
            if has_a:
                w = w + tau_a[key]
            if has_b:
                w = w + tau_b[key]
            merged[key] = w.to(base_sd[key].dtype)

    return merged


def save_merged_model(base_path, merged_sd, output_path):
    """Save merged model with all non-weight files from base."""
    os.makedirs(output_path, exist_ok=True)

    # Copy config, tokenizer, etc from base
    for f in Path(base_path).iterdir():
        if f.suffix not in ('.safetensors', '.bin') and f.name != 'trainer_state.json':
            if f.is_file():
                shutil.copy2(str(f), output_path)

    # Save merged weights
    save_file(merged_sd, os.path.join(output_path, "model.safetensors"))
    print(f"  Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=str, required=True, help="SFT base model")
    parser.add_argument("--model_a", type=str, required=True, help="GRPO model for task A")
    parser.add_argument("--model_b", type=str, required=True, help="GRPO model for task B")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--strategy", type=str, default="all",
                        help="Merge strategy: basic, scaled, depth, layer_type, ties, dare, rotation, all")
    args = parser.parse_args()

    print("Loading models...")
    base_sd = load_state_dict(args.base)
    sd_a = load_state_dict(args.model_a)
    sd_b = load_state_dict(args.model_b)

    print("Computing task vectors...")
    tau_a = compute_task_vector(base_sd, sd_a)
    tau_b = compute_task_vector(base_sd, sd_b)

    # Stats
    norm_a = sum(v.norm().item()**2 for v in tau_a.values())**0.5
    norm_b = sum(v.norm().item()**2 for v in tau_b.values())**0.5
    # Cosine similarity between flattened task vectors
    dot = sum((tau_a[k] * tau_b[k]).sum().item() for k in tau_a if k in tau_b)
    cos_sim = dot / (norm_a * norm_b) if norm_a > 0 and norm_b > 0 else 0
    print(f"  τ_A norm: {norm_a:.4f}")
    print(f"  τ_B norm: {norm_b:.4f}")
    print(f"  cos(τ_A, τ_B): {cos_sim:.4f}")

    depth_a = compute_depth_signature(tau_a)
    depth_b = compute_depth_signature(tau_b)
    num_layers = max(max(depth_a.keys(), default=0), max(depth_b.keys(), default=0)) + 1

    # Top-5 layers per task
    top_a = sorted(depth_a.items(), key=lambda x: -x[1])[:5]
    top_b = sorted(depth_b.items(), key=lambda x: -x[1])[:5]
    print(f"  Task A top layers: {[(l, f'{m:.1%}') for l,m in top_a]}")
    print(f"  Task B top layers: {[(l, f'{m:.1%}') for l,m in top_b]}")

    strategies = args.strategy.split(',') if args.strategy != 'all' else [
        'basic', 'scaled_08', 'scaled_05', 'depth', 'layer_type', 'ties', 'dare', 'rotation'
    ]

    os.makedirs(args.output_dir, exist_ok=True)

    for strategy in strategies:
        print(f"\n--- Strategy: {strategy} ---")
        out_path = os.path.join(args.output_dir, strategy)

        if strategy == 'basic':
            merged = merge_basic_addition(base_sd, tau_a, tau_b, 1.0, 1.0)
        elif strategy == 'scaled_08':
            merged = merge_basic_addition(base_sd, tau_a, tau_b, 0.8, 0.8)
        elif strategy == 'scaled_05':
            merged = merge_basic_addition(base_sd, tau_a, tau_b, 0.5, 0.5)
        elif strategy == 'depth':
            merged = merge_depth_selective(base_sd, tau_a, tau_b, depth_a, depth_b, num_layers)
        elif strategy == 'layer_type':
            merged = merge_layer_type_selective(base_sd, tau_a, tau_b)
        elif strategy == 'ties':
            merged = merge_ties(base_sd, tau_a, tau_b, density=0.5)
        elif strategy == 'dare':
            merged = merge_dare(base_sd, tau_a, tau_b, drop_rate=0.5)
        elif strategy == 'rotation':
            merged = merge_rotation_compose(base_sd, tau_a, tau_b)
        else:
            print(f"  Unknown strategy: {strategy}, skipping")
            continue

        save_merged_model(args.base, merged, out_path)

    print("\n=== All strategies saved ===")


if __name__ == "__main__":
    main()
