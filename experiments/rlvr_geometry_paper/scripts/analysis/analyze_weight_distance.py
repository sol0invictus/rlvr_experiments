"""
Analyze weight space distance between GRPO checkpoints and base SFT model.

Inspired by "Neural Thickets" (arxiv 2603.12228) which shows larger models
have RL checkpoints closer to pretrained weights.

Metrics:
1. L2 distance: ||θ_rl - θ_sft||_2
2. Normalized L2: ||θ_rl - θ_sft||_2 / ||θ_sft||_2
3. Cosine distance: 1 - cos(θ_rl, θ_sft) per layer
4. Max parameter shift: max|θ_rl - θ_sft|
5. Layer-wise breakdown

Usage:
    python experiments/rlvr_geometry_paper/scripts/analysis/analyze_weight_distance.py --base_model <sft_path> --output results.json
"""

import argparse
import json
import os
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np
from safetensors import safe_open
from tqdm import tqdm


def load_state_dict_from_safetensors(model_path: str) -> dict:
    """Load state dict from safetensors files."""
    model_path = Path(model_path)
    state_dict = {}

    # Find all safetensor files
    safetensor_files = list(model_path.glob("*.safetensors"))
    if not safetensor_files:
        raise ValueError(f"No safetensors files found in {model_path}")

    for sf_path in safetensor_files:
        with safe_open(sf_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)

    return state_dict


def compute_layer_distances(base_sd: dict, rl_sd: dict) -> dict:
    """
    Compute various distance metrics between base and RL state dicts.

    Returns dict with:
    - global metrics (L2, normalized L2, cosine)
    - per-layer metrics
    - layer-type aggregates (attention, mlp, etc.)
    """
    results = {
        "global": {},
        "per_layer": {},
        "by_layer_type": defaultdict(list),
    }

    # Get common keys
    common_keys = set(base_sd.keys()) & set(rl_sd.keys())
    print(f"Analyzing {len(common_keys)} parameters...")

    # Global metrics
    all_base_params = []
    all_rl_params = []
    all_diffs = []

    for key in tqdm(sorted(common_keys), desc="Computing distances"):
        base_tensor = base_sd[key].float()
        rl_tensor = rl_sd[key].float()

        if base_tensor.shape != rl_tensor.shape:
            print(f"  Skipping {key}: shape mismatch {base_tensor.shape} vs {rl_tensor.shape}")
            continue

        # Flatten for global metrics
        base_flat = base_tensor.flatten()
        rl_flat = rl_tensor.flatten()
        diff = rl_flat - base_flat

        all_base_params.append(base_flat)
        all_rl_params.append(rl_flat)
        all_diffs.append(diff)

        # Per-layer metrics
        l2_dist = torch.norm(diff).item()
        base_norm = torch.norm(base_flat).item()
        normalized_l2 = l2_dist / (base_norm + 1e-8)

        # Cosine distance
        cos_sim = torch.nn.functional.cosine_similarity(
            base_flat.unsqueeze(0), rl_flat.unsqueeze(0)
        ).item()
        cos_dist = 1 - cos_sim

        # Max shift
        max_shift = torch.max(torch.abs(diff)).item()
        mean_shift = torch.mean(torch.abs(diff)).item()

        layer_result = {
            "l2_distance": l2_dist,
            "normalized_l2": normalized_l2,
            "cosine_distance": cos_dist,
            "cosine_similarity": cos_sim,
            "max_shift": max_shift,
            "mean_shift": mean_shift,
            "num_params": base_flat.numel(),
            "base_norm": base_norm,
        }
        results["per_layer"][key] = layer_result

        # Categorize by layer type
        if "self_attn" in key or "attention" in key:
            layer_type = "attention"
        elif "mlp" in key:
            layer_type = "mlp"
        elif "embed" in key:
            layer_type = "embedding"
        elif "norm" in key or "ln" in key:
            layer_type = "norm"
        elif "lm_head" in key:
            layer_type = "lm_head"
        else:
            layer_type = "other"

        results["by_layer_type"][layer_type].append({
            "key": key,
            **layer_result
        })

    # Compute global metrics
    all_base = torch.cat(all_base_params)
    all_rl = torch.cat(all_rl_params)
    all_diff = torch.cat(all_diffs)

    global_l2 = torch.norm(all_diff).item()
    global_base_norm = torch.norm(all_base).item()
    global_normalized_l2 = global_l2 / (global_base_norm + 1e-8)
    global_cos_sim = torch.nn.functional.cosine_similarity(
        all_base.unsqueeze(0), all_rl.unsqueeze(0)
    ).item()

    results["global"] = {
        "l2_distance": global_l2,
        "normalized_l2": global_normalized_l2,
        "cosine_distance": 1 - global_cos_sim,
        "cosine_similarity": global_cos_sim,
        "max_shift": torch.max(torch.abs(all_diff)).item(),
        "mean_shift": torch.mean(torch.abs(all_diff)).item(),
        "std_shift": torch.std(all_diff).item(),
        "total_params": all_base.numel(),
        "base_norm": global_base_norm,
        "rl_norm": torch.norm(all_rl).item(),
    }

    # Aggregate by layer type
    results["layer_type_summary"] = {}
    for layer_type, layers in results["by_layer_type"].items():
        if layers:
            results["layer_type_summary"][layer_type] = {
                "num_layers": len(layers),
                "mean_normalized_l2": np.mean([l["normalized_l2"] for l in layers]),
                "mean_cosine_distance": np.mean([l["cosine_distance"] for l in layers]),
                "total_params": sum(l["num_params"] for l in layers),
                "max_shift_across_layers": max(l["max_shift"] for l in layers),
            }

    # Convert defaultdict to regular dict for JSON serialization
    results["by_layer_type"] = dict(results["by_layer_type"])

    return results


def analyze_all_checkpoints(base_model_path: str, checkpoint_paths: dict) -> dict:
    """
    Analyze distance from base model to all task checkpoints.

    Args:
        base_model_path: Path to SFT base model
        checkpoint_paths: Dict of {task_name: checkpoint_path}

    Returns:
        Dict with results for each task
    """
    print(f"Loading base model from {base_model_path}...")
    base_sd = load_state_dict_from_safetensors(base_model_path)
    print(f"  Loaded {len(base_sd)} parameters")

    all_results = {}

    for task_name, ckpt_path in checkpoint_paths.items():
        print(f"\n{'='*60}")
        print(f"Analyzing: {task_name}")
        print(f"Checkpoint: {ckpt_path}")
        print(f"{'='*60}")

        try:
            rl_sd = load_state_dict_from_safetensors(ckpt_path)
            results = compute_layer_distances(base_sd, rl_sd)
            results["checkpoint_path"] = str(ckpt_path)
            all_results[task_name] = results

            # Print summary
            g = results["global"]
            print(f"\nGlobal metrics:")
            print(f"  L2 distance:        {g['l2_distance']:.4f}")
            print(f"  Normalized L2:      {g['normalized_l2']:.6f}")
            print(f"  Cosine similarity:  {g['cosine_similarity']:.6f}")
            print(f"  Cosine distance:    {g['cosine_distance']:.6f}")
            print(f"  Mean |shift|:       {g['mean_shift']:.6f}")
            print(f"  Max |shift|:        {g['max_shift']:.6f}")

            print(f"\nBy layer type (normalized L2):")
            for lt, stats in results["layer_type_summary"].items():
                print(f"  {lt:12}: {stats['mean_normalized_l2']:.6f}")

        except Exception as e:
            print(f"  ERROR: {e}")
            all_results[task_name] = {"error": str(e)}

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Analyze weight distance from SFT to GRPO checkpoints")
    parser.add_argument("--base_model", type=str, required=True, help="Path to base SFT model")
    parser.add_argument("--checkpoints_dir", type=str, default="results", help="Directory containing GRPO checkpoints")
    parser.add_argument("--output", type=str, default="weight_distance_analysis.json", help="Output JSON file")
    parser.add_argument("--tasks", type=str, default=None, help="Comma-separated list of tasks to analyze (default: all)")
    args = parser.parse_args()

    # Find all GRPO checkpoints
    checkpoints_dir = Path(args.checkpoints_dir)
    checkpoint_paths = {}

    # Default task list
    default_tasks = [
        "grpo_qwen4b_maze",
        "grpo_qwen4b_mini_sudoku",
        "grpo_qwen4b_tower_of_hanoi",
        "grpo_qwen4b_shortest_path",
        "grpo_qwen4b_basic_arithmetic",
        "grpo_qwen4b_syllogism",
        "grpo_qwen_4b_instruct_countdown_4to5",
    ]

    tasks_to_analyze = args.tasks.split(",") if args.tasks else default_tasks

    for task in tasks_to_analyze:
        task_dir = checkpoints_dir / task
        if task_dir.exists():
            # Check if model files exist directly or in a checkpoint subfolder
            if list(task_dir.glob("*.safetensors")):
                checkpoint_paths[task] = str(task_dir)
            else:
                # Look for latest checkpoint
                ckpt_dirs = sorted(task_dir.glob("checkpoint-*"))
                if ckpt_dirs:
                    checkpoint_paths[task] = str(ckpt_dirs[-1])
        else:
            print(f"Warning: {task_dir} not found")

    print(f"Found {len(checkpoint_paths)} checkpoints to analyze")

    # Run analysis
    results = analyze_all_checkpoints(args.base_model, checkpoint_paths)

    # Add summary comparison
    print("\n" + "="*80)
    print("SUMMARY: Weight Distance from SFT Base")
    print("="*80)

    summary_rows = []
    for task, res in results.items():
        if "error" not in res:
            g = res["global"]
            summary_rows.append({
                "task": task.replace("grpo_qwen4b_", "").replace("grpo_qwen_4b_instruct_", ""),
                "normalized_l2": g["normalized_l2"],
                "cosine_distance": g["cosine_distance"],
                "mean_shift": g["mean_shift"],
            })

    # Sort by normalized L2
    summary_rows.sort(key=lambda x: x["normalized_l2"])

    print(f"\n{'Task':<25} {'Norm L2':>12} {'Cos Dist':>12} {'Mean Shift':>12}")
    print("-" * 65)
    for row in summary_rows:
        print(f"{row['task']:<25} {row['normalized_l2']:>12.6f} {row['cosine_distance']:>12.6f} {row['mean_shift']:>12.6f}")

    # Save results
    results["_summary"] = summary_rows

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
