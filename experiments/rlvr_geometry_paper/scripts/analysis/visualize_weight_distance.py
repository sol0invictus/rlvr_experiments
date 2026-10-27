"""
Visualize weight space distance between GRPO checkpoints and base SFT model.

Creates a 2D PCA projection showing relative positions of models in weight space,
similar to Figure 1 in Neural Thickets (arxiv 2603.12228).
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np
from safetensors import safe_open
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tqdm import tqdm


def load_state_dict_from_safetensors(model_path: str) -> dict:
    """Load state dict from safetensors files."""
    model_path = Path(model_path)
    state_dict = {}

    safetensor_files = list(model_path.glob("*.safetensors"))
    if not safetensor_files:
        raise ValueError(f"No safetensors files found in {model_path}")

    for sf_path in safetensor_files:
        with safe_open(sf_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)

    return state_dict


def flatten_state_dict(state_dict: dict, sample_rate: float = 0.01) -> np.ndarray:
    """
    Flatten state dict to a single vector.

    Args:
        state_dict: Model state dict
        sample_rate: Fraction of parameters to sample (for memory efficiency)

    Returns:
        Flattened numpy array
    """
    all_params = []
    for key in sorted(state_dict.keys()):
        tensor = state_dict[key].float().flatten()
        # Sample parameters to reduce memory
        if sample_rate < 1.0 and len(tensor) > 100:
            n_sample = max(1, int(len(tensor) * sample_rate))
            # Use random sampling instead of linspace to avoid edge cases
            step = max(1, len(tensor) // n_sample)
            indices = torch.arange(0, len(tensor), step)[:n_sample]
            tensor = tensor[indices]
        all_params.append(tensor.numpy())

    return np.concatenate(all_params)


def compute_pairwise_distances(vectors: dict) -> dict:
    """Compute pairwise L2 and cosine distances between all model vectors."""
    names = list(vectors.keys())
    n = len(names)

    l2_matrix = np.zeros((n, n))
    cos_matrix = np.zeros((n, n))

    for i, name_i in enumerate(names):
        for j, name_j in enumerate(names):
            v_i = vectors[name_i]
            v_j = vectors[name_j]

            # L2 distance
            l2_matrix[i, j] = np.linalg.norm(v_i - v_j)

            # Cosine similarity
            cos_sim = np.dot(v_i, v_j) / (np.linalg.norm(v_i) * np.linalg.norm(v_j) + 1e-8)
            cos_matrix[i, j] = 1 - cos_sim

    return {
        "names": names,
        "l2": l2_matrix,
        "cosine": cos_matrix
    }


def main():
    parser = argparse.ArgumentParser(description="Visualize weight distance in 2D")
    parser.add_argument("--base_model", type=str, required=True, help="Path to base SFT model")
    parser.add_argument("--checkpoints_dir", type=str, default="experiments/rlvr_geometry_paper/results", help="Directory containing GRPO checkpoints")
    parser.add_argument("--output", type=str, default="weight_distance_2d.png", help="Output plot file")
    parser.add_argument("--sample_rate", type=float, default=0.001, help="Fraction of params to sample (default: 0.1%)")
    parser.add_argument("--tasks", type=str, default=None, help="Comma-separated list of tasks")
    args = parser.parse_args()

    # Task performance data from transfer matrix
    task_performance = {
        "grpo_qwen4b_maze": 0.10,
        "grpo_qwen4b_mini_sudoku": 0.38,
        "grpo_qwen4b_tower_of_hanoi": 0.02,
        "grpo_qwen4b_shortest_path": 0.36,
        "grpo_qwen4b_basic_arithmetic": 0.48,
        "grpo_qwen4b_syllogism": 0.92,
        "grpo_qwen_4b_instruct_countdown_4to5": 0.68,
    }

    # Task categories
    task_categories = {
        "grpo_qwen4b_maze": "spatial",
        "grpo_qwen4b_mini_sudoku": "spatial",
        "grpo_qwen4b_tower_of_hanoi": "procedural",
        "grpo_qwen4b_shortest_path": "spatial",
        "grpo_qwen4b_basic_arithmetic": "symbolic",
        "grpo_qwen4b_syllogism": "symbolic",
        "grpo_qwen_4b_instruct_countdown_4to5": "numeric",
    }

    category_colors = {
        "spatial": "#1f77b4",      # blue
        "symbolic": "#2ca02c",     # green
        "numeric": "#ff7f0e",      # orange
        "procedural": "#d62728",   # red
        "base": "#7f7f7f",         # gray
    }

    # Find checkpoints
    checkpoints_dir = Path(args.checkpoints_dir)
    default_tasks = list(task_performance.keys())
    tasks_to_analyze = args.tasks.split(",") if args.tasks else default_tasks

    checkpoint_paths = {"base_sft": args.base_model}
    for task in tasks_to_analyze:
        task_dir = checkpoints_dir / task
        if task_dir.exists():
            if list(task_dir.glob("*.safetensors")):
                checkpoint_paths[task] = str(task_dir)
            else:
                ckpt_dirs = sorted(task_dir.glob("checkpoint-*"))
                if ckpt_dirs:
                    checkpoint_paths[task] = str(ckpt_dirs[-1])

    print(f"Loading {len(checkpoint_paths)} models...")

    # Load and flatten all models
    vectors = {}
    for name, path in tqdm(checkpoint_paths.items(), desc="Loading models"):
        try:
            sd = load_state_dict_from_safetensors(path)
            vectors[name] = flatten_state_dict(sd, sample_rate=args.sample_rate)
            print(f"  {name}: {len(vectors[name])} sampled params")
        except Exception as e:
            print(f"  ERROR loading {name}: {e}")

    # Stack vectors for PCA
    names = list(vectors.keys())
    X = np.stack([vectors[n] for n in names])

    print(f"\nRunning PCA on {X.shape} matrix...")

    # PCA to 2D
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)

    print(f"Explained variance: {pca.explained_variance_ratio_}")

    # Compute pairwise distances
    distances = compute_pairwise_distances(vectors)

    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Plot 1: 2D PCA projection
    ax1 = axes[0]

    for i, name in enumerate(names):
        x, y = X_2d[i]

        if name == "base_sft":
            color = category_colors["base"]
            marker = "*"
            size = 400
            label = "Base SFT"
        else:
            cat = task_categories.get(name, "other")
            color = category_colors.get(cat, "#999999")
            perf = task_performance.get(name, 0)
            # Size proportional to performance
            size = 100 + perf * 300
            marker = "o"
            label = name.replace("grpo_qwen4b_", "").replace("grpo_qwen_4b_instruct_", "")

        ax1.scatter(x, y, c=color, s=size, marker=marker, edgecolors='black', linewidths=1, alpha=0.8)

        # Add label
        offset = (5, 5)
        if name == "base_sft":
            offset = (-40, -15)
        ax1.annotate(label, (x, y), xytext=offset, textcoords='offset points', fontsize=9)

    # Draw lines from base to each checkpoint
    base_idx = names.index("base_sft")
    for i, name in enumerate(names):
        if name != "base_sft":
            ax1.plot([X_2d[base_idx, 0], X_2d[i, 0]],
                    [X_2d[base_idx, 1], X_2d[i, 1]],
                    'k--', alpha=0.3, linewidth=0.5)

    ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)')
    ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)')
    ax1.set_title('Weight Space: GRPO Checkpoints vs SFT Base\n(size ∝ same-task accuracy)')

    # Legend for categories
    legend_patches = [
        mpatches.Patch(color=category_colors["base"], label='Base SFT'),
        mpatches.Patch(color=category_colors["symbolic"], label='Symbolic (arith, syllogism)'),
        mpatches.Patch(color=category_colors["numeric"], label='Numeric (countdown)'),
        mpatches.Patch(color=category_colors["spatial"], label='Spatial (maze, sudoku, path)'),
        mpatches.Patch(color=category_colors["procedural"], label='Procedural (hanoi)'),
    ]
    ax1.legend(handles=legend_patches, loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Distance vs Performance scatter
    ax2 = axes[1]

    base_vec = vectors["base_sft"]
    for name in names:
        if name == "base_sft":
            continue

        # Compute normalized L2 distance
        dist = np.linalg.norm(vectors[name] - base_vec) / np.linalg.norm(base_vec)
        perf = task_performance.get(name, 0)
        cat = task_categories.get(name, "other")
        color = category_colors.get(cat, "#999999")

        label = name.replace("grpo_qwen4b_", "").replace("grpo_qwen_4b_instruct_", "")
        ax2.scatter(dist, perf, c=color, s=150, edgecolors='black', linewidths=1)
        ax2.annotate(label, (dist, perf), xytext=(5, 5), textcoords='offset points', fontsize=9)

    ax2.set_xlabel('Normalized L2 Distance from Base')
    ax2.set_ylabel('Same-Task Accuracy')
    ax2.set_title('Weight Distance vs Task Performance')
    ax2.grid(True, alpha=0.3)

    # Add trend line
    dists = []
    perfs = []
    for name in names:
        if name != "base_sft":
            dists.append(np.linalg.norm(vectors[name] - base_vec) / np.linalg.norm(base_vec))
            perfs.append(task_performance.get(name, 0))

    if len(dists) > 2:
        z = np.polyfit(dists, perfs, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(dists), max(dists), 100)
        ax2.plot(x_line, p(x_line), 'r--', alpha=0.5, label=f'Trend (slope={z[0]:.1f})')

        # Correlation
        corr = np.corrcoef(dists, perfs)[0, 1]
        ax2.text(0.05, 0.95, f'r = {corr:.2f}', transform=ax2.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax2.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to {args.output}")

    # Print distance matrix
    print("\n" + "="*80)
    print("PAIRWISE L2 DISTANCE MATRIX")
    print("="*80)

    # Print header
    short_names = [n.replace("grpo_qwen4b_", "").replace("grpo_qwen_4b_instruct_", "")[:8] for n in distances["names"]]
    header = f"{'':>12} | " + " | ".join(f"{n:>8}" for n in short_names)
    print(header)
    print("-" * len(header))

    for i, name in enumerate(distances["names"]):
        short = name.replace("grpo_qwen4b_", "").replace("grpo_qwen_4b_instruct_", "")[:12]
        row = [f"{distances['l2'][i,j]:.4f}" for j in range(len(distances["names"]))]
        print(f"{short:>12} | " + " | ".join(f"{r:>8}" for r in row))

    # Save distances to JSON
    output_json = args.output.replace('.png', '_distances.json')
    with open(output_json, 'w') as f:
        json.dump({
            "pca_coords": {names[i]: X_2d[i].tolist() for i in range(len(names))},
            "explained_variance": pca.explained_variance_ratio_.tolist(),
            "l2_matrix": distances["l2"].tolist(),
            "cosine_matrix": distances["cosine"].tolist(),
            "names": distances["names"],
        }, f, indent=2)
    print(f"Distances saved to {output_json}")


if __name__ == "__main__":
    main()
