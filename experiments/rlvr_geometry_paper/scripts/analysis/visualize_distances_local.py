#!/usr/bin/env python3
"""
Local visualization of RLVR transfer matrix and weight distance metrics.

Works entirely from pre-computed JSON data — no model loading required.

Usage:
    python experiments/rlvr_geometry_paper/scripts/analysis/visualize_distances_local.py
    python experiments/rlvr_geometry_paper/scripts/analysis/visualize_distances_local.py --output my_plots.png
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import FancyArrowPatch
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform


# ── Data paths ──────────────────────────────────────────────────────────────
EXP_DIR = Path(__file__).resolve().parents[2]  # experiments/rlvr_geometry_paper
TRANSFER_MATRIX_PATH = EXP_DIR / "outputs" / "transfer_matrix" / "transfer_matrix_7x7.json"
WEIGHT_DISTANCE_PATH = EXP_DIR / "outputs" / "weight_distance_2d_distances.json"

# ── Short display names ────────────────────────────────────────────────────
NAME_MAP = {
    "grpo_qwen_4b_instruct_countdown_4to5": "countdown",
    "grpo_qwen4b_maze": "maze",
    "grpo_qwen4b_mini_sudoku": "sudoku",
    "grpo_qwen4b_tower_of_hanoi": "hanoi",
    "grpo_qwen4b_shortest_path": "path",
    "grpo_qwen4b_basic_arithmetic": "arith",
    "grpo_qwen4b_syllogism": "syllogism",
    "base_sft": "base",
}

# ── Task cluster categories ────────────────────────────────────────────────
TASK_CATEGORY = {
    "countdown": "numeric",
    "maze": "spatial",
    "sudoku": "spatial",
    "hanoi": "procedural",
    "path": "spatial",
    "arith": "symbolic",
    "syllogism": "symbolic",
    "base": "base",
}

CATEGORY_COLORS = {
    "spatial": "#1f77b4",
    "symbolic": "#2ca02c",
    "numeric": "#ff7f0e",
    "procedural": "#d62728",
    "base": "#7f7f7f",
}

# ── Canonical task order (matches transfer matrix) ─────────────────────────
TASK_ORDER = ["countdown", "maze", "sudoku", "hanoi", "path", "arith", "syllogism"]


def short(name: str) -> str:
    return NAME_MAP.get(name, name)


def load_transfer_matrix():
    with open(TRANSFER_MATRIX_PATH) as f:
        data = json.load(f)
    # Build numpy matrix in canonical task order
    model_to_task = {
        "grpo_qwen_4b_instruct_countdown_4to5": "countdown",
        "grpo_qwen4b_maze": "maze",
        "grpo_qwen4b_mini_sudoku": "sudoku",
        "grpo_qwen4b_tower_of_hanoi": "hanoi",
        "grpo_qwen4b_shortest_path": "path",
        "grpo_qwen4b_basic_arithmetic": "arith",
        "grpo_qwen4b_syllogism": "syllogism",
    }
    task_to_eval = {
        "countdown": "countdown",
        "maze": "maze",
        "sudoku": "mini_sudoku",
        "hanoi": "tower_of_hanoi",
        "path": "shortest_path",
        "arith": "basic_arithmetic",
        "syllogism": "syllogism",
    }
    n = len(TASK_ORDER)
    mat = np.zeros((n, n))
    for model_name, scores in data["matrix"].items():
        row_task = model_to_task[model_name]
        i = TASK_ORDER.index(row_task)
        for col_task in TASK_ORDER:
            j = TASK_ORDER.index(col_task)
            eval_key = task_to_eval[col_task]
            mat[i, j] = scores[eval_key]
    return mat


def load_weight_distances():
    with open(WEIGHT_DISTANCE_PATH) as f:
        raw = f.read().strip()
    # Handle case where opening brace is missing (truncated write)
    if not raw.startswith("{"):
        raw = "{" + raw
    data = json.loads(raw)
    return data


# ── Plot 1: Transfer matrix heatmap ────────────────────────────────────────
def plot_transfer_heatmap(ax, mat):
    n = len(TASK_ORDER)
    im = ax.imshow(mat, cmap="YlOrRd", vmin=0, vmax=1, aspect="equal")

    # Annotate cells
    for i in range(n):
        for j in range(n):
            val = mat[i, j]
            color = "white" if val > 0.6 else "black"
            weight = "bold" if i == j else "normal"
            ax.text(j, i, f"{val:.0%}", ha="center", va="center",
                    fontsize=9, color=color, fontweight=weight)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(TASK_ORDER, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(TASK_ORDER, fontsize=9)
    ax.set_xlabel("Evaluated on", fontsize=10)
    ax.set_ylabel("Trained on", fontsize=10)
    ax.set_title("Transfer Matrix (accuracy)", fontsize=11, fontweight="bold")

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Accuracy", fontsize=9)
    return im


# ── Plot 2: Weight distance heatmap (L2, models only) ─────────────────────
def plot_l2_heatmap(ax, wd_data):
    names = wd_data["names"]
    l2 = np.array(wd_data["l2_matrix"])

    # Reorder: skip base, use canonical task order
    task_names = [n for n in names if n != "base_sft"]
    idx_map = {short(n): names.index(n) for n in task_names}
    ordered = [t for t in TASK_ORDER if t in idx_map]
    indices = [idx_map[t] for t in ordered]

    sub = l2[np.ix_(indices, indices)]
    n = len(ordered)

    im = ax.imshow(sub, cmap="Blues", aspect="equal")
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{sub[i,j]:.3f}", ha="center", va="center", fontsize=8)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(ordered, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(ordered, fontsize=9)
    ax.set_title("Pairwise L2 Distance (weight space)", fontsize=11, fontweight="bold")

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("L2 distance", fontsize=9)


# ── Plot 3: PCA weight space ──────────────────────────────────────────────
def plot_pca_weight_space(ax, wd_data, same_task_acc):
    coords = wd_data["pca_coords"]
    ev = wd_data["explained_variance"]

    for full_name, (x, y) in coords.items():
        name = short(full_name)
        cat = TASK_CATEGORY.get(name, "base")
        color = CATEGORY_COLORS[cat]

        if name == "base":
            ax.scatter(x, y, c=color, s=300, marker="*", edgecolors="black",
                       linewidths=1, zorder=5)
            ax.annotate(name, (x, y), xytext=(-30, -15), textcoords="offset points", fontsize=9)
        else:
            perf = same_task_acc.get(name, 0)
            size = 80 + perf * 300
            ax.scatter(x, y, c=color, s=size, marker="o", edgecolors="black",
                       linewidths=1, zorder=4, alpha=0.85)
            ax.annotate(name, (x, y), xytext=(6, 6), textcoords="offset points", fontsize=9)

    # Draw light lines from base to each checkpoint
    bx, by = coords["base_sft"]
    for full_name, (x, y) in coords.items():
        if full_name != "base_sft":
            ax.plot([bx, x], [by, y], "k--", alpha=0.2, linewidth=0.5)

    ax.set_xlabel(f"PC1 ({ev[0]:.1%} var)", fontsize=10)
    ax.set_ylabel(f"PC2 ({ev[1]:.1%} var)", fontsize=10)
    ax.set_title("Weight Space PCA (size ∝ same-task acc)", fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.3)

    # Category legend
    from matplotlib.patches import Patch
    legend = [Patch(color=CATEGORY_COLORS[c], label=c) for c in
              ["symbolic", "numeric", "spatial", "procedural", "base"]]
    ax.legend(handles=legend, loc="upper right", fontsize=8)


# ── Plot 4: Distance from base vs same-task performance ───────────────────
def plot_dist_vs_perf(ax, wd_data, same_task_acc):
    names = wd_data["names"]
    l2 = np.array(wd_data["l2_matrix"])
    base_idx = names.index("base_sft")

    dists, perfs, labels, colors = [], [], [], []
    for i, full_name in enumerate(names):
        name = short(full_name)
        if name == "base":
            continue
        d = l2[base_idx, i]
        p = same_task_acc.get(name, 0)
        cat = TASK_CATEGORY.get(name, "other")
        dists.append(d)
        perfs.append(p)
        labels.append(name)
        colors.append(CATEGORY_COLORS.get(cat, "#999"))

    ax.scatter(dists, perfs, c=colors, s=120, edgecolors="black", linewidths=1, zorder=4)
    for d, p, lab in zip(dists, perfs, labels):
        ax.annotate(lab, (d, p), xytext=(6, 6), textcoords="offset points", fontsize=9)

    # Trend line + correlation
    dists_arr, perfs_arr = np.array(dists), np.array(perfs)
    if len(dists_arr) > 2:
        z = np.polyfit(dists_arr, perfs_arr, 1)
        x_line = np.linspace(dists_arr.min() * 0.95, dists_arr.max() * 1.05, 50)
        ax.plot(x_line, np.polyval(z, x_line), "r--", alpha=0.5, linewidth=1)
        r = np.corrcoef(dists_arr, perfs_arr)[0, 1]
        ax.text(0.05, 0.95, f"r = {r:.2f}", transform=ax.transAxes, fontsize=10,
                va="top", bbox=dict(boxstyle="round", fc="wheat", alpha=0.5))

    ax.set_xlabel("L2 Distance from Base", fontsize=10)
    ax.set_ylabel("Same-Task Accuracy", fontsize=10)
    ax.set_title("Weight Distance vs Performance", fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.3)


# ── Plot 5: Transfer asymmetry ────────────────────────────────────────────
def plot_transfer_asymmetry(ax, mat):
    """Show |T(A→B) - T(B→A)| — how asymmetric transfer is."""
    n = len(TASK_ORDER)
    asym = np.abs(mat - mat.T)
    # Only upper triangle matters (avoid double counting)
    mask = np.triu(np.ones_like(asym, dtype=bool), k=1)

    im = ax.imshow(asym, cmap="Purples", vmin=0, vmax=0.7, aspect="equal")
    for i in range(n):
        for j in range(n):
            if i == j:
                ax.text(j, i, "—", ha="center", va="center", fontsize=9, color="gray")
            else:
                val = asym[i, j]
                color = "white" if val > 0.45 else "black"
                ax.text(j, i, f"{val:.0%}", ha="center", va="center", fontsize=8, color=color)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(TASK_ORDER, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(TASK_ORDER, fontsize=9)
    ax.set_title("Transfer Asymmetry |T(A→B) − T(B→A)|", fontsize=11, fontweight="bold")
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Asymmetry", fontsize=9)


# ── Plot 6: Dendrogram from transfer similarity ──────────────────────────
def plot_transfer_dendrogram(ax, mat):
    """Cluster tasks by their transfer profile similarity."""
    # Use rows of transfer matrix as feature vectors → cosine-like distance
    # Normalize rows, compute pairwise distance
    n = len(TASK_ORDER)
    # Symmetric similarity: average of T(A→B) and T(B→A)
    sym = (mat + mat.T) / 2.0
    # Convert similarity to distance
    dist = 1.0 - sym
    np.fill_diagonal(dist, 0)
    # Ensure it's a valid distance matrix
    dist = np.maximum(dist, 0)
    condensed = squareform(dist)
    Z = linkage(condensed, method="ward")
    dendrogram(Z, labels=TASK_ORDER, ax=ax, leaf_rotation=45, leaf_font_size=10,
               color_threshold=0.7)
    ax.set_title("Task Clustering (transfer similarity, Ward)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Distance", fontsize=10)


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Visualize RLVR distance metrics locally")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path (default: experiments/rlvr_geometry_paper/outputs/rlvr_distance_dashboard.png)")
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    output_path = args.output or str(EXP_DIR / "outputs" / "rlvr_distance_dashboard.png")

    # Load data
    print("Loading transfer matrix...")
    mat = load_transfer_matrix()

    print("Loading weight distances...")
    wd_data = load_weight_distances()

    # Extract same-task accuracy (diagonal of transfer matrix)
    same_task_acc = {TASK_ORDER[i]: mat[i, i] for i in range(len(TASK_ORDER))}
    print(f"Same-task accuracy: {same_task_acc}")

    # ── Create 3×2 dashboard ──
    fig, axes = plt.subplots(3, 2, figsize=(16, 20))
    fig.suptitle("RLVR Transfer Study — Distance & Transfer Dashboard\n"
                 "Qwen-4B GRPO models, 7 reasoning-gym tasks",
                 fontsize=14, fontweight="bold", y=0.98)

    plot_transfer_heatmap(axes[0, 0], mat)
    plot_l2_heatmap(axes[0, 1], wd_data)
    plot_pca_weight_space(axes[1, 0], wd_data, same_task_acc)
    plot_dist_vs_perf(axes[1, 1], wd_data, same_task_acc)
    plot_transfer_asymmetry(axes[2, 0], mat)
    plot_transfer_dendrogram(axes[2, 1], mat)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=args.dpi, bbox_inches="tight")
    print(f"\nDashboard saved to {output_path}")

    # ── Also print summary stats ──
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)

    diag = np.diag(mat)
    off_diag = mat[~np.eye(len(mat), dtype=bool)]
    print(f"Transfer matrix:")
    print(f"  Diagonal (same-task) mean: {diag.mean():.1%}")
    print(f"  Off-diagonal mean:         {off_diag.mean():.1%}")
    print(f"  Max off-diagonal:          {off_diag.max():.0%} "
          f"({TASK_ORDER[np.unravel_index(np.argmax(mat - np.diag(diag) * np.eye(len(mat))), mat.shape)[0]]} "
          f"→ {TASK_ORDER[np.unravel_index(np.argmax(mat - np.diag(diag) * np.eye(len(mat))), mat.shape)[1]]})")

    # Transfer to each task (column means, excluding diagonal)
    print(f"\nMean transfer TO each task (excl. same-task):")
    for j, task in enumerate(TASK_ORDER):
        col = mat[:, j].copy()
        col[j] = np.nan
        print(f"  {task:>10}: {np.nanmean(col):.1%}")

    # Transfer FROM each task (row means, excluding diagonal)
    print(f"\nMean transfer FROM each task (excl. same-task):")
    for i, task in enumerate(TASK_ORDER):
        row = mat[i, :].copy()
        row[i] = np.nan
        print(f"  {task:>10}: {np.nanmean(row):.1%}")

    # Weight distance summary
    names = wd_data["names"]
    l2 = np.array(wd_data["l2_matrix"])
    base_idx = names.index("base_sft")
    print(f"\nL2 distance from base SFT:")
    for i, n in enumerate(names):
        if n != "base_sft":
            print(f"  {short(n):>10}: {l2[base_idx, i]:.4f}")

    # Correlation between L2 distance and same-task accuracy
    task_dists = []
    task_perfs = []
    for i, n in enumerate(names):
        sn = short(n)
        if sn != "base" and sn in same_task_acc:
            task_dists.append(l2[base_idx, i])
            task_perfs.append(same_task_acc[sn])
    r = np.corrcoef(task_dists, task_perfs)[0, 1]
    print(f"\nCorrelation(L2_from_base, same_task_acc): r = {r:.3f}")


if __name__ == "__main__":
    main()
