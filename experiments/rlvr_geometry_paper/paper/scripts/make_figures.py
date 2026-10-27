"""
Generate paper figures for Phase 2 and Phase 3 sections.

Phase 2 (section 4):
  fig4_1  ARS bar chart + best-regime overlay (the Table-1 visual)
  fig4_2  Per-atom breakdown for 3 illustrative tasks (maze, arith, syllogism)

Phase 3 (section 5):
  fig5_1  (a) Spectral norm ratio ~1.0 across tasks (b) per-band eigenvalue shift
  fig5_2  τ projection ratio vs k, with random baseline line
  fig5_3  Basis rotation by layer type × scale (the headline F3 visual)
  fig5_4  Novel-subspace pairwise angles: heatmap per layer type
  fig5_5  Depth signature heatmap (tasks × depth quartiles) at 8B
  fig5_6  Depth migration within each family (Qwen2.5 3B->7B | Qwen3 4B->8B)

All figures saved to experiments/rlvr_geometry_paper/paper/figures/ as PDF + PNG at 300dpi.
"""
import json, os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm, colors

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 120,
})

# Resolve repo root from this file so the script is portable across machines
# (falls back to the original training-cluster path if that layout is present).
_CLUSTER = Path("/code-fsx/sunguha/rlvr_experiments/experiments/rlvr_geometry_paper")
BASE = str(_CLUSTER if _CLUSTER.exists() else Path(__file__).resolve().parents[2])
OUT = Path(f"{BASE}/paper/figures")
OUT.mkdir(parents=True, exist_ok=True)


def save(fig, name):
    pdf = OUT / f"{name}.pdf"
    png = OUT / f"{name}.png"
    fig.savefig(pdf, bbox_inches="tight", dpi=300)
    fig.savefig(png, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  wrote {pdf.name}, {png.name}")


# -------- Phase 2 figures --------

# Data from the Phase 2 Table 1
P2_TASKS = [
    # (task, ARS, base, direct_rlvr, atomic_only, atomic_warm, best_regime_abbr)
    ("syllogism",       0.74,  79.5, 96.5, 81.0, 97.0, "Warm"),
    ("knights_knaves",  0.625, 78.0, 97.5, 79.0, 95.0, "Direct"),
    ("basic_arith.",    0.66,  72.0, 32.0, 78.0, 70.0, "Atomic"),
    ("countdown",       0.56,  26.0, 70.0, 24.0, 26.0, "Direct"),
    ("maze",            0.207, 21.5, 1.0,  41.0, 38.5, "Atomic"),
    ("shortest_path",   0.14,  10.0, 44.0, 12.0, 46.0, "Warm"),
    ("mini_sudoku",     0.198, 4.0,  25.5, 17.0, 24.0, "Direct"),
]


def fig4_1_ars_and_accuracy():
    """ARS-ordered bar chart of post-training accuracy across the 4 regimes."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.5, 4.8), sharex=True,
                                    gridspec_kw={"height_ratios": [1, 3]})
    rows = sorted(P2_TASKS, key=lambda r: r[1])
    names = [r[0] for r in rows]
    ars = [r[1]*100 for r in rows]
    base = [r[2] for r in rows]
    direct = [r[3] for r in rows]
    atomic = [r[4] for r in rows]
    warm = [r[5] for r in rows]
    best = [r[6] for r in rows]

    # Top: ARS bar
    ax1.bar(range(len(names)), ars, color="#4472C4", edgecolor="black", linewidth=0.5)
    ax1.axhline(25, color="gray", linestyle=":", linewidth=0.8)
    ax1.axhline(60, color="gray", linestyle=":", linewidth=0.8)
    ax1.set_ylabel("ARS (%)")
    ax1.set_ylim(0, 100)
    ax1.text(len(names)-0.4, 25-7, "Low ARS", fontsize=7, color="gray", ha="right")
    ax1.text(len(names)-0.4, 60-7, "Medium", fontsize=7, color="gray", ha="right")
    ax1.text(len(names)-0.4, 90, "High", fontsize=7, color="gray", ha="right")

    # Bottom: 4-regime bar group
    x = np.arange(len(names))
    w = 0.2
    bars1 = ax2.bar(x - 1.5*w, base, w, label="Base (SFT)",    color="#AEAAAA", edgecolor="black", linewidth=0.4)
    bars2 = ax2.bar(x - 0.5*w, direct, w, label="Direct-RLVR", color="#D9534F", edgecolor="black", linewidth=0.4)
    bars3 = ax2.bar(x + 0.5*w, atomic, w, label="Atomic-Only", color="#5CB85C", edgecolor="black", linewidth=0.4)
    bars4 = ax2.bar(x + 1.5*w, warm, w,   label="Atomic-Warm", color="#337AB7", edgecolor="black", linewidth=0.4)

    # Mark the winning regime
    winners = {"Direct": bars2, "Atomic": bars3, "Warm": bars4}
    for i, b in enumerate(best):
        if b in winners:
            bar = winners[b][i]
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                     "*", ha="center", fontsize=13, color="black", fontweight="bold")

    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=20, ha="right")
    ax2.set_ylabel("Full-task accuracy (%)")
    ax2.set_ylim(0, 110)
    ax2.legend(loc="upper left", ncol=4, frameon=False, handlelength=1.2)
    ax2.grid(axis="y", linestyle=":", alpha=0.4)

    plt.subplots_adjust(hspace=0.05)
    save(fig, "fig4_1_ars_regime_comparison")


def fig4_2_atom_breakdown():
    """Per-atom L1/L2 accuracy for 3 illustrative tasks (maze, arithmetic, syllogism)."""
    # Data taken from Phase 2 report and from experiments/rlvr_geometry_paper/paper/data/ars/4b/ probe outputs
    tasks = {
        "maze (low ARS)": {
            "L1": [("locate_start", 0), ("locate_goal", 0), ("read_cell", 54), ("grid_size", 2)],
            "L2": [("valid_moves", 8), ("is_move_valid", 62), ("shortest_path_length", 12)],
        },
        "basic_arithmetic (high ARS, Direct fails)": {
            "L1": [("number_extraction", 62), ("operator_id", 70), ("term_count", 40), ("has_parens", 98)],
            "L2": [("single_op_eval", 76), ("order_of_ops", 2), ("innermost_eval", 82), ("sign_handling", 100)],
        },
        "syllogism (high ARS)": {
            "L1": [("premise1_subj", 100), ("premise2_subj", 100), ("quantifier_type", 86), ("conclusion_pred", 98)],
            "L2": [("single_hop_inf", 46), ("negation", 42), ("validity_check", 68)],
        },
    }
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), sharey=True)
    for ax, (name, layers) in zip(axes, tasks.items()):
        ax.set_title(name, fontsize=9)
        l1 = layers["L1"]; l2 = layers["L2"]
        # stack them, color-code L1/L2
        labels = [n for n, _ in l1] + [n for n, _ in l2]
        vals = [v for _, v in l1] + [v for _, v in l2]
        colors_ = ["#4472C4"]*len(l1) + ["#ED7D31"]*len(l2)
        y = np.arange(len(labels))
        ax.barh(y, vals, color=colors_, edgecolor="black", linewidth=0.3)
        for i, v in enumerate(vals):
            ax.text(v+1, i, f"{v}%", va="center", fontsize=7)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=7)
        ax.invert_yaxis()
        ax.set_xlim(0, 115)
        ax.axvline(50, color="gray", linestyle=":", linewidth=0.5)

    # Legend
    from matplotlib.patches import Patch
    axes[-1].legend(handles=[Patch(color="#4472C4", label="L1 (grounding)"),
                              Patch(color="#ED7D31", label="L2 (local reasoning)")],
                     loc="lower right", frameon=False, fontsize=7)
    fig.text(0.5, -0.01, "zero-shot atomic accuracy (%)", ha="center", fontsize=9)
    save(fig, "fig4_2_atom_breakdown")


# -------- Phase 3 figures --------

def load_json(p):
    return json.load(open(p))


def fig5_1_spectrum_preserved():
    """Left: spectral norm ratio ≈ 1 across 9 tasks (8B).
       Right: per-band eigenvalue shift < 0.1%."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 3.2))
    # Left
    data = load_json(f"{BASE}/results/weight_analysis/8b/eigenvalue_shifts.json")
    task_ratios = {}
    for task, layers in data.items():
        ratios = []
        for layer_name, ld in layers.items():
            s_sft = np.array(ld["s_sft"])
            s_grpo = np.array(ld["s_grpo"])
            if len(s_sft) == 0: continue
            ratios.append(s_grpo[0] / s_sft[0])
        task_ratios[task] = np.array(ratios)
    names = list(task_ratios.keys())
    means = [task_ratios[n].mean() for n in names]
    stds = [task_ratios[n].std() for n in names]
    y = np.arange(len(names))
    ax1.errorbar(means, y, xerr=stds, fmt="o", color="black", capsize=3)
    ax1.axvline(1.0, color="red", linestyle="--", linewidth=0.8)
    ax1.set_yticks(y); ax1.set_yticklabels(names, fontsize=7)
    ax1.set_xlabel(r"$\sigma_1(W_{\mathrm{GRPO}})\,/\,\sigma_1(W_{\mathrm{SFT}})$")
    ax1.set_xlim(0.995, 1.008)
    ax1.set_title("(a) Spectral norm ratio per task (8B)")
    ax1.grid(axis="x", linestyle=":", alpha=0.4)

    # Right: per-band eigenvalue shift
    bands = ["top 1%", "1–10%", "10–50%", "50–100%"]
    shifts_by_band = {b: [] for b in bands}
    for task, layers in data.items():
        for layer_name, ld in layers.items():
            s_sft = np.array(ld["s_sft"])
            s_grpo = np.array(ld["s_grpo"])
            n = len(s_sft)
            if n == 0: continue
            for lo, hi, bname in [(0, 0.01, bands[0]), (0.01, 0.1, bands[1]),
                                  (0.1, 0.5, bands[2]), (0.5, 1.0, bands[3])]:
                idx = slice(int(lo*n), int(hi*n))
                s_s = s_sft[idx]; s_g = s_grpo[idx]
                if len(s_s) == 0: continue
                # avoid div by zero for tail
                nonzero = s_s > 1e-8
                if nonzero.sum() == 0: continue
                rel_shift = np.abs(s_g[nonzero] - s_s[nonzero]) / s_s[nonzero]
                shifts_by_band[bname].extend(rel_shift.tolist())
    box_data = [np.array(shifts_by_band[b]) * 100 for b in bands]
    box_data = [d[np.isfinite(d)] for d in box_data]
    bp = ax2.boxplot(box_data, vert=True, labels=bands, showfliers=False,
                      patch_artist=True, widths=0.5)
    for patch in bp["boxes"]:
        patch.set_facecolor("#4472C4")
        patch.set_alpha(0.6)
    ax2.axhline(0.1, color="red", linestyle="--", linewidth=0.8, label="0.1%")
    ax2.set_ylabel(r"Relative shift $|\sigma_i^{\mathrm{GRPO}} - \sigma_i^{\mathrm{SFT}}| / \sigma_i^{\mathrm{SFT}}$ (%)")
    ax2.set_xlabel("Singular-value band (fraction of full rank)")
    ax2.set_title("(b) Per-band eigenvalue shift (8B)")
    ax2.set_ylim(0, 1.0)
    ax2.legend(frameon=False, loc="upper left")
    ax2.grid(axis="y", linestyle=":", alpha=0.4)
    plt.tight_layout()
    save(fig, "fig5_1_spectrum_preserved")


def fig5_2_tau_orthogonal():
    """τ projection ratio vs k vs random baseline, all scales."""
    # From Phase 3 report §4.1
    k_vals = [10, 100, 500, 1000]
    data = {
        "3B": [0.0003, 0.0034, 0.0288, 0.0917],
        "4B": [0.0001, 0.0016, 0.0155, 0.0532],
        "7B": [0.00001, 0.0004, 0.0069, 0.0259],
        "8B": [0.0001, 0.0014, 0.0102, 0.0320],
    }
    # rough ambient rank ~2560 for 4B, ~4096 for 8B etc.
    ambient = {"3B": 2048, "4B": 2560, "7B": 3584, "8B": 4096}
    fig, ax = plt.subplots(figsize=(4.8, 3.3))
    colors_ = {"3B": "#AEAAAA", "4B": "#4472C4", "7B": "#70AD47", "8B": "#ED7D31"}
    for scale, vals in data.items():
        ax.plot(k_vals, [v*100 for v in vals], "-o", color=colors_[scale], label=scale, linewidth=1.2)
    # Random baseline for 4B (representative)
    r = ambient["4B"]
    random_baseline = [(k/r)**2 * 100 for k in k_vals]
    ax.plot(k_vals, random_baseline, "--", color="gray", label="random (k/r)², r=2560",
            linewidth=1.0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("k (subspace dimension)")
    ax.set_ylabel(r"Two-sided projection $\rho_k$ (%)")
    ax.set_title("τ is near-orthogonal to W's top subspace")
    ax.legend(frameon=False)
    ax.grid(linestyle=":", alpha=0.4)
    plt.tight_layout()
    save(fig, "fig5_2_tau_orthogonal")


def fig5_3_layer_type_rotation():
    """Rotation at k=100 per layer type × scale. The headline."""
    scales = ["3B", "4B", "7B", "8B"]
    down   = [0.01, 0.02, 0.00, 0.00]
    up     = [1.47, 1.99, 2.49, 2.67]
    q      = [1.65, 1.81, 2.03, 2.07]
    x = np.arange(len(scales))
    w = 0.27
    fig, ax = plt.subplots(figsize=(5.6, 3.3))
    ax.bar(x - w, down, w, label=r"$\mathrm{mlp.down\_proj}$ (sink)",    color="#AEAAAA", edgecolor="black", linewidth=0.4)
    ax.bar(x,     up,   w, label=r"$\mathrm{mlp.up\_proj}$ (feature projector)", color="#4472C4", edgecolor="black", linewidth=0.4)
    ax.bar(x + w, q,    w, label=r"$\mathrm{attn.q\_proj}$ (feature projector)", color="#ED7D31", edgecolor="black", linewidth=0.4)
    for i, v in enumerate(down):
        ax.text(i-w, v+0.05, "≈0°", ha="center", fontsize=7, color="black")
    ax.set_xticks(x); ax.set_xticklabels(scales)
    ax.set_ylabel("Principal angle at k=100 (degrees)")
    ax.set_xlabel("Model scale")
    ax.set_title("Basis rotation is layer-type-split and scale-monotonic")
    ax.legend(frameon=False, loc="upper left", fontsize=7.5)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.set_ylim(0, 3.2)
    plt.tight_layout()
    save(fig, "fig5_3_layer_type_rotation")


def fig5_4_novel_subspace_angles():
    """Heatmap of pairwise angle between novel subspaces, one panel per layer type."""
    # From Phase 3 report §5.3 table
    pairs = ["cnt×ari", "cnt×csr", "cnt×syl", "ari×csr", "ari×syl", "csr×syl"]
    data_layers = {
        "mlp.down_proj": {
            "L0":    [18.89, 18.51, 17.70, 19.10, 18.42, 17.96],
            "L2":    [25.61, 26.75, 24.63, 26.24, 24.86, 26.18],
            "L17":   [22.85, 22.87, 22.27, 22.64, 22.15, 21.77],
            "L18":   [20.30, 20.24, 19.66, 20.05, 19.13, 18.50],
            "L27":   [29.59, 30.01, 28.74, 29.33, 27.64, 27.93],
            "L35":   [28.97, 30.49, 28.16, 29.22, 26.94, 28.04],
        },
        "mlp.up_proj": {
            "L6":    [42.33, 42.78, 40.80, 42.31, 40.21, 40.77],
            "L9":    [43.32, 42.98, 41.47, 42.07, 40.78, 40.87],
            "L17":   [39.88, 39.44, 39.13, 39.15, 37.83, 37.99],
            "L19":   [36.74, 35.90, 35.32, 35.45, 34.61, 34.12],
        },
        "attn.q_proj": {
            "L0":    [45.48, 46.08, 44.65, 46.58, 44.59, 44.69],
            "L35":   [42.09, 42.49, 39.80, 41.18, 38.01, 37.94],
        },
    }
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), sharex=True)
    cmap = plt.cm.viridis
    vmin, vmax = 15, 50
    for ax, (ltype, layers) in zip(axes, data_layers.items()):
        mat = np.array(list(layers.values()))
        labels = list(layers.keys())
        im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(pairs)))
        ax.set_xticklabels(pairs, rotation=25, ha="right", fontsize=7)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=7)
        mean_val = mat.mean()
        ax.set_title(f"{ltype}  (mean {mean_val:.1f}°)", fontsize=9)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(j, i, f"{mat[i,j]:.0f}", ha="center", va="center",
                        color="white" if mat[i,j] < 35 else "black", fontsize=6)
    # single colorbar on the right
    cbar = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02)
    cbar.set_label("Pairwise angle between novel subspaces (degrees)", fontsize=8)
    save(fig, "fig5_4_novel_subspace_angles")


def fig5_5_depth_signatures():
    """8B depth quartile heatmap across tasks."""
    # From Phase 3 §5.7 Table 5
    rows = [
        ("caesar_cipher",   88.1, 11.9, 0.0,  0.0),
        ("syllogism",       40.8, 54.7, 4.5,  0.0),
        ("knights_knaves",  34.5, 51.5, 14.0, 0.0),
        ("leg_counting",    39.6, 55.9, 4.6,  0.0),
        ("shortest_path",   28.3, 62.3, 9.4,  0.0),
        ("basic_arith.",    34.9, 18.6, 46.5, 0.0),
        ("countdown",       28.5, 19.0, 52.5, 0.0),
        ("mini_sudoku",     46.4, 9.7,  39.0, 4.9),
    ]
    names = [r[0] for r in rows]
    mat = np.array([r[1:] for r in rows])
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    im = ax.imshow(mat, aspect="auto", cmap="YlOrRd", vmin=0, vmax=90)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=8)
    ax.set_xticks(range(4))
    ax.set_xticklabels(["early\nL0-8", "mid-early\nL9-17", "mid-late\nL18-26", "late\nL27-35"],
                        fontsize=8)
    ax.set_title("8B: depth-quartile concentration (% of top-20 layer change mass)")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i,j]
            if v > 0.5:
                ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                        color="white" if v > 40 else "black", fontsize=7.5)
    # Cluster separator lines
    ax.axhline(0.5, color="black", linewidth=1.0)  # caesar cluster
    ax.axhline(4.5, color="black", linewidth=1.0)  # mid-early vs mid-late split
    ax.axhline(6.5, color="black", linewidth=1.0)  # sudoku split
    fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02, label="% of change mass")
    plt.tight_layout()
    save(fig, "fig5_5_depth_signatures_8b")


def fig5_6_scale_dependent_depth():
    """Depth migration within each model family, shown separately to remove the
    Qwen2.5-vs-Qwen3 confound. Each task's *mean depth* (center of mass over the
    four depth bands early/mid-early/mid-late/late, coded 0..3) is plotted at the
    family's smaller scale (open marker) and larger scale (filled marker); the
    connecting arrow shows the direction of movement. Within Qwen2.5 (3B->7B)
    tasks broadly deepen; within Qwen3 (4B->8B) movement goes both ways.

    Depth bins are computed from the per-task top-20 most-changed matrices in
    experiments/rlvr_geometry_paper/results/weight_analysis/{scale}/analysis_results.json (phase_a.task_magnitudes
    .<task>.per_layer), bucketed into quartiles of network depth and normalized.
    The 4-band fractions (%) used to form the means are quoted below.
    """
    # band fractions [early, mid-early, mid-late, late] per task per scale
    frac = {
        "3B": {"basic_arithmetic": [87, 13, 0, 0], "shortest_path": [53, 0, 47, 0],
               "caesar_cipher": [88, 12, 0, 0],     "syllogism": [83, 17, 0, 0]},
        "7B": {"basic_arithmetic": [41, 59, 0, 0],  "shortest_path": [30, 10, 0, 60],
               "caesar_cipher": [72, 9, 0, 18],      "syllogism": [47, 53, 0, 0]},
        "4B": {"basic_arithmetic": [39, 61, 0, 0],  "shortest_path": [54, 37, 9, 0],
               "caesar_cipher": [58, 14, 28, 0],     "syllogism": [5, 95, 0, 0]},
        "8B": {"basic_arithmetic": [35, 19, 46, 0], "shortest_path": [28, 62, 9, 0],
               "caesar_cipher": [88, 12, 0, 0],      "syllogism": [41, 55, 5, 0]},
    }
    tasks = ["basic_arithmetic", "shortest_path", "syllogism", "caesar_cipher"]
    families = [("Qwen2.5", "3B", "7B"), ("Qwen3", "4B", "8B")]
    bands = ["early", "mid-early", "mid-late", "late"]

    def com(f):  # center of mass over band index 0..3
        tot = sum(f)
        return sum(i * v for i, v in enumerate(f)) / tot if tot else float("nan")

    c_deeper, c_shallow = "#C0392B", "#2471A3"
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.2), sharex=True, sharey=True)
    for ax, (fam, s_small, s_big) in zip(axes, families):
        for yi, task in enumerate(tasks):
            x0, x1 = com(frac[s_small][task]), com(frac[s_big][task])
            col = c_deeper if x1 >= x0 else c_shallow
            ax.annotate("", xy=(x1, yi), xytext=(x0, yi),
                        arrowprops=dict(arrowstyle="-|>", color=col, lw=2.0,
                                        shrinkA=4, shrinkB=4))
            ax.scatter([x0], [yi], facecolors="white", edgecolors=col, s=55, zorder=3)
            ax.scatter([x1], [yi], facecolors=col, edgecolors=col, s=55, zorder=3)
        ax.set_title(f"{fam}  ({s_small}$\\to${s_big})")
        ax.set_xticks(range(4)); ax.set_xticklabels(bands, rotation=15)
        ax.set_xlim(-0.3, 3.3)
        ax.grid(axis="x", linestyle=":", alpha=0.4)
    axes[0].set_yticks(range(len(tasks)))
    axes[0].set_yticklabels([t.replace("_", "\n") for t in tasks])
    axes[0].set_ylim(-0.6, len(tasks) - 0.4)
    axes[0].invert_yaxis()
    # shared legend: marker shape = scale, color = direction
    from matplotlib.lines import Line2D
    handles = [
        Line2D([], [], marker="o", color="0.4", mfc="white", ls="", label="smaller scale"),
        Line2D([], [], marker="o", color="0.4", mfc="0.4", ls="", label="larger scale"),
        Line2D([], [], color=c_deeper, lw=2, label="moves deeper"),
        Line2D([], [], color=c_shallow, lw=2, label="moves shallower"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.06), fontsize=8)
    fig.supxlabel("mean depth of GRPO change (band center of mass)", fontsize=9, y=0.02)
    plt.tight_layout()
    save(fig, "fig5_6_scale_depth_migration")


# ---- run everything ----
if __name__ == "__main__":
    print("Generating Phase 2 figures...")
    fig4_1_ars_and_accuracy()
    fig4_2_atom_breakdown()
    print("Generating Phase 3 figures...")
    fig5_1_spectrum_preserved()
    fig5_2_tau_orthogonal()
    fig5_3_layer_type_rotation()
    fig5_4_novel_subspace_angles()
    fig5_5_depth_signatures()
    fig5_6_scale_dependent_depth()
    print(f"\nAll figures in {OUT}")
