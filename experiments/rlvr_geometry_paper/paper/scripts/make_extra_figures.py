#!/usr/bin/env python3
"""Generate the figures not produced by paper/scripts/make_figures.py:
the Phase-1 transfer heatmaps, the Phase-4 merge null-predictor / worked panels, the
Phase-2 shortest_path climb, and the Phase-3 per-task depth-migration appendix figure.
Most data is transcribed from the reports; the Phase-3 figure (fig_p3_pertask_depth) is
instead computed live from ../results/weight_analysis/{scale}/
analysis_results.json and is skipped gracefully if that raw data is absent."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import os
PAPER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # paper/
OUT = os.path.join(PAPER, "figures")
LAB = ["cnt", "mz", "sdk", "pth", "ari", "syl", "kk", "leg"]

M3B = np.array([
    [ 10, -4,  0,  0, -2, 10,  8,   0],   # countdown
    [  0,  2,  0,  4, -4,  0, -2, -12],   # maze
    [  0, -8,  0,  2, -8, -4, -4,  -6],   # mini_sudoku
    [  0, -2,  0, 40, -6, -8, -2,   0],   # shortest_path
    [  0, -8,  0,  0,-14, 10, -2, -10],   # basic_arith
    [  0, -4,  0,  0,-20, 20,  4, -12],   # syllogism
    [  0,-12,  0,  0, -2,  4, 18,  -6],   # knights_knaves
    [  0, -4,  0,  2,-16,  0, -8,  12],   # leg_counting
], dtype=float)

M4B = np.array([
    [ 36,  6, 14,  4,  6, 18,  4,   6],
    [-10, 12, 16,  4,  0, 12,  0,   6],
    [-16, -2, 22,-16,-26,-16,-50, -52],
    [  6,  8, 28, 28,  6, 16, -2,  14],
    [  4, -2, -4, -6,-12, 18,-16, -32],
    [ -8, -2, 26, -2, -2, 26,  2,  10],
    [ -8,  2,  4,-12, -8, 26, 12,  -4],
    [ 16,  2, 22, 12, -8, 14, -8,  28],
], dtype=float)

# SFT baseline accuracies (absolute %) per evaluated task, same column order as LAB.
B3B   = [ 0, 14,  0,  0, 46, 74, 26, 12]
B4B   = [34,  2, 14, 16, 62, 66, 74, 52]
B7Bv2 = [ 4, 12,  4,  2, 60, 64, 30, 36]
B8Bv2 = [22, 10,  0, 12, 62, 74, 74, 52]

def heatmap(ax, M, title, vmax=None, baseline=None):
    if vmax is None:
        vmax = np.abs(M).max()
    im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="equal")
    ax.set_xticks(range(8)); ax.set_xticklabels(LAB, fontsize=8)
    ax.set_xlabel("evaluated on", fontsize=9)
    ax.set_ylabel("trained on", fontsize=9)
    ax.set_title(title, fontsize=10)
    for i in range(8):
        for j in range(8):
            v = M[i, j]
            fw = "bold" if i == j else "normal"
            col = "white" if abs(v) > 0.65 * vmax else "black"
            ax.text(j, i, f"{int(v):+d}", ha="center", va="center",
                    fontsize=6.5, color=col, fontweight=fw)
    if baseline is not None:
        # Append the SFT baseline (absolute accuracy) as a separate annotation row
        # below the delta matrix, on its own so it is not read as a delta color.
        ax.axhline(7.5, color="black", lw=1.0)
        for j, b in enumerate(baseline):
            ax.text(j, 8.05, f"{int(b)}", ha="center", va="center",
                    fontsize=6.5, color="#333333")
        ax.set_yticks(list(range(8)) + [8])
        ax.set_yticklabels(LAB + ["SFT %"], fontsize=8)
        ax.set_ylim(8.7, -0.6)
    else:
        ax.set_yticks(range(8)); ax.set_yticklabels(LAB, fontsize=8)
    return im

# 7B/8B at the gentle lr=1e-6 (v2).
M7Bv2 = np.array([
    [ -2,  4,  4,  0,  0,  2,  4, -6],   # countdown
    [ -4,  6,  6,  4,  0,  2,  2, -6],   # maze
    [ -4,  0,  8,  4,  0, -2,  2, -2],   # mini_sudoku
    [ -2,  2,  8,  8, -2,  0,  2, -6],   # shortest_path
    [ -2,  2,  2,  4, -4,  2,  0, -2],   # basic_arith
    [ -4,  2,  4,  0, -2,  8,  2, -4],   # syllogism
    [ -2,  0,  6,  2,  0, -4,  4,  0],   # knights_knaves
    [ -4,  2,  4,  4,  0,  0, -4,  2],   # leg_counting
], dtype=float)

M8Bv2 = np.array([
    [  4, -2,  6, -8,  2,  0,-12, -2],   # countdown
    [  0,  0,  4, -6,  2,  0,-10, -2],   # maze
    [  4,  2,  2, -2,  4,  4, -6,  0],   # mini_sudoku
    [ -4, -4,  0, -4,  2,  4, -8, -2],   # shortest_path
    [  2,  2,  0, -6,  4,  2, -8, -2],   # basic_arith
    [  6, -2,  0, -2,  4,  2,-10,  0],   # syllogism
    [  6, -4,  4, -6,  4,  0,-14, -2],   # knights_knaves
    [  2, -4,  0, -6,  4,  2,-18,  0],   # leg_counting
], dtype=float)

# Unified 2x2 figure: all four scales on a SHARED +/-52 color scale so the
# aggressive-lr (top) vs gentle-lr (bottom) contrast is honest at a glance.
VMAX = 52
fig, axes = plt.subplots(2, 2, figsize=(9.6, 9.4))
heatmap(axes[0, 0], M3B,   "Qwen2.5-3B  ($\\mathrm{lr}=5{\\times}10^{-6}$)", vmax=VMAX, baseline=B3B)
heatmap(axes[0, 1], M4B,   "Qwen3-4B  ($\\mathrm{lr}=5{\\times}10^{-6}$)",   vmax=VMAX, baseline=B4B)
heatmap(axes[1, 0], M7Bv2, "Qwen2.5-7B  ($\\mathrm{lr}=1{\\times}10^{-6}$)", vmax=VMAX, baseline=B7Bv2)
im = heatmap(axes[1, 1], M8Bv2, "Qwen3-8B  ($\\mathrm{lr}=1{\\times}10^{-6}$)", vmax=VMAX, baseline=B8Bv2)
cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
cbar.set_label(r"$\Delta$ accuracy vs. SFT (pts)", fontsize=9)
fig.suptitle("Cross-task GRPO transfer at four scales (shared $\\pm52$ scale); "
             "diagonal = same-task; bottom row of each panel = SFT baseline accuracy",
             fontsize=11)
fig.savefig(f"{OUT}/fig_p1_transfer.pdf", bbox_inches="tight")
plt.close(fig)

# ---------------- Phase 4 ----------------
sign = np.array([0.530,0.531,0.564,0.605,0.536,0.513,0.493,
                 0.538,0.512,0.500,0.551,0.526,
                 0.547,0.508,0.531,0.522,0.507,0.521,0.530])
dmg  = np.array([18.0,22.0,12.5,4.5,7.0,18.0,24.0,
                 7.5,22.0,0.5,0.5,19.5,
                 21.0,20.0,64.5,10.5,40.0,45.5,51.0])
names = [f"T{i}" for i in range(1, 20)]

fig, (axA, axB) = plt.subplots(1, 2, figsize=(9.4, 4.2))

# (a) scatter with shrinking correlation
axA.scatter(sign[:7], dmg[:7], c="#c0392b", s=42, zorder=3, label="original $n{=}7$")
axA.scatter(sign[7:], dmg[7:], c="#2c3e50", s=42, marker="s", zorder=3,
            label="added pairs ($n{=}19$)")
for x, y, nm in zip(sign, dmg, names):
    axA.annotate(nm, (x, y), fontsize=5.5, xytext=(2, 2), textcoords="offset points")
xs = np.linspace(sign.min()-0.005, sign.max()+0.005, 50)
for idx, col, lab in [(7, "#c0392b", "fit $n{=}7$ ($r{=}{-}0.81$)"),
                      (19, "#2c3e50", "fit $n{=}19$ ($r{=}{-}0.27$)")]:
    b, a = np.polyfit(sign[:idx], dmg[:idx], 1)
    axA.plot(xs, a + b*xs, color=col, lw=1.6, ls="--", alpha=0.8, label=lab)
axA.set_xlabel("3B magnitude-weighted sign-agreement", fontsize=9)
axA.set_ylabel("4B merge damage (pts)", fontsize=9)
axA.set_title("(a) The predictor that wasn't", fontsize=10)
axA.legend(fontsize=7, loc="upper right")
axA.grid(alpha=0.25)

# (b) predictor decay with bootstrap CI
n = np.array([7, 12, 19])
r_sa = np.array([-0.810, -0.447, -0.267])
lo = np.array([-0.993, -0.872, -0.591]); hi = np.array([-0.376, 0.161, 0.140])
axB.errorbar(n, r_sa, yerr=[r_sa-lo, hi-r_sa], fmt="o-", color="#2c3e50",
             capsize=4, lw=1.8, label="sign-agreement (95% CI)")
axB.plot([7, 12], [-0.761, 0.058], "s--", color="#c0392b", lw=1.8,
         label="L1-grounding ARS")
axB.axhline(0, color="k", lw=0.8)
axB.set_xticks(n); axB.set_xlabel("number of merge pairs $n$", fontsize=9)
axB.set_ylabel("Pearson $r$ vs. merge damage", fontsize=9)
axB.set_title("(b) Both predictors collapse with $n$", fontsize=10)
axB.set_ylim(-1.05, 0.45)
axB.legend(fontsize=7, loc="lower right")
axB.grid(alpha=0.25)

fig.tight_layout()
fig.savefig(f"{OUT}/fig_p4_merge.pdf", bbox_inches="tight")
plt.close(fig)

# ---------------- Phase 4: what worked ----------------
# Three constructive results, one panel each. Data transcribed from
# report_phase4_merge_scaling.md (Parts 1, 3-5).
figW, (wA, wB, wC) = plt.subplots(1, 3, figsize=(13.6, 4.1))

# (a) Merge budget: summation composes, 1/N under-shoots (Table tab:p4-part1).
tasks_a = ["arith", "sp", "syl", "sdk", "leg", "mean*"]
sft_a   = [72.0, 10.0, 79.5,  4.0, 52.0, 42.9]
invN_a  = [69.5, 24.0, 59.5, 14.0, 45.0, 41.6]
s033_a  = [71.0, 46.5, 71.5, 28.0, 68.0, 51.3]
two_a   = [76.0, 51.0, 83.0, 17.0, 57.0, 51.2]
x = np.arange(len(tasks_a)); w = 0.2
wA.bar(x - 1.5*w, sft_a,  w, label="SFT base",        color="#95a5a6")
wA.bar(x - 0.5*w, invN_a, w, label="$1/N$ (10$\\tau$)", color="#c0392b")
wA.bar(x + 0.5*w, s033_a, w, label="$s{=}0.33$ (9$\\tau$)", color="#2980b9")
wA.bar(x + 1.5*w, two_a,  w, label="2-task sum",       color="#27ae60")
wA.set_xticks(x); wA.set_xticklabels(tasks_a, fontsize=8)
wA.set_ylabel("accuracy (%)", fontsize=9)
wA.set_title("(a) Summation composes; $1/N$ under-shoots", fontsize=10)
wA.legend(fontsize=7, loc="upper right", ncol=2)
wA.grid(alpha=0.25, axis="y")
wA.text(0.012, 0.97, "mean* = 8 non-trivial tasks", transform=wA.transAxes,
        fontsize=6.5, va="top", color="#555555")

# (b) Dropping mlp.down_proj is neutral-to-positive (Part 4 Delta-from-full table).
tasks_b = ["arith", "syl", "leg", "sp", "sdk", "cnt", "kk", "mz"]
abd_3b  = [ 2.5, 8.5, 1.0, 0.0, 0.0, -0.5, 1.5,  1.5]
abd_4b  = [ 1.0, 0.0, 2.0, 2.5,-4.5, -3.0,-6.0, -2.5]
abd_8b  = [ 5.0, 0.5, 7.0, 1.0,-2.5,  0.5, 1.5, -1.0]
xb = np.arange(len(tasks_b)); wb = 0.26
wB.bar(xb - wb, abd_3b, wb, label="3B", color="#a9cce3")
wB.bar(xb,      abd_4b, wb, label="4B", color="#5499c7")
wB.bar(xb + wb, abd_8b, wb, label="8B", color="#1f618d")
wB.axhline(0, color="k", lw=0.8)
wB.set_xticks(xb); wB.set_xticklabels(tasks_b, fontsize=8)
wB.set_ylabel(r"$\Delta$ acc. vs. full merge (pts)", fontsize=9)
wB.set_title("(b) Drop down_proj: 40/48 cells safe", fontsize=10)
wB.legend(fontsize=7, loc="lower left")
wB.grid(alpha=0.25, axis="y")

# (c) Atomic vs direct: per-task sign is cross-scale stable (Table tab:p4-f28).
tasks_c = ["arith", "leg", "kk", "cnt", "sp"]
aa_3b   = [ 2.5,  4.0,  8.5,  0.0, -33.0]
aa_4b   = [ 8.5, 11.5,  1.0,  0.5,   1.5]
aa_8b   = [10.0, 10.0,  3.0,  7.5, -18.5]
xc = np.arange(len(tasks_c)); wc = 0.26
wC.bar(xc - wc, aa_3b, wc, label="3B", color="#f5b7b1")
wC.bar(xc,      aa_4b, wc, label="4B", color="#e74c3c")
wC.bar(xc + wc, aa_8b, wc, label="8B", color="#922b21")
wC.axhline(0, color="k", lw=0.8)
wC.set_xticks(xc); wC.set_xticklabels(tasks_c, fontsize=8)
wC.set_ylabel(r"$\Delta(\mathrm{M}_{aa}-\mathrm{M}_{dd})$ (pts)", fontsize=9)
wC.set_title("(c) Atomic vs. direct: stable sign", fontsize=10)
wC.legend(fontsize=7, loc="upper right")
wC.grid(alpha=0.25, axis="y")
wC.annotate("M$_{aa}$ wins", (0.02, 0.96), xycoords="axes fraction",
            fontsize=6.5, va="top", color="#27ae60")
wC.annotate("M$_{dd}$ wins (sp)", (0.985, 0.30), xycoords="axes fraction",
            fontsize=6.5, va="center", ha="right", color="#c0392b")

figW.tight_layout()
figW.savefig(f"{OUT}/fig_p4_worked.pdf", bbox_inches="tight")
plt.close(figW)

# ---------------- Phase 2: the shortest_path craft "climb" ----------------
# 16 conditions distilled to four lessons. Data transcribed from
# report_phase2_complete.md and Appendix tab:sp. Bars sorted ascending so the
# climb from 10% (base) to 55.5% (RFT-Base) reads at a glance; color encodes the
# lesson category, dashed line marks the plain Direct-RLVR bar to beat.
from matplotlib.patches import Patch
sp_rows = [
    ("Base (SFT)",                       10.0, "ref"),
    ("Atomic-Only, short atoms",         12.0, "short"),
    ("Progressive + partial credit",     24.0, "shape"),
    ("Long atoms $\\rightarrow$ progressive", 27.5, "shape"),
    ("Partial-credit reward",            35.5, "shape"),
    ("Direct-RLVR (full task)",          44.0, "ref"),
    ("Atomic-Warm, short atoms",         46.0, "short"),
    ("Long atoms $\\rightarrow$ full (hard jump)", 51.5, "long"),
    ("RFT-Base",                         55.5, "best"),
]
cat_color = {"ref": "#95a5a6", "short": "#e67e22", "shape": "#c0392b",
             "long": "#2980b9", "best": "#27ae60"}
cat_label = {"ref": "base / Direct-RLVR", "short": "short-answer atoms",
             "shape": "softened reward or gentle ramp (hurts GRPO)",
             "long": "long-answer atoms, one hard jump", "best": "RFT-Base (best)"}
labels = [r[0] for r in sp_rows]
vals = [r[1] for r in sp_rows]
cols = [cat_color[r[2]] for r in sp_rows]
yy = np.arange(len(sp_rows))

figSP, axSP = plt.subplots(figsize=(8.6, 4.3))
axSP.barh(yy, vals, color=cols, edgecolor="black", linewidth=0.4, zorder=3)
axSP.set_yticks(yy); axSP.set_yticklabels(labels, fontsize=8)
axSP.invert_yaxis()
for y_i, v in zip(yy, vals):
    axSP.text(v + 0.7, y_i, f"{v:.1f}%", va="center", fontsize=7.5)
axSP.axvline(44, color="#34495e", lw=1.0, ls="--", zorder=1)
axSP.text(43.0, 7.4, "Direct-RLVR", rotation=90, fontsize=6.8,
          color="#34495e", va="center", ha="center")
axSP.set_xlim(0, 62)
axSP.set_xlabel("shortest_path full-task accuracy (%)", fontsize=9)
axSP.set_title("From 10% to 55.5%: 16 conditions distilled to four craft lessons",
               fontsize=10.5)
handles = [Patch(facecolor=cat_color[k], edgecolor="black", label=cat_label[k])
           for k in ["ref", "short", "shape", "long", "best"]]
axSP.legend(handles=handles, fontsize=6.8, loc="upper right", framealpha=0.95)
axSP.grid(alpha=0.25, axis="x", zorder=0)
figSP.tight_layout()
figSP.savefig(f"{OUT}/fig_p2_sp.pdf", bbox_inches="tight")
plt.close(figSP)

# ---------------- Phase 3: per-task depth migration (all tasks) ----------------
# The body shows depth migration for four representative tasks (fig5_6, panel c of
# fig:p3b). This appendix figure extends that to EVERY task with clean data, computed
# directly from the per-task top-20 most-changed matrices in
# results/weight_analysis/{scale}/analysis_results.json (the same
# source the body figure transcribes). For each (scale, task) we bucket the top-20
# matrices by transformer-block index into four depth quartiles, sum tau-norm in each,
# normalize, then take the band center of mass (0=early .. 3=late). The arrow runs from
# the family's smaller scale to its larger scale; red=deeper, blue=shallower.
import json, re, os

WA = os.path.join(PAPER, "../results/weight_analysis")   # raw geometry data
ARCH = {"3b": 36, "4b": 36, "7b": 28, "8b": 36}      # transformer blocks per scale
# tasks excluded at a given scale (not a clean GRPO run): countdown@4B was data-
# contaminated; maze was not analyzed at 8B. Drop only those (scale, task) endpoints.
EXCLUDE = {("4b", "countdown"), ("8b", "maze")}

def _depth_bands(per_layer, L):
    edges = (0.25 * L, 0.5 * L, 0.75 * L)
    b = [0.0, 0.0, 0.0, 0.0]
    for name, val in per_layer.items():
        m = re.search(r"layers\.(\d+)\.", name)
        if not m:
            continue
        i = int(m.group(1))
        k = 0 if i < edges[0] else 1 if i < edges[1] else 2 if i < edges[2] else 3
        b[k] += val
    tot = sum(b)
    return [100 * x / tot for x in b] if tot else None

def _com(b):  # band center of mass over indices 0..3
    return sum(i * v for i, v in enumerate(b)) / sum(b)

def _load_bands(scale):
    path = f"{WA}/{scale}/analysis_results.json"
    tm = json.load(open(path))["phase_a"]["task_magnitudes"]
    out = {}
    for t, v in tm.items():
        if (scale, t) in EXCLUDE:
            continue
        bb = _depth_bands(v["per_layer"], ARCH[scale])
        if bb is not None:
            out[t] = bb
    return out

if all(os.path.exists(f"{WA}/{s}/analysis_results.json") for s in ARCH):
    bands = {s: _load_bands(s) for s in ARCH}
    # consistent row order: early editor, logic/mid-early cluster, arithmetic, others
    order = ["caesar_cipher", "syllogism", "leg_counting", "knights_knaves",
             "shortest_path", "basic_arithmetic", "countdown", "mini_sudoku", "maze"]
    families = [("Qwen2.5", "3b", "7b"), ("Qwen3", "4b", "8b")]
    band_names = ["early", "mid-early", "mid-late", "late"]
    c_deeper, c_shallow = "#C0392B", "#2471A3"

    figD, axesD = plt.subplots(1, 2, figsize=(9.6, 4.6), sharex=True, sharey=True)
    for ax, (fam, s_small, s_big) in zip(axesD, families):
        for yi, task in enumerate(order):
            here = bands[s_small].get(task), bands[s_big].get(task)
            if here[0] is None or here[1] is None:
                # one endpoint missing for this family: mark the row as n/a, keep aligned
                ax.text(1.5, yi, "n/a", ha="center", va="center", fontsize=7,
                        color="0.6", style="italic")
                continue
            x0, x1 = _com(here[0]), _com(here[1])
            col = c_deeper if x1 >= x0 else c_shallow
            ax.annotate("", xy=(x1, yi), xytext=(x0, yi),
                        arrowprops=dict(arrowstyle="-|>", color=col, lw=2.0,
                                        shrinkA=3, shrinkB=3))
            ax.scatter([x0], [yi], facecolors="white", edgecolors=col, s=48, zorder=3)
            ax.scatter([x1], [yi], facecolors=col, edgecolors=col, s=48, zorder=3)
        ax.set_title(f"{fam}  ({s_small.upper()}$\\to${s_big.upper()})")
        ax.set_xticks(range(4)); ax.set_xticklabels(band_names, rotation=15)
        ax.set_xlim(-0.3, 3.3)
        ax.grid(axis="x", linestyle=":", alpha=0.4)
    axesD[0].set_yticks(range(len(order)))
    axesD[0].set_yticklabels([t.replace("_", "\n") for t in order], fontsize=8)
    axesD[0].set_ylim(-0.6, len(order) - 0.4)
    axesD[0].invert_yaxis()
    from matplotlib.lines import Line2D
    handlesD = [
        Line2D([], [], marker="o", color="0.4", mfc="white", ls="", label="smaller scale"),
        Line2D([], [], marker="o", color="0.4", mfc="0.4", ls="", label="larger scale"),
        Line2D([], [], color=c_deeper, lw=2, label="moves deeper"),
        Line2D([], [], color=c_shallow, lw=2, label="moves shallower"),
    ]
    figD.legend(handles=handlesD, loc="lower center", ncol=4, frameon=False,
                bbox_to_anchor=(0.5, -0.03), fontsize=8)
    figD.supxlabel("mean depth of GRPO change (band center of mass)", fontsize=9, y=0.04)
    figD.tight_layout(rect=(0, 0.04, 1, 1))
    figD.savefig(f"{OUT}/fig_p3_pertask_depth.pdf", bbox_inches="tight")
    plt.close(figD)
    print("wrote figures/fig_p3_pertask_depth.pdf")
else:
    print("SKIP fig_p3_pertask_depth.pdf (raw weight_analysis JSON not found)")

print("wrote figures/fig_p1_transfer.pdf, figures/fig_p4_merge.pdf, "
      "figures/fig_p4_worked.pdf, figures/fig_p2_sp.pdf")
