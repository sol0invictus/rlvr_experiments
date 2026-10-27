"""
Plots for the 4B basic_arithmetic replication runs, from the copied result folders
(trainer_state.json, evaluate_replication.py metrics, analyze_task_vector.py JSON).

    python scripts/plot_replication.py --run "Qwen3-4B=notes/overnight_qwen3_4b/rlvr:sft_qwen3_4b:grpo_basic_arithmetic_4b" \
        --run "Instruct-2507=notes/overnight/rlvr:sft_qwen3_4b_instruct2507:grpo_basic_arithmetic_4b_instruct2507" \
        --out notes/plots
"""

import argparse
import glob
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Reference palette (light mode), validated: categorical slots 1-3, chrome, ink.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
PAPER = {"sft": 72.0, "direct": 32.0, "tau": 0.852, "depth_com": 0.61,
         "rot": {"mlp.down_proj": 0.02, "mlp.up_proj": 1.99, "attn.q_proj": 1.81}}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 10, "axes.titlesize": 11, "axes.titlecolor": INK,
    "axes.titleweight": "bold", "legend.frameon": False, "legend.labelcolor": INK2, "lines.linewidth": 2,
})


def last_json(pattern):
    files = sorted(glob.glob(pattern))
    return json.load(open(files[-1])) if files else None


def wilson(p, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def rolling(xs, w=15):
    out = []
    for i in range(len(xs)):
        window = xs[max(0, i - w + 1):i + 1]
        out.append(sum(window) / len(window))
    return out


def load_run(spec):
    name, rest = spec.split("=", 1)
    root, sft, grpo = rest.split(":")
    run = {"name": name, "root": root, "sft": sft, "grpo": grpo}
    sft_metrics = [json.load(open(f)) for f in glob.glob(f"{root}/eval/{sft}/metrics_*.json")]
    run["sft_hf"] = next(d for d in sft_metrics if d["backend"] == "hf")["tasks"]["basic_arithmetic"]
    run["ckpt_eval"] = {}
    for n in range(25, 301, 25):
        d = last_json(f"{root}/{grpo}/eval/checkpoint-{n}/metrics_*.json")
        if d:
            run["ckpt_eval"][n] = d["tasks"]["basic_arithmetic"]
    run["final_hf"] = last_json(f"{root}/{grpo}/eval/final_hf/metrics_*.json")["tasks"]["basic_arithmetic"]
    state = json.load(open(f"{root}/{grpo}/checkpoint-300/trainer_state.json"))
    run["log"] = [r for r in state["log_history"] if "step_time" in r]
    run["geo_final"] = json.load(open(f"{root}/{grpo}/geometry/final.json"))
    run["geo_ckpt"] = {}
    for n in (100, 200):
        p = f"{root}/{grpo}/geometry/checkpoint-{n}.json"
        if os.path.exists(p):
            run["geo_ckpt"][n] = json.load(open(p))
    run["geo_cheap"] = {}
    for p in glob.glob(f"{root}/{grpo}/geometry/checkpoints/geometry_checkpoint-*.json"):
        n = int(p.rsplit("-", 1)[1].split(".")[0])
        run["geo_cheap"][n] = json.load(open(p))
    return run


def save(fig, out, name):
    path = os.path.join(out, name)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


def plot_accuracy(runs, out):
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for i, r in enumerate(runs):
        c = SERIES[i]
        n_sft = r["sft_hf"]["num_samples"]
        steps = [0] + sorted(r["ckpt_eval"])
        acc = [r["sft_hf"]["accuracy"]] + [r["ckpt_eval"][s]["accuracy"] for s in steps[1:]]
        ns = [n_sft] + [r["ckpt_eval"][s]["num_samples"] for s in steps[1:]]
        lo, hi = zip(*[wilson(a, n) for a, n in zip(acc, ns)])
        ax.fill_between(steps, [100 * x for x in lo], [100 * x for x in hi], color=c, alpha=0.12, linewidth=0)
        ax.plot(steps, [100 * a for a in acc], color=c, marker="o", markersize=5, label=r["name"])
        f = 100 * r["final_hf"]["accuracy"]
        ax.plot([300], [f], marker="D", markersize=8, color=c, markeredgecolor=SURFACE, markeredgewidth=1.5)
        ax.annotate(f"final (HF) {f:.0f}%", (300, f), xytext=(6, -4 if i else 6), textcoords="offset points",
                    color=INK2, fontsize=9)
    for y, lab in ((PAPER["sft"], "paper SFT 72%"), (PAPER["direct"], "paper Direct-RLVR 32%")):
        ax.axhline(y, color=MUTED, linestyle="--", linewidth=1)
        ax.text(2, y + 1, lab, color=MUTED, fontsize=8.5)
    ax.set_xlabel("GRPO step (0 = SFT model)")
    ax.set_ylabel("accuracy (%)")
    ax.set_ylim(20, 85)
    ax.set_xlim(-5, 340)
    ax.set_title("basic_arithmetic accuracy over GRPO (200 samples, 95% Wilson band)")
    ax.legend(loc="lower left")
    save(fig, out, "fig1_eval_accuracy.png")


def plot_lengths(runs, out):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for i, r in enumerate(runs):
        c = SERIES[i]
        steps = [0] + sorted(r["ckpt_eval"])
        rows = [r["sft_hf"]] + [r["ckpt_eval"][s] for s in steps[1:]]
        axes[0].plot(steps, [m["avg_response_chars"] for m in rows], color=c, marker="o", markersize=5, label=r["name"])
        axes[1].plot(steps, [m["avg_reasoning_length"] for m in rows], color=c, marker="o", markersize=5, label=r["name"])
        log = r["log"]
        axes[2].plot([x["step"] for x in log], rolling([x["completions/mean_length"] for x in log]),
                     color=c, label=r["name"])
    axes[0].set_title("eval response length")
    axes[0].set_ylabel("characters")
    axes[1].set_title("eval reasoning inside <think>")
    axes[1].set_ylabel("characters")
    axes[1].annotate("Instruct-2507 never emits <think>", (150, 4), color=INK2, fontsize=8.5, ha="center")
    axes[2].set_title("training completion length (rolling 15)")
    axes[2].set_ylabel("tokens")
    for ax in axes:
        ax.set_xlabel("GRPO step")
        ax.set_ylim(bottom=0)
    axes[0].legend(loc="upper right")
    save(fig, out, "fig2_lengths.png")


def plot_training(runs, out):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for i, r in enumerate(runs):
        c = SERIES[i]
        log = r["log"]
        st = [x["step"] for x in log]
        axes[0].plot(st, rolling([x["rewards/correctness_reward/mean"] for x in log]), color=c, label=r["name"])
        axes[1].plot(st, rolling([x["frac_reward_zero_std"] for x in log]), color=c, label=r["name"])
    axes[0].set_title("training correctness reward (rolling 15)")
    axes[0].set_ylabel("mean reward over 8 completions")
    axes[1].set_title("steps with zero reward variance (no gradient)")
    axes[1].set_ylabel("fraction (rolling 15)")
    for ax in axes:
        ax.set_xlabel("GRPO step")
        ax.set_ylim(0, 1)
    axes[0].legend(loc="lower right")
    save(fig, out, "fig3_training_reward.png")


def plot_profile(runs, out):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [1.6, 1]})
    for i, r in enumerate(runs):
        c = SERIES[i]
        log = r["log"]
        axes[0].plot([x["step"] for x in log], [x["step_time"] for x in log], color=c, linewidth=1, alpha=0.85,
                     label=r["name"])
    axes[0].set_yscale("log")
    axes[0].set_title("time per GRPO step (spikes = rollouts hitting 2048 tokens)")
    axes[0].set_ylabel("seconds (log)")
    axes[0].set_xlabel("GRPO step")
    axes[0].legend(loc="upper right")
    names = [r["name"] for r in runs]
    gen = [sum(x["timing/generate_s"] for x in r["log"]) / 60 for r in runs]
    rest = [sum(x["step_time"] - x["timing/generate_s"] for x in r["log"]) / 60 for r in runs]
    y = range(len(runs))
    axes[1].barh(y, gen, color=[SERIES[i] for i in y], height=0.5, edgecolor=SURFACE, linewidth=2)
    axes[1].barh(y, rest, left=gen, color=MUTED, height=0.5, edgecolor=SURFACE, linewidth=2)
    for i in y:
        axes[1].text(gen[i] + rest[i] + 0.5, i, f"{gen[i]:.0f} min gen + {rest[i]:.1f} min train "
                     f"({100 * gen[i] / (gen[i] + rest[i]):.0f}% gen)", va="center", color=INK2, fontsize=8.5)
    axes[1].set_yticks(list(y), names)
    axes[1].set_xlim(0, max(g + t for g, t in zip(gen, rest)) * 2.1)
    axes[1].set_xlabel("minutes, 300 steps")
    axes[1].set_title("where GRPO time goes")
    axes[1].grid(axis="y", visible=False)
    save(fig, out, "fig4_profile.png")


def plot_rotation(runs, out):
    types = ["mlp.down_proj", "mlp.up_proj", "mlp.gate_proj", "attn.q_proj", "attn.k_proj", "attn.v_proj", "attn.o_proj"]
    fig, axes = plt.subplots(1, len(runs), figsize=(6.2 * len(runs), 4.2), sharey=True)
    axes = axes if len(runs) > 1 else [axes]
    for ax, r in zip(axes, runs):
        bt = r["geo_final"]["by_type"]
        y = list(range(len(types)))
        series = [("true (float64 SVD)", "rot_deg", SERIES[0], "o"),
                  ("random update, same norm", "null_rot_deg", SERIES[2], "s"),
                  ("fp32 SVD on GPU", "rot_deg_fp32svd_arccos", SERIES[1], "^")]
        for k, (lab, key, c, mk) in enumerate(series):
            ax.scatter([bt[t].get(key, float("nan")) for t in types], [v + (k - 1) * 0.18 for v in y],
                       color=c, marker=mk, s=42, label=lab, zorder=3, edgecolor=SURFACE, linewidth=1)
        ax.scatter([PAPER["rot"].get(t, float("nan")) for t in types], y, marker="|", s=260, color=INK,
                   linewidth=2, label="paper", zorder=4)
        ax.set_xscale("log")
        ax.set_xlim(5e-4, 6)
        ax.set_yticks(y, types)
        ax.invert_yaxis()
        ax.set_xlabel("mean principal angle, top-100 subspace (degrees, log)")
        ax.set_title(f"{r['name']}: rotation by layer type, final τ")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.12))
    fig.text(0.5, -0.16, "fp32-GPU values of exactly 0 (e.g. k_proj) can't be shown on the log axis",
             ha="center", color=MUTED, fontsize=8.5)
    save(fig, out, "fig5_rotation_by_type.png")


def plot_rotation_vs_step(runs, out):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for i, r in enumerate(runs):
        c = SERIES[i]
        pts = {**r["geo_ckpt"], 300: r["geo_final"]}
        steps = sorted(pts)
        up = lambda key: [pts[s]["by_type"]["mlp.up_proj"][key] for s in steps]
        axes[0].plot(steps, up("rot_deg"), color=c, marker="o", markersize=6, label=f"{r['name']}: true")
        axes[0].plot(steps, up("rot_deg_fp32svd_arccos"), color=c, marker="^", markersize=6, linestyle="--",
                     label=f"{r['name']}: fp32 GPU SVD")
        cheap = r["geo_cheap"]
        cs = sorted(cheap)
        axes[1].plot([0] + cs, [0] + [cheap[s]["tau_fro_total_all_params"] for s in cs], color=c, marker="o",
                     markersize=5, label=r["name"])
    axes[0].set_yscale("log")
    axes[0].axhline(PAPER["rot"]["mlp.up_proj"], color=MUTED, linestyle=":", linewidth=1)
    axes[0].text(105, PAPER["rot"]["mlp.up_proj"] * 1.15, "paper 4B up_proj 1.99°", color=MUTED, fontsize=8.5)
    axes[0].set_title("up_proj rotation vs step: the fp32 number doesn't move")
    axes[0].set_ylabel("degrees (log)")
    axes[0].set_xlabel("GRPO step")
    axes[0].set_xlim(80, 320)
    axes[0].legend(fontsize=8, loc="center right")
    axes[1].axhline(PAPER["tau"], color=MUTED, linestyle=":", linewidth=1)
    axes[1].text(5, PAPER["tau"] + 0.02, "paper 4B arithmetic 0.852", color=MUTED, fontsize=8.5)
    axes[1].set_title("task-vector size ‖τ‖_F over training")
    axes[1].set_ylabel("‖W_t − W_sft‖_F, all params")
    axes[1].set_xlabel("GRPO step")
    axes[1].legend(loc="lower right")
    save(fig, out, "fig6_rotation_and_tau_vs_step.png")


def plot_depth(runs, out):
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    for i, r in enumerate(runs):
        prof = r["geo_final"]["depth_profile"]
        ax.plot(range(len(prof)), prof, color=SERIES[i], marker="o", markersize=4,
                label=f"{r['name']} (COM band {r['geo_final']['depth_com_band_weighted']:.2f})")
    ax.set_title(f"depth profile of the final τ (paper 4B arithmetic COM band {PAPER['depth_com']})")
    ax.set_xlabel("transformer block")
    ax.set_ylabel("share of Σ‖τ‖_F")
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left")
    save(fig, out, "fig7_depth_profile.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, help="NAME=ROOT:SFT_DIR:GRPO_DIR")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    runs = [load_run(s) for s in args.run]
    plot_accuracy(runs, args.out)
    plot_lengths(runs, args.out)
    plot_training(runs, args.out)
    plot_profile(runs, args.out)
    plot_rotation(runs, args.out)
    plot_rotation_vs_step(runs, args.out)
    plot_depth(runs, args.out)


if __name__ == "__main__":
    main()
