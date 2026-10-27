"""
Stream B — TIES + DARE baselines for 3 Phase-4 pairs at 4B (s=1.0).

Pairs:
  T2_sdk_kk   (worst damage)
  T5_sp_syl   (cleanest)
  T7_cnt_ari  (another bad)

For each pair and each of {TIES, DARE} we write a merged checkpoint under
experiments/rlvr_geometry_paper/results/phase4_pairs/<pair>/<method>.

Implementation:
  TIES: for each parameter, per-tau keep top-k_density by magnitude; stack
        the two taus; elect per-entry sign by majority (size-2 => sign
        agreement); average values that match elected sign. We then rescale
        the merged tau so that ||tau_merged|| == ||tau_vanilla_sum|| to match
        the vanilla s=1.0 total budget.
  DARE: drop (1-p) fraction of entries of each tau independently with
        p=0.5, rescale survivors by 1/p, sum. Uses a fixed seed.
"""

import argparse
import json
import os
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

BASE = "/scratch/sunguha/sft_qwen_4.0b_instruct"

PAIRS = {
    "T2_sdk_kk":  ("experiments/rlvr_geometry_paper/results/grpo_qwen4b_mini_sudoku", "experiments/rlvr_geometry_paper/results/grpo_qwen4b_knights_knaves"),
    "T5_sp_syl":  ("experiments/rlvr_geometry_paper/results/grpo_qwen4b_shortest_path", "experiments/rlvr_geometry_paper/results/grpo_qwen4b_syllogism"),
    "T7_cnt_ari": ("experiments/rlvr_geometry_paper/results/grpo_qwen_4b_instruct_countdown_4to5", "experiments/rlvr_geometry_paper/results/grpo_qwen4b_basic_arithmetic"),
}


def load_sd(p):
    out = {}
    for f in sorted(Path(p).glob("model*.safetensors")):
        out.update(load_file(str(f)))
    return out


def compute_tau(base, sd):
    return {
        k: (sd[k].float() - base[k].float())
        for k in base
        if k in sd and sd[k].shape == base[k].shape
    }


def save_merged(base_path, merged, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for f in Path(base_path).iterdir():
        if f.suffix in (".safetensors", ".bin"):
            continue
        if f.name == "trainer_state.json":
            continue
        if f.is_file():
            shutil.copy2(str(f), out_dir)
    save_file(merged, os.path.join(out_dir, "model.safetensors"))
    print(f"  Saved -> {out_dir}")


def trim_tau(tau, density=0.2):
    """Return a copy of tau with all but top-density by magnitude zeroed."""
    out = {}
    for k, v in tau.items():
        if v.numel() == 0:
            out[k] = v.clone()
            continue
        thresh = v.abs().float().quantile(1.0 - density)
        mask = v.abs() >= thresh
        out[k] = v * mask.to(v.dtype)
    return out


def merge_ties(base, tau_a, tau_b, density=0.2):
    # 1) Trim
    a = trim_tau(tau_a, density=density)
    b = trim_tau(tau_b, density=density)

    # 2) Elect sign + merge (mean over agreeing entries)
    tau_merged_raw = {}
    for k in base:
        if k not in a and k not in b:
            continue
        vecs = []
        if k in a:
            vecs.append(a[k])
        if k in b:
            vecs.append(b[k])
        stacked = torch.stack(vecs)
        sign_sum = stacked.sign().sum(dim=0)
        elected = sign_sum.sign()
        # Tie break: use elected=+1 for ties (safe)
        elected[elected == 0] = 1
        agree = (stacked.sign() == elected) & (stacked != 0)
        num = (stacked * agree.float()).sum(dim=0)
        denom = agree.float().sum(dim=0).clamp(min=1)
        tau_merged_raw[k] = num / denom

    # 3) Rescale to match vanilla-sum ||tau_vanilla|| budget
    vanilla_sq = 0.0
    for k in base:
        v = torch.zeros_like(base[k], dtype=torch.float32)
        if k in tau_a:
            v = v + tau_a[k]
        if k in tau_b:
            v = v + tau_b[k]
        vanilla_sq += (v.float() ** 2).sum().item()
    vanilla_norm = vanilla_sq ** 0.5

    merged_sq = sum((v.float() ** 2).sum().item() for v in tau_merged_raw.values())
    merged_norm = merged_sq ** 0.5
    scale = vanilla_norm / merged_norm if merged_norm > 0 else 1.0
    print(f"    TIES rescale factor (vanilla/merged): {scale:.4f}")

    merged = {}
    for k in base:
        w = base[k].float()
        if k in tau_merged_raw:
            w = w + scale * tau_merged_raw[k]
        merged[k] = w.to(base[k].dtype)
    return merged


def merge_dare(base, tau_a, tau_b, drop_rate=0.5, seed=1234):
    g = torch.Generator()
    g.manual_seed(seed)
    merged = {}
    p = 1.0 - drop_rate
    for k in base:
        w = base[k].float()
        for tau in (tau_a, tau_b):
            if k in tau:
                v = tau[k]
                mask = torch.bernoulli(torch.full_like(v, p), generator=g)
                w = w + v * mask / p
        merged[k] = w.to(base[k].dtype)
    return merged


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", required=True, choices=list(PAIRS.keys()))
    parser.add_argument("--method", required=True, choices=["ties", "dare"])
    args = parser.parse_args()

    model_a, model_b = PAIRS[args.pair]
    print(f"Pair {args.pair}: {model_a} + {model_b}")
    print(f"Method: {args.method}")

    base = load_sd(BASE)
    sd_a = load_sd(model_a)
    sd_b = load_sd(model_b)
    tau_a = compute_tau(base, sd_a)
    tau_b = compute_tau(base, sd_b)
    del sd_a, sd_b

    if args.method == "ties":
        merged = merge_ties(base, tau_a, tau_b, density=0.2)
    else:
        merged = merge_dare(base, tau_a, tau_b, drop_rate=0.5, seed=1234)

    out_dir = f"experiments/rlvr_geometry_paper/results/phase4_pairs/{args.pair}/{args.method}"
    save_merged(BASE, merged, out_dir)


if __name__ == "__main__":
    main()
