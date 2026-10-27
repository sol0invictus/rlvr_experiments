"""
Fix Stream B:
  1. Rebuild the 3 TIES merges using a chunked quantile (avoids 'tensor too large')
  2. Build the missing T7 DARE merge (using the correct _4to5 checkpoint)
  3. Build the T7 TIES merge

Output layout matches what pipeline_gpu.sh expects:
  experiments/rlvr_geometry_paper/paper/data/baselines/T{2,5,7}/{ties,dare}/model.safetensors
"""
import os, shutil, sys
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file, save_file

BASE_DIR = "/code-fsx/sunguha/rlvr_experiments"  # repo root; paths below are repo-relative
SFT = "/code-fsx/sunguha/sft_qwen_4.0b_instruct"

# correct T7 path uses the 4to5 variant (root contains the weights)
PAIRS = {
    "T2_sdk_kk":  ("experiments/rlvr_geometry_paper/results/grpo_qwen4b_mini_sudoku",              "experiments/rlvr_geometry_paper/results/grpo_qwen4b_knights_knaves"),
    "T5_sp_syl":  ("experiments/rlvr_geometry_paper/results/grpo_qwen4b_shortest_path",            "experiments/rlvr_geometry_paper/results/grpo_qwen4b_syllogism"),
    "T7_cnt_ari": ("experiments/rlvr_geometry_paper/results/grpo_qwen_4b_instruct_countdown_4to5", "experiments/rlvr_geometry_paper/results/grpo_qwen4b_basic_arithmetic"),
}


def load_sd(p):
    files = sorted(Path(p).glob("model*.safetensors"))
    sd = {}
    for f in files:
        sd.update(load_file(str(f)))
    return sd


def compute_tau(base, model):
    tau = {}
    for k, v in base.items():
        if k in model and model[k].shape == v.shape:
            tau[k] = model[k].float() - v.float()
    return tau


def chunked_quantile(t, q, chunk_bytes=256_000_000):
    """torch.quantile() errors out above ~16M elements. Chunk it."""
    a = t.abs().flatten()
    n = a.numel()
    # fast path for small
    if n < 16_000_000:
        return a.quantile(q)
    # bigger: sample up to 1M elements uniformly
    k = min(n, 1_000_000)
    idx = torch.randperm(n)[:k]
    return a[idx].quantile(q)


def merge_ties(base_sd, tau_a, tau_b, density=0.5):
    """
    TIES: (1) trim low-magnitude per-parameter (keep top (1-density) by |τ|),
          (2) sign-elect by majority magnitude sum, (3) disjoint merge.
    Here (for 2-task) sign-elect = take the τ with larger magnitude at each position
    if signs disagree; else keep both contributions.
    """
    merged = {}
    for k, v in base_sd.items():
        out = v.float().clone()
        ta = tau_a.get(k)
        tb = tau_b.get(k)
        if ta is None and tb is None:
            merged[k] = out.to(v.dtype); continue
        contribs = []
        for t in (ta, tb):
            if t is None: continue
            # trim: zero the lowest (1-density) fraction by magnitude
            thresh = chunked_quantile(t, 1.0 - density)
            mask = (t.abs() >= thresh).float()
            contribs.append(t * mask)
        if len(contribs) == 2:
            a, b = contribs
            # sign-elect: at each position, keep both if signs agree; else keep larger |τ|
            sign_agree = (torch.sign(a) == torch.sign(b))
            # where signs disagree, pick the larger magnitude
            pick_a = (a.abs() >= b.abs()) & ~sign_agree
            pick_b = ~pick_a & ~sign_agree
            elected_a = torch.where(sign_agree | pick_a, a, torch.zeros_like(a))
            elected_b = torch.where(sign_agree | pick_b, b, torch.zeros_like(b))
            out = out + elected_a + elected_b
        else:
            out = out + contribs[0]
        merged[k] = out.to(v.dtype)
    return merged


def merge_dare(base_sd, tau_a, tau_b, drop_rate=0.5, seed=1337):
    g = torch.Generator().manual_seed(seed)
    merged = {}
    for k, v in base_sd.items():
        out = v.float().clone()
        for t in (tau_a.get(k), tau_b.get(k)):
            if t is None: continue
            mask = (torch.rand(t.shape, generator=g) > drop_rate).float()
            out = out + (t * mask) / (1.0 - drop_rate)
        merged[k] = out.to(v.dtype)
    return merged


def save_merged(out_dir, merged, ref_ckpt):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    save_file(merged, os.path.join(out_dir, "model.safetensors"))
    # copy supporting files from reference
    for fname in ["config.json", "tokenizer.json", "tokenizer_config.json",
                  "special_tokens_map.json", "generation_config.json",
                  "chat_template.jinja", "added_tokens.json",
                  "vocab.json", "merges.txt"]:
        src = Path(ref_ckpt) / fname
        if src.exists():
            shutil.copy(src, Path(out_dir) / fname)


def main():
    print("Loading SFT base...", flush=True)
    base = load_sd(SFT)
    for pair, (pa, pb) in PAIRS.items():
        ap = os.path.join(BASE_DIR, pa)
        bp = os.path.join(BASE_DIR, pb)
        if not Path(ap).exists() or not Path(bp).exists():
            print(f"  SKIP {pair}: missing {ap if not Path(ap).exists() else bp}")
            continue
        print(f"  loading {pair} A={pa}", flush=True)
        sd_a = load_sd(ap)
        print(f"  loading {pair} B={pb}", flush=True)
        sd_b = load_sd(bp)
        tau_a = compute_tau(base, sd_a)
        tau_b = compute_tau(base, sd_b)
        del sd_a, sd_b

        ties_out = os.path.join(BASE_DIR, "experiments/rlvr_geometry_paper/paper/data/baselines", pair, "ties")
        dare_out = os.path.join(BASE_DIR, "experiments/rlvr_geometry_paper/paper/data/baselines", pair, "dare")

        if not Path(os.path.join(ties_out, "model.safetensors")).exists():
            print(f"  building TIES -> {ties_out}", flush=True)
            m = merge_ties(base, tau_a, tau_b, density=0.5)
            save_merged(ties_out, m, ap)
            del m
        else:
            print(f"  TIES exists: {ties_out}")
        if not Path(os.path.join(dare_out, "model.safetensors")).exists():
            print(f"  building DARE -> {dare_out}", flush=True)
            m = merge_dare(base, tau_a, tau_b, drop_rate=0.5)
            save_merged(dare_out, m, ap)
            del m
        else:
            print(f"  DARE exists: {dare_out}")
        del tau_a, tau_b

    print("DONE rebuild", flush=True)


if __name__ == "__main__":
    main()
