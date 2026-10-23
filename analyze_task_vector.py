"""
Weight-space geometry of a GRPO task vector tau = W_grpo - W_sft.

Re-implementation of the metrics defined in the paper's "Weight-Space Geometry:
Metric Definitions" appendix (the original analyze_*.py scripts were not kept).
All metrics are computed per weight matrix and then aggregated by layer type and
by depth.

Per matrix (2-D weights of the transformer blocks, plus the embedding):
    tau_fro, w_fro, rel_change      ||tau||_F, ||W_sft||_F, ||tau||_F / ||W_sft||_F
  with --svd:
    spec_ratio                      sigma_1(W_grpo) / sigma_1(W_sft)
    max_rel_sigma_shift             max_i |sigma_i(W_grpo) - sigma_i(W_sft)| / sigma_i(W_sft)
    d_erank                         erank(W_grpo) - erank(W_sft), erank = exp(H(sigma / sum sigma))
    rho_k, rho_null                 ||U_k U_k^T tau V_k V_k^T||_F^2 / ||tau||_F^2 and (k/r)^2
    rot_deg                         mean principal angle between top-k left singular
                                    subspaces of W_sft and W_grpo (degrees; accurate float64
                                    sin/cos estimator)
    rot_deg_arccos_fp32             same with the naive fp32 arccos estimator, whose floor is
                                    ~0.02-0.05 deg (identical subspaces read as "rotated")
    tau_sigma1, tau_decay           sigma_1(tau) and sigma_1(tau) / sigma_10(tau)

Whole model:
    tau_fro_total_*                 ||tau||_F over all params / block linears only / no embedding
    depth_profile                   sum of ||tau||_F per block, normalized to sum 1
    depth_com_band_*                top-N most-changed matrices (by ||tau||_F) bucketed into
                                    four depth quartiles (0 = early ... 3 = late); mean band,
                                    count-weighted and ||tau||-weighted

Ambiguities in the original definitions are reported both ways (see keys above).

Usage:
    python analyze_task_vector.py --sft outputs/sft_qwen3_4b --grpo results/grpo/checkpoint-300 \
        --svd --device cuda --out results/geometry/step300.json
    # many checkpoints, cheap metrics only:
    python analyze_task_vector.py --sft SFT --grpo ckpt-25 ckpt-50 ... --out_dir results/geometry
"""

import argparse
import json
import math
import os
import re
from collections import defaultdict
from typing import Dict, Iterator, List, Optional, Tuple

import torch
from safetensors import safe_open

LAYER_RE = re.compile(r"model\.layers\.(\d+)\.(self_attn|mlp)\.(\w+)\.weight$")
EMBED_KEYS = ("model.embed_tokens.weight",)


# ======================================================================
# Loading
# ======================================================================
def _shard_map(model_dir: str) -> Dict[str, str]:
    """Map tensor name -> safetensors file for a (possibly sharded) checkpoint."""
    index = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(index):
        with open(index) as f:
            weight_map = json.load(f)["weight_map"]
        return {k: os.path.join(model_dir, v) for k, v in weight_map.items()}
    single = os.path.join(model_dir, "model.safetensors")
    with safe_open(single, framework="pt") as f:
        return {k: single for k in f.keys()}


def iter_pairs(sft_dir: str, grpo_dir: str) -> Iterator[Tuple[str, torch.Tensor, torch.Tensor]]:
    """Yield (name, W_sft, W_grpo) as float32 CPU tensors, one tensor at a time."""
    sft_map, grpo_map = _shard_map(sft_dir), _shard_map(grpo_dir)
    missing = set(sft_map) ^ set(grpo_map)
    if missing:
        print(f"[warn] {len(missing)} tensors present in only one checkpoint, e.g. {sorted(missing)[:3]}")
    for name in sorted(set(sft_map) & set(grpo_map)):
        with safe_open(sft_map[name], framework="pt") as fs, safe_open(grpo_map[name], framework="pt") as fg:
            yield name, fs.get_tensor(name).float(), fg.get_tensor(name).float()


# ======================================================================
# Metrics
# ======================================================================
def erank(sigma: torch.Tensor) -> float:
    p = sigma / sigma.sum()
    p = p[p > 0]
    return float(torch.exp(-(p * p.log()).sum()))


def principal_angles_deg(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Principal angles between span(A) and span(B); A, B have orthonormal columns.

    Plain arccos(svdvals(A^T B)) has a floor of ~0.02-0.05 deg in fp32: identical
    subspaces come out "rotated". Use the sine/cosine combination (Bjorck & Golub) in
    float64: arcsin of svdvals((I - AA^T)B) for small angles, arccos for large ones.
    """
    A, B = A.double(), B.double()
    cos = torch.linalg.svdvals(A.T @ B).clamp(0.0, 1.0)              # descending
    sin = torch.linalg.svdvals(B - A @ (A.T @ B)).clamp(0.0, 1.0)    # descending
    sin = sin.flip(0)                                                # pair with cos
    theta = torch.where(sin < 0.7, torch.arcsin(sin), torch.arccos(cos))
    return torch.rad2deg(theta)


def principal_angles_deg_arccos_fp32(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """The naive estimator (arccos in fp32), kept to compare with the paper's numbers."""
    s = torch.linalg.svdvals(A.float().T @ B.float()).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.arccos(s))


def matrix_metrics(w_s: torch.Tensor, w_g: torch.Tensor, k: int, svd: bool, device: str) -> Dict:
    tau = w_g - w_s
    out = {
        "shape": list(w_s.shape),
        "tau_fro": float(tau.norm()),
        "w_fro": float(w_s.norm()),
    }
    out["rel_change"] = out["tau_fro"] / out["w_fro"] if out["w_fro"] > 0 else float("nan")
    if not svd or w_s.dim() != 2:
        return out

    w_s, w_g, tau = w_s.to(device), w_g.to(device), tau.to(device)
    U_s, S_s, Vh_s = torch.linalg.svd(w_s, full_matrices=False)
    U_g, S_g, _ = torch.linalg.svd(w_g, full_matrices=False)
    r = S_s.numel()
    kk = min(k, r)

    out["spec_ratio"] = float(S_g[0] / S_s[0])
    nz = S_s > 0
    out["max_rel_sigma_shift"] = float(((S_g[nz] - S_s[nz]).abs() / S_s[nz]).max())
    out["d_erank"] = erank(S_g) - erank(S_s)

    if out["tau_fro"] > 0:
        Uk, Vk = U_s[:, :kk], Vh_s[:kk, :].T
        proj = Uk @ (Uk.T @ tau @ Vk) @ Vk.T
        out["rho_k"] = float(proj.norm() ** 2 / tau.norm() ** 2)
        tau_s = torch.linalg.svdvals(tau)
        out["tau_sigma1"] = float(tau_s[0])
        out["tau_decay"] = float(tau_s[0] / tau_s[9]) if tau_s.numel() >= 10 and tau_s[9] > 0 else float("nan")
    else:
        out["rho_k"] = float("nan")
        out["tau_sigma1"] = 0.0
        out["tau_decay"] = float("nan")
    out["rho_null"] = (kk / r) ** 2
    out["k"] = kk
    out["rot_deg"] = float(principal_angles_deg(U_s[:, :kk], U_g[:, :kk]).mean())
    out["rot_deg_arccos_fp32"] = float(principal_angles_deg_arccos_fp32(U_s[:, :kk], U_g[:, :kk]).mean())
    return out


def classify(name: str) -> Tuple[Optional[int], str]:
    m = LAYER_RE.match(name)
    if m:
        return int(m.group(1)), f"{'attn' if m.group(2) == 'self_attn' else 'mlp'}.{m.group(3)}"
    if name in EMBED_KEYS:
        return None, "embed"
    return None, "other"


def analyze(sft_dir: str, grpo_dir: str, k: int, top_n: int, svd: bool, device: str) -> Dict:
    per_matrix = {}
    sq_all = sq_linear = sq_no_embed = 0.0
    for name, w_s, w_g in iter_pairs(sft_dir, grpo_dir):
        layer, ltype = classify(name)
        sq = float((w_g - w_s).norm() ** 2)
        sq_all += sq
        if ltype != "embed":
            sq_no_embed += sq
        if layer is None:
            continue
        sq_linear += sq
        m = matrix_metrics(w_s, w_g, k=k, svd=svd, device=device)
        m.update(layer=layer, type=ltype)
        per_matrix[name] = m

    num_layers = 1 + max(m["layer"] for m in per_matrix.values())

    # Aggregate by layer type
    by_type = defaultdict(lambda: defaultdict(list))
    for m in per_matrix.values():
        for key in ("tau_fro", "rel_change", "spec_ratio", "max_rel_sigma_shift", "d_erank",
                    "rho_k", "rho_null", "rot_deg", "rot_deg_arccos_fp32", "tau_sigma1", "tau_decay"):
            if key in m and not (isinstance(m[key], float) and math.isnan(m[key])):
                by_type[m["type"]][key].append(m[key])
    by_type_mean = {t: {key: sum(v) / len(v) for key, v in d.items()} for t, d in by_type.items()}

    # Depth profile: sum of ||tau||_F per block, normalized
    depth = [0.0] * num_layers
    for m in per_matrix.values():
        depth[m["layer"]] += m["tau_fro"]
    total = sum(depth)
    depth_profile = [d / total if total > 0 else 0.0 for d in depth]

    # Depth centre of mass: top-N matrices by ||tau||_F, bucketed into 4 quartile bands
    top = sorted(per_matrix.values(), key=lambda m: m["tau_fro"], reverse=True)[:top_n]
    bands = [min(3, m["layer"] * 4 // num_layers) for m in top]
    weights = [m["tau_fro"] for m in top]
    com_count = sum(bands) / len(bands) if bands and total > 0 else float("nan")
    com_weighted = sum(b * w for b, w in zip(bands, weights)) / sum(weights) if sum(weights) > 0 else float("nan")

    return {
        "sft": sft_dir,
        "grpo": grpo_dir,
        "k": k,
        "top_n": top_n,
        "num_layers": num_layers,
        "tau_fro_total_all_params": math.sqrt(sq_all),
        "tau_fro_total_block_linears": math.sqrt(sq_linear),
        "tau_fro_total_no_embed": math.sqrt(sq_no_embed),
        "by_type": by_type_mean,
        "depth_profile": depth_profile,
        "depth_com_band_count": com_count,
        "depth_com_band_weighted": com_weighted,
        "per_matrix": per_matrix,
    }


def summarize(res: Dict) -> str:
    lines = [f"{res['grpo']}",
             f"  ||tau||_F all/linear/no-embed: {res['tau_fro_total_all_params']:.3f} / "
             f"{res['tau_fro_total_block_linears']:.3f} / {res['tau_fro_total_no_embed']:.3f}",
             f"  depth COM (band 0-3, top-{res['top_n']}): count {res['depth_com_band_count']:.2f}, "
             f"weighted {res['depth_com_band_weighted']:.2f}"]
    nan = float("nan")
    for t in ("mlp.down_proj", "mlp.up_proj", "attn.q_proj"):
        d = res["by_type"].get(t, {})
        if "rot_deg" in d:
            lines.append(f"  {t:14s} rot {d['rot_deg']:.3f} deg (arccos-fp32 {d['rot_deg_arccos_fp32']:.3f}) | "
                         f"rel {100 * d['rel_change']:.3f}% | "
                         f"spec_ratio {d['spec_ratio']:.4f} | rho_k {d.get('rho_k', nan):.2e} "
                         f"(null {d['rho_null']:.2e})")
        elif d:
            lines.append(f"  {t:14s} rel {100 * d['rel_change']:.3f}%")
    return "\n".join(lines)


def parse_args():
    p = argparse.ArgumentParser(description="Task-vector geometry (paper App. H metrics)")
    p.add_argument("--sft", required=True, help="SFT checkpoint dir (W_sft)")
    p.add_argument("--grpo", required=True, nargs="+", help="One or more GRPO checkpoint dirs")
    p.add_argument("--k", type=int, default=100, help="Top-k subspace size (default 100)")
    p.add_argument("--top_n", type=int, default=20, help="Matrices used for depth COM (default 20)")
    p.add_argument("--svd", action="store_true", help="Compute SVD-based metrics (slow on CPU)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default=None, help="Output JSON (single --grpo)")
    p.add_argument("--out_dir", default=None, help="Output dir, one JSON per --grpo")
    return p.parse_args()


def main():
    args = parse_args()
    for grpo_dir in args.grpo:
        res = analyze(args.sft, grpo_dir, k=args.k, top_n=args.top_n, svd=args.svd, device=args.device)
        print(summarize(res))
        out = args.out if len(args.grpo) == 1 and args.out else None
        if out is None and args.out_dir:
            out = os.path.join(args.out_dir, f"geometry_{os.path.basename(os.path.normpath(grpo_dir))}.json")
        if out:
            os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
            with open(out, "w") as f:
                json.dump(res, f, indent=2)
            print(f"  saved -> {out}")


if __name__ == "__main__":
    main()
