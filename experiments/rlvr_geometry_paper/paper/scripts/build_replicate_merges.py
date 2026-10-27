"""Build M_aa, M_dd, M_mix merges at an arbitrary scale."""
import argparse, os, shutil
from pathlib import Path
import torch
from safetensors.torch import load_file, save_file


def load_sd(p):
    files = sorted(Path(p).glob("model*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no safetensors in {p}")
    sd = {}
    for f in files:
        sd.update(load_file(str(f)))
    return sd


def compute_tau(base_sd, ckpt_path):
    print(f"  loading {ckpt_path}", flush=True)
    model_sd = load_sd(ckpt_path)
    tau = {}
    for k, v in base_sd.items():
        if k in model_sd and model_sd[k].shape == v.shape:
            tau[k] = (model_sd[k].float() - v.float()).to(torch.float32)
    return tau


def tau_norm(tau):
    s = 0.0
    for v in tau.values():
        s += float((v ** 2).sum())
    return s ** 0.5


def save_merged(out_dir, merged, ref_ckpt):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    save_file(merged, os.path.join(out_dir, "model.safetensors"))
    for fname in ["config.json", "tokenizer.json", "tokenizer_config.json",
                  "special_tokens_map.json", "generation_config.json",
                  "chat_template.jinja", "added_tokens.json",
                  "vocab.json", "merges.txt"]:
        src = Path(ref_ckpt) / fname
        if src.exists():
            shutil.copy(src, Path(out_dir) / fname)


def merge_basic(base_sd, taus):
    out = {}
    for k, v in base_sd.items():
        w = v.float().clone()
        for tau in taus:
            if k in tau:
                w = w + tau[k]
        out[k] = w.to(v.dtype)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", required=True)
    ap.add_argument("--sft", required=True)
    ap.add_argument("--atomic_ari", required=True)
    ap.add_argument("--atomic_sp", required=True)
    ap.add_argument("--direct_ari", required=True)
    ap.add_argument("--direct_sp", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    print(f"=== Build merges @ {args.scale} ===")
    print(f"  SFT base: {args.sft}")
    base = load_sd(args.sft)

    tau_a_ari = compute_tau(base, args.atomic_ari)
    print(f"    ‖τ_atomic_ari‖ = {tau_norm(tau_a_ari):.3f}")
    tau_a_sp = compute_tau(base, args.atomic_sp)
    print(f"    ‖τ_atomic_sp‖ = {tau_norm(tau_a_sp):.3f}")
    tau_d_ari = compute_tau(base, args.direct_ari)
    print(f"    ‖τ_direct_ari‖ = {tau_norm(tau_d_ari):.3f}")
    tau_d_sp = compute_tau(base, args.direct_sp)
    print(f"    ‖τ_direct_sp‖ = {tau_norm(tau_d_sp):.3f}")

    os.makedirs(args.out_dir, exist_ok=True)
    for name, taus, ref in [
        ("M_aa", [tau_a_sp, tau_a_ari], args.atomic_ari),
        ("M_dd", [tau_d_sp, tau_d_ari], args.direct_ari),
        ("M_mix", [tau_a_sp, tau_d_ari], args.direct_ari),
    ]:
        out = os.path.join(args.out_dir, name)
        print(f"  -> {out}", flush=True)
        merged = merge_basic(base, taus)
        save_merged(out, merged, ref)
        del merged

    print("DONE")


if __name__ == "__main__":
    main()
