"""
Layer-selective merge: only sum tau contributions from specified layer types.

Phase 3 found mlp.down_proj basis frozen (0deg). Test whether zeroing tau's
mlp.down_proj + all non-feature-projector contributions preserves merge quality.

Variants:
  --include up      : only up_proj + gate_proj
  --include up_q    : up_proj + gate_proj + q_proj
  --include up_q_k  : up_proj + gate_proj + q_proj + k_proj
  --include all_but_down : everything except mlp.down_proj
"""

import argparse, os, shutil
from pathlib import Path
import torch
from safetensors.torch import load_file, save_file


def load_sd(p):
    sd = {}
    for f in sorted(Path(p).glob("model*.safetensors")):
        sd.update(load_file(str(f)))
    return sd


def keep_key(key, include):
    if include == "up":
        return "up_proj" in key or "gate_proj" in key
    if include == "up_q":
        return "up_proj" in key or "gate_proj" in key or ("q_proj" in key and "attn" not in key or "q_proj.weight" in key)
    if include == "up_q_clean":
        return "mlp.up_proj" in key or "mlp.gate_proj" in key or "self_attn.q_proj" in key
    if include == "all_but_down":
        return not ("mlp.down_proj" in key)
    if include == "all":
        return True
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--model_a", required=True)
    parser.add_argument("--model_b", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--include", required=True, choices=["up", "up_q_clean", "all_but_down", "all"])
    parser.add_argument("--scale", type=float, default=1.0)
    args = parser.parse_args()

    base = load_sd(args.base)
    a = load_sd(args.model_a)
    b = load_sd(args.model_b)

    merged = {}
    summed_keys = 0
    skipped_keys = 0
    for k, w_base in base.items():
        w = w_base.float()
        if keep_key(k, args.include):
            if k in a and a[k].shape == w_base.shape:
                w = w + args.scale * (a[k].float() - w_base.float())
            if k in b and b[k].shape == w_base.shape:
                w = w + args.scale * (b[k].float() - w_base.float())
            summed_keys += 1
        else:
            skipped_keys += 1
        merged[k] = w.to(w_base.dtype)

    print(f"Summed tau on {summed_keys} keys, skipped {skipped_keys}")
    os.makedirs(args.output, exist_ok=True)
    for f in Path(args.base).iterdir():
        if f.suffix not in ('.safetensors', '.bin') and f.name != 'trainer_state.json' and f.is_file():
            shutil.copy2(str(f), args.output)
    save_file(merged, os.path.join(args.output, "model.safetensors"))
    print(f"Saved: {args.output} (include={args.include}, scale={args.scale})")


if __name__ == "__main__":
    main()
