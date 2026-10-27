"""
Triple-task merge: W_sft + s*(tau_SP + tau_Arith + tau_Syllogism).

Phase 4 pair merges showed SP+Arith, SP+Syl, Arith+Syl all compose cleanly.
Test whether all three together work.
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--sp", required=True)
    parser.add_argument("--arith", required=True)
    parser.add_argument("--syl", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--scale", type=float, default=1.0)
    args = parser.parse_args()

    print("Loading base..."); base = load_sd(args.base)
    print("Loading SP..."); sp = load_sd(args.sp)
    print("Loading Arith..."); ar = load_sd(args.arith)
    print("Loading Syl..."); sy = load_sd(args.syl)

    merged = {}
    for k, w_base in base.items():
        w = w_base.float()
        for sd in [sp, ar, sy]:
            if k in sd and sd[k].shape == w_base.shape:
                w = w + args.scale * (sd[k].float() - w_base.float())
        merged[k] = w.to(w_base.dtype)

    os.makedirs(args.output, exist_ok=True)
    for f in Path(args.base).iterdir():
        if f.suffix not in ('.safetensors', '.bin') and f.name != 'trainer_state.json' and f.is_file():
            shutil.copy2(str(f), args.output)
    save_file(merged, os.path.join(args.output, "model.safetensors"))
    print(f"Saved to {args.output} (scale={args.scale})")


if __name__ == "__main__":
    main()
