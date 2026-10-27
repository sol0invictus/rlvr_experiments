"""
Merge ALL 10 Phase 1 task vectors (4B) into one model at 1/N scale.

W_merged = W_sft + (1/N) * Σ τ_i for Phase 1 i = 1..10

Default uses the Phase 1 4B GRPO checkpoints (not Phase 2 refinements).
"""

import argparse
import os
import shutil
import torch
from pathlib import Path
from safetensors.torch import load_file, save_file


PHASE1_CHECKPOINTS = {
    "basic_arithmetic": "experiments/rlvr_geometry_paper/results/grpo_qwen4b_basic_arithmetic",
    "countdown": "experiments/rlvr_geometry_paper/results/grpo_qwen_4b_instruct_countdown_4to5",
    "shortest_path": "experiments/rlvr_geometry_paper/results/grpo_qwen4b_shortest_path",
    "maze": "experiments/rlvr_geometry_paper/results/grpo_qwen4b_maze",
    "mini_sudoku": "experiments/rlvr_geometry_paper/results/grpo_qwen4b_mini_sudoku",
    "syllogism": "experiments/rlvr_geometry_paper/results/grpo_qwen4b_syllogism",
    "knights_knaves": "experiments/rlvr_geometry_paper/results/grpo_qwen4b_knights_knaves",
    "leg_counting": "experiments/rlvr_geometry_paper/results/grpo_qwen4b_leg_counting",
    "caesar_cipher": "experiments/rlvr_geometry_paper/results/grpo_qwen4b_caesar_cipher",
    "cryptarithm": "experiments/rlvr_geometry_paper/results/grpo_qwen4b_cryptarithm",
}


def load_state_dict(model_path):
    files = sorted(Path(model_path).glob("model*.safetensors"))
    sd = {}
    for f in files:
        sd.update(load_file(str(f)))
    if not sd:
        raise FileNotFoundError(f"No model files in {model_path}")
    return sd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--scale", type=float, default=None,
                        help="Per-task scale (default: 1/N over valid tasks)")
    args = parser.parse_args()

    print("Loading base model...")
    base_sd = load_state_dict(args.base)

    all_tau = {}
    task_norms = {}
    valid_tasks = []

    for task, ckpt_path in PHASE1_CHECKPOINTS.items():
        if not os.path.exists(ckpt_path):
            print(f"  SKIP {task}: {ckpt_path} not found")
            continue
        try:
            print(f"  Loading {task} from {ckpt_path}...")
            task_sd = load_state_dict(ckpt_path)
        except Exception as e:
            print(f"  SKIP {task}: {e}")
            continue

        norm_sq = 0.0
        for key in base_sd:
            if key in task_sd and base_sd[key].shape == task_sd[key].shape:
                tau = task_sd[key].float() - base_sd[key].float()
                if key not in all_tau:
                    all_tau[key] = torch.zeros_like(tau)
                all_tau[key] += tau  # accumulate unscaled; scale applied at end
                norm_sq += tau.norm().item() ** 2
        task_norms[task] = norm_sq ** 0.5
        valid_tasks.append(task)
        print(f"    ||tau|| = {task_norms[task]:.4f}")

    n = len(valid_tasks)
    scale = args.scale if args.scale is not None else (1.0 / n if n else 0.0)
    print(f"\nMerging {n} task vectors at scale={scale:.4f} (requested={args.scale})")
    print(f"Tasks: {valid_tasks}")

    merged_sd = {}
    for key in base_sd:
        if key in all_tau:
            merged_sd[key] = (base_sd[key].float() + scale * all_tau[key]).to(base_sd[key].dtype)
        else:
            merged_sd[key] = base_sd[key]

    os.makedirs(args.output, exist_ok=True)
    for f in Path(args.base).iterdir():
        if f.suffix not in ('.safetensors', '.bin') and f.name != 'trainer_state.json':
            if f.is_file():
                shutil.copy2(str(f), args.output)

    save_file(merged_sd, os.path.join(args.output, "model.safetensors"))
    print(f"\nSaved merged model to {args.output}")
    print(f"Formula: W_sft + {scale:.4f} * (Sum of {n} Phase-1 tau_i)")


if __name__ == "__main__":
    main()
