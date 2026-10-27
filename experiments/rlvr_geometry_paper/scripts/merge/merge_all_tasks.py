"""
Merge ALL task vectors into one model.

W_merged = W_sft + Σ τ_i for all tasks i

Usage:
    python experiments/rlvr_geometry_paper/scripts/merge/merge_all_tasks.py --base /path/to/sft --output /path/to/merged
"""

import argparse
import json
import os
import shutil
import torch
from pathlib import Path
from safetensors.torch import load_file, save_file


TASK_CHECKPOINTS = {
    # Best available checkpoint per task
    "shortest_path": "experiments/rlvr_geometry_paper/results/phase2_sp/idea_rft_base/grpo",       # 55.5% (Phase 2 best)
    "basic_arithmetic": "experiments/rlvr_geometry_paper/results/phase2_arith/stage2_l2",           # 78% (Atomic-Only)
    "countdown": "experiments/rlvr_geometry_paper/results/phase2_countdown/stage3_full",             # Atomic-Warm (Phase 2)
    "maze": "experiments/rlvr_geometry_paper/results/grpo_qwen4b_maze",                             # Phase 1
    "mini_sudoku": "experiments/rlvr_geometry_paper/results/grpo_qwen4b_mini_sudoku",               # Phase 1
    "syllogism": "experiments/rlvr_geometry_paper/results/grpo_qwen4b_syllogism",                   # Phase 1
    "knights_knaves": "experiments/rlvr_geometry_paper/results/grpo_qwen4b_knights_knaves",         # Phase 1
    "leg_counting": "experiments/rlvr_geometry_paper/results/grpo_qwen4b_leg_counting",             # Phase 1
    "caesar_cipher": "experiments/rlvr_geometry_paper/results/grpo_qwen4b_caesar_cipher",           # Phase 1
}


def load_state_dict(model_path):
    safetensor_files = sorted(Path(model_path).glob("model*.safetensors"))
    if safetensor_files:
        state_dict = {}
        for f in safetensor_files:
            state_dict.update(load_file(str(f)))
        return state_dict
    raise FileNotFoundError(f"No model files in {model_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--scale", type=float, default=1.0, help="Scale factor per task vector")
    args = parser.parse_args()

    print("Loading base model...")
    base_sd = load_state_dict(args.base)

    # Accumulate task vectors
    all_tau = {}
    task_norms = {}
    valid_tasks = []

    for task, ckpt_path in TASK_CHECKPOINTS.items():
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
                all_tau[key] += args.scale * tau
                norm_sq += tau.norm().item() ** 2

        task_norms[task] = norm_sq ** 0.5
        valid_tasks.append(task)
        print(f"    ||τ|| = {task_norms[task]:.4f}")

    print(f"\nMerging {len(valid_tasks)} task vectors...")
    print(f"Tasks: {valid_tasks}")

    # Pairwise cosine similarities (sampled)
    print("\nPairwise cos(τ_i, τ_j) — computing from accumulated...")

    # Apply merged tau to base
    merged_sd = {}
    for key in base_sd:
        if key in all_tau:
            merged_sd[key] = (base_sd[key].float() + all_tau[key]).to(base_sd[key].dtype)
        else:
            merged_sd[key] = base_sd[key]

    # Save
    os.makedirs(args.output, exist_ok=True)
    for f in Path(args.base).iterdir():
        if f.suffix not in ('.safetensors', '.bin') and f.name != 'trainer_state.json':
            if f.is_file():
                shutil.copy2(str(f), args.output)

    save_file(merged_sd, os.path.join(args.output, "model.safetensors"))
    print(f"\nSaved merged model to {args.output}")
    print(f"Formula: W_sft + {args.scale}×(Σ τ_i) for {len(valid_tasks)} tasks")


if __name__ == "__main__":
    main()
