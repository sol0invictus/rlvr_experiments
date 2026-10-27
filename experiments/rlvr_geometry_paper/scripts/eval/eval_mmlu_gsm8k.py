#!/usr/bin/env python3
"""
Evaluate RLVR checkpoints on MMLU and GSM8K using lm-eval-harness.
Covers Phase 1, 2, 3, 4 models.
"""

import argparse
import json
import os
import subprocess
from pathlib import Path

# ============================================================================
# Checkpoint Registry
# ============================================================================

CHECKPOINTS = {
    # Phase 1: SFT bases
    "sft_3b": "/scratch/sunguha/rlvr_experiments/outputs/sft_qwen25_3b",
    "sft_4b": "/scratch/sunguha/sft_qwen_4.0b_instruct",
    "sft_8b": "/scratch/sunguha/rlvr_experiments/outputs/sft_qwen3_8b",

    # Phase 1: 4B GRPO checkpoints (10 tasks)
    "grpo_4b_arithmetic": "/scratch/sunguha/rlvr_experiments/outputs/eval_4b/grpo_qwen4b_basic_arithmetic",
    "grpo_4b_countdown": "/scratch/sunguha/rlvr_experiments/outputs/eval_4b/grpo_qwen_4b_instruct_countdown_4to5",
    "grpo_4b_shortest_path": "/scratch/sunguha/rlvr_experiments/outputs/eval_4b/grpo_qwen4b_shortest_path",
    "grpo_4b_maze": "/scratch/sunguha/rlvr_experiments/outputs/eval_4b/grpo_qwen4b_maze",
    "grpo_4b_mini_sudoku": "/scratch/sunguha/rlvr_experiments/outputs/eval_4b/grpo_qwen4b_mini_sudoku",
    "grpo_4b_syllogism": "/scratch/sunguha/rlvr_experiments/outputs/eval_4b/grpo_qwen4b_syllogism",
    "grpo_4b_knights_knaves": "/scratch/sunguha/rlvr_experiments/outputs/eval_4b/grpo_qwen4b_knights_knaves",
    "grpo_4b_leg_counting": "/scratch/sunguha/rlvr_experiments/outputs/eval_4b/grpo_qwen4b_leg_counting",
    "grpo_4b_caesar_cipher": "/scratch/sunguha/rlvr_experiments/outputs/eval_4b/grpo_qwen4b_caesar_cipher",
    "grpo_4b_cryptarithm": "/scratch/sunguha/rlvr_experiments/outputs/eval_4b/grpo_qwen4b_cryptarithm",

    # Phase 1: 3B GRPO checkpoints
    "grpo_3b_arithmetic": "/scratch/sunguha/rlvr_experiments/outputs/eval_3b/grpo_qwen25_3b_basic_arithmetic",
    "grpo_3b_countdown": "/scratch/sunguha/rlvr_experiments/outputs/eval_3b/grpo_qwen25_3b_countdown",
    "grpo_3b_shortest_path": "/scratch/sunguha/rlvr_experiments/outputs/eval_3b/grpo_qwen25_3b_shortest_path",
    "grpo_3b_maze": "/scratch/sunguha/rlvr_experiments/outputs/eval_3b/grpo_qwen25_3b_maze",
    "grpo_3b_mini_sudoku": "/scratch/sunguha/rlvr_experiments/outputs/eval_3b/grpo_qwen25_3b_mini_sudoku",
    "grpo_3b_syllogism": "/scratch/sunguha/rlvr_experiments/outputs/eval_3b/grpo_qwen25_3b_syllogism",
    "grpo_3b_knights_knaves": "/scratch/sunguha/rlvr_experiments/outputs/eval_3b/grpo_qwen25_3b_knights_knaves",
    "grpo_3b_leg_counting": "/scratch/sunguha/rlvr_experiments/outputs/eval_3b/grpo_qwen25_3b_leg_counting",
    "grpo_3b_caesar_cipher": "/scratch/sunguha/rlvr_experiments/outputs/eval_3b/grpo_qwen25_3b_caesar_cipher",
    "grpo_3b_cryptarithm": "/scratch/sunguha/rlvr_experiments/outputs/eval_3b/grpo_qwen25_3b_cryptarithm",

    # Phase 1: 8B GRPO checkpoints
    "grpo_8b_arithmetic": "/scratch/sunguha/rlvr_experiments/outputs/eval_8b_v2/grpo_qwen3_8b_v2_basic_arithmetic",
    "grpo_8b_countdown": "/scratch/sunguha/rlvr_experiments/outputs/eval_8b_v2/grpo_qwen3_8b_v2_countdown",
    "grpo_8b_shortest_path": "/scratch/sunguha/rlvr_experiments/outputs/eval_8b_v2/grpo_qwen3_8b_v2_shortest_path",
    "grpo_8b_maze": "/scratch/sunguha/rlvr_experiments/outputs/eval_8b_v2/grpo_qwen3_8b_v2_maze",
    "grpo_8b_mini_sudoku": "/scratch/sunguha/rlvr_experiments/outputs/eval_8b_v2/grpo_qwen3_8b_v2_mini_sudoku",
    "grpo_8b_syllogism": "/scratch/sunguha/rlvr_experiments/outputs/eval_8b_v2/grpo_qwen3_8b_v2_syllogism",
    "grpo_8b_knights_knaves": "/scratch/sunguha/rlvr_experiments/outputs/eval_8b_v2/grpo_qwen3_8b_v2_knights_knaves",
    "grpo_8b_leg_counting": "/scratch/sunguha/rlvr_experiments/outputs/eval_8b_v2/grpo_qwen3_8b_v2_leg_counting",
    "grpo_8b_caesar_cipher": "/scratch/sunguha/rlvr_experiments/outputs/eval_8b_v2/grpo_qwen3_8b_v2_caesar_cipher",
    "grpo_8b_cryptarithm": "/scratch/sunguha/rlvr_experiments/outputs/eval_8b_v2/grpo_qwen3_8b_v2_cryptarithm",

    # Phase 2: Atomic curriculum checkpoints (4B, 7 tasks)
    # Arithmetic
    "p2_arith_direct": "/scratch/sunguha/rlvr_experiments/results/phase2_arith/direct_rlvr",
    "p2_arith_atomic_only": "/scratch/sunguha/rlvr_experiments/results/phase2_arith/stage2_l2",
    "p2_arith_atomic_warm": "/scratch/sunguha/rlvr_experiments/results/phase2_arith/stage3_full",
    # Countdown
    "p2_countdown_direct": "/scratch/sunguha/rlvr_experiments/results/phase2_countdown/direct_rlvr",
    # Shortest path
    "p2_sp_direct": "/scratch/sunguha/rlvr_experiments/results/phase2_sp/direct_rlvr",
    "p2_sp_atomic_warm": "/scratch/sunguha/rlvr_experiments/results/phase2_sp/atomic_warm_300",
    "p2_sp_rft_base": "/scratch/sunguha/rlvr_experiments/results/phase2_sp/idea_rft_base/grpo",
    # Phase 2 Extended (4 new tasks)
    "p2ext_syllogism_direct": "/scratch/sunguha/rlvr_experiments/results/phase2_ext/syllogism/direct_rlvr",
    "p2ext_syllogism_atomic_only": "/scratch/sunguha/rlvr_experiments/results/phase2_ext/syllogism/stage2_l2",
    "p2ext_syllogism_atomic_warm": "/scratch/sunguha/rlvr_experiments/results/phase2_ext/syllogism/stage3_full",
    "p2ext_knights_direct": "/scratch/sunguha/rlvr_experiments/results/phase2_ext/knights_knaves/direct_rlvr",
    "p2ext_knights_atomic_only": "/scratch/sunguha/rlvr_experiments/results/phase2_ext/knights_knaves/stage2_l2",
    "p2ext_knights_atomic_warm": "/scratch/sunguha/rlvr_experiments/results/phase2_ext/knights_knaves/stage3_full",
    "p2ext_maze_direct": "/scratch/sunguha/rlvr_experiments/results/phase2_ext/maze/direct_rlvr",
    "p2ext_maze_atomic_only": "/scratch/sunguha/rlvr_experiments/results/phase2_ext/maze/stage2_l2",
    "p2ext_maze_atomic_warm": "/scratch/sunguha/rlvr_experiments/results/phase2_ext/maze/stage3_full",
    "p2ext_sudoku_direct": "/scratch/sunguha/rlvr_experiments/results/phase2_ext/mini_sudoku/direct_rlvr",
    "p2ext_sudoku_atomic_only": "/scratch/sunguha/rlvr_experiments/results/phase2_ext/mini_sudoku/stage2_l2",
    "p2ext_sudoku_atomic_warm": "/scratch/sunguha/rlvr_experiments/results/phase2_ext/mini_sudoku/stage3_full",

    # Phase 4: Merged models (4B)
    "p4_merge_2task": "/scratch/sunguha/rlvr_experiments/results/phase4_pairs/sp_arith_s1.0",
    "p4_merge_9task_s033": "/scratch/sunguha/rlvr_experiments/results/phase4_pairs/all_tasks_s0.33",
}

# Subsets for different evaluation runs
PHASE1_MODELS = [k for k in CHECKPOINTS if k.startswith("sft_") or k.startswith("grpo_")]
PHASE2_MODELS = [k for k in CHECKPOINTS if k.startswith("p2_") or k.startswith("p2ext_")] + ["sft_4b"]
PHASE4_MODELS = [k for k in CHECKPOINTS if k.startswith("p4_")] + ["sft_4b"]


def run_eval(model_name: str, model_path: str, tasks: str, output_dir: str, batch_size: int = 4):
    """Run lm-eval-harness on a single model."""
    output_path = Path(output_dir) / f"{model_name}.json"

    if output_path.exists():
        print(f"  [SKIP] {model_name} already evaluated")
        return

    print(f"  [EVAL] {model_name} on {tasks}...")

    cmd = [
        "python", "-m", "lm_eval",
        "--model", "hf",
        "--model_args", f"pretrained={model_path},trust_remote_code=True,dtype=bfloat16",
        "--tasks", tasks,
        "--batch_size", str(batch_size),
        "--output_path", str(output_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"  [ERROR] {model_name}: {result.stderr[:500]}")
        return None

    print(f"  [DONE] {model_name}")
    return output_path


def run_phase(phase: str, tasks: str, output_dir: str, batch_size: int = 4):
    """Run evaluation for a specific phase."""

    if phase == "1":
        models = PHASE1_MODELS
    elif phase == "2":
        models = PHASE2_MODELS
    elif phase == "4":
        models = PHASE4_MODELS
    elif phase == "all":
        models = list(CHECKPOINTS.keys())
    else:
        print(f"Unknown phase: {phase}")
        return

    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Phase {phase}: Evaluating {len(models)} models on {tasks}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}\n")

    for model_name in models:
        model_path = CHECKPOINTS[model_name]
        if not os.path.exists(model_path):
            print(f"  [SKIP] {model_name} - path not found: {model_path}")
            continue
        run_eval(model_name, model_path, tasks, output_dir, batch_size)


def summarize_results(output_dir: str):
    """Summarize all results in a directory."""
    results = {}

    for f in Path(output_dir).glob("*.json"):
        try:
            with open(f) as fp:
                data = json.load(fp)

            model_name = f.stem
            results[model_name] = {}

            if "results" in data:
                for task, metrics in data["results"].items():
                    if "acc" in metrics:
                        results[model_name][task] = metrics["acc"]
                    elif "acc_norm" in metrics:
                        results[model_name][task] = metrics["acc_norm"]
                    elif "exact_match" in metrics:
                        results[model_name][task] = metrics["exact_match"]
        except Exception as e:
            print(f"Error reading {f}: {e}")

    return results


def print_summary_table(results: dict):
    """Print results as a table."""
    if not results:
        print("No results found.")
        return

    # Get all tasks
    all_tasks = set()
    for model_results in results.values():
        all_tasks.update(model_results.keys())
    all_tasks = sorted(all_tasks)

    # Print header
    header = "Model".ljust(35) + " | " + " | ".join(t[:12].ljust(12) for t in all_tasks)
    print(header)
    print("-" * len(header))

    # Print rows
    for model_name in sorted(results.keys()):
        row = model_name.ljust(35) + " | "
        for task in all_tasks:
            val = results[model_name].get(task, None)
            if val is not None:
                row += f"{val*100:5.1f}%".ljust(12) + " | "
            else:
                row += "N/A".ljust(12) + " | "
        print(row)


def main():
    parser = argparse.ArgumentParser(description="Evaluate RLVR checkpoints on MMLU/GSM8K")
    parser.add_argument("--phase", type=str, default="all",
                        help="Phase to evaluate: 1, 2, 4, or all")
    parser.add_argument("--tasks", type=str, default="gsm8k,mmlu",
                        help="Tasks to evaluate (comma-separated)")
    parser.add_argument("--output_dir", type=str,
                        default="/scratch/sunguha/rlvr_experiments/outputs/mmlu_gsm8k_eval",
                        help="Output directory")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for evaluation")
    parser.add_argument("--summarize", action="store_true",
                        help="Just summarize existing results")
    args = parser.parse_args()

    if args.summarize:
        results = summarize_results(args.output_dir)
        print_summary_table(results)
        return

    run_phase(args.phase, args.tasks, args.output_dir, args.batch_size)

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    results = summarize_results(args.output_dir)
    print_summary_table(results)


if __name__ == "__main__":
    main()
