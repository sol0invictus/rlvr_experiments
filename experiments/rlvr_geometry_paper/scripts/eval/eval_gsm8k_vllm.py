#!/usr/bin/env python3
"""
Fast GSM8K evaluation using vLLM with tensor parallelism (8 GPUs).
~10-15 min per model instead of 5+ hours.
"""

import argparse
import json
import re
import gc
from datasets import load_dataset
from pathlib import Path
from vllm import LLM, SamplingParams


CHECKPOINTS = {
    # SFT Baselines
    "sft_3b": "/scratch/sunguha/rlvr_experiments/outputs/sft_qwen25_3b",
    "sft_4b": "/scratch/sunguha/sft_qwen_4.0b_instruct",
    "sft_8b": "/scratch/sunguha/rlvr_experiments/outputs/sft_qwen3_8b",

    # Phase 1: 4B GRPO
    "grpo_4b_arithmetic": "/scratch/sunguha/rlvr_experiments/results/grpo_qwen4b_basic_arithmetic",
    "grpo_4b_countdown": "/scratch/sunguha/rlvr_experiments/results/grpo_qwen_4b_instruct_countdown_4to5",
    "grpo_4b_shortest_path": "/scratch/sunguha/rlvr_experiments/results/grpo_qwen4b_shortest_path",
    "grpo_4b_syllogism": "/scratch/sunguha/rlvr_experiments/results/grpo_qwen4b_syllogism",
    "grpo_4b_maze": "/scratch/sunguha/rlvr_experiments/results/grpo_qwen4b_maze",
    "grpo_4b_knights_knaves": "/scratch/sunguha/rlvr_experiments/results/grpo_qwen4b_knights_knaves",
    "grpo_4b_mini_sudoku": "/scratch/sunguha/rlvr_experiments/results/grpo_qwen4b_mini_sudoku",
    "grpo_4b_leg_counting": "/scratch/sunguha/rlvr_experiments/results/grpo_qwen4b_leg_counting",
    "grpo_4b_caesar_cipher": "/scratch/sunguha/rlvr_experiments/results/grpo_qwen4b_caesar_cipher",
    "grpo_4b_cryptarithm": "/scratch/sunguha/rlvr_experiments/results/grpo_qwen4b_cryptarithm",

    # Phase 1: 3B GRPO
    "grpo_3b_arithmetic": "/scratch/sunguha/rlvr_experiments/results/grpo_qwen25_3b_basic_arithmetic",
    "grpo_3b_countdown": "/scratch/sunguha/rlvr_experiments/results/grpo_qwen25_3b_countdown",
    "grpo_3b_shortest_path": "/scratch/sunguha/rlvr_experiments/results/grpo_qwen25_3b_shortest_path",
    "grpo_3b_syllogism": "/scratch/sunguha/rlvr_experiments/results/grpo_qwen25_3b_syllogism",
    "grpo_3b_maze": "/scratch/sunguha/rlvr_experiments/results/grpo_qwen25_3b_maze",
    "grpo_3b_knights_knaves": "/scratch/sunguha/rlvr_experiments/results/grpo_qwen25_3b_knights_knaves",
    "grpo_3b_mini_sudoku": "/scratch/sunguha/rlvr_experiments/results/grpo_qwen25_3b_mini_sudoku",
    "grpo_3b_leg_counting": "/scratch/sunguha/rlvr_experiments/results/grpo_qwen25_3b_leg_counting",

    # Phase 1: 8B GRPO
    "grpo_8b_arithmetic": "/scratch/sunguha/rlvr_experiments/results/grpo_qwen3_8b_basic_arithmetic",
    "grpo_8b_countdown": "/scratch/sunguha/rlvr_experiments/results/grpo_qwen3_8b_countdown",
    "grpo_8b_shortest_path": "/scratch/sunguha/rlvr_experiments/results/grpo_qwen3_8b_shortest_path",
    "grpo_8b_syllogism": "/scratch/sunguha/rlvr_experiments/results/grpo_qwen3_8b_syllogism",
    "grpo_8b_knights_knaves": "/scratch/sunguha/rlvr_experiments/results/grpo_qwen3_8b_knights_knaves",
    "grpo_8b_mini_sudoku": "/scratch/sunguha/rlvr_experiments/results/grpo_qwen3_8b_mini_sudoku",
    "grpo_8b_leg_counting": "/scratch/sunguha/rlvr_experiments/results/grpo_qwen3_8b_leg_counting",

    # Phase 2: Atomic curriculum
    "p2_arith_atomic_only": "/scratch/sunguha/rlvr_experiments/results/phase2_arith/stage2_l2",
    "p2_arith_atomic_warm": "/scratch/sunguha/rlvr_experiments/results/phase2_arith/stage3_full",
    "p2_sp_atomic_warm": "/scratch/sunguha/rlvr_experiments/results/phase2_sp/atomic_warm_300",
    "p2_sp_rft_base": "/scratch/sunguha/rlvr_experiments/results/phase2_sp/idea_rft_base/grpo",
    "p2_sp_idea_b": "/scratch/sunguha/rlvr_experiments/results/phase2_sp/idea_b/stage2_full",

    # Phase 2 Extended
    "p2ext_syllogism_direct": "/scratch/sunguha/rlvr_experiments/results/phase2_ext/syllogism/direct_rlvr",
    "p2ext_syllogism_atomic_only": "/scratch/sunguha/rlvr_experiments/results/phase2_ext/syllogism/stage2_l2",
    "p2ext_syllogism_atomic_warm": "/scratch/sunguha/rlvr_experiments/results/phase2_ext/syllogism/stage3_full",
    "p2ext_knights_knaves_direct": "/scratch/sunguha/rlvr_experiments/results/phase2_ext/knights_knaves/direct_rlvr",
    "p2ext_knights_knaves_atomic_only": "/scratch/sunguha/rlvr_experiments/results/phase2_ext/knights_knaves/stage2_l2",
    "p2ext_knights_knaves_atomic_warm": "/scratch/sunguha/rlvr_experiments/results/phase2_ext/knights_knaves/stage3_full",
    "p2ext_maze_direct": "/scratch/sunguha/rlvr_experiments/results/phase2_ext/maze/direct_rlvr",
    "p2ext_maze_atomic_only": "/scratch/sunguha/rlvr_experiments/results/phase2_ext/maze/stage2_l2",
    "p2ext_maze_atomic_warm": "/scratch/sunguha/rlvr_experiments/results/phase2_ext/maze/stage3_full",
    "p2ext_mini_sudoku_direct": "/scratch/sunguha/rlvr_experiments/results/phase2_ext/mini_sudoku/direct_rlvr",
    "p2ext_mini_sudoku_atomic_only": "/scratch/sunguha/rlvr_experiments/results/phase2_ext/mini_sudoku/stage2_l2",
    "p2ext_mini_sudoku_atomic_warm": "/scratch/sunguha/rlvr_experiments/results/phase2_ext/mini_sudoku/stage3_full",

    # Phase 4: Layer-selective merges
    "p4_layersel_all_but_down": "/scratch/sunguha/rlvr_experiments/results/phase4_layersel/all_but_down",
    "p4_layersel_up": "/scratch/sunguha/rlvr_experiments/results/phase4_layersel/up",
    "p4_layersel_up_q_clean": "/scratch/sunguha/rlvr_experiments/results/phase4_layersel/up_q_clean",
}


def extract_answer(text: str) -> str:
    """Extract numerical answer from GSM8K response."""
    match = re.search(r'####\s*(-?[\d,]+\.?\d*)', text)
    if match:
        return match.group(1).replace(',', '')
    match = re.search(r'(?:the answer is|answer:)\s*(-?[\d,]+\.?\d*)', text, re.IGNORECASE)
    if match:
        return match.group(1).replace(',', '')
    numbers = re.findall(r'-?[\d,]+\.?\d*', text)
    if numbers:
        return numbers[-1].replace(',', '')
    return ""


def extract_gold_answer(answer_text: str) -> str:
    match = re.search(r'####\s*(-?[\d,]+\.?\d*)', answer_text)
    if match:
        return match.group(1).replace(',', '')
    return ""


def evaluate_model(model_name: str, model_path: str, prompts: list, gold_answers: list,
                   output_dir: Path, tensor_parallel_size: int = 1):
    """Evaluate a model using vLLM with tensor parallelism."""
    output_file = output_dir / f"{model_name}.json"

    if output_file.exists():
        print(f"  [SKIP] {model_name} - already evaluated")
        return None

    if not Path(model_path).exists():
        print(f"  [SKIP] {model_name} - path not found")
        return None

    if not (Path(model_path) / "config.json").exists():
        print(f"  [SKIP] {model_name} - no config.json")
        return None

    print(f"  [EVAL] {model_name} (TP={tensor_parallel_size})...")

    try:
        llm = LLM(
            model=model_path,
            trust_remote_code=True,
            dtype="bfloat16",
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=0.85,
            max_model_len=2048,
        )

        sampling_params = SamplingParams(
            max_tokens=512,
            temperature=0,
        )

        outputs = llm.generate(prompts, sampling_params)

        correct = 0
        for i, output in enumerate(outputs):
            response = output.outputs[0].text
            pred = extract_answer(response)
            if pred == gold_answers[i]:
                correct += 1

        accuracy = correct / len(gold_answers)

        output_data = {
            "model": model_name,
            "path": model_path,
            "accuracy": accuracy,
            "correct": correct,
            "total": len(gold_answers),
        }
        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=2)

        print(f"  [DONE] {model_name}: {accuracy*100:.1f}% ({correct}/{len(gold_answers)})")

        del llm
        gc.collect()
        import torch
        torch.cuda.empty_cache()

        return accuracy

    except Exception as e:
        print(f"  [ERROR] {model_name}: {e}")
        return None


def summarize_results(output_dir: Path):
    results = {}
    for f in output_dir.glob("*.json"):
        try:
            with open(f) as fp:
                data = json.load(fp)
            results[data["model"]] = data["accuracy"]
        except:
            pass

    if not results:
        print("No results found.")
        return

    print("\n" + "="*60)
    print("GSM8K Results Summary")
    print("="*60)
    print(f"{'Model':<35} | {'Accuracy':>10}")
    print("-"*50)
    for model in sorted(results.keys()):
        print(f"{model:<35} | {results[model]*100:>8.1f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="/scratch/sunguha/rlvr_experiments/outputs/gsm8k_eval")
    parser.add_argument("--tensor_parallel", type=int, default=1, help="Tensor parallel size (1, 2, 4, or 8)")
    parser.add_argument("--models", type=str, default=None)
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.summarize:
        summarize_results(output_dir)
        return

    print("Loading GSM8K test set...")
    dataset = load_dataset("openai/gsm8k", "main", split="test")
    print(f"Loaded {len(dataset)} samples")

    # Build prompts once
    prompts = []
    gold_answers = []
    for item in dataset:
        prompt = f"Question: {item['question']}\n\nLet's solve this step by step.\n\nAnswer:"
        prompts.append(prompt)
        gold_answers.append(extract_gold_answer(item["answer"]))

    if args.models:
        models = {k: v for k, v in CHECKPOINTS.items() if k in args.models.split(",")}
    else:
        models = CHECKPOINTS

    print(f"\nEvaluating {len(models)} models (TP={args.tensor_parallel})...")
    print("="*60)

    for model_name, model_path in models.items():
        evaluate_model(model_name, model_path, prompts, gold_answers, output_dir, args.tensor_parallel)

    summarize_results(output_dir)


if __name__ == "__main__":
    main()
