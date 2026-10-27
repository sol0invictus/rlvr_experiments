#!/usr/bin/env python3
"""
Custom GSM8K evaluation script for RLVR checkpoints.
Handles transformers 5.6.2 tokenizer issue by passing tokenizer explicitly.
"""

import argparse
import json
import re
import torch
from datasets import load_dataset
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# Checkpoint registry
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
    """Extract numerical answer from GSM8K-style response."""
    # Try to find #### pattern first (GSM8K format)
    match = re.search(r'####\s*(-?[\d,]+\.?\d*)', text)
    if match:
        return match.group(1).replace(',', '')

    # Try to find "the answer is X" pattern
    match = re.search(r'(?:the answer is|answer:)\s*(-?[\d,]+\.?\d*)', text, re.IGNORECASE)
    if match:
        return match.group(1).replace(',', '')

    # Try to find last number in the text
    numbers = re.findall(r'-?[\d,]+\.?\d*', text)
    if numbers:
        return numbers[-1].replace(',', '')

    return ""


def extract_gold_answer(answer_text: str) -> str:
    """Extract gold answer from GSM8K answer field."""
    match = re.search(r'####\s*(-?[\d,]+\.?\d*)', answer_text)
    if match:
        return match.group(1).replace(',', '')
    return ""


def evaluate_model(model_name: str, model_path: str, dataset, output_dir: Path,
                   max_samples: int = None, batch_size: int = 1):
    """Evaluate a single model on GSM8K."""
    output_file = output_dir / f"{model_name}.json"

    if output_file.exists():
        print(f"  [SKIP] {model_name} - already evaluated")
        return None

    if not Path(model_path).exists():
        print(f"  [SKIP] {model_name} - path not found: {model_path}")
        return None

    if not (Path(model_path) / "config.json").exists():
        print(f"  [SKIP] {model_name} - no config.json at: {model_path}")
        return None

    print(f"  [EVAL] {model_name}...")

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.eval()

    # Ensure pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Evaluate
    correct = 0
    total = 0
    results = []

    samples = list(dataset)
    if max_samples:
        samples = samples[:max_samples]

    for item in tqdm(samples, desc=f"  {model_name}"):
        question = item["question"]
        gold_answer = extract_gold_answer(item["answer"])

        # Format prompt (simple CoT prompt)
        prompt = f"Question: {question}\n\nLet's solve this step by step.\n\nAnswer:"

        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # Generate with explicit tokenizer
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                tokenizer=tokenizer,  # Key fix for transformers 5.6.2
            )

        # Decode
        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        pred_answer = extract_answer(response)

        # Check correctness
        is_correct = pred_answer == gold_answer
        if is_correct:
            correct += 1
        total += 1

        results.append({
            "question": question,
            "gold": gold_answer,
            "pred": pred_answer,
            "correct": is_correct,
            "response": response[:500]  # Truncate for storage
        })

    accuracy = correct / total if total > 0 else 0

    # Save results
    output_data = {
        "model": model_name,
        "path": model_path,
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "results": results
    }

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"  [DONE] {model_name}: {accuracy*100:.1f}% ({correct}/{total})")

    # Clean up
    del model
    del tokenizer
    torch.cuda.empty_cache()

    return accuracy


def summarize_results(output_dir: Path):
    """Summarize all GSM8K results."""
    results = {}

    for f in output_dir.glob("*.json"):
        try:
            with open(f) as fp:
                data = json.load(fp)
            results[data["model"]] = data["accuracy"]
        except Exception as e:
            print(f"Error reading {f}: {e}")

    if not results:
        print("No results found.")
        return

    print("\n" + "="*60)
    print("GSM8K Results Summary")
    print("="*60)
    print(f"{'Model':<35} | {'Accuracy':>10}")
    print("-"*50)

    for model in sorted(results.keys()):
        acc = results[model]
        print(f"{model:<35} | {acc*100:>8.1f}%")


def main():
    parser = argparse.ArgumentParser(description="Evaluate RLVR checkpoints on GSM8K")
    parser.add_argument("--output_dir", type=str,
                        default="/scratch/sunguha/rlvr_experiments/outputs/gsm8k_eval",
                        help="Output directory")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max samples to evaluate (None = all)")
    parser.add_argument("--models", type=str, default=None,
                        help="Comma-separated model names (None = all)")
    parser.add_argument("--summarize", action="store_true",
                        help="Just summarize existing results")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.summarize:
        summarize_results(output_dir)
        return

    # Load GSM8K test set
    print("Loading GSM8K test set...")
    dataset = load_dataset("openai/gsm8k", "main", split="test")
    print(f"Loaded {len(dataset)} test samples")

    # Select models
    if args.models:
        model_names = args.models.split(",")
        models = {k: v for k, v in CHECKPOINTS.items() if k in model_names}
    else:
        models = CHECKPOINTS

    print(f"\nEvaluating {len(models)} models...")
    print("="*60)

    for model_name, model_path in models.items():
        evaluate_model(model_name, model_path, dataset, output_dir, args.max_samples)

    # Print summary
    summarize_results(output_dir)


if __name__ == "__main__":
    main()
