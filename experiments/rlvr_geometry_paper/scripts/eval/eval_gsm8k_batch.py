#!/usr/bin/env python3
"""
Batched GSM8K evaluation script for RLVR checkpoints.
Uses vLLM for fast inference.
"""

import argparse
import json
import re
import os
from datasets import load_dataset
from pathlib import Path
from tqdm import tqdm

# Try vLLM first, fall back to HF
try:
    from vllm import LLM, SamplingParams
    USE_VLLM = True
except ImportError:
    USE_VLLM = False
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer


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
    # Try #### pattern
    match = re.search(r'####\s*(-?[\d,]+\.?\d*)', text)
    if match:
        return match.group(1).replace(',', '')
    # Try "the answer is X"
    match = re.search(r'(?:the answer is|answer:)\s*(-?[\d,]+\.?\d*)', text, re.IGNORECASE)
    if match:
        return match.group(1).replace(',', '')
    # Last number
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


def evaluate_model_vllm(model_name: str, model_path: str, dataset, output_dir: Path):
    """Evaluate using vLLM (fast batched inference)."""
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

    print(f"  [EVAL] {model_name}...")

    # Build prompts
    prompts = []
    gold_answers = []
    for item in dataset:
        question = item["question"]
        gold = extract_gold_answer(item["answer"])
        prompt = f"Question: {question}\n\nLet's solve this step by step.\n\nAnswer:"
        prompts.append(prompt)
        gold_answers.append(gold)

    # Load model with vLLM
    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        dtype="bfloat16",
        gpu_memory_utilization=0.9,
    )

    sampling_params = SamplingParams(
        max_tokens=512,
        temperature=0,
    )

    # Generate all at once
    outputs = llm.generate(prompts, sampling_params)

    # Evaluate
    correct = 0
    results = []
    for i, output in enumerate(outputs):
        response = output.outputs[0].text
        pred = extract_answer(response)
        gold = gold_answers[i]
        is_correct = pred == gold
        if is_correct:
            correct += 1
        results.append({
            "gold": gold,
            "pred": pred,
            "correct": is_correct,
        })

    accuracy = correct / len(results)

    # Save
    output_data = {
        "model": model_name,
        "path": model_path,
        "accuracy": accuracy,
        "correct": correct,
        "total": len(results),
    }
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"  [DONE] {model_name}: {accuracy*100:.1f}% ({correct}/{len(results)})")

    # Cleanup
    del llm
    import gc
    gc.collect()
    if not USE_VLLM:
        import torch
        torch.cuda.empty_cache()

    return accuracy


def evaluate_model_hf_batch(model_name: str, model_path: str, dataset, output_dir: Path, batch_size: int = 8):
    """Evaluate using HuggingFace with batching."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

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

    print(f"  [EVAL] {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    correct = 0
    total = 0
    samples = list(dataset)

    for i in tqdm(range(0, len(samples), batch_size), desc=f"  {model_name}"):
        batch = samples[i:i+batch_size]

        prompts = [f"Question: {item['question']}\n\nLet's solve this step by step.\n\nAnswer:" for item in batch]
        golds = [extract_gold_answer(item["answer"]) for item in batch]

        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                tokenizer=tokenizer,
            )

        for j, output in enumerate(outputs):
            response = tokenizer.decode(output[inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            pred = extract_answer(response)
            if pred == golds[j]:
                correct += 1
            total += 1

    accuracy = correct / total if total > 0 else 0

    output_data = {
        "model": model_name,
        "path": model_path,
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
    }
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"  [DONE] {model_name}: {accuracy*100:.1f}% ({correct}/{total})")

    del model, tokenizer
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
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--models", type=str, default=None)
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--use_vllm", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.summarize:
        summarize_results(output_dir)
        return

    print("Loading GSM8K test set...")
    dataset = load_dataset("openai/gsm8k", "main", split="test")
    print(f"Loaded {len(dataset)} samples")

    if args.models:
        models = {k: v for k, v in CHECKPOINTS.items() if k in args.models.split(",")}
    else:
        models = CHECKPOINTS

    print(f"\nEvaluating {len(models)} models (batch_size={args.batch_size})...")

    for model_name, model_path in models.items():
        if args.use_vllm and USE_VLLM:
            evaluate_model_vllm(model_name, model_path, dataset, output_dir)
        else:
            evaluate_model_hf_batch(model_name, model_path, dataset, output_dir, args.batch_size)

    summarize_results(output_dir)


if __name__ == "__main__":
    main()
