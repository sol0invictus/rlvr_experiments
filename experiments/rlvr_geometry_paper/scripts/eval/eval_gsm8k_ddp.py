#!/usr/bin/env python3
"""
GSM8K evaluation with Data Parallel (DDP) - each GPU evaluates 1/8 of the samples.
Run with: torchrun --nproc_per_node=8 experiments/rlvr_geometry_paper/scripts/eval/eval_gsm8k_ddp.py
"""

import argparse
import json
import os
import re
import torch
import torch.distributed as dist
from datasets import load_dataset
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


CHECKPOINTS = {
    "sft_3b": "/scratch/sunguha/rlvr_experiments/outputs/sft_qwen25_3b",
    "sft_4b": "/scratch/sunguha/sft_qwen_4.0b_instruct",
    "sft_8b": "/scratch/sunguha/rlvr_experiments/outputs/sft_qwen3_8b",
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
    "grpo_3b_arithmetic": "/scratch/sunguha/rlvr_experiments/results/grpo_qwen25_3b_basic_arithmetic",
    "grpo_3b_countdown": "/scratch/sunguha/rlvr_experiments/results/grpo_qwen25_3b_countdown",
    "grpo_3b_shortest_path": "/scratch/sunguha/rlvr_experiments/results/grpo_qwen25_3b_shortest_path",
    "grpo_3b_syllogism": "/scratch/sunguha/rlvr_experiments/results/grpo_qwen25_3b_syllogism",
    "grpo_3b_maze": "/scratch/sunguha/rlvr_experiments/results/grpo_qwen25_3b_maze",
    "grpo_3b_knights_knaves": "/scratch/sunguha/rlvr_experiments/results/grpo_qwen25_3b_knights_knaves",
    "grpo_3b_mini_sudoku": "/scratch/sunguha/rlvr_experiments/results/grpo_qwen25_3b_mini_sudoku",
    "grpo_3b_leg_counting": "/scratch/sunguha/rlvr_experiments/results/grpo_qwen25_3b_leg_counting",
    "grpo_8b_arithmetic": "/scratch/sunguha/rlvr_experiments/results/grpo_qwen3_8b_basic_arithmetic",
    "grpo_8b_countdown": "/scratch/sunguha/rlvr_experiments/results/grpo_qwen3_8b_countdown",
    "grpo_8b_shortest_path": "/scratch/sunguha/rlvr_experiments/results/grpo_qwen3_8b_shortest_path",
    "grpo_8b_syllogism": "/scratch/sunguha/rlvr_experiments/results/grpo_qwen3_8b_syllogism",
    "grpo_8b_knights_knaves": "/scratch/sunguha/rlvr_experiments/results/grpo_qwen3_8b_knights_knaves",
    "grpo_8b_mini_sudoku": "/scratch/sunguha/rlvr_experiments/results/grpo_qwen3_8b_mini_sudoku",
    "grpo_8b_leg_counting": "/scratch/sunguha/rlvr_experiments/results/grpo_qwen3_8b_leg_counting",
    "p2_arith_atomic_only": "/scratch/sunguha/rlvr_experiments/results/phase2_arith/stage2_l2",
    "p2_arith_atomic_warm": "/scratch/sunguha/rlvr_experiments/results/phase2_arith/stage3_full",
    "p2_sp_atomic_warm": "/scratch/sunguha/rlvr_experiments/results/phase2_sp/atomic_warm_300",
    "p2_sp_rft_base": "/scratch/sunguha/rlvr_experiments/results/phase2_sp/idea_rft_base/grpo",
    "p2_sp_idea_b": "/scratch/sunguha/rlvr_experiments/results/phase2_sp/idea_b/stage2_full",
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
    "p4_layersel_all_but_down": "/scratch/sunguha/rlvr_experiments/results/phase4_layersel/all_but_down",
    "p4_layersel_up": "/scratch/sunguha/rlvr_experiments/results/phase4_layersel/up",
    "p4_layersel_up_q_clean": "/scratch/sunguha/rlvr_experiments/results/phase4_layersel/up_q_clean",
}


def extract_answer(text: str) -> str:
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


def evaluate_model(model_name: str, model_path: str, dataset, output_dir: Path, rank: int, world_size: int):
    """Each rank evaluates a shard of the dataset."""
    output_file = output_dir / f"{model_name}.json"

    if output_file.exists() and rank == 0:
        print(f"  [SKIP] {model_name} - already evaluated")
        return None

    if not Path(model_path).exists():
        if rank == 0:
            print(f"  [SKIP] {model_name} - path not found")
        return None

    if not (Path(model_path) / "config.json").exists():
        if rank == 0:
            print(f"  [SKIP] {model_name} - no config.json")
        return None

    if rank == 0:
        print(f"  [EVAL] {model_name} (DDP, world_size={world_size})...")

    # Load model on this GPU
    device = f"cuda:{rank}"
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).to(device)
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Shard the dataset
    samples = list(dataset)
    my_samples = samples[rank::world_size]  # Each rank gets every world_size-th sample

    local_correct = 0
    local_total = 0

    for item in tqdm(my_samples, desc=f"Rank {rank}", disable=(rank != 0)):
        question = item["question"]
        gold = extract_gold_answer(item["answer"])
        prompt = f"Question: {question}\n\nLet's solve this step by step.\n\nAnswer:"

        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                tokenizer=tokenizer,
            )

        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        pred = extract_answer(response)

        if pred == gold:
            local_correct += 1
        local_total += 1

    # Gather results from all ranks
    local_tensor = torch.tensor([local_correct, local_total], device=device)
    gathered = [torch.zeros(2, device=device) for _ in range(world_size)]
    dist.all_gather(gathered, local_tensor)

    if rank == 0:
        total_correct = sum(t[0].item() for t in gathered)
        total_samples = sum(t[1].item() for t in gathered)
        accuracy = total_correct / total_samples if total_samples > 0 else 0

        output_data = {
            "model": model_name,
            "path": model_path,
            "accuracy": accuracy,
            "correct": int(total_correct),
            "total": int(total_samples),
        }
        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=2)

        print(f"  [DONE] {model_name}: {accuracy*100:.1f}% ({int(total_correct)}/{int(total_samples)})")

    # Cleanup
    del model, tokenizer
    torch.cuda.empty_cache()

    dist.barrier()
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
    parser.add_argument("--models", type=str, default=None)
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.summarize:
        summarize_results(output_dir)
        return

    # Initialize DDP
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if rank == 0:
        print("Loading GSM8K test set...")
    dataset = load_dataset("openai/gsm8k", "main", split="test")
    if rank == 0:
        print(f"Loaded {len(dataset)} samples, distributing across {world_size} GPUs")

    if args.models:
        models = {k: v for k, v in CHECKPOINTS.items() if k in args.models.split(",")}
    else:
        models = CHECKPOINTS

    if rank == 0:
        print(f"\nEvaluating {len(models)} models...")
        print("="*60)

    for model_name, model_path in models.items():
        evaluate_model(model_name, model_path, dataset, output_dir, rank, world_size)

    if rank == 0:
        summarize_results(output_dir)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
