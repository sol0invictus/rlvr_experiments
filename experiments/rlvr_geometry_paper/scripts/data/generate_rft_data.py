"""
Generate Rejection-Sampled Fine-Tuning (RFT) data for shortest_path.

Samples K completions per problem from a trained checkpoint, verifies each
against BFS ground truth, keeps only correct ones. Oversamples "frontier"
problems (low pass rate) to focus learning on hard-but-solvable cases.

Uses file-based gather (each rank writes a shard, rank 0 merges) to avoid
DDP timeout issues.

Usage:
    torchrun --nproc_per_node=8 experiments/rlvr_geometry_paper/scripts/data/generate_rft_data.py \
        --model experiments/rlvr_geometry_paper/results/phase2_sp/idea_b/stage2_full \
        --num_problems 300 --k 8 --output rft_data.jsonl
"""

import argparse
import json
import os
import re
import random
import time
import torch
from pathlib import Path
from tqdm import tqdm

try:
    import reasoning_gym
except ImportError:
    raise ImportError("reasoning-gym required")

from transformers import AutoModelForCausalLM, AutoTokenizer


def extract_answer(text):
    matches = re.findall(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if matches:
        return matches[-1].strip()
    return None


def verify_answer(extracted, ground_truth):
    if extracted is None:
        return False
    ext = extracted.strip().lower()
    gt = ground_truth.strip().lower()
    if ext == gt:
        return True
    if gt == "infeasible" and "infeasible" in ext:
        return True
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--num_problems", type=int, default=300)
    parser.add_argument("--k", type=int, default=8, help="Samples per problem")
    parser.add_argument("--seed", type=int, default=7777)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=1000, help="Max SFT samples to keep")
    args = parser.parse_args()

    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    device = torch.device(f"cuda:{local_rank}")

    if rank == 0:
        print(f"RFT data generation")
        print(f"Model: {args.model}")
        print(f"Problems: {args.num_problems}, K={args.k}, Temp={args.temperature}")
        print(f"GPUs: {world_size}")

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16
    ).to(device)
    model.eval()
    model.generation_config.stop_strings = None

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    dataset = reasoning_gym.create_dataset(
        "shortest_path", size=args.num_problems, seed=args.seed
    )

    sys_prompt = (
        "You are a helpful assistant. You must always think step-by-step. Put "
        "your thinking process inside <think> and </think> tags, and put your final answer "
        "inside <answer> and </answer> tags."
    )

    all_items = list(dataset)
    my_items = all_items[rank::world_size]
    if rank == 0:
        print(f"Each GPU handles {len(my_items)} problems")

    local_results = []

    for item in tqdm(my_items, desc=f"GPU{rank}", disable=rank != 0):
        question = item["question"]
        gt = str(item["answer"])

        msgs = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": question},
        ]
        input_text = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(input_text, return_tensors="pt").to(device)
        prompt_len = inputs["input_ids"].shape[1]

        correct_completions = []
        with torch.no_grad():
            for i in range(args.k):
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                    tokenizer=tokenizer,
                )
                resp = tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True)
                extracted = extract_answer(resp)
                if verify_answer(extracted, gt):
                    correct_completions.append(resp)

        pass_rate = len(correct_completions) / args.k

        if correct_completions:
            local_results.append({
                "question": question,
                "ground_truth": gt,
                "correct_completions": correct_completions,
                "pass_rate": pass_rate,
                "num_correct": len(correct_completions),
            })

    # File-based gather: each rank writes a shard
    output_dir = str(Path(args.output).parent)
    shard_file = os.path.join(output_dir, f"rft_shard_rank{rank}.json")
    with open(shard_file, 'w') as f:
        json.dump(local_results, f)
    print(f"Rank {rank}: wrote {len(local_results)} results to {shard_file}")

    # Rank 0 waits for all shards then merges
    if rank == 0:
        for wait_rank in range(world_size):
            wait_file = os.path.join(output_dir, f"rft_shard_rank{wait_rank}.json")
            for _ in range(600):
                if os.path.exists(wait_file):
                    break
                time.sleep(1)

        all_results = []
        for r in range(world_size):
            shard = os.path.join(output_dir, f"rft_shard_rank{r}.json")
            with open(shard) as f:
                all_results.extend(json.load(f))

        total_problems = len(all_results)
        total_correct = sum(r["num_correct"] for r in all_results)
        avg_pass_rate = sum(r["pass_rate"] for r in all_results) / max(total_problems, 1)

        print(f"\n{'='*50}")
        print(f"Problems with at least 1 correct: {total_problems}/{args.num_problems}")
        print(f"Total correct completions: {total_correct}")
        print(f"Average pass rate: {avg_pass_rate:.1%}")

        sft_samples = []
        for r in all_results:
            pr = r["pass_rate"]
            if pr <= 2 / args.k:
                weight = 4
            elif pr <= 4 / args.k:
                weight = 2
            else:
                weight = 1

            comp = random.choice(r["correct_completions"])
            for _ in range(weight):
                sft_samples.append({
                    "messages": [
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": r["question"]},
                        {"role": "assistant", "content": comp},
                    ]
                })

        random.shuffle(sft_samples)
        sft_samples = sft_samples[:args.max_samples]

        frontier = sum(1 for r in all_results if r["pass_rate"] <= 2 / args.k)
        medium = sum(1 for r in all_results if 2 / args.k < r["pass_rate"] <= 4 / args.k)
        easy = sum(1 for r in all_results if r["pass_rate"] > 4 / args.k)
        print(f"Difficulty: frontier={frontier}, medium={medium}, easy={easy}")
        print(f"SFT samples after weighting: {len(sft_samples)}")

        with open(args.output, 'w') as f:
            for s in sft_samples:
                f.write(json.dumps(s) + '\n')
        print(f"Saved to {args.output}")

        for r in range(world_size):
            os.remove(os.path.join(output_dir, f"rft_shard_rank{r}.json"))
        print(f"{'='*50}")


if __name__ == "__main__":
    main()
