"""
Best-of-N evaluation for shortest_path.

Generates N responses per problem, picks the best one using heuristics:
1. If any response has a valid direction path (not "infeasible"), prefer it
2. Among direction paths, pick most common (majority vote)
3. If all say "infeasible", go with infeasible

Uses batch generation with num_return_sequences for speed.
Supports DDP via torchrun for problem sharding across GPUs.

Usage:
    torchrun --nproc_per_node=8 experiments/rlvr_geometry_paper/scripts/eval/eval_best_of_n_sp.py --model ... --n 8 --num_samples 100
"""

import argparse
import json
import os
import re
import torch
import torch.distributed as dist
from collections import Counter
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

try:
    import reasoning_gym
except ImportError:
    raise ImportError("reasoning-gym required")

from transformers import AutoModelForCausalLM, AutoTokenizer


def is_distributed():
    return dist.is_initialized()

def get_rank():
    return dist.get_rank() if is_distributed() else 0

def get_world_size():
    return dist.get_world_size() if is_distributed() else 1

def is_main():
    return get_rank() == 0


def extract_answer(text):
    matches = re.findall(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if matches:
        return matches[-1].strip()
    return None


def is_direction_answer(answer):
    if not answer:
        return False
    if "infeasible" in answer.lower():
        return False
    dirs = re.findall(r'\b(up|down|left|right)\b', answer.lower())
    return len(dirs) > 0


def verify_answer(extracted, ground_truth):
    if extracted is None:
        return False
    ext = extracted.strip().lower()
    gt = ground_truth.strip().lower()
    if ext == gt:
        return True
    if gt == "infeasible" and "infeasible" in ext:
        return True
    return ext == gt


def best_of_n_select(answers):
    if not answers:
        return None
    direction_answers = [a for a in answers if is_direction_answer(a)]
    infeasible_answers = [a for a in answers if a and "infeasible" in a.lower()]
    if direction_answers:
        counter = Counter(direction_answers)
        return counter.most_common(1)[0][0]
    elif infeasible_answers:
        return "infeasible"
    else:
        valid = [a for a in answers if a is not None]
        if valid:
            counter = Counter(valid)
            return counter.most_common(1)[0][0]
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--num_candidates", type=int, default=8)
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=9999)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")

    args.n = args.num_candidates
    if is_main():
        print(f"Best-of-{args.n} eval on shortest_path")
        print(f"Model: {args.model}")
        print(f"Samples: {args.num_samples}, Seed: {args.seed}, Temp: {args.temperature}")

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
        "shortest_path", size=args.num_samples, seed=args.seed
    )

    sys_prompt = (
        "You are a helpful assistant. Think step by step inside "
        "<think>...</think> tags, then provide your final answer "
        "inside <answer>...</answer> tags."
    )

    all_items = list(dataset)
    rank = get_rank()
    world_size = get_world_size()
    if world_size > 1:
        all_items = all_items[rank::world_size]
        if rank == 0:
            print(f"DDP: {world_size} ranks, {len(all_items)} problems per rank")

    correct_greedy = 0
    correct_bon = 0
    total = 0

    for item in tqdm(all_items, desc="Best-of-N", disable=rank != 0):
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

        answers = []
        with torch.no_grad():
            for i in range(args.n):
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                    tokenizer=tokenizer,
                )
                resp = tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True)
                answers.append(extract_answer(resp))

        # pass@1 = first sample
        if verify_answer(answers[0], gt):
            correct_greedy += 1

        selected = best_of_n_select(answers)
        if verify_answer(selected, gt):
            correct_bon += 1

        total += 1

    if is_distributed():
        cg = torch.tensor([correct_greedy], device=device)
        cb = torch.tensor([correct_bon], device=device)
        tt = torch.tensor([total], device=device)
        dist.all_reduce(cg, op=dist.ReduceOp.SUM)
        dist.all_reduce(cb, op=dist.ReduceOp.SUM)
        dist.all_reduce(tt, op=dist.ReduceOp.SUM)
        correct_greedy = cg.item()
        correct_bon = cb.item()
        total = tt.item()

    if is_main():
        greedy_acc = correct_greedy / max(total, 1)
        bon_acc = correct_bon / max(total, 1)
        print(f"\n{'='*50}")
        print(f"Model: {args.model}")
        print(f"pass@1 (temp={args.temperature}): {greedy_acc:.1%} ({correct_greedy}/{total})")
        print(f"best-of-{args.n}:          {bon_acc:.1%} ({correct_bon}/{total})")
        print(f"Lift: +{bon_acc - greedy_acc:.1%}")
        print(f"{'='*50}")

        if args.output_dir:
            out_dir = Path(args.output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = Path(args.model).name
            result = {
                "model": args.model,
                "n": args.n,
                "temperature": args.temperature,
                "num_samples": args.num_samples,
                "seed": args.seed,
                "pass_at_1": greedy_acc,
                "best_of_n": bon_acc,
                "lift": bon_acc - greedy_acc,
            }
            with open(out_dir / f"bon_{model_name}_{ts}.json", "w") as f:
                json.dump(result, f, indent=2)

    if is_distributed():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
