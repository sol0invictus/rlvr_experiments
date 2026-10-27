"""
Evaluate arith models on training-like data (seed=42, same distribution as GRPO training)
vs eval data (seed=9999) to check if GRPO improves in-domain.
"""
import json
import os
import torch
import torch.distributed as dist
import reasoning_gym
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import re

def is_distributed():
    return dist.is_initialized()
def get_rank():
    return dist.get_rank() if is_distributed() else 0
def get_world_size():
    return dist.get_world_size() if is_distributed() else 1

def eval_model_on_arith(model_path, label, seeds=[42, 9999], n=50):
    """Eval one model on basic_arithmetic with different seeds."""
    if "RANK" in os.environ:
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

    rank = get_rank()
    world_size = get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if rank == 0:
        print(f"\n{'='*60}")
        print(f"Model: {label} ({model_path})")
        print(f"{'='*60}")

    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16
    ).to(f"cuda:{local_rank}")
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    system_prompt = (
        "You are a helpful assistant. Think step by step inside "
        "<think>...</think> tags, then provide your final answer "
        "inside <answer>...</answer> tags."
    )

    for seed in seeds:
        dataset = reasoning_gym.create_dataset("basic_arithmetic", size=n, seed=seed)
        all_items = list(dataset)

        # Shard across ranks
        my_items = all_items[rank::world_size]

        correct = 0
        for item in my_items:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": item["question"]}
            ]
            input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs, max_new_tokens=2048, temperature=0.1,
                    do_sample=True, pad_token_id=tokenizer.pad_token_id,
                    tokenizer=tokenizer,
                )
            response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

            # Extract answer
            match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
            extracted = match.group(1).strip() if match else None

            # Verify
            if extracted is not None:
                truth = str(item["answer"])
                try:
                    if abs(float(extracted.replace(",","")) - float(truth.replace(",",""))) < 1e-6:
                        correct += 1
                except:
                    if extracted.strip() == truth.strip():
                        correct += 1

        # Aggregate
        if is_distributed():
            t = torch.tensor([correct], device=f"cuda:{local_rank}")
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            correct = t.item()

        if rank == 0:
            seed_label = "TRAIN-LIKE (seed=42)" if seed == 42 else "EVAL (seed=9999)"
            print(f"  {seed_label}: {correct}/{n} = {correct/n:.0%}")

    if is_distributed():
        dist.destroy_process_group()


if __name__ == "__main__":
    import sys
    # Usage: torchrun ... experiments/rlvr_geometry_paper/scripts/eval/eval_arith_train_domain.py <model_path> <label>
    model_path = sys.argv[1]
    label = sys.argv[2] if len(sys.argv) > 2 else model_path
    eval_model_on_arith(model_path, label)
