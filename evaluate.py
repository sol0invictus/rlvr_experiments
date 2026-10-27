"""
evaluate.py — GRPO checkpoint evaluation using environment reward functions.

Supports DDP (DistributedDataParallel) for multi-GPU evaluation via torchrun.
Only uses the correctness reward function for final scoring.

Usage (single GPU):
    python evaluate.py configs/eval_countdown.yaml
    python evaluate.py configs/eval_countdown.yaml --model outputs/grpo_qwen3_countdown/checkpoint-300

Usage (multi-GPU with torchrun):
    torchrun --nproc_per_node=8 evaluate.py configs/eval_countdown.yaml
    torchrun --nproc_per_node=4 evaluate.py configs/eval_countdown.yaml --model outputs/grpo_qwen3_countdown/checkpoint-300 --samples 500
"""

import argparse
import json
import os
import sys

import torch
import torch.distributed as dist
import yaml
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from environments import load_environment


# ── Distributed helpers ────────────────────────────────────────────────────

def is_dist_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_dist_initialized() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_dist_initialized() else 1


def is_main_process() -> bool:
    return get_rank() == 0


def setup_distributed():
    """Initialize the distributed process group if launched via torchrun."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        if is_main_process():
            print(f"DDP initialized: world_size={get_world_size()}")
        return True
    return False


def cleanup_distributed():
    if is_dist_initialized():
        dist.destroy_process_group()


def print_rank0(*args, **kwargs):
    """Print only on rank 0."""
    if is_main_process():
        print(*args, **kwargs)


# ── Utilities ──────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_prompt(tokenizer, messages: list, system_prompt: str) -> str:
    if system_prompt:
        full_messages = [{"role": "system", "content": system_prompt}] + messages
    else:
        full_messages = messages
    return tokenizer.apply_chat_template(
        full_messages, tokenize=False, add_generation_prompt=True
    )


@torch.no_grad()
def generate_batch(model, tokenizer, prompt_texts: list[str], gen_cfg: dict, device: torch.device) -> list[str]:
    """Generate completions for a batch of prompts."""
    orig_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    inputs = tokenizer(
        prompt_texts,
        return_tensors="pt",
        padding=True,
        truncation=False,
    ).to(device)

    tokenizer.padding_side = orig_padding_side

    # For DDP-wrapped models, access the underlying module for .generate()
    gen_model = model.module if hasattr(model, "module") else model

    out_ids = gen_model.generate(
        **inputs,
        tokenizer=tokenizer,
        max_new_tokens=gen_cfg.get("max_new_tokens", 512),
        temperature=gen_cfg.get("temperature", 0.0),
        do_sample=gen_cfg.get("temperature", 0.0) > 0,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        use_cache=True,
    )

    prompt_len = inputs["input_ids"].shape[1]
    completions = []
    for seq in out_ids:
        new_tokens = seq[prompt_len:]
        completions.append(tokenizer.decode(new_tokens, skip_special_tokens=False))
    return completions


def run_correctness_reward(reward_fns, completion: str, sample: dict) -> float:
    """
    Run ONLY the correctness reward function and return its score.

    The correctness reward function is identified by name: `correctness_reward`.
    """
    import inspect

    completion_wrapped = [[{"role": "assistant", "content": completion}]]

    for fn in reward_fns:
        if fn.__name__ == "correctness_reward":
            sig = inspect.signature(fn)
            kwargs = {}
            if "ground_truth" in sig.parameters:
                kwargs["ground_truth"] = [sample["ground_truth"]]
            if "metadata" in sig.parameters:
                kwargs["metadata"] = [sample.get("metadata", {})]
            return fn(completion_wrapped, **kwargs)[0]

    # Fallback: if no function named correctness_reward, use the first one
    fn = reward_fns[0]
    sig = inspect.signature(fn)
    kwargs = {}
    if "ground_truth" in sig.parameters:
        kwargs["ground_truth"] = [sample["ground_truth"]]
    if "metadata" in sig.parameters:
        kwargs["metadata"] = [sample.get("metadata", {})]
    return fn(completion_wrapped, **kwargs)[0]


def shard_dataset(dataset, rank: int, world_size: int):
    """Shard a HuggingFace Dataset across DDP ranks."""
    total = len(dataset)
    per_rank = (total + world_size - 1) // world_size  # ceil division
    start = rank * per_rank
    end = min(start + per_rank, total)
    return dataset.select(range(start, end))


def gather_results(local_samples: list[dict], world_size: int) -> list[dict]:
    """Gather sample results from all ranks to rank 0 using dist.gather_object."""
    if world_size == 1:
        return local_samples

    gathered = [None] * world_size if is_main_process() else None
    dist.gather_object(local_samples, gathered, dst=0)

    if is_main_process():
        all_samples = []
        for rank_samples in gathered:
            all_samples.extend(rank_samples)
        return all_samples
    return []


def evaluate(config_path: str, model_path_override: str = None, num_samples_override: int = None):
    # ── Distributed setup ──────────────────────────────────────────────────
    use_ddp = setup_distributed()
    rank = get_rank()
    world_size = get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if use_ddp:
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = load_config(config_path)

    model_path = model_path_override or cfg["model"]["name_or_path"]
    eval_cfg   = cfg.get("evaluation", {})
    gen_cfg    = cfg.get("generation", {})
    system_prompt = cfg.get("system_prompt", "").strip()
    output_dir = cfg.get("output_dir", "outputs/evaluation")

    num_samples = num_samples_override or eval_cfg.get("num_samples", 200)
    seed        = eval_cfg.get("seed", 42)

    # ── Environment & dataset ──────────────────────────────────────────────
    env_data_cfg = dict(cfg)
    if "environment" not in env_data_cfg:
        raise ValueError("Config must have an `environment:` block (same format as training configs).")

    env_data_cfg["environment"] = dict(env_data_cfg["environment"])
    env_data_cfg["environment"]["samples"] = num_samples
    env_data_cfg["seed"] = seed

    env     = load_environment(env_data_cfg)
    dataset = env.get_dataset(env_data_cfg)
    reward_fns = env.get_reward_functions()

    print_rank0(f"\nModel       : {model_path}")
    print_rank0(f"Environment : {cfg['environment'].get('name')} / {cfg['environment'].get('task_name', '')}")
    print_rank0(f"Samples     : {len(dataset)}")
    print_rank0(f"Reward fn   : correctness_reward (only)")
    print_rank0(f"World size  : {world_size}\n")

    # ── Shard dataset across ranks ─────────────────────────────────────────
    if use_ddp:
        dataset = shard_dataset(dataset, rank, world_size)
        print(f"[Rank {rank}] processing {len(dataset)} samples")

    # ── Global inference optimisations ────────────────────────────────────
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # ── Model ──────────────────────────────────────────────────────────────
    model_cfg  = cfg.get("model", {})
    torch_dtype = model_cfg.get("torch_dtype", "bfloat16")
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(torch_dtype, torch.bfloat16)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Try attention implementations in order; prefer config setting, then flash_attention_2 > sdpa > eager
    preferred_attn = model_cfg.get("attn_implementation", "flash_attention_2")
    attn_fallback_order = ["flash_attention_2", "sdpa", "eager"]
    if preferred_attn in attn_fallback_order:
        attn_fallback_order.remove(preferred_attn)
        attn_fallback_order.insert(0, preferred_attn)
    for attn_impl in attn_fallback_order:
        try:
            if use_ddp:
                # DDP: load model onto the local GPU (no device_map="auto")
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=dtype,
                    attn_implementation=attn_impl,
                ).to(device)
            else:
                # Single-process: use device_map="auto" for potential multi-GPU sharding
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=dtype,
                    device_map="auto",
                    attn_implementation=attn_impl,
                )
            print_rank0(f"Attention implementation : {attn_impl}")
            break
        except (ImportError, ValueError, NotImplementedError):
            continue

    model.eval()

    # Wrap with DDP — no gradient sync needed for eval, but DDP wrapper
    # ensures consistent API and future-proofs for any distributed ops
    if use_ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank
        )

    # Optional: torch.compile
    if model_cfg.get("compile", False):
        print_rank0("Compiling model with torch.compile (mode=reduce-overhead)…")
        compile_target = model.module if use_ddp else model
        compile_target = torch.compile(compile_target, mode="reduce-overhead")

    # ── Evaluation loop ────────────────────────────────────────────────────
    local_samples = []
    batch_size = gen_cfg.get("batch_size", 8)

    pbar = tqdm(
        range(0, len(dataset), batch_size),
        desc=f"[Rank {rank}] Evaluating" if use_ddp else "Evaluating",
        disable=not is_main_process() and use_ddp,
    )

    for batch_start in pbar:
        batch = dataset[batch_start : batch_start + batch_size]
        # dataset may return a dict-of-lists (HF datasets) or a list of dicts
        if isinstance(batch, dict):
            batch = [
                {k: batch[k][i] for k in batch}
                for i in range(len(next(iter(batch.values()))))
            ]

        prompt_texts = [build_prompt(tokenizer, s["prompt"], system_prompt) for s in batch]
        completions  = generate_batch(model, tokenizer, prompt_texts, gen_cfg, device)

        for sample, prompt_text, completion in zip(batch, prompt_texts, completions):
            score = run_correctness_reward(reward_fns, completion, sample)

            local_samples.append({
                "prompt":       prompt_text,
                "completion":   completion,
                "ground_truth": sample["ground_truth"],
                "metadata":     sample.get("metadata", {}),
                "correctness":  score,
            })

    # ── Gather results from all ranks ──────────────────────────────────────
    if use_ddp:
        dist.barrier()

    all_samples = gather_results(local_samples, world_size)

    # ── Metrics (rank 0 only) ──────────────────────────────────────────────
    if is_main_process():
        n = len(all_samples)
        correct = sum(s["correctness"] for s in all_samples)
        accuracy = correct / n if n > 0 else 0.0

        metrics = {
            "correctness_reward": accuracy,
            "num_correct": int(correct),
            "num_samples": n,
        }

        print("\n" + "=" * 50)
        print("EVALUATION RESULTS")
        print("=" * 50)
        for name, val in metrics.items():
            if isinstance(val, float):
                print(f"  {name:<30} {val:.4f}")
            else:
                print(f"  {name:<30} {val}")
        print("=" * 50)

        # ── Save results ───────────────────────────────────────────────────
        os.makedirs(output_dir, exist_ok=True)
        metrics_path = os.path.join(output_dir, "metrics.json")
        samples_path = os.path.join(output_dir, "samples.jsonl")

        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        with open(samples_path, "w") as f:
            for s in all_samples:
                f.write(json.dumps(s) + "\n")

        print(f"\nSaved metrics → {metrics_path}")
        print(f"Saved samples → {samples_path}")
    else:
        metrics = None
        all_samples = []

    cleanup_distributed()
    return metrics, all_samples


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a GRPO checkpoint.")
    parser.add_argument("config", help="Path to eval config YAML")
    parser.add_argument("--model", default=None, help="Override model path from config")
    parser.add_argument("--samples", type=int, default=None, help="Override num_samples from config")
    args = parser.parse_args()

    evaluate(args.config, model_path_override=args.model, num_samples_override=args.samples)
