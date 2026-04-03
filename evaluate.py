"""
evaluate.py — GRPO checkpoint evaluation using environment reward functions.

Usage:
    python evaluate.py configs/eval_countdown.yaml
    python evaluate.py configs/eval_countdown.yaml --model outputs/grpo_qwen3_countdown/checkpoint-300
    python evaluate.py configs/eval_countdown.yaml --samples 100
"""

import argparse
import json
import os
import sys

import torch
import yaml
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from environments import load_environment


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
def generate(model, tokenizer, prompt_text: str, gen_cfg: dict) -> str:
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    out_ids = model.generate(
        **inputs,
        max_new_tokens=gen_cfg.get("max_new_tokens", 512),
        temperature=gen_cfg.get("temperature", 0.0),
        do_sample=gen_cfg.get("temperature", 0.0) > 0,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    new_tokens = out_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=False)


def run_reward_fns(reward_fns, completion: str, sample: dict) -> dict:
    """Run all reward functions and return {fn_name: score}."""
    import inspect

    completion_wrapped = [[{"role": "assistant", "content": completion}]]
    scores = {}
    for fn in reward_fns:
        sig = inspect.signature(fn)
        kwargs = {}
        if "ground_truth" in sig.parameters:
            kwargs["ground_truth"] = [sample["ground_truth"]]
        if "metadata" in sig.parameters:
            kwargs["metadata"] = [sample.get("metadata", {})]
        scores[fn.__name__] = fn(completion_wrapped, **kwargs)[0]
    return scores


def evaluate(config_path: str, model_path_override: str = None, num_samples_override: int = None):
    cfg = load_config(config_path)

    model_path = model_path_override or cfg["model"]["name_or_path"]
    eval_cfg   = cfg.get("evaluation", {})
    gen_cfg    = cfg.get("generation", {})
    system_prompt = cfg.get("system_prompt", "").strip()
    output_dir = cfg.get("output_dir", "outputs/evaluation")

    num_samples = num_samples_override or eval_cfg.get("num_samples", 200)
    seed        = eval_cfg.get("seed", 42)

    # ── Environment & dataset ──────────────────────────────────────────────
    # The eval config reuses the same `environment:` block as training configs.
    env_data_cfg = dict(cfg)
    if "environment" not in env_data_cfg:
        raise ValueError("Config must have an `environment:` block (same format as training configs).")

    # Override sample count for eval
    env_data_cfg["environment"] = dict(env_data_cfg["environment"])
    env_data_cfg["environment"]["samples"] = num_samples
    env_data_cfg["seed"] = seed

    env     = load_environment(env_data_cfg)
    dataset = env.get_dataset(env_data_cfg)
    reward_fns = env.get_reward_functions()

    print(f"\nModel       : {model_path}")
    print(f"Environment : {cfg['environment'].get('name')} / {cfg['environment'].get('task_name', '')}")
    print(f"Samples     : {len(dataset)}")
    print(f"Reward fns  : {[fn.__name__ for fn in reward_fns]}\n")

    # ── Model ──────────────────────────────────────────────────────────────
    torch_dtype = cfg.get("model", {}).get("torch_dtype", "bfloat16")
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(torch_dtype, "auto")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=dtype, device_map="auto"
    )
    model.eval()

    # ── Evaluation loop ────────────────────────────────────────────────────
    all_samples = []
    reward_sums = {}

    for sample in tqdm(dataset, desc="Evaluating"):
        prompt_text = build_prompt(tokenizer, sample["prompt"], system_prompt)
        completion  = generate(model, tokenizer, prompt_text, gen_cfg)
        scores      = run_reward_fns(reward_fns, completion, sample)

        for name, val in scores.items():
            reward_sums[name] = reward_sums.get(name, 0.0) + val

        all_samples.append({
            "prompt":       prompt_text,
            "completion":   completion,
            "ground_truth": sample["ground_truth"],
            "metadata":     sample.get("metadata", {}),
            "scores":       scores,
            "total_reward": sum(scores.values()),
        })

    n = len(all_samples)

    # ── Metrics ────────────────────────────────────────────────────────────
    metrics = {name: total / n for name, total in reward_sums.items()}
    metrics["total_reward"] = sum(metrics.values())
    metrics["num_samples"]  = n

    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    for name, val in metrics.items():
        print(f"  {name:<30} {val:.4f}")
    print("=" * 50)

    # ── Save results ───────────────────────────────────────────────────────
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

    return metrics, all_samples


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a GRPO checkpoint.")
    parser.add_argument("config", help="Path to eval config YAML")
    parser.add_argument("--model", default=None, help="Override model path from config")
    parser.add_argument("--samples", type=int, default=None, help="Override num_samples from config")
    args = parser.parse_args()

    evaluate(args.config, model_path_override=args.model, num_samples_override=args.samples)
