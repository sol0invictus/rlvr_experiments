"""
Evaluate Cross-Task Transfer for Reasoning Skill Experiments

Evaluates a trained model on multiple reasoning-gym tasks to measure
how training on one task transfers to others.

Usage:
    # Evaluate on default pilot tasks
    python evaluate_transfer.py --model results/grpo_countdown

    # Evaluate on specific tasks
    python evaluate_transfer.py --model results/grpo_countdown \
        --tasks countdown,basic_arithmetic,knights_and_knaves

    # With config file (for model settings)
    python evaluate_transfer.py configs/eval_transfer.yaml \
        --model results/grpo_countdown
"""

import sys
import os
import yaml
import json
import torch
import torch.distributed as dist
import re
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from tqdm import tqdm

try:
    import reasoning_gym
    REASONING_GYM_AVAILABLE = True
except ImportError:
    REASONING_GYM_AVAILABLE = False

# Try Unsloth first for faster inference
try:
    from unsloth import FastLanguageModel
    UNSLOTH_AVAILABLE = True
except ImportError:
    UNSLOTH_AVAILABLE = False
    from transformers import AutoModelForCausalLM, AutoTokenizer


# ======================================================================
# Default pilot tasks (used when --tasks is not specified)
# ======================================================================
PILOT_TASKS = [
    "countdown",
    "basic_arithmetic",
    "knights_and_knaves",
    "graph_coloring",
    "number_sequences",
    "sudoku",
]

# Task-specific parameters for reasoning-gym dataset generation.
# Tasks not listed here use defaults.
TASK_PARAMS = {
    "countdown": {
        "min_numbers": 4,
        "max_numbers": 5,
        "min_target": 50,
        "max_target": 200,
    },
    "sudoku": {
        "size": 4,  # 4x4 for pilot — full 9x9 is too hard for 1.5B
    },
}


# ======================================================================
# DDP helpers
# ======================================================================
def is_distributed():
    return dist.is_initialized()

def get_rank():
    return dist.get_rank() if is_distributed() else 0

def get_world_size():
    return dist.get_world_size() if is_distributed() else 1

def is_main():
    return get_rank() == 0


# ======================================================================
# Model loading
# ======================================================================
def load_model(model_path: str, max_seq_length: int = 2048):
    """Load model for inference. Uses local GPU in DDP mode."""
    rank = get_rank()
    if is_main():
        print(f"Loading model from {model_path}...")

    if UNSLOTH_AVAILABLE and not is_distributed():
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=max_seq_length,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(model)
    else:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = torch.device(f"cuda:{local_rank}")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
        ).to(device)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(model_path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Left padding is required for correct batched generation
    tokenizer.padding_side = "left"

    return model, tokenizer


# ======================================================================
# Generation
# ======================================================================
def generate_response(
    model,
    tokenizer,
    prompt: str,
    system_prompt: Optional[str] = None,
    max_new_tokens: int = 1024,
    temperature: float = 0.1,
) -> str:
    """Generate model response for a single prompt."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            tokenizer=tokenizer,
        )

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )
    return response


def generate_responses_batch(
    model,
    tokenizer,
    prompts: List[str],
    system_prompt: Optional[str] = None,
    max_new_tokens: int = 1024,
    temperature: float = 0.1,
) -> List[str]:
    """Generate model responses for a batch of prompts."""
    all_input_texts = []
    for prompt in prompts:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        input_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        all_input_texts.append(input_text)

    inputs = tokenizer(
        all_input_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            tokenizer=tokenizer,
        )

    responses = []
    for i in range(len(prompts)):
        prompt_len = inputs["attention_mask"][i].sum().item()
        response = tokenizer.decode(
            outputs[i][prompt_len:],
            skip_special_tokens=True,
        )
        responses.append(response)

    return responses


# ======================================================================
# Answer extraction (generalized)
# ======================================================================
def extract_answer(text: str) -> Optional[str]:
    """
    Extract answer from model response.

    Priority:
    1. <answer>...</answer> tags  (DeepSeek-R1 format)
    2. #### marker                (GSM8K format)
    3. Last non-empty line        (fallback)
    """
    # 1. <answer> tags — take the LAST match (the model sometimes echoes
    #    the chat template which contains <answer>...</answer> placeholders)
    matches = re.findall(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if matches:
        return matches[-1].strip()

    # 2. #### marker
    if "####" in text:
        parts = text.split("####")
        if len(parts) > 1:
            answer = parts[-1].strip().split("\n")[0].strip()
            return answer

    # 3. Last non-empty line
    lines = text.strip().split("\n")
    for line in reversed(lines):
        line = line.strip()
        if line and not line.startswith("<"):
            return line

    return None


# ======================================================================
# Correctness verification (task-dispatched)
# ======================================================================
def normalize(s: str) -> str:
    """Normalize a string for comparison."""
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = s.replace("$", "").replace("%", "").rstrip(".")
    return s


def verify_answer(
    task_name: str,
    extracted: str,
    ground_truth: str,
    metadata: Optional[Dict] = None,
) -> bool:
    """
    Check if an extracted answer is correct for a given task.

    Dispatches to task-specific logic where needed, falls back to
    normalized string comparison otherwise.
    """
    if extracted is None:
        return False

    # --- Task-specific verification ---

    if task_name == "countdown":
        return _verify_countdown(extracted, ground_truth, metadata)

    if task_name in ("basic_arithmetic", "chain_sum", "fraction_simplification"):
        return _verify_numeric(extracted, ground_truth)

    if task_name == "number_sequences":
        return _verify_numeric(extracted, ground_truth)

    # --- Default: normalized string match ---
    return normalize(extracted) == normalize(ground_truth)


def _verify_countdown(
    extracted: str, truth: str, metadata: Optional[Dict]
) -> bool:
    """Verify countdown: evaluate expression, check if result == target."""
    target = None
    if isinstance(metadata, dict):
        target = metadata.get("target")

    try:
        cleaned = extracted.strip()
        # Only allow safe arithmetic characters
        if re.match(r"^[\d\s\+\-\*\/\(\)\.]+$", cleaned):
            result = eval(cleaned)
            if target is not None:
                return result == target
            # Fallback: compare to truth expression
            return result == eval(truth)
    except Exception:
        pass

    # Last resort: string match
    return normalize(extracted) == normalize(truth)


def _verify_numeric(extracted: str, truth: str) -> bool:
    """Verify by comparing as numbers, then as strings."""
    try:
        ext_num = float(extracted.replace(",", "").strip())
        truth_num = float(truth.replace(",", "").strip())
        return abs(ext_num - truth_num) < 1e-6
    except (ValueError, TypeError):
        pass
    return normalize(extracted) == normalize(truth)


# ======================================================================
# Single-task evaluation
# ======================================================================
def evaluate_task(
    model,
    tokenizer,
    task_name: str,
    num_samples: int = 200,
    seed: int = 9999,
    system_prompt: Optional[str] = None,
    max_new_tokens: int = 1024,
    verbose: bool = True,
    batch_size: int = 8,
) -> Tuple[Dict[str, Any], List[Dict]]:
    """
    Evaluate model on one reasoning-gym task using batched generation.

    Uses a different seed than training (default 9999) to avoid
    evaluating on memorized examples.

    Returns (metrics_dict, list_of_per_example_results).
    """
    if not REASONING_GYM_AVAILABLE:
        raise ImportError("reasoning-gym required. pip install reasoning-gym")

    task_kw = TASK_PARAMS.get(task_name, {})

    print(f"\n{'='*60}")
    print(f"Evaluating: {task_name}  ({num_samples} samples, seed={seed}, batch={batch_size})")
    print(f"{'='*60}")

    try:
        dataset = reasoning_gym.create_dataset(
            task_name,
            size=num_samples,
            seed=seed,
            **task_kw,
        )
    except TypeError:
        dataset = reasoning_gym.create_dataset(
            task_name,
            size=num_samples,
            seed=seed,
        )

    correct = 0
    valid_format = 0
    reasoning_lengths = []
    results = []

    # Collect all items, then shard across DDP ranks
    all_items = list(dataset)
    rank = get_rank()
    world_size = get_world_size()
    if world_size > 1:
        all_items = all_items[rank::world_size]
        if rank == 0:
            print(f"  DDP: {world_size} ranks, {len(all_items)} samples per rank")

    num_batches = (len(all_items) + batch_size - 1) // batch_size

    for batch_idx in tqdm(range(num_batches), desc=task_name, disable=rank != 0):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, len(all_items))
        batch_items = all_items[batch_start:batch_end]

        # Batch generate
        batch_questions = [item["question"] for item in batch_items]
        batch_responses = generate_responses_batch(
            model,
            tokenizer,
            batch_questions,
            system_prompt=system_prompt,
            max_new_tokens=max_new_tokens,
        )

        # Process each response
        for item, response in zip(batch_items, batch_responses):
            question = item["question"]
            ground_truth = str(item["answer"])
            metadata = item.get("metadata", {})

            extracted = extract_answer(response)

            if extracted is not None:
                valid_format += 1

            is_correct = verify_answer(task_name, extracted, ground_truth, metadata)
            if is_correct:
                correct += 1

            think_match = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
            if think_match:
                reasoning_lengths.append(len(think_match.group(1)))

            result = {
                "question": question[:200],
                "ground_truth": ground_truth[:200],
                "response": response[:500],
                "extracted": extracted,
                "correct": is_correct,
            }
            results.append(result)

            if verbose and len(results) <= 2:
                print(f"\n--- Example {len(results)} ---")
                print(f"Q: {question[:120]}...")
                print(f"Truth: {ground_truth[:80]}")
                print(f"Extracted: {extracted}")
                print(f"Correct: {is_correct}")

    # Gather results across DDP ranks
    if is_distributed():
        # Sum correct and valid_format across ranks
        correct_tensor = torch.tensor([correct], device=f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}")
        format_tensor = torch.tensor([valid_format], device=correct_tensor.device)
        dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(format_tensor, op=dist.ReduceOp.SUM)
        correct = correct_tensor.item()
        valid_format = format_tensor.item()

    accuracy = correct / max(num_samples, 1)
    format_rate = valid_format / max(num_samples, 1)
    avg_reasoning = (
        sum(reasoning_lengths) / len(reasoning_lengths)
        if reasoning_lengths
        else 0
    )

    metrics = {
        "task": task_name,
        "accuracy": accuracy,
        "valid_format_rate": format_rate,
        "avg_reasoning_length": avg_reasoning,
        "num_samples": num_samples,
        "num_correct": correct,
    }

    if is_main():
        print(f"\n  Accuracy:     {accuracy:.2%}  ({correct}/{num_samples})")
        print(f"  Format rate:  {format_rate:.2%}")
        print(f"  Avg reason:   {avg_reasoning:.0f} chars")

    return metrics, results


# ======================================================================
# Main
# ======================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate cross-task transfer on reasoning-gym tasks"
    )
    parser.add_argument(
        "config",
        nargs="?",
        default=None,
        help="Optional YAML config file for model/eval settings",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to trained model (overrides config)",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default=None,
        help="Comma-separated list of tasks (default: pilot set)",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=200,
        help="Samples per task (default: 200)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=9999,
        help="Eval seed — must differ from training seed (default: 9999)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/transfer_eval",
        help="Where to save results",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=1024,
        help="Max tokens per generation (default: 1024)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for generation (default: 8)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Init DDP if launched with torchrun
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

    # Load config if provided
    config = {}
    if args.config and os.path.exists(args.config):
        with open(args.config, "r") as f:
            config = yaml.safe_load(f) or {}

    # Determine model path
    model_path = args.model or config.get("model", {}).get("name_or_path")
    if not model_path:
        print("ERROR: Provide --model or a config with model.name_or_path")
        sys.exit(1)

    # Determine tasks
    if args.tasks:
        tasks = [t.strip() for t in args.tasks.split(",")]
    else:
        tasks = config.get("evaluation", {}).get("tasks", PILOT_TASKS)

    # System prompt
    system_prompt = config.get("system_prompt")
    if system_prompt is None:
        system_prompt = (
            "You are a helpful assistant. Think step by step inside "
            "<think>...</think> tags, then provide your final answer "
            "inside <answer>...</answer> tags."
        )

    max_seq_length = config.get("model", {}).get("max_seq_length", 2048)

    if is_main():
        print(f"Model:   {model_path}")
        print(f"Tasks:   {tasks}")
        print(f"Samples: {args.num_samples} per task")
        print(f"Seed:    {args.seed}")
        print(f"DDP:     {get_world_size()} ranks")

    # Load model
    model, tokenizer = load_model(model_path, max_seq_length=max_seq_length)

    # Evaluate each task
    all_metrics = {}
    all_results = {}

    for task_name in tasks:
        metrics, results = evaluate_task(
            model,
            tokenizer,
            task_name,
            num_samples=args.num_samples,
            seed=args.seed,
            system_prompt=system_prompt,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size,
        )
        all_metrics[task_name] = metrics
        all_results[task_name] = results

    # Only rank 0 prints and saves
    if is_main():
        print("\n" + "=" * 60)
        print("TRANSFER EVALUATION SUMMARY")
        print("=" * 60)
        print(f"Model: {model_path}")
        print(f"{'Task':<25} {'Accuracy':>10} {'Format':>10}")
        print("-" * 50)
        for task_name in tasks:
            m = all_metrics[task_name]
            print(f"{task_name:<25} {m['accuracy']:>9.2%} {m['valid_format_rate']:>9.2%}")
        print("=" * 60)

        # Save results
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = Path(model_path).name

        metrics_file = output_dir / f"metrics_{model_name}_{timestamp}.json"
        with open(metrics_file, "w") as f:
            json.dump(
                {
                    "model": model_path,
                    "timestamp": timestamp,
                    "eval_seed": args.seed,
                    "num_samples": args.num_samples,
                    "tasks": all_metrics,
                },
                f,
                indent=2,
            )
        print(f"\nMetrics saved to {metrics_file}")

        details_file = output_dir / f"details_{model_name}_{timestamp}.json"
        with open(details_file, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"Details saved to {details_file}")

    if is_distributed():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
