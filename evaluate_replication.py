"""
Replication eval: same tasks, seeds and sampling as evaluate_transfer.py (the
paper-era eval), plus batched HF / vLLM generation and every response scored
under four answer-extraction rules from the same samples:
    last_fallback  last <answer>, else '####', else last line -- the paper's
                   post-fix rule (evaluate_transfer.py); reported as `accuracy`
    last           last <answer>...</answer> only (strict)
    first          first <answer>...</answer> only (what the GRPO training reward uses)
    legacy         first <answer>, else '####', else last line (the pre-fix eval)

Usage:
    # Evaluate on default pilot tasks
    python evaluate_replication.py --model results/grpo_countdown

    # Evaluate on specific tasks
    python evaluate_replication.py --model results/grpo_countdown \
        --tasks countdown,basic_arithmetic,knights_and_knaves

    # Batched vLLM generation (much faster; cross-check against --backend hf)
    python evaluate_replication.py --model results/grpo_countdown --backend vllm

    # With config file (for model settings / system prompt)
    python evaluate_replication.py configs/eval_transfer.yaml \
        --model results/grpo_countdown
"""

import sys
import os
import yaml
import json
import torch
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

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant. Think step by step inside "
    "<think>...</think> tags, then provide your final answer "
    "inside <answer>...</answer> tags."
)


# ======================================================================
# Generation backends
# ======================================================================
def build_prompt(tokenizer, question: str, system_prompt: Optional[str]) -> str:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": question})
    # No chat_template_kwargs: for Qwen3 hybrid models enable_thinking stays at its
    # default (True), so the model opens <think> itself, as in training.
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def model_sampling_defaults(model_path: str) -> Dict[str, Any]:
    """Sampling knobs HF generate inherits from the checkpoint's generation_config.json
    when the caller only sets temperature (as the original eval did): e.g. Qwen3-4B
    top_k=20 / top_p=0.95, Qwen2.5 repetition_penalty 1.05-1.1. The vLLM backend
    applies the same values so both backends sample from the same distribution."""
    from transformers import GenerationConfig
    try:
        gc = GenerationConfig.from_pretrained(model_path)
    except OSError:
        gc = GenerationConfig()
    return {
        "top_k": gc.top_k if gc.top_k is not None else 50,
        "top_p": gc.top_p if gc.top_p is not None else 1.0,
        "min_p": gc.min_p if gc.min_p is not None else 0.0,
        "repetition_penalty": gc.repetition_penalty if gc.repetition_penalty is not None else 1.0,
    }


class HFGenerator:
    """Batched HF generate with left padding."""

    def __init__(self, model_path: str, batch_size: int, seed: int, stop_strings: Optional[List[str]]):
        print(f"Loading model from {model_path} (HF)...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map="auto"
        )
        self.model.eval()
        # GRPO checkpoints carry stop_strings=['</answer>'] in generation_config.json
        # (train_grpo.py sets it before training); SFT checkpoints don't. Set it
        # explicitly so every model is evaluated the same way.
        self.model.generation_config.stop_strings = stop_strings
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.batch_size = batch_size
        self.sampling = {**model_sampling_defaults(model_path), "stop_strings": stop_strings}
        torch.manual_seed(seed)

    def generate(self, prompts: List[str], max_new_tokens: int, temperature: float) -> List[str]:
        responses = []
        for i in tqdm(range(0, len(prompts), self.batch_size), desc="generate"):
            batch = prompts[i:i + self.batch_size]
            inputs = self.tokenizer(batch, return_tensors="pt", padding=True).to(self.model.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    pad_token_id=self.tokenizer.pad_token_id,
                    tokenizer=self.tokenizer,
                )
            new_tokens = outputs[:, inputs["input_ids"].shape[1]:]
            responses.extend(self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True))
        return responses


class VLLMGenerator:
    """Batched vLLM generation (all prompts in one call)."""

    def __init__(self, model_path: str, seed: int, gpu_memory_utilization: float,
                 stop_strings: Optional[List[str]]):
        from vllm import LLM
        print(f"Loading model from {model_path} (vLLM)...")
        self.llm = LLM(model=model_path, dtype="bfloat16", seed=seed,
                       gpu_memory_utilization=gpu_memory_utilization)
        self.tokenizer = self.llm.get_tokenizer()
        self.seed = seed
        self.sampling = {**model_sampling_defaults(model_path), "stop_strings": stop_strings}

    def generate(self, prompts: List[str], max_new_tokens: int, temperature: float) -> List[str]:
        from vllm import SamplingParams
        s = self.sampling
        params = SamplingParams(temperature=temperature, top_p=s["top_p"], top_k=s["top_k"],
                                min_p=s["min_p"], repetition_penalty=s["repetition_penalty"],
                                stop=s["stop_strings"], include_stop_str_in_output=True,
                                max_tokens=max_new_tokens, seed=self.seed)
        outputs = self.llm.generate(prompts, params)
        return [o.outputs[0].text for o in outputs]


# ======================================================================
# Answer extraction
# ======================================================================
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def extract_answer(text: str, rule: str = "last") -> Optional[str]:
    """
    Extract answer from model response.

    rule="last":   last <answer>...</answer> (post-fix eval; an echoed template tag
                   such as "inside <answer>...</answer> tags" can't hijack it)
    rule="first":  first <answer>...</answer> (training reward behaviour)
    rule="legacy": first <answer>, else '####' marker, else last non-empty line
    rule="last_fallback": last <answer>, else '####' marker, else last non-empty line
                   (the paper-era post-fix rule in evaluate_transfer.py)
    """
    matches = ANSWER_RE.findall(text)
    if rule == "last":
        return matches[-1].strip() if matches else None
    if rule == "first":
        return matches[0].strip() if matches else None

    # legacy / last_fallback
    if matches:
        return (matches[-1] if rule == "last_fallback" else matches[0]).strip()
    if "####" in text:
        parts = text.split("####")
        if len(parts) > 1:
            return parts[-1].strip().split("\n")[0].strip()
    for line in reversed(text.strip().split("\n")):
        line = line.strip()
        if line and not line.startswith("<"):
            return line
    return None


EXTRACTION_RULES = ("last_fallback", "last", "first", "legacy")


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
    except (ValueError, TypeError, OverflowError):
        pass
    return normalize(extracted) == normalize(truth)


# ======================================================================
# Single-task evaluation
# ======================================================================
def evaluate_task(
    generator,
    task_name: str,
    num_samples: int = 200,
    seed: int = 9999,
    system_prompt: Optional[str] = None,
    max_new_tokens: int = 1024,
    temperature: float = 0.1,
    verbose: bool = True,
) -> Tuple[Dict[str, Any], List[Dict]]:
    """
    Evaluate model on one reasoning-gym task.

    Uses a different seed than training (default 9999) to avoid
    evaluating on memorized examples.

    Returns (metrics_dict, list_of_per_example_results).
    """
    if not REASONING_GYM_AVAILABLE:
        raise ImportError("reasoning-gym required. pip install reasoning-gym")

    task_kw = TASK_PARAMS.get(task_name, {})

    print(f"\n{'='*60}")
    print(f"Evaluating: {task_name}  ({num_samples} samples, seed={seed})")
    print(f"{'='*60}")

    try:
        dataset = reasoning_gym.create_dataset(
            task_name,
            size=num_samples,
            seed=seed,
            **task_kw,
        )
    except TypeError:
        # Task doesn't accept extra kwargs
        dataset = reasoning_gym.create_dataset(
            task_name,
            size=num_samples,
            seed=seed,
        )
    items = list(dataset)

    prompts = [build_prompt(generator.tokenizer, item["question"], system_prompt) for item in items]
    responses = generator.generate(prompts, max_new_tokens=max_new_tokens, temperature=temperature)

    correct = {rule: 0 for rule in EXTRACTION_RULES}
    valid_format = {rule: 0 for rule in EXTRACTION_RULES}
    reasoning_lengths, response_lengths, response_tokens = [], [], []
    results = []

    for item, response in zip(items, responses):
        ground_truth = str(item["answer"])
        metadata = item.get("metadata", {})

        result = {
            "question": item["question"],
            "ground_truth": ground_truth,
            "response": response,
        }
        for rule in EXTRACTION_RULES:
            extracted = extract_answer(response, rule)
            is_correct = verify_answer(task_name, extracted, ground_truth, metadata)
            valid_format[rule] += extracted is not None
            correct[rule] += is_correct
            result[f"extracted_{rule}"] = extracted
            result[f"correct_{rule}"] = is_correct
        results.append(result)

        # Track reasoning length
        think_match = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
        if think_match:
            reasoning_lengths.append(len(think_match.group(1)))
        response_lengths.append(len(response))
        response_tokens.append(len(generator.tokenizer(response, add_special_tokens=False)["input_ids"]))

        # Show first few examples
        if verbose and len(results) <= 2:
            print(f"\n--- Example {len(results)} ---")
            print(f"Q: {item['question'][:120]}...")
            print(f"Truth: {ground_truth[:80]}")
            print(f"Extracted (last_fallback): {result['extracted_last_fallback']}")
            print(f"Correct (last_fallback): {result['correct_last_fallback']}")

    n = max(num_samples, 1)
    mean = lambda xs: sum(xs) / len(xs) if xs else 0
    metrics = {
        "task": task_name,
        "accuracy": correct["last_fallback"] / n,
        "accuracy_strict": correct["last"] / n,
        "valid_format_rate": valid_format["last"] / n,
        "accuracy_by_rule": {rule: correct[rule] / n for rule in EXTRACTION_RULES},
        "valid_format_by_rule": {rule: valid_format[rule] / n for rule in EXTRACTION_RULES},
        "avg_reasoning_length": mean(reasoning_lengths),
        "avg_response_chars": mean(response_lengths),
        "avg_response_tokens": mean(response_tokens),
        "num_samples": num_samples,
        "num_correct": correct["last_fallback"],
    }

    print(f"\n  Accuracy (last_fallback/last/first/legacy): "
          + " / ".join(f"{metrics['accuracy_by_rule'][r]:.2%}" for r in EXTRACTION_RULES))
    print(f"  Format rate:  {metrics['valid_format_rate']:.2%}")
    print(f"  Avg reason:   {metrics['avg_reasoning_length']:.0f} chars, "
          f"response {metrics['avg_response_chars']:.0f} chars / {metrics['avg_response_tokens']:.0f} tokens")

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
        "--temperature",
        type=float,
        default=0.1,
        help="Sampling temperature; 0 = greedy (default: 0.1, the original script's value)",
    )
    parser.add_argument(
        "--backend",
        choices=["hf", "vllm"],
        default="hf",
        help="Generation backend (default: hf)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="HF backend batch size (default: 32)",
    )
    parser.add_argument(
        "--stop_at_answer",
        action="store_true",
        help="Stop generation at the first </answer> (default: off, generate to EOS / max_new_tokens)",
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.85,
        help="vLLM backend GPU memory fraction (default: 0.85)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

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
        system_prompt = DEFAULT_SYSTEM_PROMPT
    system_prompt = system_prompt.strip()

    print(f"Model:   {model_path}")
    print(f"Tasks:   {tasks}")
    print(f"Samples: {args.num_samples} per task")
    print(f"Seed:    {args.seed}")
    print(f"Backend: {args.backend}, temperature={args.temperature}, max_new_tokens={args.max_new_tokens}")

    # Load model
    stop_strings = ["</answer>"] if args.stop_at_answer else None
    if args.backend == "vllm":
        generator = VLLMGenerator(model_path, seed=args.seed,
                                  gpu_memory_utilization=args.gpu_memory_utilization,
                                  stop_strings=stop_strings)
    else:
        generator = HFGenerator(model_path, batch_size=args.batch_size, seed=args.seed,
                                stop_strings=stop_strings)
    print(f"Sampling: temperature={args.temperature}, {generator.sampling}")

    # Evaluate each task
    all_metrics = {}
    all_results = {}

    for task_name in tasks:
        metrics, results = evaluate_task(
            generator,
            task_name,
            num_samples=args.num_samples,
            seed=args.seed,
            system_prompt=system_prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        all_metrics[task_name] = metrics
        all_results[task_name] = results

    # Print summary
    print("\n" + "=" * 60)
    print("TRANSFER EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"{'Task':<25} {'Acc(paper)':>10} {'Acc(last)':>10} {'Acc(first)':>11} {'Acc(legacy)':>12} {'Format':>8}")
    print("-" * 80)
    for task_name in tasks:
        m = all_metrics[task_name]
        r = m["accuracy_by_rule"]
        print(f"{task_name:<25} {r['last_fallback']:>9.2%} {r['last']:>9.2%} {r['first']:>10.2%} {r['legacy']:>11.2%} "
              f"{m['valid_format_rate']:>7.2%}")
    print("=" * 60)

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = Path(model_path).name

    run_info = {
        "model": model_path,
        "timestamp": timestamp,
        "eval_seed": args.seed,
        "num_samples": args.num_samples,
        "backend": args.backend,
        "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens,
        "sampling": generator.sampling,
        "system_prompt": system_prompt,
    }

    # Save compact metrics (for transfer matrix construction)
    metrics_file = output_dir / f"metrics_{model_name}_{timestamp}.json"
    with open(metrics_file, "w") as f:
        json.dump({**run_info, "tasks": all_metrics}, f, indent=2)
    print(f"\nMetrics saved to {metrics_file}")

    # Save detailed per-example results with full responses (larger file)
    details_file = output_dir / f"details_{model_name}_{timestamp}.json"
    with open(details_file, "w") as f:
        json.dump({**run_info, "tasks": all_results}, f, indent=2)
    print(f"Details saved to {details_file}")


if __name__ == "__main__":
    main()
