"""
Generate SFT Warmup Data for Base Models

Creates a small JSONL dataset (~500 examples) with <think>...</think> and
<answer>...</answer> structured outputs to teach a pretrained-only model
(e.g. Qwen3-4B-Base) the expected output format before GRPO training.

Uses reasoning-gym tasks (countdown, basic_arithmetic) to programmatically
construct training examples with step-by-step reasoning traces.

Usage:
    python data_gen/generate_sft_warmup.py [--output data_gen/data/sft_warmup.jsonl] [--num_samples 500]
"""

import json
import argparse
import random
from pathlib import Path

try:
    import reasoning_gym
except ImportError:
    raise ImportError("reasoning-gym required. Install with: pip install reasoning-gym")


SYSTEM_PROMPT = (
    "You are a helpful assistant. Think step by step inside "
    "<think>...</think> tags, then provide your final answer "
    "inside <answer>...</answer> tags."
)


def generate_countdown_examples(num_samples: int, seed: int = 42) -> list:
    """Generate countdown examples with synthetic reasoning traces."""
    dataset = reasoning_gym.create_dataset(
        "countdown",
        size=num_samples,
        seed=seed,
        min_numbers=3,
        max_numbers=4,
        min_target=10,
        max_target=100,
    )

    examples = []
    for item in dataset:
        question = item["question"]
        answer = str(item["answer"])
        metadata = item.get("metadata", {})
        target = metadata.get("target", "")
        numbers = metadata.get("numbers", [])

        # Build a synthetic reasoning trace
        think_lines = []
        think_lines.append(f"I need to find an expression using {numbers} that equals {target}.")
        think_lines.append(f"Let me try different combinations of these numbers with +, -, *, /.")
        think_lines.append(f"The answer is: {answer}")
        think_lines.append(f"Let me verify: {answer} = {target}. Correct!")

        think_content = "\n".join(think_lines)
        assistant_response = f"<think>\n{think_content}\n</think>\n<answer>{answer}</answer>"

        examples.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
                {"role": "assistant", "content": assistant_response},
            ]
        })

    return examples


def generate_arithmetic_examples(num_samples: int, seed: int = 42) -> list:
    """Generate basic arithmetic examples with reasoning traces."""
    dataset = reasoning_gym.create_dataset(
        "basic_arithmetic",
        size=num_samples,
        seed=seed,
    )

    examples = []
    for item in dataset:
        question = item["question"]
        answer = str(item["answer"])

        think_content = (
            f"I need to solve: {question}\n"
            f"Computing step by step...\n"
            f"The result is {answer}."
        )
        assistant_response = f"<think>\n{think_content}\n</think>\n<answer>{answer}</answer>"

        examples.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
                {"role": "assistant", "content": assistant_response},
            ]
        })

    return examples


def generate_simple_qa_examples(num_samples: int, seed: int = 42) -> list:
    """
    Generate trivially simple Q&A examples to teach the format.
    No external dependency needed — just arithmetic and logic.
    """
    rng = random.Random(seed)
    examples = []

    for _ in range(num_samples):
        a = rng.randint(1, 100)
        b = rng.randint(1, 100)
        op = rng.choice(["+", "-", "*"])

        if op == "+":
            result = a + b
        elif op == "-":
            result = a - b
        else:
            result = a * b

        question = f"What is {a} {op} {b}?"
        think_content = f"I need to compute {a} {op} {b}.\n{a} {op} {b} = {result}"
        assistant_response = f"<think>\n{think_content}\n</think>\n<answer>{result}</answer>"

        examples.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
                {"role": "assistant", "content": assistant_response},
            ]
        })

    return examples


def main():
    parser = argparse.ArgumentParser(description="Generate SFT warmup data")
    parser.add_argument(
        "--output",
        type=str,
        default="data_gen/data/sft_warmup.jsonl",
        help="Output JSONL path",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=500,
        help="Total number of examples to generate",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Split across sources: 40% countdown, 30% arithmetic, 30% simple QA
    n_countdown = int(args.num_samples * 0.4)
    n_arithmetic = int(args.num_samples * 0.3)
    n_simple = args.num_samples - n_countdown - n_arithmetic

    print(f"Generating {args.num_samples} warmup examples...")
    print(f"  Countdown:  {n_countdown}")
    print(f"  Arithmetic: {n_arithmetic}")
    print(f"  Simple QA:  {n_simple}")

    all_examples = []
    all_examples.extend(generate_countdown_examples(n_countdown, seed=args.seed))
    all_examples.extend(generate_arithmetic_examples(n_arithmetic, seed=args.seed + 1))
    all_examples.extend(generate_simple_qa_examples(n_simple, seed=args.seed + 2))

    # Shuffle
    rng = random.Random(args.seed)
    rng.shuffle(all_examples)

    # Write
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for ex in all_examples:
            f.write(json.dumps(ex) + "\n")

    print(f"\n✓ Wrote {len(all_examples)} examples to {output_path}")


if __name__ == "__main__":
    main()
