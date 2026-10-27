"""
Atomic Readiness Score (ARS) probe.

Zero-shot evaluation of a model on atomic sub-skills for basic_arithmetic.
No training — just measures what the base model already knows.

Usage:
    python experiments/rlvr_geometry_paper/scripts/eval/probe_atomics.py --model_path /scratch/sunguha/sft_qwen_4.0b_instruct --num_samples 50
"""

import argparse
import json
import random
import re
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from environments.atomic_templates import TASK_ATOMS, ALL_ATOMS

try:
    import reasoning_gym
except ImportError:
    print("ERROR: pip install reasoning-gym")
    sys.exit(1)


def generate_atomic_samples(atom_fn, num_samples=50, seed=9999, task='basic_arithmetic', task_params=None):
    """Generate atomic Q&A pairs from reasoning_gym samples."""
    ds = reasoning_gym.create_dataset(task, size=num_samples * 2, seed=seed, **(task_params or {}))
    rng = random.Random(seed)
    samples = []
    for item in ds:
        try:
            q, a = atom_fn(item, rng)
            samples.append({'question': q, 'answer': a, 'source': item['metadata'].get('expression', item['metadata'].get('source_dataset', ''))})
        except Exception as e:
            continue
        if len(samples) >= num_samples:
            break
    return samples


def evaluate_atom(model, tokenizer, samples, max_new_tokens=128):
    """Evaluate model on atomic Q&A pairs. Returns accuracy."""
    correct = 0
    total = len(samples)
    results = []

    # Clear stop_strings from generation_config to avoid tokenizer requirement
    if hasattr(model, 'generation_config') and hasattr(model.generation_config, 'stop_strings'):
        model.generation_config.stop_strings = None

    for sample in samples:
        prompt = f"Answer this question concisely.\n\nQuestion: {sample['question']}\n\nProvide your answer inside <answer></answer> tags."
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
            )

        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        extracted = _extract_answer(response)
        is_correct = _check_answer(extracted, sample['answer'])
        correct += int(is_correct)

        results.append({
            'question': sample['question'],
            'expected': sample['answer'],
            'extracted': extracted,
            'correct': is_correct,
            'full_response': response[:200],
        })

    accuracy = correct / total if total > 0 else 0
    return accuracy, results


def _extract_answer(text):
    match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip().split('\n')[-1].strip()


def _check_answer(extracted, expected):
    if extracted is None:
        return False
    ext = extracted.strip().lower().rstrip('.')
    exp = expected.strip().lower().rstrip('.')
    if ext == exp:
        return True
    try:
        if abs(float(ext.replace(',', '')) - float(exp.replace(',', ''))) < 1e-6:
            return True
    except (ValueError, TypeError):
        pass
    ext_words = set(ext.replace(',', ' ').replace(';', ' ').split())
    exp_words = set(exp.replace(',', ' ').replace(';', ' ').split())
    if ext_words and exp_words and ext_words == exp_words:
        return True
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='/scratch/sunguha/sft_qwen_4.0b_instruct')
    parser.add_argument('--task', type=str, default='basic_arithmetic',
                        choices=list(TASK_ATOMS.keys()),
                        help='Which task to probe atoms for')
    parser.add_argument('--num_samples', type=int, default=50)
    parser.add_argument('--seed', type=int, default=9999)
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()

    task_atom_registry = TASK_ATOMS[args.task]

    # Task-specific params for dataset generation
    task_params_map = {
        'basic_arithmetic': {},
        'countdown': {'min_numbers': 4, 'max_numbers': 5, 'min_target': 10, 'max_target': 200},
        'shortest_path': {},
    }
    task_params = task_params_map.get(args.task, {})

    print(f"Loading model: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    print(f"\n{'='*60}")
    print(f"ATOMIC READINESS SCORE (ARS) PROBE — {args.task}")
    print(f"Model: {args.model_path}")
    print(f"Samples per atom: {args.num_samples}")
    print(f"{'='*60}\n")

    all_results = {}
    layer_scores = {'L1': [], 'L2': []}

    for layer_name, atoms in task_atom_registry.items():
        print(f"\n--- {layer_name} Atoms ---")
        for atom_name, atom_fn in atoms.items():
            samples = generate_atomic_samples(atom_fn, num_samples=args.num_samples, seed=args.seed,
                                              task=args.task, task_params=task_params)
            accuracy, results = evaluate_atom(model, tokenizer, samples)
            full_name = f"{layer_name}_{atom_name}"
            all_results[full_name] = {
                'accuracy': accuracy,
                'num_samples': len(samples),
                'results': results[:5],  # save first 5 for inspection
            }
            layer_scores[layer_name].append(accuracy)
            print(f"  {atom_name:30s} {accuracy*100:5.1f}% ({sum(1 for r in results if r['correct'])}/{len(results)})")

            # Show 2 wrong examples
            wrong = [r for r in results if not r['correct']][:2]
            for w in wrong:
                print(f"    WRONG: Q={w['question'][:80]}")
                print(f"           Expected={w['expected']}, Got={w['extracted'][:50]}")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    l1_avg = sum(layer_scores['L1']) / len(layer_scores['L1']) if layer_scores['L1'] else 0
    l2_avg = sum(layer_scores['L2']) / len(layer_scores['L2']) if layer_scores['L2'] else 0
    overall = (l1_avg + l2_avg) / 2
    print(f"  L1 (Grounding) avg:       {l1_avg*100:.1f}%")
    print(f"  L2 (Local Reasoning) avg:  {l2_avg*100:.1f}%")
    print(f"  ARS (overall):             {overall*100:.1f}%")
    print()

    if args.output:
        with open(args.output, 'w') as f:
            json.dump({
                'model': args.model_path,
                'num_samples': args.num_samples,
                'seed': args.seed,
                'per_atom': {k: {'accuracy': v['accuracy']} for k, v in all_results.items()},
                'L1_avg': l1_avg,
                'L2_avg': l2_avg,
                'ARS': overall,
            }, f, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == '__main__':
    main()
