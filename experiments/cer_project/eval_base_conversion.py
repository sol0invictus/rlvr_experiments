"""Evaluate a checkpoint on base_conversion task with a fresh seed (greedy decoding)."""
import json, os, sys, re, argparse, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

import reasoning_gym


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", help="Path to model checkpoint")
    parser.add_argument("--num_samples", type=int, default=200)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=9999)
    args = parser.parse_args()

    print(f"Loading model from {args.model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
    ).cuda().eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Generating {args.num_samples} base_conversion problems (seed={args.seed})...")
    dataset = reasoning_gym.create_dataset(
        "base_conversion", size=args.num_samples, seed=args.seed
    )

    system_prompt = "You are a helpful assistant. You must always think step-by-step. Put your thinking process inside <think> and </think> tags, and put your final answer inside <answer> and </answer> tags."

    correct = 0
    total = 0
    format_ok = 0

    print(f"Evaluating {args.num_samples} samples with greedy decoding...")
    with torch.no_grad():
        for i, sample in enumerate(dataset):
            prompt = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": sample["question"]},
            ]
            text = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(text, return_tensors="pt").to("cuda")
            output_ids = model.generate(
                **inputs, max_new_tokens=args.max_new_tokens, do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
            new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
            completion = tokenizer.decode(new_ids, skip_special_tokens=True)

            has_think = '<think>' in completion and '</think>' in completion
            has_answer = '<answer>' in completion and '</answer>' in completion
            if has_think and has_answer:
                format_ok += 1

            answer_match = re.search(r'<answer>(.*?)</answer>', completion, re.DOTALL)
            if answer_match:
                extracted = answer_match.group(1).strip().lower()
                truth = str(sample['answer']).strip().lower()
                if extracted == truth:
                    correct += 1

            total += 1
            if (i + 1) % 20 == 0:
                print(f"  [{i+1}/{args.num_samples}] acc={correct/total:.1%} format={format_ok/total:.1%}")

    print(f"\n{'='*50}")
    print(f"FINAL: {correct}/{total} = {correct/total:.1%} accuracy")
    print(f"FORMAT: {format_ok}/{total} = {format_ok/total:.1%}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
