"""Evaluate a checkpoint on GSM8K test set with greedy decoding."""
import json, os, sys, argparse, torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_from_disk, load_dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from environments.gsm8k import GSM8KEnvironment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", help="Path to model checkpoint")
    parser.add_argument("--num_samples", type=int, default=200)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--data_path", type=str, default="/scratch/sunguha/cer_project/data/gsm8k_test")
    args = parser.parse_args()

    print(f"Loading model from {args.model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
    ).cuda().eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if os.path.exists(args.data_path):
        dataset = load_from_disk(args.data_path)
    else:
        dataset = load_dataset("openai/gsm8k", "main", split="test")

    if args.num_samples < len(dataset):
        dataset = dataset.select(range(args.num_samples))

    system_prompt = "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>"

    env = GSM8KEnvironment({"data": {"answer_column": "answer"}})
    reward_funcs = env.get_reward_functions()

    correct = 0
    total = 0
    format_ok = 0

    print(f"Evaluating {len(dataset)} samples with greedy decoding...")
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

            completions = [[{"role": "assistant", "content": completion}]]
            answers = [sample["answer"]]

            for func in reward_funcs:
                name = getattr(func, "__name__", str(func))
                scores = func(completions, answer=answers)
                if "accuracy" in name:
                    if scores[0] > 0:
                        correct += 1
                if "format" in name:
                    if scores[0] > 0:
                        format_ok += 1
            total += 1

            if (i + 1) % 20 == 0:
                print(f"  [{i+1}/{len(dataset)}] acc={correct/total:.1%} format={format_ok/total:.1%}")

    print(f"\n{'='*50}")
    print(f"FINAL: {correct}/{total} = {correct/total:.1%} accuracy")
    print(f"FORMAT: {format_ok}/{total} = {format_ok/total:.1%}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
