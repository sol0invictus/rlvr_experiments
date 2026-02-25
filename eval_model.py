"""Quick inference test for a trained GRPO checkpoint."""
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = sys.argv[1] if len(sys.argv) > 1 else "results/hp_sweep/A1_grpo_baseline"

print(f"Loading model from {model_path}...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto")

system_prompt = (
    "You are a helpful assistant. Think step by step inside <think>...</think> tags, "
    "then provide your final answer inside <answer>...</answer> tags."
)

test_prompts = [
    "Using the numbers [3, 7, 25, 50], create an expression that equals 176.",
    "Using the numbers [2, 5, 10, 100], create an expression that equals 45.",
    "Using the numbers [1, 4, 6, 8], create an expression that equals 42.",
]

for i, prompt in enumerate(test_prompts):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt").to(model.device)
    output = model.generate(
        input_ids, max_new_tokens=512, temperature=0.7, do_sample=True,
        tokenizer=tokenizer, stop_strings=["</answer>"],
    )
    response = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)
    print(f"\n{'='*60}")
    print(f"Prompt {i+1}: {prompt}")
    print(f"{'='*60}")
    print(response)
