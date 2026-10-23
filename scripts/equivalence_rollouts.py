"""
Phase B2 equivalence check: are HF-generate and vLLM rollouts drawn from the same
distribution? Same model, same prompts (training set, seed 42), same sampling
settings as GRPO (T=1.0, top_p=1, top_k=0, stop at </answer>, eos=<|im_end|>).

Compares, per backend: correctness reward, length bonus, completion tokens,
</answer> hit rate and truncation rate, with bootstrap CIs on the differences and
a two-sample KS test on completion length.

    python scripts/equivalence_rollouts.py --model /data/rlvr/smoke_sft_0.5b \
        --config configs/replication/smoke_grpo_qwen25_0.5b.yaml --prompts 64 --out results.json
"""

import argparse
import json
import os
import random
import sys

import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environments.reasoning_gym_env import ReasoningGymEnvironment  # noqa: E402


def hf_rollouts(model_path, prompts, n, max_new, stop, seed, batch_size):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_path, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto").eval()
    torch.manual_seed(seed)
    texts, ntoks = [], []
    flat = [p for p in prompts for _ in range(n)]
    for i in range(0, len(flat), batch_size):
        enc = tok(flat[i:i + batch_size], return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=max_new, do_sample=True, temperature=1.0, top_p=1.0,
                                 top_k=0, stop_strings=stop, tokenizer=tok, eos_token_id=tok.eos_token_id,
                                 pad_token_id=tok.pad_token_id)
        for row in out[:, enc["input_ids"].shape[1]:].tolist():
            # same post-processing as training: cut at EOS, strip trailing pad
            if tok.eos_token_id in row:
                row = row[:row.index(tok.eos_token_id) + 1]
            while row and row[-1] == tok.pad_token_id:
                row.pop()
            ntoks.append(len(row))
            texts.append(tok.decode(row, skip_special_tokens=True))
    del model
    torch.cuda.empty_cache()
    return texts, ntoks


def vllm_rollouts(model_path, prompts, n, max_new, stop, seed):
    from vllm import LLM, SamplingParams
    llm = LLM(model=model_path, dtype="bfloat16", seed=seed, gpu_memory_utilization=0.5)
    params = SamplingParams(n=n, temperature=1.0, top_p=1.0, top_k=0, max_tokens=max_new,
                            stop=stop, include_stop_str_in_output=True, seed=seed)
    texts, ntoks = [], []
    for o in llm.generate(prompts, params):
        for c in o.outputs:
            texts.append(c.text)
            ntoks.append(len(c.token_ids))
    return texts, ntoks


def summarize(env, texts, ntoks, truths, max_new):
    corr = env.correctness_reward(texts, truths)
    length = env.length_penalty_reward(texts, truths)
    return {
        "n": len(texts),
        "correct": corr,
        "length_bonus": length,
        "tokens": ntoks,
        "stop_hit": [float("</answer>" in t) for t in texts],
        "truncated": [float(k >= max_new) for k in ntoks],
    }


def mean(xs):
    return sum(xs) / len(xs)


def bootstrap_diff(a, b, iters=2000, seed=0):
    rng = random.Random(seed)
    diffs = sorted(mean([rng.choice(a) for _ in a]) - mean([rng.choice(b) for _ in b]) for _ in range(iters))
    return mean(a) - mean(b), diffs[int(0.025 * iters)], diffs[int(0.975 * iters)]


def ks_stat(a, b):
    a, b = sorted(a), sorted(b)
    pts = sorted(set(a + b))
    import bisect
    return max(abs(bisect.bisect_right(a, x) / len(a) - bisect.bisect_right(b, x) / len(b)) for x in pts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--config", required=True, help="GRPO config (environment, system prompt, generation)")
    ap.add_argument("--prompts", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    config = yaml.safe_load(open(args.config))
    env = ReasoningGymEnvironment(config)
    ds = env.get_dataset(config).select(range(args.prompts))
    n = config["generation"]["num_generations"]
    max_new = config["generation"]["max_completion_length"]
    stop = config["generation"].get("stop_strings", ["</answer>"])

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    system = config.get("system_prompt", "").strip()
    prompts = [tok.apply_chat_template([{"role": "system", "content": system}] + row["prompt"],
                                       tokenize=False, add_generation_prompt=True) for row in ds]
    truths = [t for t in ds["ground_truth"] for _ in range(n)]

    res = {}
    for name, fn in (("hf", lambda: hf_rollouts(args.model, prompts, n, max_new, stop, args.seed, args.batch_size)),
                     ("vllm", lambda: vllm_rollouts(args.model, prompts, n, max_new, stop, args.seed))):
        texts, ntoks = fn()
        res[name] = summarize(env, texts, ntoks, truths, max_new)
        res[name]["examples"] = texts[:8]

    report = {"model": args.model, "prompts": args.prompts, "num_generations": n}
    for key in ("correct", "length_bonus", "tokens", "stop_hit", "truncated"):
        d, lo, hi = bootstrap_diff(res["hf"][key], res["vllm"][key])
        report[key] = {"hf": mean(res["hf"][key]), "vllm": mean(res["vllm"][key]),
                       "diff": d, "ci95": [lo, hi], "ci_contains_0": lo <= 0 <= hi}
    report["tokens"]["ks_stat"] = ks_stat(res["hf"]["tokens"], res["vllm"]["tokens"])
    for k, v in report.items():
        print(k, v)
    with open(args.out, "w") as f:
        json.dump({"report": report, "raw": res}, f, indent=2)


if __name__ == "__main__":
    main()
