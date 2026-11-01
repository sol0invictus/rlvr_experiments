"""
Phase C generation: sample the model on atoms and full tasks, score every sample, save everything.

Two modes:
    atoms  the gate run: every unique atom question in a build (k_atom samples each) and every
           instance's full task (k_full samples), at the gate temperature (T = 1.0)
    full   the pilot: full tasks only, instances drawn straight from reasoning-gym, so tasks without
           a solver can be screened too

Both modes can add one T = 0.1 sample per instance (--t01), the paper's accuracy-eval convention.

Usage (from the repo root):
    python -m experiments.atom_decomposition.eval_atoms atoms --model /data/rlvr/sft_qwen3_4b \
        --build_dir experiments/atom_decomposition/results/atoms_seed2027 \
        --out_dir experiments/atom_decomposition/results/eval_seed2027 --t01
    python -m experiments.atom_decomposition.eval_atoms full --model /data/rlvr/sft_qwen3_4b \
        --tasks basic_arithmetic,maze --n 200 --seed 2026 --k 4 --t01 \
        --out_dir experiments/atom_decomposition/results/pilot_seed2026

Outputs in --out_dir (JSONL, appended chunk by chunk; a rerun skips what is already there):
    atom_results.jsonl  {key, task, check, answer, correct, correct_strict, has_answer_tag, n_tokens, responses}
    full_results.jsonl  {task, idx, temperature, answer, correct, correct_strict, rg_score,
                         has_answer_tag, n_tokens, responses}   (one row per instance and temperature)
    manifest.json       arguments, sampling settings, counts, wall time

Scoring: extraction rule last_fallback (the paper's; correct_strict uses `last` only). Atoms use the
checkers in common.py. Full tasks from the paper's 10 use evaluate_replication.verify_answer (the paper's
rules); other tasks count as correct when reasoning-gym's own scorer gives 1.0. rg_score is always logged.
"""

import argparse
import json
import os
import random
import re
import sys
import time
from collections import defaultdict

import reasoning_gym as rg

from .common import CHECKERS
from .tasks import TASKS  # noqa: F401  (registers the paren_tree checker)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from evaluate_replication import (DEFAULT_SYSTEM_PROMPT, build_prompt,  # noqa: E402
                                  extract_answer, model_sampling_defaults, verify_answer)

ANSWER_TAG = re.compile(r"<answer>.*?</answer>", re.DOTALL)

# The paper's 10 tasks keep the paper's verification rule; anything else uses rg's scorer.
PAPER_TASKS = {"basic_arithmetic", "countdown", "syllogism", "knights_knaves", "maze", "mini_sudoku",
               "shortest_path", "leg_counting", "caesar_cipher", "cryptarithm"}


# ----------------------------------------------------------------------
# Backends: generate(prompts, n, max_new_tokens, temperature) -> per prompt, n (text, n_tokens) pairs
# ----------------------------------------------------------------------

class VLLMBackend:
    """One request per prompt with n samples. (Repeating a prompt n times with a fixed per-request
    seed, as evaluate_replication.VLLMGenerator would, returns n identical samples.)"""

    def __init__(self, model, seed, gpu_memory_utilization, max_model_len):
        from vllm import LLM
        self.llm = LLM(model=model, dtype="bfloat16", seed=seed, gpu_memory_utilization=gpu_memory_utilization,
                       max_model_len=max_model_len, enable_prefix_caching=True)
        self.tokenizer = self.llm.get_tokenizer()
        self.sampling = model_sampling_defaults(model)
        self.seed = seed

    def generate(self, prompts, n, max_new_tokens, temperature):
        from vllm import SamplingParams
        s = self.sampling
        params = SamplingParams(n=n, temperature=temperature, top_p=s["top_p"], top_k=s["top_k"],
                                min_p=s["min_p"], repetition_penalty=s["repetition_penalty"],
                                max_tokens=max_new_tokens, seed=self.seed)
        outs = self.llm.generate(prompts, params)
        return [[(c.text, len(c.token_ids)) for c in o.outputs] for o in outs]


class HFBackend:
    """Slow; for local smoke tests."""

    def __init__(self, model, seed, batch_size):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.model = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch.bfloat16, device_map="auto").eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model, padding_side="left")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.sampling = model_sampling_defaults(model)
        self.batch_size = batch_size
        torch.manual_seed(seed)

    def generate(self, prompts, n, max_new_tokens, temperature):
        out = []
        s = self.sampling
        for i in range(0, len(prompts), self.batch_size):
            batch = prompts[i:i + self.batch_size]
            enc = self.tokenizer(batch, return_tensors="pt", padding=True).to(self.model.device)
            with self.torch.no_grad():
                gen = self.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=True,
                                          temperature=temperature, top_p=s["top_p"], top_k=s["top_k"],
                                          repetition_penalty=s["repetition_penalty"], num_return_sequences=n,
                                          pad_token_id=self.tokenizer.pad_token_id)
            new = gen[:, enc["input_ids"].shape[1]:]
            for j in range(len(batch)):
                rows = new[j * n:(j + 1) * n]
                out.append([(self.tokenizer.decode(r, skip_special_tokens=True),
                             int((r != self.tokenizer.pad_token_id).sum())) for r in rows])
        return out


class FakeBackend:
    """No model: answers the gold with probability p_correct, else a wrong answer. For plumbing tests."""

    def __init__(self, gold_by_prompt, p_correct=0.6, seed=0):
        self.gold, self.p, self.rng = gold_by_prompt, p_correct, random.Random(seed)
        self.tokenizer = None

    def generate(self, prompts, n, max_new_tokens, temperature):
        out = []
        for p in prompts:
            gold = self.gold[p]
            samples = []
            for _ in range(n):
                ans = gold if self.rng.random() < self.p else "wrong"
                text = f"<think>fake</think>\n<answer>{ans}</answer>"
                samples.append((text, len(text.split())))
            out.append(samples)
        return out


def make_prompt(backend, question):
    if backend.tokenizer is None:
        return f"[system] {DEFAULT_SYSTEM_PROMPT}\n[user] {question}"
    return build_prompt(backend.tokenizer, question, DEFAULT_SYSTEM_PROMPT)


# ----------------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------------

def score_atom(check, gold, text):
    return (int(CHECKERS[check](extract_answer(text, "last_fallback"), gold)),
            int(CHECKERS[check](extract_answer(text, "last"), gold)))


class FullScorer:
    def __init__(self):
        self.datasets = {}

    def __call__(self, task, entry, text):
        ext = extract_answer(text, "last_fallback")
        strict = extract_answer(text, "last")
        ds = self.datasets.setdefault(task, rg.create_dataset(task, size=1, seed=0))
        rg_score = float(ds.score_answer(answer=ext, entry=entry)) if ext is not None else 0.0
        if task in PAPER_TASKS:
            correct = verify_answer(task, ext, entry["answer"], entry["metadata"])
            correct_strict = verify_answer(task, strict, entry["answer"], entry["metadata"])
        else:
            correct = rg_score >= 1.0
            correct_strict = strict is not None and float(ds.score_answer(answer=strict, entry=entry)) >= 1.0
        return int(correct), int(correct_strict), rg_score


# ----------------------------------------------------------------------
# Work units, chunking and resume
# ----------------------------------------------------------------------

def _done_keys(path, keyfn):
    if not os.path.exists(path):
        return set()
    return {keyfn(json.loads(l)) for l in open(path)}


def run_units(backend, units, n, max_new_tokens, temperature, out_path, keyfn, finish, chunk):
    """units: list of dicts with 'question'. finish(unit, samples) -> output row."""
    done = _done_keys(out_path, keyfn)
    todo = [u for u in units if keyfn(u) not in done]
    print(f"{os.path.basename(out_path)} T={temperature}: {len(todo)} to do, {len(units) - len(todo)} done",
          flush=True)
    with open(out_path, "a") as f:
        for i in range(0, len(todo), chunk):
            part = todo[i:i + chunk]
            t0 = time.time()
            samples = backend.generate([make_prompt(backend, u["question"]) for u in part], n,
                                       max_new_tokens, temperature)
            for u, sm in zip(part, samples):
                f.write(json.dumps(finish(u, sm)) + "\n")
            f.flush()
            print(f"  {min(i + chunk, len(todo))}/{len(todo)} in {time.time() - t0:.0f}s", flush=True)


def full_units(rows):
    return [{"task": r["task"], "idx": r["idx"], "question": r["question"], "answer": r["answer"],
             "metadata": r["metadata"]} for r in rows]


def run_full(backend, rows, temperatures_n, max_new_tokens, out_dir, chunk):
    scorer = FullScorer()
    path = os.path.join(out_dir, "full_results.jsonl")
    for temperature, n in temperatures_n:
        def finish(u, sm, temperature=temperature):
            scored = [scorer(u["task"], u, t) for t, _ in sm]
            return {"task": u["task"], "idx": u["idx"], "temperature": temperature, "answer": u["answer"],
                    "correct": [c for c, _, _ in scored], "correct_strict": [c for _, c, _ in scored],
                    "rg_score": [r for _, _, r in scored],
                    "has_answer_tag": [int(bool(ANSWER_TAG.search(t))) for t, _ in sm],
                    "n_tokens": [k for _, k in sm], "responses": [t for t, _ in sm]}
        run_units(backend, [dict(u, temperature=temperature) for u in full_units(rows)], n, max_new_tokens,
                  temperature, path, lambda r: (r["task"], r["idx"], r["temperature"]), finish, chunk)


def run_atoms(backend, build_dir, k_atom, max_new_tokens, temperature, out_dir, chunk, limit=None):
    seen, units = set(), []
    for l in open(os.path.join(build_dir, "atoms.jsonl")):
        a = json.loads(l)
        if limit and a["idx"] >= limit:
            continue
        if a["key"] not in seen:
            seen.add(a["key"])
            units.append({"key": a["key"], "task": a["task"], "question": a["question"],
                          "answer": a["answer"], "check": a["check"]})

    def finish(u, sm):
        scored = [score_atom(u["check"], u["answer"], t) for t, _ in sm]
        return {"key": u["key"], "task": u["task"], "check": u["check"], "answer": u["answer"],
                "correct": [c for c, _ in scored], "correct_strict": [c for _, c in scored],
                "has_answer_tag": [int(bool(ANSWER_TAG.search(t))) for t, _ in sm],
                "n_tokens": [k for _, k in sm], "responses": [t for t, _ in sm]}
    run_units(backend, units, k_atom, max_new_tokens, temperature,
              os.path.join(out_dir, "atom_results.jsonl"), lambda r: r["key"], finish, chunk)
    return len(units)


def summarize(out_dir):
    """Per-task accuracy and token counts, for the manifest and the log."""
    summ = defaultdict(lambda: defaultdict(list))
    p = os.path.join(out_dir, "full_results.jsonl")
    if os.path.exists(p):
        for l in open(p):
            r = json.loads(l)
            d = summ[f"full/{r['task']}/T{r['temperature']}"]
            d["correct"] += r["correct"]
            d["tags"] += r["has_answer_tag"]
            d["tokens"] += r["n_tokens"]
    p = os.path.join(out_dir, "atom_results.jsonl")
    if os.path.exists(p):
        for l in open(p):
            r = json.loads(l)
            d = summ[f"atoms/{r['task']}"]
            d["correct"] += r["correct"]
            d["tags"] += r["has_answer_tag"]
            d["tokens"] += r["n_tokens"]
    return {k: {"samples": len(v["correct"]), "accuracy": sum(v["correct"]) / len(v["correct"]),
                "format_rate": sum(v["tags"]) / len(v["tags"]),
                "mean_tokens": sum(v["tokens"]) / len(v["tokens"])} for k, v in sorted(summ.items())}


# ----------------------------------------------------------------------

def make_backend(a, gold_by_question=None):
    if a.backend == "vllm":
        return VLLMBackend(a.model, a.seed_gen, a.gpu_memory_utilization, a.max_model_len)
    if a.backend == "hf":
        return HFBackend(a.model, a.seed_gen, a.batch_size)
    gold = {f"[system] {DEFAULT_SYSTEM_PROMPT}\n[user] {q}": g for q, g in gold_by_question.items()}
    return FakeBackend(gold, a.fake_p_correct, a.seed_gen)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="mode", required=True)
    for name in ("atoms", "full"):
        q = sub.add_parser(name)
        q.add_argument("--model", default=None)
        q.add_argument("--backend", choices=("vllm", "hf", "fake"), default="vllm")
        q.add_argument("--out_dir", required=True)
        q.add_argument("--temperature", type=float, default=1.0, help="gate-sample temperature")
        q.add_argument("--t01", action="store_true", help="also 1 full-task sample per instance at T=0.1")
        q.add_argument("--max_new_tokens_full", type=int, default=2048)
        q.add_argument("--seed_gen", type=int, default=0)
        q.add_argument("--chunk", type=int, default=4096, help="prompts per generate call / per flush")
        q.add_argument("--limit", type=int, default=None, help="first N instances/atoms only (smoke tests)")
        q.add_argument("--gpu_memory_utilization", type=float, default=0.9)
        q.add_argument("--max_model_len", type=int, default=4096)
        q.add_argument("--batch_size", type=int, default=8, help="HF backend only")
        q.add_argument("--fake_p_correct", type=float, default=0.6)
    pa = sub.choices["atoms"]
    pa.add_argument("--build_dir", required=True)
    pa.add_argument("--k_atom", type=int, default=8)
    pa.add_argument("--k_full", type=int, default=8)
    pa.add_argument("--max_new_tokens_atom", type=int, default=1024)
    pf = sub.choices["full"]
    pf.add_argument("--tasks", required=True)
    pf.add_argument("--n", type=int, default=200)
    pf.add_argument("--seed", type=int, default=2026)
    pf.add_argument("--task_configs", default="{}")
    pf.add_argument("--k", type=int, default=4)
    a = p.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    t0 = time.time()

    if a.mode == "atoms":
        rows = [json.loads(l) for l in open(os.path.join(a.build_dir, "instances.jsonl"))]
        atom_rows = [json.loads(l) for l in open(os.path.join(a.build_dir, "atoms.jsonl"))]
    else:
        cfgs = json.loads(a.task_configs)
        rows = []
        for task in a.tasks.split(","):
            ds = rg.create_dataset(task, size=a.n, seed=a.seed, **cfgs.get(task, {}))
            rows += [{"task": task, "idx": i, **ds[i]} for i in range(a.n)]
        atom_rows = []
    if a.limit:   # first N instances per task
        rows = [r for r in rows if r["idx"] < a.limit]
    gold = {r["question"]: r["answer"] for r in rows}
    gold.update({x["question"]: x["answer"] for x in atom_rows})
    backend = make_backend(a, gold)

    n_atoms = 0
    if a.mode == "atoms":
        n_atoms = run_atoms(backend, a.build_dir, a.k_atom, a.max_new_tokens_atom, a.temperature, a.out_dir,
                            a.chunk, a.limit)
        full_plan = [(a.temperature, a.k_full)]
    else:
        full_plan = [(a.temperature, a.k)]
    if a.t01:
        full_plan.append((0.1, 1))
    run_full(backend, rows, full_plan, a.max_new_tokens_full, a.out_dir, a.chunk)

    summary = summarize(a.out_dir)
    manifest = {"args": vars(a), "system_prompt": DEFAULT_SYSTEM_PROMPT,
                "sampling_defaults": getattr(backend, "sampling", None), "n_instances": len(rows),
                "n_unique_atoms": n_atoms, "wall_s": round(time.time() - t0), "summary": summary}
    with open(os.path.join(a.out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
