"""
ValidationCallback — runs validation every N training steps.

Generates completions on a held-out validation set, scores them with the
environment's reward functions, and reports mean reward to stdout + an
optional JSONL file.  The metrics are also appended to the trainer's
log_history so they appear in TensorBoard / W&B runs.

Config block (inside the top-level YAML):

    validation:
      enabled: true
      eval_steps: 100        # validate every N steps (default 100)
      num_samples: 64        # how many val examples to score (default 64)
      temperature: 0.0       # generation temperature; 0 = greedy (default)
      log_dir: ...           # where to write JSONL (default <output_dir>/validation)
      val_seed: 9999         # seed for val-set generation (ReasoningGym only)
"""

import json
import os
import random
from typing import Any, Callable, Dict, List, Optional

import torch
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments


class ValidationCallback(TrainerCallback):
    """
    Runs greedy-decoding inference on a small validation set every
    `eval_steps` training steps and logs mean reward.

    Parameters
    ----------
    val_dataset : HF Dataset
        Validation examples in the same format as the training dataset
        (must have a 'prompt' column; any extra columns are forwarded to
        reward functions as keyword arguments).
    reward_funcs : list of callables
        Same reward functions as used during training.
    tokenizer :
        Tokenizer matching the model.
    eval_steps : int
        Run validation every this many global training steps.
    num_samples : int
        Number of examples sampled from val_dataset per eval.
    max_new_tokens : int
        Token budget for generation.
    log_dir : str or None
        If provided, write per-step JSONL logs here.
    temperature : float
        Sampling temperature; 0.0 = greedy decoding.
    """

    def __init__(
        self,
        val_dataset,
        reward_funcs: List[Callable],
        tokenizer,
        eval_steps: int = 100,
        num_samples: int = 64,
        max_new_tokens: int = 512,
        log_dir: Optional[str] = None,
        temperature: float = 0.0,
    ):
        self.val_dataset = val_dataset
        self.reward_funcs = reward_funcs
        self.tokenizer = tokenizer
        self.eval_steps = eval_steps
        self.num_samples = min(num_samples, len(val_dataset))
        self.max_new_tokens = max_new_tokens
        self.log_dir = log_dir
        self.temperature = temperature

    # ------------------------------------------------------------------
    # TrainerCallback hooks
    # ------------------------------------------------------------------

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if state.global_step == 0 or state.global_step % self.eval_steps != 0:
            return
        # Only run on the main process (rank 0) to avoid redundant work
        if args.process_index != 0:
            return

        model = kwargs.get("model")
        if model is None:
            return

        # Unwrap DDP / Accelerate wrapper to get the raw model
        if hasattr(model, "module"):
            unwrapped = model.module
        else:
            unwrapped = model

        self._run_validation(unwrapped, state)

    # ------------------------------------------------------------------
    # Core logic
    # ------------------------------------------------------------------

    def _run_validation(self, model, state: TrainerState):
        step = state.global_step

        # Reproducible sampling (varies by step so we see different examples)
        rng = random.Random(step)
        indices = list(range(len(self.val_dataset)))
        if len(indices) > self.num_samples:
            indices = rng.sample(indices, self.num_samples)
        samples = [self.val_dataset[i] for i in indices]

        # Generate completions (wrapped as [{"role": "assistant", "content": ...}])
        completions = self._generate(model, samples)

        # Build extra kwargs for reward functions (all columns except 'prompt')
        extra_kwargs: Dict[str, List[Any]] = {}
        if samples:
            for key in samples[0].keys():
                if key != "prompt":
                    extra_kwargs[key] = [s[key] for s in samples]

        prompts = [s["prompt"] for s in samples]

        # Score with each reward function
        rewards_by_func: Dict[str, List[float]] = {}
        for func in self.reward_funcs:
            name = getattr(func, "__name__", str(func))
            scores = func(completions, prompts=prompts, **extra_kwargs)
            rewards_by_func[name] = scores

        # Aggregate
        n = len(completions)
        func_names = list(rewards_by_func.keys())
        total_rewards = [
            sum(rewards_by_func[fn][i] for fn in func_names)
            for i in range(n)
        ]
        mean_reward = sum(total_rewards) / n if n > 0 else 0.0
        func_means = {
            fn: (sum(rewards_by_func[fn]) / len(rewards_by_func[fn]))
            for fn in func_names
            if rewards_by_func[fn]
        }

        # Print summary
        func_str = "  ".join(f"{k}={v:.4f}" for k, v in func_means.items())
        print(
            f"\n[Validation] step={step}  mean_reward={mean_reward:.4f}"
            f"  n={n}  {func_str}"
        )

        # Append to trainer log_history so TensorBoard/W&B picks it up
        val_metrics: Dict[str, Any] = {"step": step, "val/mean_reward": mean_reward}
        val_metrics.update({f"val/{fn}": v for fn, v in func_means.items()})
        state.log_history.append(val_metrics)

        # Optionally write per-example JSONL
        if self.log_dir:
            self._write_jsonl(step, samples, completions, rewards_by_func, total_rewards)

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def _generate(self, model, samples: list) -> list:
        """Return completions as [{"role": "assistant", "content": str}]."""
        model.eval()
        completions = []

        gen_kwargs: Dict[str, Any] = dict(
            max_new_tokens=self.max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        if self.temperature > 0.0:
            gen_kwargs.update(do_sample=True, temperature=self.temperature)
        else:
            gen_kwargs["do_sample"] = False

        with torch.no_grad():
            for sample in samples:
                prompt = sample["prompt"]
                text = self.tokenizer.apply_chat_template(
                    prompt,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                inputs = self.tokenizer(text, return_tensors="pt").to(model.device)
                output_ids = model.generate(**inputs, **gen_kwargs)
                new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
                decoded = self.tokenizer.decode(new_ids, skip_special_tokens=True)
                completions.append([{"role": "assistant", "content": decoded}])

        model.train()
        return completions

    # ------------------------------------------------------------------
    # JSONL logging
    # ------------------------------------------------------------------

    def _write_jsonl(
        self,
        step: int,
        samples: list,
        completions: list,
        rewards_by_func: Dict[str, List[float]],
        total_rewards: List[float],
    ):
        os.makedirs(self.log_dir, exist_ok=True)
        out_path = os.path.join(self.log_dir, f"val_step_{step:07d}.jsonl")
        with open(out_path, "w") as fh:
            for i, (sample, comp) in enumerate(zip(samples, completions)):
                rec = {
                    "step": step,
                    "prompt": sample["prompt"],
                    "completion": comp[0]["content"] if comp else "",
                    "rewards": {
                        fn: scores[i]
                        for fn, scores in rewards_by_func.items()
                        if i < len(scores)
                    },
                    "total_reward": total_rewards[i],
                }
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[Validation] saved {len(samples)} records → {out_path}")
