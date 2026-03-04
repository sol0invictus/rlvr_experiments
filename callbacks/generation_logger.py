"""
GenerationLoggingCallback — logs GRPO generations to console and disk.

Works by wrapping each reward function so that every time completions are
scored the texts and their per-function rewards are captured.  The callback
then flushes the captured batch to a JSONL file and (optionally) prints a
few examples to stdout.

Config block expected under `generation_logging` in the YAML config:

    generation_logging:
      enabled: true              # default false
      log_dir: runs/generations  # default: <output_dir>/generations
      every_n_steps: 10          # log every N training steps (default 10)
      print_n_examples: 2        # examples to print to stdout per flush (default 2)
"""

import json
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments


@dataclass
class _Batch:
    """One batch of captured generations."""
    step: int
    prompts: List[Any]
    completions: List[str]
    rewards_by_func: Dict[str, List[float]] = field(default_factory=dict)


class GenerationLoggingCallback(TrainerCallback):
    """
    Captures model generations during GRPO training and saves them to disk.

    Usage
    -----
    Instantiate **before** creating GRPOTrainer, then:
      1. Call `callback.wrap_reward_funcs(reward_funcs)` to get instrumented
         versions of the reward functions to pass to GRPOTrainer.
      2. Pass the callback instance in the `callbacks` list of GRPOTrainer.

    The callback writes one JSONL file per flush to `log_dir`.  Each line is a
    JSON object with keys: step, prompt, completion, rewards (dict).
    """

    def __init__(
        self,
        log_dir: str,
        every_n_steps: int = 10,
        print_n_examples: int = 2,
    ):
        self.log_dir = log_dir
        self.every_n_steps = every_n_steps
        self.print_n_examples = print_n_examples

        # Storage filled by wrapped reward functions
        self._pending: List[_Batch] = []
        self._current_step: int = 0

    # ------------------------------------------------------------------
    # Reward-function wrapping
    # ------------------------------------------------------------------

    def wrap_reward_funcs(
        self, reward_funcs: List[Callable]
    ) -> List[Callable]:
        """
        Return instrumented copies of `reward_funcs`.

        The first function in the list is treated as the "primary" collector —
        it captures prompts and completions.  All functions record their
        reward scores under their function name.
        """
        wrapped = []
        for idx, func in enumerate(reward_funcs):
            wrapped.append(self._make_wrapper(func, idx == 0))
        return wrapped

    def _make_wrapper(self, func: Callable, is_primary: bool) -> Callable:
        func_name = getattr(func, "__name__", f"reward_func_{id(func)}")
        callback = self  # closure reference

        def wrapper(completions, prompts=None, **kwargs):
            rewards = func(completions, prompts=prompts, **kwargs)

            step = callback._current_step
            if is_primary:
                # Start a new batch record
                batch = _Batch(
                    step=step,
                    prompts=list(prompts) if prompts is not None else [],
                    completions=list(completions),
                )
                batch.rewards_by_func[func_name] = list(rewards)
                callback._pending.append(batch)
            else:
                # Append scores to the most recent batch
                if callback._pending:
                    callback._pending[-1].rewards_by_func[func_name] = list(rewards)

            return rewards

        wrapper.__name__ = func_name
        return wrapper

    # ------------------------------------------------------------------
    # Trainer callbacks
    # ------------------------------------------------------------------

    def on_step_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        self._current_step = state.global_step

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        self._current_step = state.global_step
        if state.global_step % self.every_n_steps == 0 and self._pending:
            self._flush(state.global_step)

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if self._pending:
            self._flush(state.global_step, final=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _flush(self, step: int, final: bool = False):
        os.makedirs(self.log_dir, exist_ok=True)

        tag = "final" if final else f"step_{step:07d}"
        out_path = os.path.join(self.log_dir, f"generations_{tag}.jsonl")

        records = self._build_records()
        with open(out_path, "w") as fh:
            for rec in records:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

        print(
            f"\n[GenerationLogger] step={step}  "
            f"saved {len(records)} generation(s) → {out_path}"
        )
        self._print_examples(records)
        self._pending.clear()

    def _build_records(self) -> List[Dict]:
        records = []
        for batch in self._pending:
            n = len(batch.completions)
            for i in range(n):
                prompt = batch.prompts[i] if i < len(batch.prompts) else None
                rec = {
                    "step": batch.step,
                    "prompt": prompt,
                    "completion": batch.completions[i],
                    "rewards": {
                        name: scores[i]
                        for name, scores in batch.rewards_by_func.items()
                        if i < len(scores)
                    },
                }
                records.append(rec)
        return records

    def _print_examples(self, records: List[Dict]):
        if not records or self.print_n_examples <= 0:
            return

        examples = records[: self.print_n_examples]
        sep = "=" * 70
        print(f"\n{sep}")
        for idx, rec in enumerate(examples):
            print(f"--- Example {idx + 1} (step {rec['step']}) ---")
            prompt = rec["prompt"]
            if isinstance(prompt, list):
                # Chat format: print last user turn
                user_msgs = [m for m in prompt if m.get("role") == "user"]
                prompt_str = user_msgs[-1]["content"] if user_msgs else str(prompt)
            else:
                prompt_str = str(prompt) if prompt is not None else "(no prompt)"
            # Truncate long prompts for readability
            if len(prompt_str) > 300:
                prompt_str = prompt_str[:300] + " …"
            print(f"PROMPT   : {prompt_str}")
            completion_str = rec["completion"]
            if len(completion_str) > 600:
                completion_str = completion_str[:600] + " …"
            print(f"COMPLETION: {completion_str}")
            rewards_str = "  ".join(
                f"{k}={v:.3f}" for k, v in rec["rewards"].items()
            )
            print(f"REWARDS  : {rewards_str}")
            print()
        print(sep)
