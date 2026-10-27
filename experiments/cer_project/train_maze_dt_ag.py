"""
DT-AG: Dual-Temperature Advantage-Gated training on Maze task.

The core insight: not all positives are equally valuable for SFT, and not all
negatives are equally valuable for GRPO suppression. DT-AG routes loss based
on completion confidence (measured by mean log-probability):

- High-surprise positives (low mean logprob) → strong SFT weight
  These are diverse, "exploratory" correct solutions — the kind a high-temperature
  generation would produce. They prevent overfitting to a narrow mode.

- High-confidence negatives (high mean logprob) → strong GRPO weight
  These are errors the model is confident about — the kind a low-temperature
  generation would produce. Suppressing these is critical because they represent
  entrenched wrong patterns.

This achieves the dual-temperature effect without requiring two generation passes:
the single generation at temperature=1.0 produces a mix of confident and exploratory
completions, and DT-AG assigns them asymmetric importance based on their confidence.
"""

import sys
import os
import json
import argparse
import torch
import torch.nn.functional as F
import yaml
import numpy as np
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback, TrainerState, TrainerControl
from trl import GRPOConfig, GRPOTrainer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from environments.reasoning_gym_env import ReasoningGymEnvironment


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


class DTAGMetricsCallback(TrainerCallback):
    def __init__(self, log_dir: str, log_every_n_steps: int = 5):
        self.log_dir = log_dir
        self.log_every_n_steps = log_every_n_steps
        self.accumulated = []
        os.makedirs(log_dir, exist_ok=True)

    def add_batch(self, stats):
        self.accumulated.append(stats)

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        if state.global_step % self.log_every_n_steps != 0:
            return
        if not self.accumulated:
            return
        self._flush(state.global_step)

    def _flush(self, step):
        mean_sft = np.mean([s['sft_loss'] for s in self.accumulated])
        mean_rl = np.mean([s['rl_loss'] for s in self.accumulated])
        mean_sft_temp_w = np.mean([s['mean_sft_temp_weight'] for s in self.accumulated])
        mean_rl_temp_w = np.mean([s['mean_rl_temp_weight'] for s in self.accumulated])
        mean_logp = np.mean([s['mean_completion_logp'] for s in self.accumulated])

        out = {
            'step': step,
            'sft_loss': float(mean_sft),
            'rl_loss': float(mean_rl),
            'mean_sft_temp_weight': float(mean_sft_temp_w),
            'mean_rl_temp_weight': float(mean_rl_temp_w),
            'mean_completion_logp': float(mean_logp),
        }
        path = os.path.join(self.log_dir, f"dt_ag_metrics_step_{step:06d}.json")
        with open(path, 'w') as f:
            json.dump(out, f, indent=2)

        print(f"\n[DT-AG] step={step} | sft={out['sft_loss']:.4f} rl={out['rl_loss']:.4f} | "
              f"sft_w={out['mean_sft_temp_weight']:.3f} rl_w={out['mean_rl_temp_weight']:.3f} "
              f"logp={out['mean_completion_logp']:.2f}")
        self.accumulated.clear()


class DTAGTrainer(GRPOTrainer):
    """
    Dual-Temperature Advantage-Gated Loss.

    Standard AG gating (positive → SFT, negative → GRPO) but with
    confidence-based asymmetric weighting:

    - SFT weight ∝ surprise^sft_temp_power (high-surprise = diverse = more valuable for SFT)
    - GRPO weight ∝ confidence^rl_temp_power (high-confidence = entrenched error = more important to suppress)

    Confidence = mean per-token log-prob of the completion under current policy.
    Surprise = -confidence (normalized).
    """

    def __init__(self, *args, dt_config=None, dt_metrics_callback=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.dt_config = dt_config or {}
        self.dt_metrics_callback = dt_metrics_callback
        self.sft_weight = self.dt_config.get('sft_weight', 1.0)
        self.rl_weight = self.dt_config.get('rl_weight', 1.0)
        self.sft_temp_power = self.dt_config.get('sft_temp_power', 1.5)
        self.rl_temp_power = self.dt_config.get('rl_temp_power', 1.5)
        self.temp_weight_max = self.dt_config.get('temp_weight_max', 3.0)

    def _compute_loss(self, model, inputs):
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)
        mask = completion_mask if "tool_mask" not in inputs else completion_mask * inputs["tool_mask"]

        per_token_logps, _ = self._get_per_token_logps_and_entropies(
            model, input_ids, attention_mask, logits_to_keep, compute_entropy=False,
        )

        advantages = inputs["advantages"]
        if advantages.dim() == 1:
            advantages = advantages.unsqueeze(1)

        old_per_token_logps = inputs.get("old_per_token_logps")
        old_per_token_logps = per_token_logps.detach() if old_per_token_logps is None else old_per_token_logps

        pos_adv = (advantages > 0).float()
        neg_adv = (advantages <= 0).float()
        denom = mask.sum(dim=1).clamp(min=1)

        # Compute per-completion mean log-prob (confidence measure)
        with torch.no_grad():
            clamped_logps = per_token_logps.detach().clamp(min=-20.0)
            mean_logp = (clamped_logps * mask).sum(dim=1) / denom  # shape: [batch]
            mean_logp = mean_logp.clamp(min=-20.0, max=0.0)

            # Normalize to [0, 1] within batch
            logp_min = mean_logp.min()
            logp_max = mean_logp.max()
            logp_range = (logp_max - logp_min).clamp(min=1e-4)

            # Surprise = 1 - normalized_confidence (so low logp → high surprise)
            normalized_confidence = ((mean_logp - logp_min) / logp_range).clamp(0.0, 1.0)
            surprise = 1.0 - normalized_confidence  # [0, 1]

            # SFT temperature weight: surprise^power, scaled to [1, temp_weight_max]
            sft_temp_weight = 1.0 + (self.temp_weight_max - 1.0) * (surprise ** self.sft_temp_power)
            sft_temp_weight = sft_temp_weight.clamp(max=self.temp_weight_max).unsqueeze(1)

            # GRPO temperature weight: confidence^power, scaled to [1, temp_weight_max]
            rl_temp_weight = 1.0 + (self.temp_weight_max - 1.0) * (normalized_confidence ** self.rl_temp_power)
            rl_temp_weight = rl_temp_weight.clamp(max=self.temp_weight_max).unsqueeze(1)

        # SFT on positive-advantage completions (weighted by surprise)
        sft_per_token = -per_token_logps * pos_adv * sft_temp_weight * mask
        sft_loss = (sft_per_token.sum(dim=1) / denom).mean()

        # GRPO on negative-advantage completions (weighted by confidence)
        log_ratio = (per_token_logps - old_per_token_logps).clamp(-10.0, 10.0)
        coef_1 = torch.exp(log_ratio)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
        grpo_per_token = -torch.min(coef_1 * advantages, coef_2 * advantages)
        rl_per_token = grpo_per_token * neg_adv * rl_temp_weight * mask
        rl_loss = (rl_per_token.sum(dim=1) / denom).mean()

        # Combined
        loss = self.sft_weight * sft_loss + self.rl_weight * rl_loss

        # KL penalty
        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            )
            kl_loss = ((per_token_kl * mask).sum(dim=1) / denom).mean()
            loss = loss + self.beta * kl_loss

        mode = "train" if self.model.training else "eval"
        self._metrics[mode]["loss"].append(loss.detach().item())

        if self.dt_metrics_callback is not None:
            with torch.no_grad():
                sft_w_mean = (sft_temp_weight.squeeze() * pos_adv.squeeze()).sum() / pos_adv.sum().clamp(min=1)
                rl_w_mean = (rl_temp_weight.squeeze() * neg_adv.squeeze()).sum() / neg_adv.sum().clamp(min=1)
            self.dt_metrics_callback.add_batch({
                'sft_loss': sft_loss.item(),
                'rl_loss': rl_loss.item(),
                'mean_sft_temp_weight': sft_w_mean.item(),
                'mean_rl_temp_weight': rl_w_mean.item(),
                'mean_completion_logp': mean_logp.mean().item(),
            })

        return loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", nargs="?", default="configs/maze_dt_ag.yaml")
    parser.add_argument("--max_steps", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)

    env = ReasoningGymEnvironment(config)
    dataset = env.get_dataset(config)
    reward_funcs = env.get_reward_functions()

    system_prompt = config.get('system_prompt', '').strip()
    if system_prompt:
        def add_system_prompt(example):
            prompt = example['prompt']
            if not any(m.get('role') == 'system' for m in prompt):
                prompt = [{'role': 'system', 'content': system_prompt}] + prompt
                example['prompt'] = prompt
            return example
        dataset = dataset.map(add_system_prompt)

    dt_conf = config.get('dt_ag', {})
    training_conf = config['training']

    dt_metrics_callback = DTAGMetricsCallback(
        log_dir=os.path.join(training_conf['output_dir'], 'dt_ag_metrics'),
        log_every_n_steps=dt_conf.get('log_every_n_steps', 5),
    )

    model_path = config['model']['name_or_path']
    print(f"Loading model: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=config['model'].get('torch_dtype', 'auto'),
        attn_implementation=config['model'].get('attn_implementation', None),
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    gen_conf = config.get('generation', {})
    max_steps = args.max_steps if args.max_steps is not None else training_conf.get('max_steps', -1)

    num_generations = gen_conf.get('num_generations', 8)
    per_device_train_batch_size = training_conf.get('per_device_train_batch_size', 1)
    generation_batch_size = gen_conf.get('generation_batch_size', num_generations * per_device_train_batch_size)

    training_args = GRPOConfig(
        output_dir=training_conf['output_dir'],
        learning_rate=float(training_conf['learning_rate']),
        loss_type=training_conf.get('loss_type', 'grpo'),
        beta=training_conf.get('beta', 0.04),
        remove_unused_columns=False,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=training_conf.get('gradient_accumulation_steps', 1),
        num_train_epochs=training_conf.get('num_train_epochs', 2),
        bf16=training_conf.get('bf16', True),
        max_completion_length=gen_conf.get('max_completion_length', 2048),
        num_generations=num_generations,
        generation_batch_size=generation_batch_size,
        temperature=gen_conf.get('temperature', 1.0),
        report_to=training_conf.get('report_to', []),
        logging_steps=training_conf.get('logging_steps', 1),
        push_to_hub=False,
        save_strategy=training_conf.get('save_strategy', 'steps'),
        save_steps=training_conf.get('save_steps', 50),
        save_total_limit=3,
        max_steps=max_steps,
        warmup_ratio=training_conf.get('warmup_ratio', 0.1),
        max_grad_norm=training_conf.get('max_grad_norm', 1.0),
        lr_scheduler_type=training_conf.get('lr_scheduler_type', 'constant'),
        gradient_checkpointing=training_conf.get('gradient_checkpointing', True),
        dataloader_num_workers=training_conf.get('dataloader_num_workers', 4),
        ddp_find_unused_parameters=False,
        mask_truncated_completions=training_conf.get('mask_truncated_completions', True),
    )

    print("Initializing DT-AG Trainer for Maze...")
    trainer = DTAGTrainer(
        model=model,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        callbacks=[dt_metrics_callback],
        dt_config=dt_conf,
        dt_metrics_callback=dt_metrics_callback,
    )

    print(f"Starting DT-AG Maze Training")
    print(f"  Model: {model_path}")
    print(f"  Steps: {max_steps}")
    print(f"  GPUs: {torch.cuda.device_count()}")
    print(f"  sft_temp_power={dt_conf.get('sft_temp_power', 1.5)}, rl_temp_power={dt_conf.get('rl_temp_power', 1.5)}, temp_weight_max={dt_conf.get('temp_weight_max', 3.0)}")
    trainer.train()

    trainer.save_model(training_conf['output_dir'])
    print(f"\nTraining complete. Model saved to {training_conf['output_dir']}")


if __name__ == "__main__":
    main()
