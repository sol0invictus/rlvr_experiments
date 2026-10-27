"""
AG-AMS: Advantage-Gated with Advantage Magnitude Scaling on Maze task.

Extension of AG where SFT gradient is scaled by |advantage|^alpha and GRPO
gradient is scaled by |advantage|^beta. This concentrates SFT on clear winners
(large positive advantage) and spreads GRPO uniformly across negatives.

Additionally adds a format bonus (+0.1) to advantages of format-correct completions
before the gating decision, so format-correct-but-wrong completions are slightly
preferred over format-broken ones as SFT targets.
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


class AGAMSMetricsCallback(TrainerCallback):
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
        mean_sft_scale = np.mean([s['mean_sft_scale'] for s in self.accumulated])
        mean_rl_scale = np.mean([s['mean_rl_scale'] for s in self.accumulated])
        mean_format_bonus_applied = np.mean([s['format_bonus_applied'] for s in self.accumulated])

        out = {
            'step': step,
            'sft_loss': float(mean_sft),
            'rl_loss': float(mean_rl),
            'mean_sft_scale': float(mean_sft_scale),
            'mean_rl_scale': float(mean_rl_scale),
            'format_bonus_applied': float(mean_format_bonus_applied),
        }
        path = os.path.join(self.log_dir, f"ag_ams_metrics_step_{step:06d}.json")
        with open(path, 'w') as f:
            json.dump(out, f, indent=2)

        print(f"\n[AG-AMS] step={step} | sft={out['sft_loss']:.4f} rl={out['rl_loss']:.4f} | "
              f"sft_scale={out['mean_sft_scale']:.3f} rl_scale={out['mean_rl_scale']:.3f} "
              f"fmt_bonus={out['format_bonus_applied']:.1%}")
        self.accumulated.clear()


class AGAMSTrainer(GRPOTrainer):
    """
    Advantage-Gated with Advantage Magnitude Scaling.

    - Positive advantage → SFT loss scaled by |advantage|^alpha
    - Negative advantage → GRPO loss scaled by |advantage|^beta
    - Format bonus: +0.1 added to advantages of format-correct completions

    With alpha > 1 (default 2.0): SFT concentrates on clear winners.
    With beta < 1 (default 0.5): GRPO spreads more uniformly across negatives.
    """

    def __init__(self, *args, ams_config=None, ams_metrics_callback=None, tokenizer=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.ams_config = ams_config or {}
        self.ams_metrics_callback = ams_metrics_callback
        self.sft_weight = self.ams_config.get('sft_weight', 1.0)
        self.rl_weight = self.ams_config.get('rl_weight', 1.0)
        self.alpha = self.ams_config.get('alpha', 2.0)
        self.beta_ams = self.ams_config.get('beta_scale', 0.5)
        self.format_bonus = self.ams_config.get('format_bonus', 0.1)
        self._ams_tokenizer = tokenizer

    def _check_format(self, completion_ids):
        """Check if completion has <answer>...</answer> tags."""
        if self._ams_tokenizer is None:
            return torch.zeros(completion_ids.shape[0], device=completion_ids.device)

        has_format = []
        for i in range(completion_ids.shape[0]):
            text = self._ams_tokenizer.decode(completion_ids[i], skip_special_tokens=True)
            has_answer = '<answer>' in text and '</answer>' in text
            has_format.append(1.0 if has_answer else 0.0)
        return torch.tensor(has_format, device=completion_ids.device)

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

        # Apply format bonus before gating
        with torch.no_grad():
            format_mask = self._check_format(completion_ids)
            format_bonus_vec = format_mask * self.format_bonus
            adjusted_advantages = advantages.squeeze(-1) + format_bonus_vec if advantages.dim() > 1 else advantages + format_bonus_vec
            if adjusted_advantages.dim() == 1:
                adjusted_advantages = adjusted_advantages.unsqueeze(1)
            frac_format_bonus = format_mask.mean().item()

        pos_adv = (adjusted_advantages > 0).float()
        neg_adv = (adjusted_advantages <= 0).float()
        denom = mask.sum(dim=1).clamp(min=1)

        # Magnitude scaling for SFT (alpha power on positive advantages)
        with torch.no_grad():
            pos_magnitudes = adjusted_advantages.abs().clamp(min=1e-6)
            sft_scale = (pos_magnitudes ** self.alpha) * pos_adv
            # Normalize so mean scale is ~1 (prevents gradient explosion)
            sft_scale_sum = sft_scale.sum()
            if sft_scale_sum > 1e-6:
                sft_scale = sft_scale * (pos_adv.sum() / sft_scale_sum)

        # Magnitude scaling for GRPO (beta power on negative advantages)
        with torch.no_grad():
            neg_magnitudes = adjusted_advantages.abs().clamp(min=1e-6)
            rl_scale = (neg_magnitudes ** self.beta_ams) * neg_adv
            # Normalize
            rl_scale_sum = rl_scale.sum()
            if rl_scale_sum > 1e-6:
                rl_scale = rl_scale * (neg_adv.sum() / rl_scale_sum)

        # SFT on positive-advantage completions (magnitude-scaled)
        sft_per_token = -per_token_logps * sft_scale * mask
        sft_loss = (sft_per_token.sum(dim=1) / denom).mean()

        # GRPO on negative-advantage completions (magnitude-scaled)
        log_ratio = per_token_logps - old_per_token_logps
        coef_1 = torch.exp(log_ratio)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
        grpo_per_token = -torch.min(coef_1 * adjusted_advantages, coef_2 * adjusted_advantages)
        rl_per_token = grpo_per_token * rl_scale * mask
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

        if self.ams_metrics_callback is not None:
            with torch.no_grad():
                mean_sft_s = sft_scale[pos_adv.bool()].mean().item() if pos_adv.any() else 0.0
                mean_rl_s = rl_scale[neg_adv.bool()].mean().item() if neg_adv.any() else 0.0
            self.ams_metrics_callback.add_batch({
                'sft_loss': sft_loss.item(),
                'rl_loss': rl_loss.item(),
                'mean_sft_scale': mean_sft_s,
                'mean_rl_scale': mean_rl_s,
                'format_bonus_applied': frac_format_bonus,
            })

        return loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", nargs="?", default="configs/maze_ag_ams.yaml")
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

    ams_conf = config.get('ag_ams', {})
    training_conf = config['training']

    ams_metrics_callback = AGAMSMetricsCallback(
        log_dir=os.path.join(training_conf['output_dir'], 'ag_ams_metrics'),
        log_every_n_steps=ams_conf.get('log_every_n_steps', 5),
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

    print("Initializing AG-AMS Trainer for Maze...")
    trainer = AGAMSTrainer(
        model=model,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        callbacks=[ams_metrics_callback],
        ams_config=ams_conf,
        ams_metrics_callback=ams_metrics_callback,
        tokenizer=tokenizer,
    )

    print(f"Starting AG-AMS Maze Training")
    print(f"  Model: {model_path}")
    print(f"  Steps: {max_steps}")
    print(f"  GPUs: {torch.cuda.device_count()}")
    print(f"  alpha={ams_conf.get('alpha', 2.0)}, beta_scale={ams_conf.get('beta_scale', 0.5)}, format_bonus={ams_conf.get('format_bonus', 0.1)}")
    trainer.train()

    trainer.save_model(training_conf['output_dir'])
    print(f"\nTraining complete. Model saved to {training_conf['output_dir']}")


if __name__ == "__main__":
    main()
