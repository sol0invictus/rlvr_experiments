"""
ERA: Entropy-Routed Advantage for unified SFT+RL training.

Per-token routing between SFT and GRPO based on model entropy at that position:
- High entropy tokens (model uncertain) → NLL (SFT) loss — direct teaching
- Low entropy tokens (model confident) → GRPO loss — nudge direction

Combined with advantage gating:
- Positive advantage completions: high-H → SFT drill, low-H → GRPO reinforce
- Negative advantage completions: low-H → GRPO suppress hard (confident mistakes),
  high-H → lighter suppression (already uncertain)

Threshold is adaptive: percentile of per-batch entropy distribution.
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


class ERAMetricsCallback(TrainerCallback):
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
        mean_entropy = np.mean([s['mean_entropy'] for s in self.accumulated])
        mean_threshold = np.mean([s['entropy_threshold'] for s in self.accumulated])
        mean_frac_sft_tokens = np.mean([s['frac_sft_tokens'] for s in self.accumulated])
        mean_frac_rl_tokens = np.mean([s['frac_rl_tokens'] for s in self.accumulated])

        out = {
            'step': step,
            'sft_loss': float(mean_sft),
            'rl_loss': float(mean_rl),
            'mean_entropy': float(mean_entropy),
            'entropy_threshold': float(mean_threshold),
            'frac_sft_tokens': float(mean_frac_sft_tokens),
            'frac_rl_tokens': float(mean_frac_rl_tokens),
        }
        path = os.path.join(self.log_dir, f"era_metrics_step_{step:06d}.json")
        with open(path, 'w') as f:
            json.dump(out, f, indent=2)

        print(f"\n[ERA] step={step} | sft={out['sft_loss']:.4f} rl={out['rl_loss']:.4f} | "
              f"H_mean={out['mean_entropy']:.3f} H_thresh={out['entropy_threshold']:.3f} | "
              f"sft_tok={out['frac_sft_tokens']:.1%} rl_tok={out['frac_rl_tokens']:.1%}")
        self.accumulated.clear()


class ERATrainer(GRPOTrainer):
    """
    Entropy-Routed Advantage Trainer.

    Routes each token to SFT or GRPO loss based on model entropy:
    - Positive advantage + high entropy → SFT (teach uncertain tokens)
    - Positive advantage + low entropy → GRPO reinforce (nudge confident tokens)
    - Negative advantage + low entropy → GRPO suppress (fix confident mistakes)
    - Negative advantage + high entropy → light GRPO suppress (already uncertain)

    The entropy threshold is computed as a percentile of the batch entropy distribution.
    """

    def __init__(self, *args, era_config=None, era_metrics_callback=None, tokenizer=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.era_config = era_config or {}
        self.era_metrics_callback = era_metrics_callback
        self.sft_weight = self.era_config.get('sft_weight', 1.0)
        self.rl_weight = self.era_config.get('rl_weight', 1.0)
        self.entropy_percentile = self.era_config.get('entropy_percentile', 50.0)
        self.high_entropy_sft_scale = self.era_config.get('high_entropy_sft_scale', 2.0)
        self.low_entropy_rl_scale = self.era_config.get('low_entropy_rl_scale', 2.0)
        self.format_bonus = self.era_config.get('format_bonus', 0.1)
        self._era_tokenizer = tokenizer

    def _check_format(self, completion_ids):
        if self._era_tokenizer is None:
            return torch.zeros(completion_ids.shape[0], device=completion_ids.device)
        has_format = []
        for i in range(completion_ids.shape[0]):
            text = self._era_tokenizer.decode(completion_ids[i], skip_special_tokens=True)
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

        # Use parent's method with compute_entropy=True
        per_token_logps, per_token_entropy = self._get_per_token_logps_and_entropies(
            model, input_ids, attention_mask, logits_to_keep, compute_entropy=True,
        )

        advantages = inputs["advantages"]
        if advantages.dim() == 1:
            advantages = advantages.unsqueeze(1)

        old_per_token_logps = inputs.get("old_per_token_logps")
        old_per_token_logps = per_token_logps.detach() if old_per_token_logps is None else old_per_token_logps

        # Format bonus
        with torch.no_grad():
            format_mask = self._check_format(completion_ids)
            format_bonus_vec = format_mask * self.format_bonus
            adjusted_advantages = advantages.squeeze(-1) + format_bonus_vec if advantages.dim() > 1 else advantages + format_bonus_vec
            if adjusted_advantages.dim() == 1:
                adjusted_advantages = adjusted_advantages.unsqueeze(1)

        # Compute entropy threshold (adaptive percentile of valid tokens)
        with torch.no_grad():
            valid_entropies = per_token_entropy[mask.bool()]
            if valid_entropies.numel() > 0:
                entropy_threshold = torch.quantile(valid_entropies, self.entropy_percentile / 100.0)
            else:
                entropy_threshold = torch.tensor(1.0, device=mask.device)

            # Per-token routing masks
            high_entropy = (per_token_entropy >= entropy_threshold).float()
            low_entropy = (per_token_entropy < entropy_threshold).float()

        # Advantage gating (sequence-level)
        pos_adv = (adjusted_advantages > 0).float()  # [B, 1]
        neg_adv = (adjusted_advantages <= 0).float()  # [B, 1]
        denom = mask.sum(dim=1).clamp(min=1)

        # === POSITIVE ADVANTAGE COMPLETIONS ===
        # High entropy tokens → SFT (teach the uncertain ones)
        pos_sft_mask = pos_adv * high_entropy * mask  # [B, seq]
        pos_sft_loss = (-per_token_logps * pos_sft_mask * self.high_entropy_sft_scale)

        # Low entropy tokens → GRPO reinforce (nudge the confident ones)
        pos_rl_mask = pos_adv * low_entropy * mask
        log_ratio_pos = per_token_logps - old_per_token_logps
        coef_1_pos = torch.exp(log_ratio_pos.clamp(max=10.0))
        coef_2_pos = torch.clamp(coef_1_pos, 1 - self.epsilon_low, 1 + self.epsilon_high)
        grpo_pos = -torch.min(coef_1_pos * adjusted_advantages, coef_2_pos * adjusted_advantages)
        pos_rl_loss = grpo_pos * pos_rl_mask

        # === NEGATIVE ADVANTAGE COMPLETIONS ===
        # Low entropy tokens → GRPO suppress hard (confident mistakes)
        neg_rl_low_mask = neg_adv * low_entropy * mask
        log_ratio_neg = per_token_logps - old_per_token_logps
        coef_1_neg = torch.exp(log_ratio_neg.clamp(max=10.0))
        coef_2_neg = torch.clamp(coef_1_neg, 1 - self.epsilon_low, 1 + self.epsilon_high)
        grpo_neg = -torch.min(coef_1_neg * adjusted_advantages, coef_2_neg * adjusted_advantages)
        neg_rl_hard_loss = grpo_neg * neg_rl_low_mask * self.low_entropy_rl_scale

        # High entropy tokens → light GRPO suppress (already uncertain, less signal)
        neg_rl_high_mask = neg_adv * high_entropy * mask
        neg_rl_light_loss = grpo_neg * neg_rl_high_mask  # scale = 1.0

        # Combine
        total_sft = (pos_sft_loss.sum(dim=1) / denom).mean()
        total_rl = ((pos_rl_loss + neg_rl_hard_loss + neg_rl_light_loss).sum(dim=1) / denom).mean()

        loss = self.sft_weight * total_sft + self.rl_weight * total_rl

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

        if self.era_metrics_callback is not None:
            with torch.no_grad():
                frac_sft = pos_sft_mask.sum() / mask.sum().clamp(min=1)
                frac_rl = (pos_rl_mask + neg_rl_low_mask + neg_rl_high_mask).sum() / mask.sum().clamp(min=1)
            self.era_metrics_callback.add_batch({
                'sft_loss': total_sft.item(),
                'rl_loss': total_rl.item(),
                'mean_entropy': valid_entropies.mean().item() if valid_entropies.numel() > 0 else 0.0,
                'entropy_threshold': entropy_threshold.item(),
                'frac_sft_tokens': frac_sft.item(),
                'frac_rl_tokens': frac_rl.item(),
            })

        return loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", nargs="?", default="configs/maze_era.yaml")
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

    era_conf = config.get('era', {})
    training_conf = config['training']

    era_metrics_callback = ERAMetricsCallback(
        log_dir=os.path.join(training_conf['output_dir'], 'era_metrics'),
        log_every_n_steps=era_conf.get('log_every_n_steps', 5),
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

    print("Initializing ERA Trainer...")
    trainer = ERATrainer(
        model=model,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        callbacks=[era_metrics_callback],
        era_config=era_conf,
        era_metrics_callback=era_metrics_callback,
        tokenizer=tokenizer,
    )

    print(f"Starting ERA Training")
    print(f"  Model: {model_path}")
    print(f"  Steps: {max_steps}")
    print(f"  GPUs: {torch.cuda.device_count()}")
    print(f"  entropy_percentile={era_conf.get('entropy_percentile', 50.0)}")
    print(f"  high_entropy_sft_scale={era_conf.get('high_entropy_sft_scale', 2.0)}")
    print(f"  low_entropy_rl_scale={era_conf.get('low_entropy_rl_scale', 2.0)}")
    trainer.train()

    trainer.save_model(training_conf['output_dir'])
    print(f"\nTraining complete. Model saved to {training_conf['output_dir']}")


if __name__ == "__main__":
    main()
