"""
SWAG: Surprise-Weighted Advantage Gating for unified SFT+RL training.

Per-token credit assignment using generation-time surprise (old log-probs):
- Positive completions: surprising tokens (low old_logp) get amplified SFT
  → "aha moments" that need drilling
- Negative completions: confident tokens (high old_logp) get amplified GRPO
  → "confident mistakes" that need unlearning
- Tokens the model was already sure about (correct) or uncertain about (wrong)
  get minimal gradient — they're not the learning bottleneck.

Key insight: not all tokens in a correct completion are equally informative.
Generation-time surprise identifies WHERE the model got lucky vs WHERE it knew.
Concentrating gradient on "lucky correct" tokens accelerates genuine learning.
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


class SWAGMetricsCallback(TrainerCallback):
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
        mean_surprise = np.mean([s['mean_token_surprise'] for s in self.accumulated])
        mean_sft_weight = np.mean([s['mean_sft_token_weight'] for s in self.accumulated])
        mean_rl_weight = np.mean([s['mean_rl_token_weight'] for s in self.accumulated])

        out = {
            'step': step,
            'sft_loss': float(mean_sft),
            'rl_loss': float(mean_rl),
            'mean_token_surprise': float(mean_surprise),
            'mean_sft_token_weight': float(mean_sft_weight),
            'mean_rl_token_weight': float(mean_rl_weight),
        }
        path = os.path.join(self.log_dir, f"swag_metrics_step_{step:06d}.json")
        with open(path, 'w') as f:
            json.dump(out, f, indent=2)

        print(f"\n[SWAG] step={step} | sft={out['sft_loss']:.4f} rl={out['rl_loss']:.4f} | "
              f"surprise={out['mean_token_surprise']:.3f} | "
              f"sft_w={out['mean_sft_token_weight']:.3f} rl_w={out['mean_rl_token_weight']:.3f}")
        self.accumulated.clear()


class SWAGTrainer(GRPOTrainer):
    """
    Surprise-Weighted Advantage Gating Trainer.

    Routes gradient intensity per-token based on generation-time surprise:
    - For positive-advantage completions:
      token_weight = (-old_logp)^alpha (normalized)
      → Tokens that were surprising when generated get strong SFT
    - For negative-advantage completions:
      token_weight = (old_logp_clipped)^beta (normalized, inverted: confident = high weight)
      → Tokens the model was confident about (but wrong) get strong GRPO suppression

    Combined with standard advantage gating (pos → SFT, neg → GRPO).
    """

    def __init__(self, *args, swag_config=None, swag_metrics_callback=None, tokenizer=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.swag_config = swag_config or {}
        self.swag_metrics_callback = swag_metrics_callback
        self.sft_weight = self.swag_config.get('sft_weight', 1.0)
        self.rl_weight = self.swag_config.get('rl_weight', 1.0)
        self.alpha_surprise = self.swag_config.get('alpha_surprise', 1.0)
        self.beta_confidence = self.swag_config.get('beta_confidence', 1.0)
        self.surprise_clip = self.swag_config.get('surprise_clip', 10.0)
        self.confidence_floor = self.swag_config.get('confidence_floor', 0.1)
        self.format_bonus = self.swag_config.get('format_bonus', 0.1)
        self._swag_tokenizer = tokenizer

    def _check_format(self, completion_ids):
        if self._swag_tokenizer is None:
            return torch.zeros(completion_ids.shape[0], device=completion_ids.device)
        has_format = []
        for i in range(completion_ids.shape[0]):
            text = self._swag_tokenizer.decode(completion_ids[i], skip_special_tokens=True)
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

        # Format bonus
        with torch.no_grad():
            format_mask = self._check_format(completion_ids)
            format_bonus_vec = format_mask * self.format_bonus
            adjusted_advantages = advantages.squeeze(-1) + format_bonus_vec if advantages.dim() > 1 else advantages + format_bonus_vec
            if adjusted_advantages.dim() == 1:
                adjusted_advantages = adjusted_advantages.unsqueeze(1)

        # Compute per-token surprise weights from OLD log-probs
        with torch.no_grad():
            # Surprise = -old_logp (high when model was uncertain at generation time)
            token_surprise = (-old_per_token_logps).clamp(min=0, max=self.surprise_clip)

            # Normalize surprise to [0, 1] per-sequence using min-max
            seq_min = (token_surprise * mask + (~mask.bool()).float() * 999).min(dim=1, keepdim=True).values
            seq_max = (token_surprise * mask).max(dim=1, keepdim=True).values
            surprise_range = (seq_max - seq_min).clamp(min=1e-6)
            surprise_norm = ((token_surprise - seq_min) / surprise_range).clamp(0, 1)

            # SFT weights: linear map from surprise_norm to [1-alpha, 1+alpha]
            # High surprise → weight up to 1+alpha, low surprise → weight down to 1-alpha
            # alpha_surprise controls the spread (e.g., 0.5 → weights in [0.5, 1.5])
            sft_token_weights = 1.0 + self.alpha_surprise * (2 * surprise_norm - 1)

            # RL weights: linear map from CONFIDENCE (1 - surprise_norm) to [1-beta, 1+beta]
            # High confidence (low surprise) → weight up to 1+beta
            confidence_norm = 1.0 - surprise_norm
            rl_token_weights = 1.0 + self.beta_confidence * (2 * confidence_norm - 1)

        # Advantage gating
        pos_adv = (adjusted_advantages > 0).float()
        neg_adv = (adjusted_advantages <= 0).float()
        denom = mask.sum(dim=1).clamp(min=1)

        # SFT on positive-advantage completions (surprise-weighted per token)
        sft_per_token = -per_token_logps * sft_token_weights * pos_adv * mask
        sft_loss = (sft_per_token.sum(dim=1) / denom).mean()

        # GRPO on negative-advantage completions (confidence-weighted per token)
        log_ratio = per_token_logps - old_per_token_logps
        coef_1 = torch.exp(log_ratio.clamp(max=10.0))
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
        grpo_per_token = -torch.min(coef_1 * adjusted_advantages, coef_2 * adjusted_advantages)
        rl_per_token = grpo_per_token * rl_token_weights * neg_adv * mask
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

        if self.swag_metrics_callback is not None:
            with torch.no_grad():
                valid_surprise = token_surprise[mask.bool()]
                valid_sft_w = sft_token_weights[(pos_adv.bool().expand_as(mask)) & mask.bool()]
                valid_rl_w = rl_token_weights[(neg_adv.bool().expand_as(mask)) & mask.bool()]
            self.swag_metrics_callback.add_batch({
                'sft_loss': sft_loss.item(),
                'rl_loss': rl_loss.item(),
                'mean_token_surprise': valid_surprise.mean().item() if valid_surprise.numel() > 0 else 0.0,
                'mean_sft_token_weight': valid_sft_w.mean().item() if valid_sft_w.numel() > 0 else 0.0,
                'mean_rl_token_weight': valid_rl_w.mean().item() if valid_rl_w.numel() > 0 else 0.0,
            })

        return loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", nargs="?", default="configs/maze_swag.yaml")
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

    swag_conf = config.get('swag', {})
    training_conf = config['training']

    swag_metrics_callback = SWAGMetricsCallback(
        log_dir=os.path.join(training_conf['output_dir'], 'swag_metrics'),
        log_every_n_steps=swag_conf.get('log_every_n_steps', 5),
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

    print("Initializing SWAG Trainer...")
    trainer = SWAGTrainer(
        model=model,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        callbacks=[swag_metrics_callback],
        swag_config=swag_conf,
        swag_metrics_callback=swag_metrics_callback,
        tokenizer=tokenizer,
    )

    print(f"Starting SWAG Training")
    print(f"  Model: {model_path}")
    print(f"  Steps: {max_steps}")
    print(f"  GPUs: {torch.cuda.device_count()}")
    print(f"  alpha_surprise={swag_conf.get('alpha_surprise', 1.0)}")
    print(f"  beta_confidence={swag_conf.get('beta_confidence', 1.0)}")
    trainer.train()

    trainer.save_model(training_conf['output_dir'])
    print(f"\nTraining complete. Model saved to {training_conf['output_dir']}")


if __name__ == "__main__":
    main()
