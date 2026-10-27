"""
CER + Advantage-Gated hybrid on Maze task.

Combines both ideas:
- Advantage-Gated: positive advantage → SFT, negative → GRPO
- CER (soft Gaussian): weight tokens by entropy proximity to competence edge

This means: SFT reinforcement is strongest on tokens at the competence edge
of correct completions (where the model is "almost there"), and GRPO suppression
is strongest on tokens at the competence edge of incorrect completions.
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


class CERAGMetricsCallback(TrainerCallback):
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
        n_sft = sum(s['n_sft_tokens'] for s in self.accumulated)
        n_rl = sum(s['n_rl_tokens'] for s in self.accumulated)
        total = n_sft + n_rl + 1e-8
        mean_sft_loss = np.mean([s['sft_loss'] for s in self.accumulated])
        mean_rl_loss = np.mean([s['rl_loss'] for s in self.accumulated])
        mean_weight = np.mean([s['mean_weight'] for s in self.accumulated])
        mu = self.accumulated[-1].get('mu', 0)

        out = {
            'step': step,
            'frac_sft': n_sft / total,
            'frac_rl': n_rl / total,
            'mean_sft_loss': float(mean_sft_loss),
            'mean_rl_loss': float(mean_rl_loss),
            'mean_weight': float(mean_weight),
            'mu': float(mu),
        }
        path = os.path.join(self.log_dir, f"cerag_metrics_step_{step:06d}.json")
        with open(path, 'w') as f:
            json.dump(out, f, indent=2)

        print(f"\n[CER-AG] step={step} | sft={out['frac_sft']:.1%} rl={out['frac_rl']:.1%} | "
              f"mu={out['mu']:.3f} mean_w={out['mean_weight']:.3f} | "
              f"sft_loss={out['mean_sft_loss']:.4f} rl_loss={out['mean_rl_loss']:.4f}")
        self.accumulated.clear()


class CERAGTrainer(GRPOTrainer):
    """
    CER + Advantage-Gated:
    - Positive advantage → SFT loss, weighted by entropy proximity to mu
    - Negative advantage → GRPO loss, weighted by entropy proximity to mu
    """

    def __init__(self, *args, cerag_config=None, cerag_metrics_callback=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.cerag_config = cerag_config or {}
        self.cerag_metrics_callback = cerag_metrics_callback
        self.sft_weight = self.cerag_config.get('sft_weight', 1.0)
        self.rl_weight = self.cerag_config.get('rl_weight', 1.0)
        self.mu = self.cerag_config.get('mu_init', 0.4)
        self.sigma = self.cerag_config.get('sigma', 0.5)
        self.min_weight = self.cerag_config.get('min_weight', 0.1)
        self.adapt_mu = self.cerag_config.get('adapt_mu', True)
        self.mu_ema = self.cerag_config.get('mu_ema', 0.99)
        self.mu_percentile = self.cerag_config.get('mu_percentile', 50)
        self.mu_update_freq = self.cerag_config.get('mu_update_freq', 20)

    def _compute_loss(self, model, inputs):
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)
        mask = completion_mask if "tool_mask" not in inputs else completion_mask * inputs["tool_mask"]

        per_token_logps, entropies = self._get_per_token_logps_and_entropies(
            model, input_ids, attention_mask, logits_to_keep, compute_entropy=True,
        )

        advantages = inputs["advantages"]
        if advantages.dim() == 1:
            advantages = advantages.unsqueeze(1)

        old_per_token_logps = inputs.get("old_per_token_logps")
        old_per_token_logps = per_token_logps.detach() if old_per_token_logps is None else old_per_token_logps

        # CER soft weighting (entropy-based)
        weights = self.min_weight + (1.0 - self.min_weight) * torch.exp(
            -((entropies - self.mu) ** 2) / (2 * self.sigma ** 2)
        )
        weights = weights * mask

        # Advantage gating
        pos_adv = (advantages > 0).float()
        neg_adv = (advantages <= 0).float()

        # SFT on positive-advantage completions, entropy-weighted
        sft_per_token = -per_token_logps * pos_adv * weights

        # GRPO on negative-advantage completions, entropy-weighted
        log_ratio = per_token_logps - old_per_token_logps
        coef_1 = torch.exp(log_ratio)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
        grpo_per_token = -torch.min(coef_1 * advantages, coef_2 * advantages)
        rl_per_token = grpo_per_token * neg_adv * weights

        combined = self.sft_weight * sft_per_token + self.rl_weight * rl_per_token
        denom = mask.sum(dim=1).clamp(min=1)
        loss = (combined.sum(dim=1) / denom).mean()

        # KL penalty
        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            )
            kl_loss = ((per_token_kl * mask).sum(dim=1) / denom).mean()
            loss = loss + self.beta * kl_loss

        # Adapt mu
        if self.adapt_mu and self.state.global_step % self.mu_update_freq == 0:
            with torch.no_grad():
                valid_entropies = entropies[mask.bool()]
                if valid_entropies.numel() > 100:
                    new_mu = torch.quantile(valid_entropies.float(), self.mu_percentile / 100.0).item()
                    self.mu = self.mu_ema * self.mu + (1 - self.mu_ema) * new_mu

        mode = "train" if self.model.training else "eval"
        self._metrics[mode]["loss"].append(loss.detach().item())

        sft_loss_val = (sft_per_token.sum(dim=1) / denom).mean().item()
        rl_loss_val = (rl_per_token.sum(dim=1) / denom).mean().item()

        if self.cerag_metrics_callback is not None:
            with torch.no_grad():
                n_sft = (pos_adv * mask).sum().item()
                n_rl = (neg_adv * mask).sum().item()
                mean_w = weights[mask.bool()].mean().item() if mask.any() else 0
            self.cerag_metrics_callback.add_batch({
                'n_sft_tokens': int(n_sft),
                'n_rl_tokens': int(n_rl),
                'sft_loss': sft_loss_val,
                'rl_loss': rl_loss_val,
                'mean_weight': mean_w,
                'mu': self.mu,
            })

        return loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", nargs="?", default="configs/maze_cer_ag.yaml")
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

    cerag_conf = config.get('cer_ag', {})
    training_conf = config['training']

    cerag_metrics_callback = CERAGMetricsCallback(
        log_dir=os.path.join(training_conf['output_dir'], 'cerag_metrics'),
        log_every_n_steps=cerag_conf.get('log_every_n_steps', 5),
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

    print("Initializing CER-AG Trainer for Maze...")
    trainer = CERAGTrainer(
        model=model,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        callbacks=[cerag_metrics_callback],
        cerag_config=cerag_conf,
        cerag_metrics_callback=cerag_metrics_callback,
    )

    print(f"Starting CER+AG Maze Training")
    print(f"  Model: {model_path}")
    print(f"  Steps: {max_steps}")
    print(f"  GPUs: {torch.cuda.device_count()}")
    print(f"  sft_weight={cerag_conf.get('sft_weight', 1.0)}, rl_weight={cerag_conf.get('rl_weight', 1.0)}")
    print(f"  CER: mu_init={cerag_conf.get('mu_init', 0.4)}, sigma={cerag_conf.get('sigma', 0.5)}")
    trainer.train()

    trainer.save_model(training_conf['output_dir'])
    print(f"\nTraining complete. Model saved to {training_conf['output_dir']}")


if __name__ == "__main__":
    main()
