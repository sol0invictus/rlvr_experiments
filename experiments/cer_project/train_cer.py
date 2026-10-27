"""
Phase 2: Competence-Edge Routing (CER) Training

Per-token loss routing based on measured competence (entropy):
  - Low entropy (mastered) → skip (zero loss)
  - Mid entropy (competence edge) → RL loss (GRPO)
  - High entropy (beyond reach) → SFT loss (cross-entropy with best-of-K expert)

Subclasses GRPOTrainer and overrides _compute_loss to implement routing.
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
from environments.gsm8k import GSM8KEnvironment
from callbacks.generation_logger import GenerationLoggingCallback
from callbacks.validation_callback import ValidationCallback


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_dataset_local_or_hub(config):
    from datasets import load_from_disk, load_dataset
    data_conf = config['data']
    local_path = data_conf.get('local_train_path')
    if local_path and os.path.exists(local_path):
        print(f"Loading dataset from local cache: {local_path}")
        return load_from_disk(local_path)
    return load_dataset(data_conf['dataset_name'], data_conf.get('subset'), split=data_conf.get('split', 'train'))


def load_val_dataset_local_or_hub(config):
    from datasets import load_from_disk, load_dataset
    data_conf = config['data']
    local_path = data_conf.get('local_test_path')
    if local_path and os.path.exists(local_path):
        print(f"Loading val dataset from local cache: {local_path}")
        return load_from_disk(local_path)
    return load_dataset(data_conf['dataset_name'], data_conf.get('subset'), split='test')


class CERMetricsCallback(TrainerCallback):
    """Logs CER-specific metrics: fraction of tokens in each regime, SFT loss component."""

    def __init__(self, log_dir: str, log_every_n_steps: int = 10):
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

    def on_train_end(self, args, state, control, **kwargs):
        if self.accumulated:
            self._flush(state.global_step, final=True)

    def _flush(self, step, final=False):
        # Aggregate
        total_skip = sum(s['n_skip'] for s in self.accumulated)
        total_rl = sum(s['n_rl'] for s in self.accumulated)
        total_sft = sum(s['n_sft'] for s in self.accumulated)
        total = total_skip + total_rl + total_sft + 1e-8
        mean_sft_loss = np.mean([s['sft_loss'] for s in self.accumulated if s['sft_loss'] > 0]) if any(s['sft_loss'] > 0 for s in self.accumulated) else 0

        tag = "final" if final else f"step_{step:06d}"
        out = {
            'step': step,
            'frac_skip': total_skip / total,
            'frac_rl': total_rl / total,
            'frac_sft': total_sft / total,
            'mean_sft_loss': float(mean_sft_loss),
            'tau_low': self.accumulated[-1].get('tau_low', 0),
            'tau_high': self.accumulated[-1].get('tau_high', 0),
        }

        path = os.path.join(self.log_dir, f"cer_metrics_{tag}.json")
        with open(path, 'w') as f:
            json.dump(out, f, indent=2)

        print(f"\n[CER] step={step} | skip={out['frac_skip']:.1%} rl={out['frac_rl']:.1%} "
              f"sft={out['frac_sft']:.1%} | τ=[{out['tau_low']:.3f}, {out['tau_high']:.3f}] | "
              f"sft_loss={out['mean_sft_loss']:.4f}")

        self.accumulated.clear()


class CERTrainer(GRPOTrainer):
    """
    GRPOTrainer with soft Competence-Edge Weighting.

    All tokens get RL loss, but weighted by a function of entropy:
    - Low entropy (mastered) → downweighted (less gradient)
    - Mid entropy (competence edge) → full weight (most gradient)
    - High entropy (beyond reach) → downweighted (less gradient)

    This is a soft version: no tokens are zeroed out, just reweighted.
    """

    def __init__(self, *args, cer_config=None, cer_metrics_callback=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.cer_config = cer_config or {}
        self.cer_metrics_callback = cer_metrics_callback

        # Soft weighting params
        self.mu = self.cer_config.get('mu_init', 0.4)
        self.sigma = self.cer_config.get('sigma', 0.5)
        self.min_weight = self.cer_config.get('min_weight', 0.1)
        self.adapt_mu = self.cer_config.get('adapt_mu', True)
        self.mu_ema = self.cer_config.get('mu_ema', 0.99)
        self.mu_percentile = self.cer_config.get('mu_percentile', 50)
        self.mu_update_freq = self.cer_config.get('mu_update_freq', 20)

    def _compute_loss(self, model, inputs):
        """GRPO loss with soft entropy-based per-token weighting."""
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

        # === SOFT WEIGHTING ===
        # Gaussian bell curve centered at mu (competence edge), width sigma
        # w(e) = min_weight + (1 - min_weight) * exp(-(e - mu)^2 / (2*sigma^2))
        weights = self.min_weight + (1.0 - self.min_weight) * torch.exp(
            -((entropies - self.mu) ** 2) / (2 * self.sigma ** 2)
        )
        weights = weights * mask

        # === STANDARD GRPO LOSS (weighted) ===
        log_ratio = per_token_logps - old_per_token_logps
        coef_1 = torch.exp(log_ratio)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
        per_token_loss = -torch.min(coef_1 * advantages, coef_2 * advantages)

        # Apply soft weights
        weighted_loss = per_token_loss * weights
        denom = mask.sum(dim=1).clamp(min=1)
        rl_loss = (weighted_loss.sum(dim=1) / denom).mean()

        # === KL PENALTY ===
        kl_loss = torch.tensor(0.0, device=per_token_logps.device)
        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            )
            kl_loss = ((per_token_kl * mask).sum(dim=1) / denom).mean()

        loss = rl_loss + self.beta * kl_loss

        # === ADAPT MU ===
        if self.adapt_mu and self.state.global_step % self.mu_update_freq == 0:
            with torch.no_grad():
                valid_entropies = entropies[mask.bool()]
                if valid_entropies.numel() > 100:
                    new_mu = torch.quantile(valid_entropies.float(), self.mu_percentile / 100.0).item()
                    self.mu = self.mu_ema * self.mu + (1 - self.mu_ema) * new_mu

        # === METRICS ===
        mode = "train" if self.model.training else "eval"
        self._metrics[mode]["loss"].append(loss.detach().item())
        self._metrics[mode]["rl_loss"].append(rl_loss.detach().item())

        # Log CER stats
        if self.cer_metrics_callback is not None:
            with torch.no_grad():
                low_mask = (entropies < self.mu - self.sigma) & mask.bool()
                mid_mask = (entropies >= self.mu - self.sigma) & (entropies <= self.mu + self.sigma) & mask.bool()
                high_mask = (entropies > self.mu + self.sigma) & mask.bool()
                mean_weight = weights[mask.bool()].mean().item() if mask.any() else 0
            self.cer_metrics_callback.add_batch({
                'n_skip': int(low_mask.sum().item()),
                'n_rl': int(mid_mask.sum().item()),
                'n_sft': int(high_mask.sum().item()),
                'sft_loss': mean_weight,
                'tau_low': self.mu - self.sigma,
                'tau_high': self.mu + self.sigma,
            })

        return loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", nargs="?", default="configs/cer_qwen3b_gsm8k.yaml")
    parser.add_argument("--max_steps", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)

    # Load dataset
    dataset = load_dataset_local_or_hub(config)
    system_prompt = config.get('system_prompt', '').strip()
    def format_as_conversation(example):
        prompt_col = config['data']['prompt_column']
        prompt = [{"role": "user", "content": example[prompt_col]}]
        if system_prompt:
            prompt = [{"role": "system", "content": system_prompt}] + prompt
        example['prompt'] = prompt
        return example
    dataset = dataset.map(format_as_conversation)

    val_conf = config.get('validation', {})
    val_dataset = None
    if val_conf.get('enabled', False):
        val_dataset = load_val_dataset_local_or_hub(config)
        if val_dataset is not None:
            num_samples = val_conf.get('num_samples', 200)
            if num_samples and num_samples < len(val_dataset):
                val_dataset = val_dataset.select(range(num_samples))
            val_dataset = val_dataset.map(format_as_conversation)

    # Environment for reward functions
    env = GSM8KEnvironment(config)
    reward_funcs = env.get_reward_functions()

    # Callbacks
    gen_log_conf = config.get('generation_logging', {})
    generation_logging_callback = None
    if gen_log_conf.get('enabled', False):
        log_dir = gen_log_conf.get('log_dir', os.path.join(config['training']['output_dir'], 'generations'))
        generation_logging_callback = GenerationLoggingCallback(
            log_dir=log_dir,
            every_n_steps=gen_log_conf.get('every_n_steps', 20),
            print_n_examples=gen_log_conf.get('print_n_examples', 2),
        )
        reward_funcs = generation_logging_callback.wrap_reward_funcs(reward_funcs)

    cer_conf = config.get('cer', {})
    cer_metrics_callback = CERMetricsCallback(
        log_dir=os.path.join(config['training']['output_dir'], 'cer_metrics'),
        log_every_n_steps=cer_conf.get('log_every_n_steps', 10),
    )

    # Model
    model_path = config['model']['name_or_path']
    print(f"Loading model: {model_path}")
    attn_implementation = config['model'].get('attn_implementation', None)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=config['model'].get('torch_dtype', 'auto'),
        **({"attn_implementation": attn_implementation} if attn_implementation else {}),
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Note: NOT adding </answer> subword tokens to eos_token_id — that breaks
    # generation because ">" (token 29) appears in <think> tags.
    # Instead, rely on max_completion_length for stopping.

    # Training config
    training_conf = config['training']
    gen_conf = config.get('generation', {})
    max_steps = args.max_steps if args.max_steps is not None else training_conf.get('max_steps', -1)

    num_generations = gen_conf.get('num_generations', 8)
    per_device_train_batch_size = training_conf.get('per_device_train_batch_size', 2)
    generation_batch_size = num_generations * per_device_train_batch_size

    training_args = GRPOConfig(
        output_dir=training_conf['output_dir'],
        learning_rate=float(training_conf['learning_rate']),
        loss_type=training_conf.get('loss_type', 'grpo'),
        beta=training_conf.get('beta', 0.04),
        remove_unused_columns=False,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=training_conf.get('gradient_accumulation_steps', 2),
        num_train_epochs=training_conf.get('num_train_epochs', 1),
        bf16=training_conf.get('bf16', True),
        max_completion_length=gen_conf.get('max_completion_length', 512),
        num_generations=num_generations,
        generation_batch_size=generation_batch_size,
        temperature=gen_conf.get('temperature', 0.7),
        report_to=training_conf.get('report_to', []),
        logging_steps=training_conf.get('logging_steps', 5),
        push_to_hub=False,
        save_strategy=training_conf.get('save_strategy', 'steps'),
        save_steps=training_conf.get('save_steps', 100),
        save_total_limit=3,
        max_steps=max_steps,
        warmup_steps=training_conf.get('warmup_steps', 20),
        max_grad_norm=training_conf.get('max_grad_norm', 1.0),
        lr_scheduler_type=training_conf.get('lr_scheduler_type', 'cosine'),
        gradient_checkpointing=training_conf.get('gradient_checkpointing', False),
        dataloader_num_workers=training_conf.get('dataloader_num_workers', 4),
        ddp_find_unused_parameters=False,
    )

    # Validation callback
    validation_callback = None
    if val_dataset is not None:
        val_log_dir = val_conf.get('log_dir', os.path.join(training_conf['output_dir'], 'validation'))
        validation_callback = ValidationCallback(
            val_dataset=val_dataset,
            reward_funcs=env.get_reward_functions(),
            tokenizer=tokenizer,
            eval_steps=val_conf.get('eval_steps', 50),
            num_samples=val_conf.get('num_samples', 200),
            max_new_tokens=gen_conf.get('max_completion_length', 512),
            log_dir=val_log_dir,
            temperature=val_conf.get('temperature', 0.0),
        )

    # Trainer
    print("Initializing CERTrainer...")
    extra_callbacks = [cb for cb in [generation_logging_callback, validation_callback, cer_metrics_callback] if cb is not None]

    trainer = CERTrainer(
        model=model,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        callbacks=extra_callbacks,
        cer_config=cer_conf,
        cer_metrics_callback=cer_metrics_callback,
    )

    print(f"Starting CER v4 Training (soft Gaussian weighting)")
    print(f"  Model: {model_path}")
    print(f"  Steps: {max_steps}")
    print(f"  GPUs: {torch.cuda.device_count()}")
    print(f"  CER config: mu_init={cer_conf.get('mu_init', 0.4)}, "
          f"sigma={cer_conf.get('sigma', 0.5)}, "
          f"min_weight={cer_conf.get('min_weight', 0.1)}")
    trainer.train()

    trainer.save_model(training_conf['output_dir'])
    print(f"\nCER training complete. Model saved to {training_conf['output_dir']}")


if __name__ == "__main__":
    main()
