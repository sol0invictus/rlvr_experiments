"""
Online Rejection SFT on Maze task.

Generate K completions per prompt, keep only those with reward > 0,
do NLL (SFT) on them. No RL loss at all.

On sparse-reward tasks, this is the purest signal: directly train on
the model's own successful completions. No policy gradient noise,
no ratio clipping, no advantage estimation.
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
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from environments.reasoning_gym_env import ReasoningGymEnvironment


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


class RejectionSFTMetricsCallback(TrainerCallback):
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
        n_skipped = sum(s['n_skipped_tokens'] for s in self.accumulated)
        total = n_sft + n_skipped + 1e-8
        mean_loss = np.mean([s['sft_loss'] for s in self.accumulated if s['sft_loss'] > 0]) if any(s['sft_loss'] > 0 for s in self.accumulated) else 0
        frac_positive = np.mean([s['frac_positive'] for s in self.accumulated])

        out = {
            'step': step,
            'frac_sft': n_sft / total,
            'frac_skipped': n_skipped / total,
            'mean_sft_loss': float(mean_loss),
            'frac_positive_completions': float(frac_positive),
        }
        path = os.path.join(self.log_dir, f"rsft_metrics_step_{step:06d}.json")
        with open(path, 'w') as f:
            json.dump(out, f, indent=2)

        print(f"\n[RSFT] step={step} | sft_frac={out['frac_sft']:.1%} | "
              f"positive_completions={out['frac_positive_completions']:.1%} | "
              f"sft_loss={out['mean_sft_loss']:.4f}")
        self.accumulated.clear()


class RejectionSFTTrainer(GRPOTrainer):
    """
    Online Rejection SFT: NLL only on positive-reward completions.
    Negative/zero-reward completions contribute zero loss.

    This subclasses GRPOTrainer to reuse its generation + reward infrastructure,
    but replaces the loss with pure SFT on accepted completions.
    """

    def __init__(self, *args, rsft_config=None, rsft_metrics_callback=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.rsft_config = rsft_config or {}
        self.rsft_metrics_callback = rsft_metrics_callback
        self.kl_weight = self.rsft_config.get('kl_weight', 0.04)

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

        # Only do SFT on completions with positive advantage (= above-group-mean reward)
        pos_mask = (advantages > 0).float()  # [batch, 1]
        sft_per_token = -per_token_logps * pos_mask * mask

        denom = mask.sum(dim=1).clamp(min=1)
        sft_loss = (sft_per_token.sum(dim=1) / denom).mean()

        # KL regularization to prevent collapse
        kl_loss = torch.tensor(0.0, device=per_token_logps.device)
        if self.kl_weight > 0 and "ref_per_token_logps" in inputs:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            )
            kl_loss = ((per_token_kl * mask).sum(dim=1) / denom).mean()

        loss = sft_loss + self.kl_weight * kl_loss

        mode = "train" if self.model.training else "eval"
        self._metrics[mode]["loss"].append(loss.detach().item())

        if self.rsft_metrics_callback is not None:
            with torch.no_grad():
                n_sft = (pos_mask * mask).sum().item()
                n_skipped = ((1 - pos_mask) * mask).sum().item()
                batch_size = advantages.shape[0]
                frac_pos = (advantages > 0).float().mean().item()
            self.rsft_metrics_callback.add_batch({
                'n_sft_tokens': int(n_sft),
                'n_skipped_tokens': int(n_skipped),
                'sft_loss': sft_loss.item(),
                'frac_positive': frac_pos,
            })

        return loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", nargs="?", default="configs/maze_rejection_sft.yaml")
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

    rsft_conf = config.get('rejection_sft', {})
    training_conf = config['training']

    rsft_metrics_callback = RejectionSFTMetricsCallback(
        log_dir=os.path.join(training_conf['output_dir'], 'rsft_metrics'),
        log_every_n_steps=rsft_conf.get('log_every_n_steps', 5),
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

    print("Initializing RejectionSFTTrainer for Maze...")
    trainer = RejectionSFTTrainer(
        model=model,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        callbacks=[rsft_metrics_callback],
        rsft_config=rsft_conf,
        rsft_metrics_callback=rsft_metrics_callback,
    )

    print(f"Starting Online Rejection SFT on Maze")
    print(f"  Model: {model_path}")
    print(f"  Steps: {max_steps}")
    print(f"  GPUs: {torch.cuda.device_count()}")
    print(f"  KL weight: {rsft_conf.get('kl_weight', 0.04)}")
    trainer.train()

    trainer.save_model(training_conf['output_dir'])
    print(f"\nTraining complete. Model saved to {training_conf['output_dir']}")


if __name__ == "__main__":
    main()
