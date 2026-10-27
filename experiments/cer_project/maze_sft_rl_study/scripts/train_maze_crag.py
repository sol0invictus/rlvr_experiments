"""
CRAG: Contrastive-Ranked Advantage Gating on Maze task.

Same AG structure (SFT on positives, GRPO on negatives), but the GRPO
component uses reward-reweighted advantages. Completions further from
the best get stronger negative gradient; near-misses get gentle suppression.

This makes the GRPO signal more informative when most negatives look identical
(all getting reward=0), by using the reward gap as a curriculum within the
contrastive component.
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


class CRAGMetricsCallback(TrainerCallback):
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
        mean_reweight = np.mean([s['mean_reweight_factor'] for s in self.accumulated])
        mean_adv_spread = np.mean([s['adv_spread'] for s in self.accumulated])

        out = {
            'step': step,
            'sft_loss': float(mean_sft),
            'rl_loss': float(mean_rl),
            'mean_reweight_factor': float(mean_reweight),
            'adv_spread': float(mean_adv_spread),
        }
        path = os.path.join(self.log_dir, f"crag_metrics_step_{step:06d}.json")
        with open(path, 'w') as f:
            json.dump(out, f, indent=2)

        print(f"\n[CRAG] step={step} | sft={out['sft_loss']:.4f} rl={out['rl_loss']:.4f} | "
              f"reweight={out['mean_reweight_factor']:.3f} adv_spread={out['adv_spread']:.3f}")
        self.accumulated.clear()


class CRAGTrainer(GRPOTrainer):
    """
    Contrastive-Ranked Advantage Gating.

    - Positive advantage → SFT loss (same as AG)
    - Negative advantage → GRPO loss with reweighted advantages

    Reweighting: for negatives, multiply advantage by (1 + |margin_from_best|^rank_power).
    This makes completions further from the best receive stronger GRPO suppression,
    while near-misses receive gentler gradient.
    """

    def __init__(self, *args, crag_config=None, crag_metrics_callback=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.crag_config = crag_config or {}
        self.crag_metrics_callback = crag_metrics_callback
        self.sft_weight = self.crag_config.get('sft_weight', 1.0)
        self.rl_weight = self.crag_config.get('rl_weight', 1.0)
        self.rank_power = self.crag_config.get('rank_power', 1.0)
        self.max_reweight = self.crag_config.get('max_reweight', 3.0)

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

        # SFT on positive-advantage completions (same as AG)
        sft_per_token = -per_token_logps * pos_adv * mask
        sft_loss = (sft_per_token.sum(dim=1) / denom).mean()

        # Reweight negative advantages by distance from best
        with torch.no_grad():
            best_adv = advantages.max()
            # margin: how far each completion is from the best (always >= 0 for negatives)
            margin = (best_adv - advantages).clamp(min=0)
            # Normalize margin to [0, 1] range
            margin_range = margin.max()
            if margin_range > 1e-6:
                norm_margin = margin / margin_range
            else:
                norm_margin = torch.zeros_like(margin)
            # Reweight factor: completions further from best get higher weight
            reweight = 1.0 + (self.max_reweight - 1.0) * (norm_margin ** self.rank_power)
            reweight = reweight.clamp(max=self.max_reweight)

        # Reweighted advantages for GRPO
        reweighted_advantages = advantages * reweight

        # GRPO on negatives with reweighted advantages
        log_ratio = per_token_logps - old_per_token_logps
        coef_1 = torch.exp(log_ratio)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
        grpo_per_token = -torch.min(coef_1 * reweighted_advantages, coef_2 * reweighted_advantages)
        rl_per_token = grpo_per_token * neg_adv * mask
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

        if self.crag_metrics_callback is not None:
            with torch.no_grad():
                mean_rw = reweight[neg_adv.bool()].mean().item() if neg_adv.any() else 1.0
                adv_spread = (advantages.max() - advantages.min()).item()
            self.crag_metrics_callback.add_batch({
                'sft_loss': sft_loss.item(),
                'rl_loss': rl_loss.item(),
                'mean_reweight_factor': mean_rw,
                'adv_spread': adv_spread,
            })

        return loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", nargs="?", default="configs/maze_crag.yaml")
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

    crag_conf = config.get('crag', {})
    training_conf = config['training']

    crag_metrics_callback = CRAGMetricsCallback(
        log_dir=os.path.join(training_conf['output_dir'], 'crag_metrics'),
        log_every_n_steps=crag_conf.get('log_every_n_steps', 5),
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

    print("Initializing CRAG Trainer for Maze...")
    trainer = CRAGTrainer(
        model=model,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        callbacks=[crag_metrics_callback],
        crag_config=crag_conf,
        crag_metrics_callback=crag_metrics_callback,
    )

    print(f"Starting CRAG Maze Training")
    print(f"  Model: {model_path}")
    print(f"  Steps: {max_steps}")
    print(f"  GPUs: {torch.cuda.device_count()}")
    print(f"  rank_power={crag_conf.get('rank_power', 1.0)}, max_reweight={crag_conf.get('max_reweight', 3.0)}")
    trainer.train()

    trainer.save_model(training_conf['output_dir'])
    print(f"\nTraining complete. Model saved to {training_conf['output_dir']}")


if __name__ == "__main__":
    main()
