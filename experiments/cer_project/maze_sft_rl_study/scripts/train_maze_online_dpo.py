"""
Online DPO + SFT Anchor on Maze task.

From each group of K generations per prompt:
  - Pick best (highest reward) as chosen
  - Pick worst (lowest reward) as rejected
  - DPO contrastive loss on the pair (RL component)
  - NLL on the chosen completion (SFT component)

loss = dpo_loss(chosen, rejected) + sft_alpha * nll(chosen)

DPO provides cleaner contrastive signal than GRPO (no policy gradient noise,
direct log-ratio comparison between best and worst).
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


class OnlineDPOMetricsCallback(TrainerCallback):
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
        mean_dpo = np.mean([s['dpo_loss'] for s in self.accumulated])
        mean_sft = np.mean([s['sft_loss'] for s in self.accumulated])
        mean_margin = np.mean([s['reward_margin'] for s in self.accumulated])
        frac_distinct = np.mean([s['frac_distinct_pairs'] for s in self.accumulated])

        out = {
            'step': step,
            'dpo_loss': float(mean_dpo),
            'sft_loss': float(mean_sft),
            'reward_margin': float(mean_margin),
            'frac_distinct_pairs': float(frac_distinct),
        }
        path = os.path.join(self.log_dir, f"dpo_metrics_step_{step:06d}.json")
        with open(path, 'w') as f:
            json.dump(out, f, indent=2)

        print(f"\n[DPO+SFT] step={step} | dpo_loss={out['dpo_loss']:.4f} sft_loss={out['sft_loss']:.4f} | "
              f"margin={out['reward_margin']:.3f} distinct_pairs={out['frac_distinct_pairs']:.1%}")
        self.accumulated.clear()


class OnlineDPOSFTTrainer(GRPOTrainer):
    """
    Online DPO + SFT Anchor.

    Uses GRPO's generation + reward infrastructure, but replaces the loss:
    - From K completions per prompt, select best/worst by advantage
    - DPO contrastive loss between them
    - NLL on the best completion (SFT anchor)

    When all completions have equal reward (no contrast), falls back to
    pure SFT on the best completion or zero loss if all are equally bad.
    """

    def __init__(self, *args, dpo_config=None, dpo_metrics_callback=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.dpo_config = dpo_config or {}
        self.dpo_metrics_callback = dpo_metrics_callback
        self.sft_alpha = self.dpo_config.get('sft_alpha', 0.5)
        self.dpo_beta = self.dpo_config.get('dpo_beta', 0.1)

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

        # Get reference log probs for DPO
        ref_per_token_logps = inputs.get("ref_per_token_logps")
        if ref_per_token_logps is None:
            ref_per_token_logps = per_token_logps.detach()

        denom = mask.sum(dim=1).clamp(min=1)
        batch_size = advantages.shape[0]

        # Per-sequence log probs (sum over tokens)
        seq_logps = (per_token_logps * mask).sum(dim=1)
        ref_seq_logps = (ref_per_token_logps * mask).sum(dim=1)

        # Identify chosen (best) and rejected (worst) by advantage
        # advantages is [batch, 1] — squeeze for comparison
        adv_flat = advantages.squeeze(1)  # [batch]

        # Create masks for best and worst in each group
        # With GRPO, completions from the same prompt are contiguous in the batch
        # We identify chosen/rejected per prompt group
        # Simple approach: best = max advantage, worst = min advantage
        best_mask = (adv_flat == adv_flat.max()).float()
        worst_mask = (adv_flat == adv_flat.min()).float()

        # Check if there's actual contrast (best != worst reward)
        has_contrast = (adv_flat.max() - adv_flat.min()).abs() > 1e-6

        if has_contrast:
            # DPO loss: -log sigmoid(beta * (log_ratio_chosen - log_ratio_rejected))
            # log_ratio = log_pi(y|x) - log_ref(y|x)
            log_ratios = seq_logps - ref_seq_logps  # [batch]

            # Weighted average for chosen and rejected
            best_norm = best_mask.sum().clamp(min=1)
            worst_norm = worst_mask.sum().clamp(min=1)
            chosen_log_ratio = (log_ratios * best_mask).sum() / best_norm
            rejected_log_ratio = (log_ratios * worst_mask).sum() / worst_norm

            dpo_loss = -F.logsigmoid(self.dpo_beta * (chosen_log_ratio - rejected_log_ratio))
            reward_margin = (adv_flat.max() - adv_flat.min()).item()
        else:
            dpo_loss = torch.tensor(0.0, device=per_token_logps.device)
            reward_margin = 0.0

        # SFT on best completions (chosen)
        sft_per_token = -per_token_logps * best_mask.unsqueeze(1) * mask
        sft_loss = (sft_per_token.sum(dim=1) / denom).mean()

        # Combined loss
        loss = dpo_loss + self.sft_alpha * sft_loss

        # KL penalty (mild, prevents drift)
        if self.beta != 0.0:
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            )
            kl_loss = ((per_token_kl * mask).sum(dim=1) / denom).mean()
            loss = loss + self.beta * kl_loss

        mode = "train" if self.model.training else "eval"
        self._metrics[mode]["loss"].append(loss.detach().item())

        if self.dpo_metrics_callback is not None:
            self.dpo_metrics_callback.add_batch({
                'dpo_loss': dpo_loss.item(),
                'sft_loss': sft_loss.item(),
                'reward_margin': reward_margin,
                'frac_distinct_pairs': 1.0 if has_contrast else 0.0,
            })

        return loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", nargs="?", default="configs/maze_online_dpo.yaml")
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

    dpo_conf = config.get('online_dpo', {})
    training_conf = config['training']

    dpo_metrics_callback = OnlineDPOMetricsCallback(
        log_dir=os.path.join(training_conf['output_dir'], 'dpo_metrics'),
        log_every_n_steps=dpo_conf.get('log_every_n_steps', 5),
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

    print("Initializing OnlineDPO+SFT Trainer for Maze...")
    trainer = OnlineDPOSFTTrainer(
        model=model,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        callbacks=[dpo_metrics_callback],
        dpo_config=dpo_conf,
        dpo_metrics_callback=dpo_metrics_callback,
    )

    print(f"Starting Online DPO + SFT Maze Training")
    print(f"  Model: {model_path}")
    print(f"  Steps: {max_steps}")
    print(f"  GPUs: {torch.cuda.device_count()}")
    print(f"  sft_alpha={dpo_conf.get('sft_alpha', 0.5)}, dpo_beta={dpo_conf.get('dpo_beta', 0.1)}")
    trainer.train()

    trainer.save_model(training_conf['output_dir'])
    print(f"\nTraining complete. Model saved to {training_conf['output_dir']}")


if __name__ == "__main__":
    main()
