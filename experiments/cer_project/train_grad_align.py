"""
Experiment: Gradient Alignment Regularizer (PCGrad-style)

Standard GRPO + SFT anchor on best-of-K completions.
When GRPO gradient conflicts with SFT gradient (would unlearn good behavior),
project GRPO gradient to remove the conflicting component.

Implementation: two-loss approach with per-parameter gradient projection.
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


class GradAlignMetricsCallback(TrainerCallback):
    """Logs gradient alignment metrics."""

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

    def _flush(self, step):
        mean_cos_sim = np.mean([s['cos_sim'] for s in self.accumulated])
        mean_n_conflicts = np.mean([s['n_conflicts'] for s in self.accumulated])
        mean_proj_ratio = np.mean([s['proj_ratio'] for s in self.accumulated])

        out = {
            'step': step,
            'mean_cos_sim': float(mean_cos_sim),
            'mean_n_conflicts': float(mean_n_conflicts),
            'mean_proj_ratio': float(mean_proj_ratio),
        }
        path = os.path.join(self.log_dir, f"ga_metrics_step_{step:06d}.json")
        with open(path, 'w') as f:
            json.dump(out, f, indent=2)

        print(f"\n[GA] step={step} | cos_sim={out['mean_cos_sim']:.4f} | "
              f"conflicts={out['mean_n_conflicts']:.0f} | proj_ratio={out['mean_proj_ratio']:.3f}")
        self.accumulated.clear()


class GradAlignTrainer(GRPOTrainer):
    """
    GRPO with Gradient Alignment Regularizer.

    Computes GRPO loss + SFT loss on positive-advantage completions.
    Uses PCGrad-style projection: when per-parameter GRPO gradient conflicts
    with SFT gradient direction, project out the conflicting component.

    This prevents RL from unlearning correct behavior.
    """

    def __init__(self, *args, ga_config=None, ga_metrics_callback=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.ga_config = ga_config or {}
        self.ga_metrics_callback = ga_metrics_callback
        self.sft_alpha = self.ga_config.get('sft_alpha', 0.3)
        self.project_grads = self.ga_config.get('project_grads', True)
        self.proj_every_n_steps = self.ga_config.get('proj_every_n_steps', 1)

    def _compute_loss(self, model, inputs):
        """
        GRPO + SFT with gradient projection.

        Strategy: compute combined loss that enables projection at backward time.
        - GRPO loss on all completions (standard)
        - SFT loss on positive-advantage completions (anchor)
        - When not projecting: loss = grpo + alpha * sft
        - When projecting: use custom backward with PCGrad
        """
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

        denom = mask.sum(dim=1).clamp(min=1)

        # === STANDARD GRPO LOSS ===
        log_ratio = per_token_logps - old_per_token_logps
        coef_1 = torch.exp(log_ratio)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
        per_token_loss = -torch.min(coef_1 * advantages, coef_2 * advantages)
        grpo_loss = ((per_token_loss * mask).sum(dim=1) / denom).mean()

        # === SFT ANCHOR on positive-advantage completions ===
        pos_adv = (advantages > 0).float()
        sft_per_token = -per_token_logps * pos_adv * mask
        # Normalize by total tokens (not just positive-adv tokens) to keep scale stable
        sft_loss = (sft_per_token.sum(dim=1) / denom).mean()

        # === KL PENALTY ===
        kl_loss = torch.tensor(0.0, device=per_token_logps.device)
        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            )
            kl_loss = ((per_token_kl * mask).sum(dim=1) / denom).mean()

        # === GRADIENT PROJECTION (PCGrad-style) ===
        do_project = (
            self.project_grads
            and self.state.global_step % self.proj_every_n_steps == 0
            and self.model.training
        )

        if do_project:
            # Compute gradients of both losses separately
            grpo_grads = torch.autograd.grad(
                grpo_loss, model.parameters(), retain_graph=True, allow_unused=True
            )
            sft_grads = torch.autograd.grad(
                sft_loss, model.parameters(), retain_graph=True, allow_unused=True
            )

            # PCGrad projection: remove conflicting component from GRPO gradient
            cos_sims = []
            n_conflicts = 0
            n_total = 0
            proj_norms_before = 0.0
            proj_norms_after = 0.0

            projected_grads = []
            for g_grpo, g_sft in zip(grpo_grads, sft_grads):
                if g_grpo is None or g_sft is None:
                    projected_grads.append(g_grpo)
                    continue

                g_grpo_flat = g_grpo.reshape(-1)
                g_sft_flat = g_sft.reshape(-1)

                dot = torch.dot(g_grpo_flat, g_sft_flat)
                sft_norm_sq = torch.dot(g_sft_flat, g_sft_flat)

                proj_norms_before += g_grpo_flat.norm().item()

                if dot < 0 and sft_norm_sq > 1e-10:
                    # Conflict: project GRPO grad onto plane orthogonal to SFT grad
                    g_proj = g_grpo_flat - (dot / sft_norm_sq) * g_sft_flat
                    projected_grads.append(g_proj.reshape(g_grpo.shape))
                    n_conflicts += 1
                    cos_sim = (dot / (g_grpo_flat.norm() * g_sft_flat.norm() + 1e-8)).item()
                    cos_sims.append(cos_sim)
                else:
                    projected_grads.append(g_grpo)

                proj_norms_after += (projected_grads[-1].reshape(-1).norm().item()
                                     if projected_grads[-1] is not None else 0)
                n_total += 1

            # Now combine: projected GRPO grad + alpha * SFT grad
            # We achieve this by setting model.grad directly
            # First zero existing grads
            model.zero_grad()
            for param, g_proj, g_sft in zip(model.parameters(), projected_grads, sft_grads):
                if g_proj is not None:
                    combined_grad = g_proj
                    if g_sft is not None:
                        combined_grad = combined_grad + self.sft_alpha * g_sft
                    # Add KL gradient
                    param.grad = combined_grad.detach()

            # Add KL gradient
            if self.beta != 0.0:
                kl_grads = torch.autograd.grad(
                    self.beta * kl_loss, model.parameters(), allow_unused=True
                )
                for param, g_kl in zip(model.parameters(), kl_grads):
                    if g_kl is not None:
                        if param.grad is not None:
                            param.grad = param.grad + g_kl.detach()
                        else:
                            param.grad = g_kl.detach()

            # Return a dummy loss that won't trigger another backward
            # The trainer will call loss.backward() but we've already set grads
            loss = torch.tensor(0.0, device=per_token_logps.device, requires_grad=True)
            # Actually we need a loss that produces zero grad when .backward() is called
            # Use a detached surrogate
            loss = (grpo_loss + self.sft_alpha * sft_loss + self.beta * kl_loss).detach().requires_grad_(True)

            # Metrics
            mean_cos = np.mean(cos_sims) if cos_sims else 0.0
            proj_ratio = proj_norms_after / (proj_norms_before + 1e-8)
            if self.ga_metrics_callback:
                self.ga_metrics_callback.add_batch({
                    'cos_sim': mean_cos,
                    'n_conflicts': n_conflicts,
                    'proj_ratio': proj_ratio,
                })

            real_loss_val = (grpo_loss + self.sft_alpha * sft_loss + self.beta * kl_loss).detach().item()
        else:
            # No projection: standard combined loss
            loss = grpo_loss + self.sft_alpha * sft_loss + self.beta * kl_loss
            real_loss_val = loss.detach().item()

            if self.ga_metrics_callback:
                self.ga_metrics_callback.add_batch({
                    'cos_sim': 0.0,
                    'n_conflicts': 0,
                    'proj_ratio': 1.0,
                })

        # === METRICS ===
        mode = "train" if self.model.training else "eval"
        self._metrics[mode]["loss"].append(real_loss_val)
        self._metrics[mode]["grpo_loss"].append(grpo_loss.detach().item())
        self._metrics[mode]["sft_loss"].append(sft_loss.detach().item())

        return loss

    def training_step(self, model, inputs, num_items_in_batch=None):
        """Override to handle manual gradient setting from PCGrad projection."""
        model.train()
        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            loss = self._compute_loss(model, inputs)

        # If we did projection, grads are already set — skip backward
        if loss.grad_fn is None:
            # Projected path: grads already set manually
            # Just need to do gradient accumulation scaling
            if self.args.gradient_accumulation_steps > 1:
                for param in model.parameters():
                    if param.grad is not None:
                        param.grad = param.grad / self.args.gradient_accumulation_steps
            return loss.detach()
        else:
            # Normal path: do backward as usual
            if self.use_apex:
                from apex import amp
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                self.accelerator.backward(loss)
            return loss.detach()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", nargs="?", default="configs/grad_align.yaml")
    parser.add_argument("--max_steps", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)

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

    env = GSM8KEnvironment(config)
    reward_funcs = env.get_reward_functions()

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

    ga_conf = config.get('grad_align', {})
    ga_metrics_callback = GradAlignMetricsCallback(
        log_dir=os.path.join(config['training']['output_dir'], 'ga_metrics'),
        log_every_n_steps=ga_conf.get('log_every_n_steps', 10),
    )

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

    print("Initializing GradAlignTrainer...")
    extra_callbacks = [cb for cb in [generation_logging_callback, validation_callback, ga_metrics_callback] if cb is not None]

    trainer = GradAlignTrainer(
        model=model,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        callbacks=extra_callbacks,
        ga_config=ga_conf,
        ga_metrics_callback=ga_metrics_callback,
    )

    print(f"Starting Gradient Alignment Training")
    print(f"  Model: {model_path}")
    print(f"  Steps: {max_steps}")
    print(f"  GPUs: {torch.cuda.device_count()}")
    print(f"  sft_alpha={ga_conf.get('sft_alpha', 0.3)}, project_grads={ga_conf.get('project_grads', True)}")
    trainer.train()

    trainer.save_model(training_conf['output_dir'])
    print(f"\nGradient Alignment training complete. Model saved to {training_conf['output_dir']}")


if __name__ == "__main__":
    main()
