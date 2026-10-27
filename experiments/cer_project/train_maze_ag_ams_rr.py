"""
AG-AMS-RR: Advantage-Gated Magnitude Scaling with Reference Refresh.

Same as AG-AMS but at a specified step (default: 150), the KL reference model
is updated to the current model weights. This "resets the KL budget" and allows
the model to continue improving past the point where KL penalty would normally
constrain it.

Intuition: at step 150, the model has learned a better policy than the base model.
The KL penalty should prevent rapid change from THIS point, not from the original base.
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
from environments.gsm8k import GSM8KEnvironment


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


class RefreshCallback(TrainerCallback):
    """Refreshes the reference model at specified steps."""

    def __init__(self, refresh_steps, log_dir):
        self.refresh_steps = refresh_steps if isinstance(refresh_steps, list) else [refresh_steps]
        self.log_dir = log_dir
        self.refreshed_at = []
        os.makedirs(log_dir, exist_ok=True)

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        if state.global_step not in self.refresh_steps:
            return
        trainer = kwargs.get('model', None)
        # Access the trainer through the model's trainer reference
        # Actually we need to access it differently - we'll handle this in the trainer class
        pass


class AGAMSRRMetricsCallback(TrainerCallback):
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
        mean_kl = np.mean([s['kl'] for s in self.accumulated])

        out = {
            'step': step,
            'sft_loss': float(mean_sft),
            'rl_loss': float(mean_rl),
            'kl': float(mean_kl),
        }
        path = os.path.join(self.log_dir, f"rr_metrics_step_{step:06d}.json")
        with open(path, 'w') as f:
            json.dump(out, f, indent=2)

        print(f"\n[AG-AMS-RR] step={step} | sft={out['sft_loss']:.4f} rl={out['rl_loss']:.4f} | kl={out['kl']:.5f}")
        self.accumulated.clear()


class AGAMSRRTrainer(GRPOTrainer):
    """
    AG-AMS with Reference Refresh.

    Identical to AG-AMS but refreshes the KL reference model at specified steps.
    This allows continued improvement past the KL ceiling.
    """

    def __init__(self, *args, rr_config=None, rr_metrics_callback=None, tokenizer=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.rr_config = rr_config or {}
        self.rr_metrics_callback = rr_metrics_callback
        self.sft_weight = self.rr_config.get('sft_weight', 1.0)
        self.rl_weight = self.rr_config.get('rl_weight', 1.0)
        self.alpha = self.rr_config.get('alpha', 2.0)
        self.beta_ams = self.rr_config.get('beta_scale', 0.5)
        self.format_bonus = self.rr_config.get('format_bonus', 0.1)
        self.refresh_steps = self.rr_config.get('refresh_steps', [150])
        if isinstance(self.refresh_steps, int):
            self.refresh_steps = [self.refresh_steps]
        self._rr_tokenizer = tokenizer
        self._refreshed = set()

    def _maybe_refresh_ref(self):
        """Check if we should refresh the reference model at this step."""
        if not hasattr(self, 'state') or self.state is None:
            return
        step = self.state.global_step
        if step in self.refresh_steps and step not in self._refreshed:
            if self.ref_model is not None:
                print(f"\n{'='*50}")
                print(f"[AG-AMS-RR] REFRESHING reference model at step {step}")
                print(f"{'='*50}\n")
                # Copy current model weights to reference model
                with torch.no_grad():
                    for ref_param, model_param in zip(self.ref_model.parameters(), self.model.parameters()):
                        ref_param.data.copy_(model_param.data)
                self._refreshed.add(step)
            else:
                print(f"\n[AG-AMS-RR] WARNING: No ref_model to refresh at step {step} (beta=0 or PEFT?)")

    def _check_format(self, completion_ids):
        if self._rr_tokenizer is None:
            return torch.zeros(completion_ids.shape[0], device=completion_ids.device)
        has_format = []
        for i in range(completion_ids.shape[0]):
            text = self._rr_tokenizer.decode(completion_ids[i], skip_special_tokens=True)
            has_answer = '<answer>' in text and '</answer>' in text
            has_format.append(1.0 if has_answer else 0.0)
        return torch.tensor(has_format, device=completion_ids.device)

    def _compute_loss(self, model, inputs):
        # Check for reference refresh before computing loss
        self._maybe_refresh_ref()

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

        pos_adv = (adjusted_advantages > 0).float()
        neg_adv = (adjusted_advantages <= 0).float()
        denom = mask.sum(dim=1).clamp(min=1)

        # Magnitude scaling for SFT
        with torch.no_grad():
            pos_magnitudes = adjusted_advantages.abs().clamp(min=1e-6)
            sft_scale = (pos_magnitudes ** self.alpha) * pos_adv
            sft_scale_sum = sft_scale.sum()
            if sft_scale_sum > 1e-6:
                sft_scale = sft_scale * (pos_adv.sum() / sft_scale_sum)

        # Magnitude scaling for GRPO
        with torch.no_grad():
            neg_magnitudes = adjusted_advantages.abs().clamp(min=1e-6)
            rl_scale = (neg_magnitudes ** self.beta_ams) * neg_adv
            rl_scale_sum = rl_scale.sum()
            if rl_scale_sum > 1e-6:
                rl_scale = rl_scale * (neg_adv.sum() / rl_scale_sum)

        # SFT on positive-advantage completions
        sft_per_token = -per_token_logps * sft_scale * mask
        sft_loss = (sft_per_token.sum(dim=1) / denom).mean()

        # GRPO on negative-advantage completions
        log_ratio = per_token_logps - old_per_token_logps
        coef_1 = torch.exp(log_ratio)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
        grpo_per_token = -torch.min(coef_1 * adjusted_advantages, coef_2 * adjusted_advantages)
        rl_per_token = grpo_per_token * rl_scale * mask
        rl_loss = (rl_per_token.sum(dim=1) / denom).mean()

        loss = self.sft_weight * sft_loss + self.rl_weight * rl_loss

        # KL penalty
        kl_val = 0.0
        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            )
            kl_loss = ((per_token_kl * mask).sum(dim=1) / denom).mean()
            kl_val = kl_loss.item()
            loss = loss + self.beta * kl_loss

        mode = "train" if self.model.training else "eval"
        self._metrics[mode]["loss"].append(loss.detach().item())

        if self.rr_metrics_callback is not None:
            self.rr_metrics_callback.add_batch({
                'sft_loss': sft_loss.item(),
                'rl_loss': rl_loss.item(),
                'kl': kl_val,
            })

        return loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", nargs="?", default="configs/maze_ag_ams_rr.yaml")
    parser.add_argument("--max_steps", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)

    env_name = config.get('environment', {}).get('name', 'reasoning_gym') if isinstance(config.get('environment'), dict) else config.get('environment', 'reasoning_gym')
    if env_name == 'gsm8k':
        env = GSM8KEnvironment(config)
    else:
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

    rr_conf = config.get('ag_ams_rr', {})
    training_conf = config['training']

    rr_metrics_callback = AGAMSRRMetricsCallback(
        log_dir=os.path.join(training_conf['output_dir'], 'rr_metrics'),
        log_every_n_steps=rr_conf.get('log_every_n_steps', 5),
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

    print("Initializing AG-AMS-RR Trainer...")
    trainer = AGAMSRRTrainer(
        model=model,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        callbacks=[rr_metrics_callback],
        rr_config=rr_conf,
        rr_metrics_callback=rr_metrics_callback,
        tokenizer=tokenizer,
    )

    print(f"Starting AG-AMS-RR Training")
    print(f"  Model: {model_path}")
    print(f"  Steps: {max_steps}")
    print(f"  GPUs: {torch.cuda.device_count()}")
    print(f"  alpha={rr_conf.get('alpha', 2.0)}, beta_scale={rr_conf.get('beta_scale', 0.5)}")
    print(f"  refresh_steps={rr_conf.get('refresh_steps', [150])}")
    trainer.train()

    trainer.save_model(training_conf['output_dir'])
    print(f"\nTraining complete. Model saved to {training_conf['output_dir']}")


if __name__ == "__main__":
    main()
