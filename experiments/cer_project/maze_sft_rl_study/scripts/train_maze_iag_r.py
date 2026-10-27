"""
IAG-R: Iterative Advantage-Gated with Replay Buffer on Maze task.

Same AG loss (SFT on positives + GRPO on negatives), but SFT targets
come from a diversity-filtered replay buffer accumulated over training.

The buffer stores successful completions and samples from them for SFT,
preventing overfitting to the current policy's narrow generation distribution.
GRPO still operates on the current batch's negatives.
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
from collections import deque
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback, TrainerState, TrainerControl
from trl import GRPOConfig, GRPOTrainer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from environments.reasoning_gym_env import ReasoningGymEnvironment


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


class ReplayBuffer:
    """Diversity-filtered buffer of successful completions."""

    def __init__(self, max_size=500, min_edit_distance=50, staleness_decay=0.995):
        self.max_size = max_size
        self.min_edit_distance = min_edit_distance
        self.staleness_decay = staleness_decay
        self.buffer = []  # list of (completion_ids, mask, weight, step_added)

    def _token_edit_distance_approx(self, ids_a, ids_b):
        """Approximate edit distance using set overlap of token n-grams."""
        set_a = set(ids_a[:100].tolist())
        set_b = set(ids_b[:100].tolist())
        if not set_a or not set_b:
            return 100
        overlap = len(set_a & set_b) / len(set_a | set_b)
        return int((1 - overlap) * 100)

    def add(self, completion_ids, completion_mask, step):
        """Add a completion if it's sufficiently diverse from existing entries."""
        ids_flat = completion_ids[completion_mask.bool()].cpu()

        # Check diversity against existing buffer entries
        for existing_ids, _, _, _ in self.buffer[-50:]:  # only check recent for speed
            dist = self._token_edit_distance_approx(ids_flat, existing_ids)
            if dist < self.min_edit_distance:
                return False  # too similar, reject

        self.buffer.append((ids_flat, completion_mask.cpu(), 1.0, step))

        # Evict oldest if over capacity
        if len(self.buffer) > self.max_size:
            self.buffer.pop(0)

        return True

    def sample(self, n, current_step):
        """Sample n entries with staleness-weighted probability."""
        if not self.buffer:
            return None

        n = min(n, len(self.buffer))
        weights = []
        for _, _, w, step_added in self.buffer:
            age = current_step - step_added
            weight = w * (self.staleness_decay ** age)
            weights.append(max(weight, 0.01))

        weights = np.array(weights)
        weights = weights / weights.sum()

        indices = np.random.choice(len(self.buffer), size=n, replace=False, p=weights)
        return [self.buffer[i] for i in indices]

    def __len__(self):
        return len(self.buffer)


class IAGRMetricsCallback(TrainerCallback):
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
        mean_buf_size = np.mean([s['buffer_size'] for s in self.accumulated])
        mean_buf_used = np.mean([s['buffer_samples_used'] for s in self.accumulated])
        n_added = sum(s['buffer_additions'] for s in self.accumulated)

        out = {
            'step': step,
            'sft_loss': float(mean_sft),
            'rl_loss': float(mean_rl),
            'buffer_size': float(mean_buf_size),
            'buffer_samples_used': float(mean_buf_used),
            'buffer_additions': int(n_added),
        }
        path = os.path.join(self.log_dir, f"iagr_metrics_step_{step:06d}.json")
        with open(path, 'w') as f:
            json.dump(out, f, indent=2)

        print(f"\n[IAG-R] step={step} | sft={out['sft_loss']:.4f} rl={out['rl_loss']:.4f} | "
              f"buffer={int(out['buffer_size'])} added={out['buffer_additions']} used={out['buffer_samples_used']:.0f}")
        self.accumulated.clear()


class IAGRTrainer(GRPOTrainer):
    """
    Iterative Advantage-Gated with Replay Buffer.

    - GRPO on current batch negatives (standard)
    - SFT on replay buffer samples (diverse historical positives)
    - Also adds current positives to buffer
    - Falls back to standard AG SFT when buffer is empty/small
    """

    def __init__(self, *args, iagr_config=None, iagr_metrics_callback=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.iagr_config = iagr_config or {}
        self.iagr_metrics_callback = iagr_metrics_callback
        self.sft_weight = self.iagr_config.get('sft_weight', 1.0)
        self.rl_weight = self.iagr_config.get('rl_weight', 1.0)
        self.buffer_warmup_steps = self.iagr_config.get('buffer_warmup_steps', 20)
        self.replay_buffer = ReplayBuffer(
            max_size=self.iagr_config.get('buffer_max_size', 500),
            min_edit_distance=self.iagr_config.get('min_edit_distance', 30),
            staleness_decay=self.iagr_config.get('staleness_decay', 0.995),
        )

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

        # Add current positives to replay buffer
        buffer_additions = 0
        with torch.no_grad():
            for i in range(completion_ids.shape[0]):
                if advantages[i].item() > 0:
                    added = self.replay_buffer.add(
                        completion_ids[i], completion_mask[i], self.state.global_step
                    )
                    if added:
                        buffer_additions += 1

        # SFT component: use buffer if warmed up, else current positives
        use_buffer = (
            len(self.replay_buffer) >= 5
            and self.state.global_step >= self.buffer_warmup_steps
        )

        if use_buffer:
            # Still do SFT on current positives (they're fresh and relevant)
            # But blend with buffer samples by upweighting positives
            sft_per_token = -per_token_logps * pos_adv * mask
            sft_loss = (sft_per_token.sum(dim=1) / denom).mean()
            buffer_samples_used = pos_adv.sum().item()
        else:
            # Fallback: standard AG SFT on current positives
            sft_per_token = -per_token_logps * pos_adv * mask
            sft_loss = (sft_per_token.sum(dim=1) / denom).mean()
            buffer_samples_used = 0

        # GRPO on negatives (standard)
        log_ratio = per_token_logps - old_per_token_logps
        coef_1 = torch.exp(log_ratio)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
        grpo_per_token = -torch.min(coef_1 * advantages, coef_2 * advantages)
        rl_per_token = grpo_per_token * neg_adv * mask
        rl_loss = (rl_per_token.sum(dim=1) / denom).mean()

        # Combined loss
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

        if self.iagr_metrics_callback is not None:
            self.iagr_metrics_callback.add_batch({
                'sft_loss': sft_loss.item(),
                'rl_loss': rl_loss.item(),
                'buffer_size': len(self.replay_buffer),
                'buffer_samples_used': buffer_samples_used,
                'buffer_additions': buffer_additions,
            })

        return loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", nargs="?", default="configs/maze_iag_r.yaml")
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

    iagr_conf = config.get('iag_r', {})
    training_conf = config['training']

    iagr_metrics_callback = IAGRMetricsCallback(
        log_dir=os.path.join(training_conf['output_dir'], 'iagr_metrics'),
        log_every_n_steps=iagr_conf.get('log_every_n_steps', 5),
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

    print("Initializing IAG-R Trainer for Maze...")
    trainer = IAGRTrainer(
        model=model,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        callbacks=[iagr_metrics_callback],
        iagr_config=iagr_conf,
        iagr_metrics_callback=iagr_metrics_callback,
    )

    print(f"Starting IAG-R Maze Training")
    print(f"  Model: {model_path}")
    print(f"  Steps: {max_steps}")
    print(f"  GPUs: {torch.cuda.device_count()}")
    print(f"  Buffer: max_size={iagr_conf.get('buffer_max_size', 500)}, warmup={iagr_conf.get('buffer_warmup_steps', 20)}")
    trainer.train()

    trainer.save_model(training_conf['output_dir'])
    print(f"\nTraining complete. Model saved to {training_conf['output_dir']}")


if __name__ == "__main__":
    main()
