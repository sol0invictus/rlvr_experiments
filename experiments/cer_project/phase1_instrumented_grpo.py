"""
Phase 1: Instrumented GRPO Training (TRL 1.3.0 compatible)

Runs standard GRPO on GSM8K with Qwen2.5-1.5B-Instruct, logging per-token
entropy and gradient-proxy signals to validate the inverted-U hypothesis.

Hooks into GRPOTrainer._get_per_token_logps_and_entropies to capture data
without reimplementing any loss logic.
"""

import sys
import os
import json
import argparse
import torch
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


def get_environment(config):
    env_config = config.get('environment', {})
    name = env_config.get('name', '')
    if name == 'gsm8k' or config.get('data', {}).get('dataset_name') == 'openai/gsm8k':
        return GSM8KEnvironment(config)
    else:
        raise ValueError(f"Unknown environment: {name}")


def load_dataset_local_or_hub(config):
    """Load dataset from local arrow cache if available, else HF Hub."""
    from datasets import load_from_disk, load_dataset
    data_conf = config['data']
    local_path = data_conf.get('local_train_path')
    if local_path and os.path.exists(local_path):
        print(f"Loading dataset from local cache: {local_path}")
        return load_from_disk(local_path)
    print(f"Loading dataset from HF Hub: {data_conf['dataset_name']}")
    return load_dataset(data_conf['dataset_name'], data_conf.get('subset'), split=data_conf.get('split', 'train'))


def load_val_dataset_local_or_hub(config):
    """Load validation dataset from local arrow cache if available."""
    from datasets import load_from_disk, load_dataset
    data_conf = config['data']
    local_path = data_conf.get('local_test_path')
    if local_path and os.path.exists(local_path):
        print(f"Loading val dataset from local cache: {local_path}")
        return load_from_disk(local_path)
    print(f"Loading val dataset from HF Hub: {data_conf['dataset_name']}")
    return load_dataset(data_conf['dataset_name'], data_conf.get('subset'), split='test')


class TokenDiagnosticsCallback(TrainerCallback):
    """Accumulates per-token entropy and gradient proxy data, writes binned summaries."""

    def __init__(self, log_dir: str, log_every_n_steps: int = 5, num_bins: int = 20):
        self.log_dir = log_dir
        self.log_every_n_steps = log_every_n_steps
        self.num_bins = num_bins
        self.accumulated_data = []
        os.makedirs(log_dir, exist_ok=True)

    def add_batch(self, entropies, grad_proxies, advantages_abs):
        self.accumulated_data.append({
            'entropies': entropies,
            'grad_proxies': grad_proxies,
            'advantages_abs': advantages_abs,
        })

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        if state.global_step % self.log_every_n_steps != 0:
            return
        if not self.accumulated_data:
            return
        self._flush(state.global_step)

    def on_train_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        if self.accumulated_data:
            self._flush(state.global_step, final=True)

    def _flush(self, step, final=False):
        all_entropies = []
        all_grad_proxies = []
        all_advantages = []
        for rec in self.accumulated_data:
            all_entropies.extend(rec['entropies'])
            all_grad_proxies.extend(rec['grad_proxies'])
            all_advantages.extend(rec['advantages_abs'])

        if not all_entropies:
            self.accumulated_data.clear()
            return

        ent_arr = np.array(all_entropies)
        gp_arr = np.array(all_grad_proxies)
        adv_arr = np.array(all_advantages)

        p99 = np.percentile(ent_arr, 99)
        bin_edges = np.linspace(0, p99 + 1e-8, self.num_bins + 1)
        bin_indices = np.digitize(ent_arr, bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, self.num_bins - 1)

        bin_stats = []
        for b in range(self.num_bins):
            mask = bin_indices == b
            if mask.sum() > 0:
                bin_stats.append({
                    'bin': b,
                    'entropy_lo': float(bin_edges[b]),
                    'entropy_hi': float(bin_edges[b+1]),
                    'count': int(mask.sum()),
                    'mean_grad_proxy': float(gp_arr[mask].mean()),
                    'std_grad_proxy': float(gp_arr[mask].std()),
                    'median_grad_proxy': float(np.median(gp_arr[mask])),
                    'mean_entropy': float(ent_arr[mask].mean()),
                    'mean_advantage': float(adv_arr[mask].mean()),
                })

        tag = "final" if final else f"step_{step:06d}"
        summary_path = os.path.join(self.log_dir, f"binned_{tag}.json")
        with open(summary_path, 'w') as f:
            json.dump({
                'step': step,
                'total_tokens': len(all_entropies),
                'mean_entropy': float(ent_arr.mean()),
                'mean_grad_proxy': float(gp_arr.mean()),
                'bins': bin_stats,
            }, f, indent=2)

        print(f"\n[TokenDiagnostics] step={step} | "
              f"{len(all_entropies)} tokens | "
              f"ent=[{ent_arr.min():.2f}, {ent_arr.mean():.2f}, {ent_arr.max():.2f}] | "
              f"grad_proxy mean={gp_arr.mean():.5f}")

        self.accumulated_data.clear()


class InstrumentedGRPOTrainer(GRPOTrainer):
    """
    GRPOTrainer that captures per-token entropy × advantage to validate
    the inverted-U hypothesis. Does NOT modify loss computation.
    """

    def __init__(self, *args, diagnostics_callback=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.diagnostics_callback = diagnostics_callback
        self._capture_next = True

    def _compute_loss(self, model, inputs):
        """Standard loss, plus diagnostic capture."""
        # Capture diagnostics before calling super
        if self.diagnostics_callback is not None and self._capture_next:
            self._capture_diagnostics(model, inputs)

        # Call parent's _compute_loss unchanged
        return super()._compute_loss(model, inputs)

    @torch.no_grad()
    def _capture_diagnostics(self, model, inputs):
        """Extract per-token entropy and grad proxy from completion tokens."""
        try:
            prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
            completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
            input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
            attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
            logits_to_keep = completion_ids.size(1)

            per_token_logps, entropies = self._get_per_token_logps_and_entropies(
                model, input_ids, attention_mask, logits_to_keep, compute_entropy=True,
            )

            advantages = inputs["advantages"]
            if advantages.dim() == 1:
                advantages = advantages.unsqueeze(1)

            mask = completion_mask
            # grad_proxy ≈ |log_prob_t| * |advantage| — proxy for how much this token
            # would contribute to the GRPO gradient update
            grad_proxy = per_token_logps.abs() * advantages.abs()

            batch_entropies = []
            batch_grad_proxies = []
            batch_advantages = []

            for i in range(entropies.shape[0]):
                token_mask = mask[i].bool()
                if token_mask.sum() == 0:
                    continue
                ent_vals = entropies[i][token_mask].cpu().float().numpy().tolist()
                gp_vals = grad_proxy[i][token_mask].cpu().float().numpy().tolist()
                adv_val = float(advantages[i].abs().mean().cpu())
                batch_entropies.extend(ent_vals)
                batch_grad_proxies.extend(gp_vals)
                batch_advantages.extend([adv_val] * len(ent_vals))

            if batch_entropies:
                self.diagnostics_callback.add_batch(batch_entropies, batch_grad_proxies, batch_advantages)

        except Exception as e:
            print(f"[Diagnostics] Warning: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", nargs="?", default="configs/phase1_qwen15b_gsm8k.yaml")
    parser.add_argument("--max_steps", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    env = get_environment(config)

    # Load dataset: prefer local cache to avoid HF rate limits
    dataset = load_dataset_local_or_hub(config)

    # Format as conversations for GRPO
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

    instr_conf = config.get('instrumentation', {})
    diagnostics_callback = TokenDiagnosticsCallback(
        log_dir=instr_conf.get('log_dir', os.path.join(config['training']['output_dir'], 'token_diagnostics')),
        log_every_n_steps=instr_conf.get('log_every_n_steps', 5),
        num_bins=instr_conf.get('num_entropy_bins', 20),
    )

    model_path = config['model']['name_or_path']
    model_conf = config['model']
    print(f"Loading model: {model_path}")

    attn_implementation = model_conf.get('attn_implementation', None)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=config['model'].get('torch_dtype', 'auto'),
        **({"attn_implementation": attn_implementation} if attn_implementation else {}),
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Add </answer> token as EOS for generation stopping
    # TRL 1.3.0 doesn't pass tokenizer to model.generate(), so stop_strings won't work
    answer_end_ids = tokenizer.encode("</answer>", add_special_tokens=False)
    if answer_end_ids:
        eos_ids = model.generation_config.eos_token_id
        if isinstance(eos_ids, int):
            eos_ids = [eos_ids]
        elif eos_ids is None:
            eos_ids = []
        eos_ids = list(set(eos_ids + answer_end_ids))
        model.generation_config.eos_token_id = eos_ids
        print(f"  EOS token IDs: {eos_ids}")

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

    print("Initializing InstrumentedGRPOTrainer...")
    extra_callbacks = [cb for cb in [generation_logging_callback, validation_callback, diagnostics_callback] if cb is not None]

    trainer = InstrumentedGRPOTrainer(
        model=model,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        callbacks=extra_callbacks,
        diagnostics_callback=diagnostics_callback,
    )

    print(f"Starting Phase 1: Instrumented GRPO")
    print(f"  Model: {model_path}")
    print(f"  Steps: {max_steps}")
    print(f"  GPUs: {torch.cuda.device_count()}")
    print(f"  Diagnostics: every {instr_conf.get('log_every_n_steps', 5)} steps → {instr_conf.get('log_dir')}")
    trainer.train()

    trainer.save_model(training_conf['output_dir'])
    print(f"\nPhase 1 complete. Model saved to {training_conf['output_dir']}")


if __name__ == "__main__":
    main()
