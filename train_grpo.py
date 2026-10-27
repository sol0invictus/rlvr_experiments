import sys
import argparse
import os
import time
import torch
import transformers
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer
from peft import LoraConfig
from environments import load_environment
from callbacks.generation_logger import GenerationLoggingCallback
from callbacks.validation_callback import ValidationCallback

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def _warmup_kwargs(training_conf):
    warmup = training_conf.get('warmup_steps', training_conf.get('warmup_ratio', 0))
    if int(transformers.__version__.split('.')[0]) >= 5 or float(warmup).is_integer():
        return {'warmup_steps': int(warmup) if float(warmup).is_integer() else warmup}
    return {'warmup_ratio': float(warmup)}


def parse_args():
    parser = argparse.ArgumentParser(
        description="GRPO Training\n\n"
        "Single GPU:  python train_grpo.py configs/config.yaml\n"
        "Multi-GPU:   torchrun --nproc_per_node=8 train_grpo.py configs/config.yaml\n"
        "             accelerate launch --num_processes=8 train_grpo.py configs/config.yaml"
    )
    parser.add_argument("config", nargs="?", default="config.yaml", help="Path to YAML config")
    parser.add_argument("--max_steps", type=int, default=None, help="Override max_steps (for smoke testing)")
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = args.config

    print(f"Loading config from {config_path}")
    config = load_config(config_path)

    # Initialize Environment
    print("Initializing Environment...")
    env = load_environment(config)
    
    # Load and Process Dataset
    dataset = env.get_dataset(config)
    
    # Inject system_prompt into dataset rows if specified in config
    system_prompt = config.get('system_prompt')
    if system_prompt:
        system_prompt = system_prompt.strip()
        def add_system_prompt(example):
            prompt = example['prompt']
            # Only add if not already present
            if not any(m.get('role') == 'system' for m in prompt):
                prompt = [{'role': 'system', 'content': system_prompt}] + prompt
                example['prompt'] = prompt
            return example
        dataset = dataset.map(add_system_prompt)
    
    # Validation dataset (optional)
    val_conf = config.get('validation', {})
    val_dataset = None
    if val_conf.get('enabled', False):
        val_dataset = env.get_val_dataset(config)
        if val_dataset is not None:
            if system_prompt:
                val_dataset = val_dataset.map(add_system_prompt)
            print(f"[Validation] dataset ready: {len(val_dataset)} samples")
        else:
            print("[Validation] environment does not support get_val_dataset — skipping.")

    # Reward Functions
    reward_funcs = env.get_reward_functions()

    # Generation logging callback (optional)
    gen_log_conf = config.get('generation_logging', {})
    generation_logging_callback = None
    if gen_log_conf.get('enabled', False):
        log_dir = gen_log_conf.get(
            'log_dir',
            os.path.join(config['training']['output_dir'], 'generations'),
        )
        generation_logging_callback = GenerationLoggingCallback(
            log_dir=log_dir,
            every_n_steps=gen_log_conf.get('every_n_steps', 10),
            print_n_examples=gen_log_conf.get('print_n_examples', 2),
        )
        reward_funcs = generation_logging_callback.wrap_reward_funcs(reward_funcs)
        print(f"[GenerationLogger] enabled — saving to {log_dir}")
    
    # Load Model
    # NOTE: device_map must NOT be set for DDP — Accelerate places each replica
    # on its assigned GPU. Using device_map='auto' would enable pipeline/tensor
    # parallelism and break DDP.  Ignoring any device_map from the config here.
    model_path = config['model']['name_or_path']
    model_conf = config['model']
    print(f"Loading model: {model_path}")

    # Flash attention: set attn_implementation="flash_attention_2" in model config
    # to enable. Requires flash-attn package and a supported GPU (Ampere+).
    attn_implementation = model_conf.get('attn_implementation', None)
    if attn_implementation:
        print(f"  attn_implementation: {attn_implementation}")

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=config['model'].get('torch_dtype', 'auto'),
        **({"attn_implementation": attn_implementation} if attn_implementation else {}),
    )
    
    # Load tokenizer (needed for stop_strings and reward decoding)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Configure stop_strings on model's generation_config
    # This makes HF generate() stop as soon as </answer> is produced
    # Not used with vLLM rollouts: a stop string ends the sequence without EOS,
    # which TRL's vLLM path treats as truncated. Instruct models emit EOS after
    # </answer> on their own.
    #
    # vllm.stop_strings: true keeps the original recipe's stop at </answer> on the
    # vLLM path too (include_stop_str_in_output keeps "</answer>" in the text so the
    # reward regex still matches). Safe only with mask_truncated_completions off.
    vllm_conf = config.get('vllm', {})
    use_vllm = vllm_conf.get('enabled', False)
    stop_strings = config.get('generation', {}).get('stop_strings', ['</answer>'])
    generation_kwargs = None
    if not use_vllm:
        model.generation_config.stop_strings = stop_strings
        print(f"  stop_strings: {stop_strings}")
    elif vllm_conf.get('stop_strings', False):
        generation_kwargs = {"stop": stop_strings, "include_stop_str_in_output": True}
        print(f"  stop_strings (vLLM): {stop_strings}")
    
    # Training Arguments
    print("Configuring training arguments...")
    training_conf = config['training']
    gen_conf = config.get('generation', {})

    # Allow CLI override of max_steps (for smoke testing)
    max_steps = args.max_steps if args.max_steps is not None else training_conf.get('max_steps', -1)

    num_generations = gen_conf.get('num_generations', 4)
    per_device_train_batch_size = training_conf.get('per_device_train_batch_size', 1)

    # Default generation_batch_size = num_generations * per_device_train_batch_size
    # generates all rollouts for a micro-batch in one pass. Set
    # generation.generation_batch_size explicitly to make each optimizer step one
    # fully on-policy generation batch (generation_batch_size =
    # per_device_train_batch_size * gradient_accumulation_steps).
    generation_batch_size = gen_conf.get(
        'generation_batch_size', num_generations * per_device_train_batch_size
    )
    print(f"  generation_batch_size: {generation_batch_size} "
          f"(num_generations={num_generations} × per_device_bs={per_device_train_batch_size})")

    training_args = GRPOConfig(
        output_dir=training_conf['output_dir'],
        learning_rate=float(training_conf['learning_rate']),
        loss_type=training_conf.get('loss_type', 'grpo'),
        beta=training_conf.get('beta', 0.04),
        remove_unused_columns=False,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=training_conf.get('gradient_accumulation_steps', 1),
        num_train_epochs=training_conf.get('num_train_epochs', 1),
        bf16=training_conf.get('bf16', False),
        max_completion_length=gen_conf.get('max_completion_length', 1024),
        num_generations=num_generations,
        generation_batch_size=generation_batch_size,
        temperature=gen_conf.get('temperature', 0.7),
        report_to=training_conf.get('report_to', []),
        logging_steps=training_conf.get('logging_steps', 10),
        push_to_hub=training_conf.get('push_to_hub', False),
        save_strategy=training_conf.get('save_strategy', 'steps'),
        save_steps=training_conf.get('save_steps', 10),
        save_total_limit=training_conf.get('save_total_limit', None),
        save_only_model=training_conf.get('save_only_model', False),
        max_steps=max_steps,
        # transformers 5 dropped warmup_ratio; warmup_steps < 1 is read as a ratio.
        # transformers 4.x only accepts an int warmup_steps, so pass a ratio separately.
        **_warmup_kwargs(training_conf),
        weight_decay=training_conf.get('weight_decay', 0.0),
        scale_rewards=training_conf.get('scale_rewards', 'group'),
        mask_truncated_completions=training_conf.get('mask_truncated_completions', False),
        epsilon=training_conf.get('epsilon', 0.2),
        epsilon_high=training_conf.get('epsilon_high', None),
        reward_weights=training_conf.get('reward_weights', None),
        seed=config.get('seed', 42),
        max_grad_norm=training_conf.get('max_grad_norm', 1.0),
        lr_scheduler_type=training_conf.get('lr_scheduler_type', 'cosine'),
        gradient_checkpointing=training_conf.get('gradient_checkpointing', False),
        use_liger_kernel=training_conf.get('use_liger_kernel', False),
        dataloader_num_workers=training_conf.get('dataloader_num_workers', 4),
        generation_kwargs=generation_kwargs,
        # DDP: disable unused-parameter detection to avoid hangs when some
        # parameters don't receive a gradient on every step (e.g. embeddings).
        ddp_find_unused_parameters=False,
        num_iterations=training_conf.get('num_iterations', 1),
        # vLLM rollouts (colocate = same GPU as training; sleep mode frees the
        # vLLM KV cache during the backward pass)
        use_vllm=use_vllm,
        **({
            'vllm_mode': config['vllm'].get('mode', 'colocate'),
            'vllm_gpu_memory_utilization': config['vllm'].get('gpu_memory_utilization', 0.3),
            'vllm_enable_sleep_mode': config['vllm'].get('enable_sleep_mode', False),
            'vllm_max_model_length': config['vllm'].get('max_model_length', None),
            # TRL default is True (truncated importance-sampling correction). The
            # HF-generate runs had no sampler/trainer mismatch, so replication
            # configs set it False.
            **({'vllm_importance_sampling_correction': config['vllm']['importance_sampling_correction']}
               if 'importance_sampling_correction' in config['vllm'] else {}),
        } if use_vllm else {}),
    )

    # LoRA (optional). Recommended for RL: all-linear targets, low rank,
    # ~10x the full-FT learning rate (see experiments/countdown_lora/config.yaml).
    lora_conf = config.get('lora', {})
    peft_config = None
    if lora_conf.get('enabled', False):
        peft_config = LoraConfig(
            r=lora_conf.get('r', 16),
            lora_alpha=lora_conf.get('alpha', 32),
            lora_dropout=lora_conf.get('dropout', 0.0),
            target_modules=lora_conf.get('target_modules', 'all-linear'),
            use_dora=lora_conf.get('use_dora', False),
            task_type="CAUSAL_LM",
        )
        print(f"[LoRA] r={peft_config.r} alpha={peft_config.lora_alpha} "
              f"targets={peft_config.target_modules} dora={peft_config.use_dora}")

    # Validation callback (optional)
    validation_callback = None
    if val_dataset is not None:
        val_log_dir = val_conf.get(
            'log_dir',
            os.path.join(training_conf['output_dir'], 'validation'),
        )
        validation_callback = ValidationCallback(
            val_dataset=val_dataset,
            reward_funcs=env.get_reward_functions(),
            tokenizer=tokenizer,
            eval_steps=val_conf.get('eval_steps', 100),
            num_samples=val_conf.get('num_samples', 64),
            max_new_tokens=gen_conf.get('max_completion_length', 512),
            log_dir=val_log_dir,
            temperature=val_conf.get('temperature', 0.0),
            batch_size=val_conf.get('batch_size', 16),
        )
        print(
            f"[Validation] callback ready — every {val_conf.get('eval_steps', 100)} steps, "
            f"{val_conf.get('num_samples', 64)} samples, logs → {val_log_dir}"
        )

    # Trainer
    print("Initializing GRPOTrainer...")
    extra_callbacks = [
        cb for cb in [generation_logging_callback, validation_callback] if cb is not None
    ]
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        callbacks=extra_callbacks,
        peft_config=peft_config,
    )
    if peft_config is not None:
        trainer.model.print_trainable_parameters()

    # Patch trainer's generate to inject tokenizer for stop_strings support
    # TRL's GRPOTrainer.generate() doesn't pass tokenizer= to model.generate(),
    # but transformers requires it when stop_strings are in generation_config
    # generation.pad_after_stop: "eos" (default) writes EOS after a stop string (below);
    # "strip" leaves the pads for the wrapper further down to drop.
    pad_after_stop = gen_conf.get('pad_after_stop', 'eos')
    _orig_generate_single = trainer._generate_single_turn
    def _patched_generate_single(*args, **kwargs):
        unwrapped = trainer.accelerator.unwrap_model(trainer.model)
        _orig_gen = unwrapped.generate
        def _gen_with_tok(*args, **kwargs):
            kwargs.setdefault('tokenizer', tokenizer)
            out = _orig_gen(*args, **kwargs)
            # A stop_string ends the sequence without emitting EOS; generate()
            # then right-fills with pad. When pad != eos (true for Qwen
            # Instruct models), TRL's completion mask never closes and the loss
            # trains on every pad token. Write EOS into the first pad slot of
            # each finished completion so the mask ends there.
            if (pad_after_stop == 'eos' and tokenizer.pad_token_id != tokenizer.eos_token_id
                    and 'input_ids' in kwargs):
                completion = out[:, kwargs['input_ids'].shape[1]:]
                is_pad = completion == tokenizer.pad_token_id
                needs_eos = is_pad.any(dim=1) & ~(completion == tokenizer.eos_token_id).any(dim=1)
                rows = needs_eos.nonzero(as_tuple=True)[0]
                completion[rows, is_pad[rows].int().argmax(dim=1)] = tokenizer.eos_token_id
            return out
        unwrapped.generate = _gen_with_tok
        try:
            result = _orig_generate_single(*args, **kwargs)
        finally:
            unwrapped.generate = _orig_gen
        return result
    if not use_vllm:
        trainer._generate_single_turn = _patched_generate_single

    # pad_after_stop "strip": drop the trailing pad tokens instead, so a completion
    # that stopped at </answer> carries no EOS -- what the original 8-GPU runs had (one
    # completion per GPU, so no padding at all). The replication runs used "strip".
    pad_id, eos_id = tokenizer.pad_token_id, tokenizer.eos_token_id
    _inner_generate_single = trainer._generate_single_turn

    def _strip_trailing_pad(ids):
        end = len(ids)
        while end > 0 and ids[end - 1] == pad_id:
            end -= 1
        return ids[:end]

    def _timed_generate_single(*args, **kwargs):
        t0 = time.perf_counter()
        result = _inner_generate_single(*args, **kwargs)
        mode = "train" if trainer.model.training else "eval"
        trainer._metrics[mode]["timing/generate_s"].append(time.perf_counter() - t0)
        if use_vllm or pad_after_stop != 'strip' or pad_id is None or pad_id == eos_id:
            return result
        # TRL >=0.29 returns (completion_ids, logprobs); <=0.28 returns
        # (prompt_ids, completion_ids, logprobs, extra_fields).
        result = list(result)
        idx = 0 if len(result) == 2 else 1
        result[idx] = [_strip_trailing_pad(ids) for ids in result[idx]]
        return tuple(result)
    trainer._generate_single_turn = _timed_generate_single

    # Train
    print("Starting training...")
    trainer.train()
    
    # Save Model
    trainer.save_model(config['training']['output_dir'])
    print(f"Model saved to {config['training']['output_dir']}")

if __name__ == "__main__":
    main()
