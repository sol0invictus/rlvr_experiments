import sys
import argparse
import os
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer
from environments.reasoning_gym_env import ReasoningGymEnvironment
from callbacks.generation_logger import GenerationLoggingCallback
from callbacks.validation_callback import ValidationCallback

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_environment(config):
    env_config = config.get('environment', {})
    if not env_config:
        # Fallback for legacy config or direct dataset inference
        if config.get('data', {}).get('dataset_name') == 'openai/gsm8k':
            from environments.gsm8k import GSM8KEnvironment
            return GSM8KEnvironment(config)
        else:
            raise ValueError("Could not determine environment from config.")

    name = env_config.get('name')
    if name == 'gsm8k':
        from environments.gsm8k import GSM8KEnvironment
        return GSM8KEnvironment(config)
    elif name == 'reasoning_gym':
        return ReasoningGymEnvironment(config)
    else:
        raise ValueError(f"Unknown environment: {name}")


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
    env = get_environment(config)
    
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
    
    # Load tokenizer (needed for reward decoding)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Training Arguments
    print("Configuring training arguments...")
    training_conf = config['training']
    gen_conf = config.get('generation', {})

    # Allow CLI override of max_steps (for smoke testing)
    max_steps = args.max_steps if args.max_steps is not None else training_conf.get('max_steps', -1)

    num_generations = gen_conf.get('num_generations', 4)
    per_device_train_batch_size = training_conf.get('per_device_train_batch_size', 1)

    # generation_batch_size = num_generations * per_device_train_batch_size ensures
    # all rollouts for a micro-batch are produced in a single forward pass, avoiding
    # chunked generation that would interleave KV-cache evictions and reloads.
    generation_batch_size = num_generations * per_device_train_batch_size
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
        max_steps=max_steps,
        warmup_steps=training_conf.get('warmup_steps', 0),
        max_grad_norm=training_conf.get('max_grad_norm', 1.0),
        lr_scheduler_type=training_conf.get('lr_scheduler_type', 'cosine'),
        gradient_checkpointing=training_conf.get('gradient_checkpointing', False),
        dataloader_num_workers=training_conf.get('dataloader_num_workers', 4),
        # DDP: disable unused-parameter detection to avoid hangs when some
        # parameters don't receive a gradient on every step (e.g. embeddings).
        ddp_find_unused_parameters=False,
    )

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
    )

    # Train
    print("Starting training...")
    trainer.train()
    
    # Save Model
    trainer.save_model(config['training']['output_dir'])
    print(f"Model saved to {config['training']['output_dir']}")

if __name__ == "__main__":
    main()
