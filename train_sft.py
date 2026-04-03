import os
import sys
import argparse
import yaml
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer
from datasets import load_dataset


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def parse_args():
    parser = argparse.ArgumentParser(
        description="SFT Training\n\n"
        "Single GPU:  python train_sft.py configs/sft_maze.yaml\n"
        "Multi-GPU:   torchrun --nproc_per_node=8 train_sft.py configs/sft_maze.yaml\n"
        "             accelerate launch --num_processes=8 train_sft.py configs/sft_maze.yaml\n"
        "Multi-node:  torchrun --nnodes=N --nproc_per_node=8 train_sft.py configs/sft_maze.yaml"
    )
    parser.add_argument("config", nargs="?", default="config.yaml", help="Path to YAML config")
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = args.config

    print(f"Loading config from {config_path}")
    config = load_config(config_path)

    # Load SFT data
    sft_file = config.get('sft', {}).get('output_file', 'sft_llm_data.json')
    if not os.path.exists(sft_file) and os.path.exists('sft_data.jsonl'):
        sft_file = 'sft_data.jsonl'

    print(f"Loading SFT data from {sft_file}")
    dataset = load_dataset('json', data_files=sft_file, split='train')

    # Map raw columns to "messages" format if needed
    if "system_prompt" in dataset.column_names and "question" in dataset.column_names:
        def format_to_messages(example):
            return {
                "messages": [
                    {"role": "system",    "content": example["system_prompt"]},
                    {"role": "user",      "content": example["question"]},
                    {"role": "assistant", "content": example["response"]},
                ]
            }
        dataset = dataset.map(format_to_messages)

    # Model
    model_path = config['model']['name_or_path']
    print(f"Loading model: {model_path}")

    train_args_conf = config.get('sft_training', config.get('training', {}))
    sft_conf        = config.get('sft', {})
    output_dir      = train_args_conf.get('output_dir', sft_conf.get('output_dir', 'sft_output'))

    training_args = SFTConfig(
        output_dir=output_dir,
        learning_rate=float(train_args_conf.get('learning_rate', 2e-5)),
        num_train_epochs=train_args_conf.get('num_train_epochs', 1),
        bf16=train_args_conf.get('bf16', False),
        logging_steps=train_args_conf.get('logging_steps', 10),
        per_device_train_batch_size=train_args_conf.get('batch_size', 4),
        gradient_accumulation_steps=train_args_conf.get('gradient_accumulation_steps', 1),
        save_strategy=train_args_conf.get('save_strategy', 'steps'),
        save_steps=train_args_conf.get('save_steps', 50),
        report_to=train_args_conf.get('report_to', []),
        # DDP: disable unused-parameter detection to avoid hangs when some
        # parameters don't receive a gradient on every step (e.g. embeddings).
        ddp_find_unused_parameters=False,
    )

    # NOTE: device_map must NOT be set for DDP/multi-node — Accelerate places
    # each process on its assigned GPU. device_map='auto' enables pipeline/tensor
    # parallelism and breaks DDP. Ignoring any device_map from the config here.
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=config['model'].get('torch_dtype', 'auto'),
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Initializing SFTTrainer...")
    # SFTTrainer automatically masks the user prompt when using 'messages' format.
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        processing_class=tokenizer,
    )

    print("Starting SFT...")
    trainer.train()

    print(f"Saving model to {output_dir}")
    trainer.save_model(output_dir)


if __name__ == "__main__":
    main()
