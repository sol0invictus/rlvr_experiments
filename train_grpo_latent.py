
import sys
import torch
import yaml
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer
from environments.gsm8k import GSM8KEnvironment
from environments.maze_env import MazeEnvironment
from environments.syllogism_env import SyllogismEnvironment
from environments.battleship_env import BattleshipEnvironment
from latent_qwen import LatentQwen2ForCausalLM

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_environment(config):
    env_config = config.get('environment', {})
    if not env_config:
        # Fallback for legacy config or direct dataset inference
        if config.get('data', {}).get('dataset_name') == 'openai/gsm8k':
            return GSM8KEnvironment(config)
        else:
            raise ValueError("Could not determine environment from config.")
            
    name = env_config.get('name')
    if name == 'gsm8k':
        return GSM8KEnvironment(config)
    elif name == 'maze':
        return MazeEnvironment(config)
    elif name == 'syllogism':
        return SyllogismEnvironment(config)
    elif name == 'battleship':
        return BattleshipEnvironment(config)
    else:
        raise ValueError(f"Unknown environment: {name}")

def main():
    if len(sys.argv) < 2:
        config_path = "configs/train_latent_gsm8k.yaml" 
    else:
        config_path = sys.argv[1]

    print(f"Loading config from {config_path}")
    config = load_config(config_path)

    # Initialize Environment
    print("Initializing Environment...")
    env = get_environment(config)
    
    # Load and Process Dataset
    dataset = env.get_dataset(config)
    
    # Reward Functions
    reward_funcs = env.get_reward_functions()
    
    # Load Tokenizer
    model_name = config['model']['name_or_path']
    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Add special tokens
    special_tokens = ["<think>", "</think>", "<answer>", "</answer>"]
    # Check if they exist, if not add them
    num_added_toks = tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    print(f"Added {num_added_toks} special tokens: {special_tokens}")
    
    # Prepare Model
    print(f"Loading LatentQwen2ForCausalLM: {model_name}")
    # We pass config overrides here
    
    # We load the base model weights into our class. 
    # Since LatentQwen2ForCausalLM inherits from Qwen2ForCausalLM, from_pretrained should work 
    # if we point to a Qwen2 checkpoint.
    # However, we need to pass `num_latent_thoughts` to the constructor config?
    # Or set it after.
    # `from_pretrained` passes kwargs to config.
    
    model = LatentQwen2ForCausalLM.from_pretrained(
        model_name,
        torch_dtype=config['model'].get('torch_dtype', 'auto'),
        device_map=config['model'].get('device_map', 'auto'),
        num_latent_thoughts=config['model'].get('num_latent_thoughts', 0)
    )
    
    # Resize embeddings if we added tokens
    if num_added_toks > 0:
        model.resize_token_embeddings(len(tokenizer))
        
    # Set the think token ID on the model instance
    think_token_id = tokenizer.convert_tokens_to_ids("<think>")
    close_think_id = tokenizer.convert_tokens_to_ids("</think>")
    answer_id = tokenizer.convert_tokens_to_ids("<answer>")
    
    model.set_special_token_ids(think_token_id)
    model.close_think_id = close_think_id
    model.answer_id = answer_id
    
    print(f"Set <think>: {think_token_id}, </think>: {close_think_id}, <answer>: {answer_id}")
    
    # Training Arguments
    print("Configuring training arguments...")
    training_conf = config['training']
    
    training_args = GRPOConfig(
        output_dir=training_conf['output_dir'],
        learning_rate=float(training_conf['learning_rate']),
        remove_unused_columns=False,
        gradient_accumulation_steps=training_conf.get('gradient_accumulation_steps', 1),
        num_train_epochs=training_conf.get('num_train_epochs', 1),
        bf16=training_conf.get('bf16', False),
        max_completion_length=config['generation']['max_completion_length'],
        num_generations=config['generation']['num_generations'],
        max_prompt_length=config['generation'].get('max_prompt_length', 128),
        report_to=training_conf.get('report_to', []),
        logging_steps=training_conf.get('logging_steps', 10),
        push_to_hub=training_conf.get('push_to_hub', False),
        save_strategy=training_conf.get('save_strategy', 'steps'),
        save_steps=training_conf.get('save_steps', 10),
        max_steps=training_conf.get('max_steps', -1),
    )

    # Trainer
    print("Initializing GRPOTrainer...")
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer, # Use updated processing_class arg name (newer trl) or tokenizer
    )

    # Train
    print("Starting training...")
    trainer.train()
    
    # Save Model
    trainer.save_model(config['training']['output_dir'])
    tokenizer.save_pretrained(config['training']['output_dir'])
    print(f"Model and tokenizer saved to {config['training']['output_dir']}")

if __name__ == "__main__":
    main()
