
import sys
import os
import yaml
import torch
import json
import tqdm
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from transformers import AutoModelForCausalLM, AutoTokenizer
from environments.maze_env import MazeEnvironment
from environments.syllogism_env import SyllogismEnvironment
from environments.gsm8k import GSM8KEnvironment

@dataclass
class ModelConfig:
    name_or_path: str
    dtype: str = "auto"
    device_map: str = "auto"

@dataclass
class EnvironmentConfig:
    name: str
    params: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SFTConfig:
    output_file: str = "sft_data.jsonl"
    max_samples: Optional[int] = None
    think_start_token: str = "<think>"
    think_end_token: str = "</think>"
    answer_start_token: str = "<answer>"
    answer_end_token: str = "</answer>"
    system_prompt: str = "You are a helpful assistant. You are given a problem and its correct solution. Your task is to generate the step-by-step reasoning that leads to this solution. Output the reasoning inside {think_start}{think_end} tags and the final answer inside {answer_start}{answer_end} tags."

@dataclass
class DataGenConfig:
    model: ModelConfig
    environment: EnvironmentConfig
    sft: SFTConfig

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "DataGenConfig":
        model_cfg = ModelConfig(
            name_or_path=config_dict['model']['name_or_path'],
            dtype=config_dict['model'].get('torch_dtype', 'auto'),
            device_map=config_dict['model'].get('device_map', 'auto')
        )
        
        env_dict = config_dict.get('environment', {})
        # Fallback for legacy config format where dataset_name implies environment
        if not env_dict:
             if config_dict.get('data', {}).get('dataset_name') == 'openai/gsm8k':
                  env_dict = {'name': 'gsm8k'}
             else:
                  # Default to empty or unknown, caught later
                  pass
                  
        env_name = env_dict.get('name', 'unknown')
        env_params = {k: v for k, v in env_dict.items() if k != 'name'}
        # Also grab any other top-level keys that might be env params if needed, 
        # but let's stick to the structure for now.
        
        environment_cfg = EnvironmentConfig(name=env_name, params=env_params)
        # Note: Some environments might read top-level keys or other keys. 
        # The existing code passed 'config' (the whole dict) to the environment.
        # We should probably still pass the whole dict or ensure params has everything needed.
        # For now, let's keep passing the original config dict to environment classes 
        # until they are refactored to use EnvironmentConfig.
        # But we want to use the typed config here. 
        # Actually, let's just make sure we can reconstitute what's needed.
        
        sft_dict = config_dict.get('sft', {})
        sft_cfg = SFTConfig(
            output_file=sft_dict.get('output_file', 'sft_data.jsonl'),
            max_samples=sft_dict.get('max_samples'),
            think_start_token=sft_dict.get('think_start_token', '<think>'),
            think_end_token=sft_dict.get('think_end_token', '</think>'),
            answer_start_token=sft_dict.get('answer_start_token', '<answer>'),
            answer_end_token=sft_dict.get('answer_end_token', '</answer>'),
            system_prompt=sft_dict.get('system_prompt', "You are a helpful assistant. You are given a problem and its correct solution. Your task is to generate the step-by-step reasoning that leads to this solution. Output the reasoning inside {think_start}{think_end} tags and the final answer inside {answer_start}{answer_end} tags.")
        )

        return cls(model=model_cfg, environment=environment_cfg, sft=sft_cfg)


def load_config(config_path) -> Dict[str, Any]:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_environment(config: Dict[str, Any]) -> Any:
    # Need to maintain compat with classes taking dict
    env_config = config.get('environment', {})
    if not env_config:
        if config.get('data', {}).get('dataset_name') == 'openai/gsm8k':
            return GSM8KEnvironment(config)
            
    name = env_config.get('name')
    if name == 'gsm8k':
        return GSM8KEnvironment(config)
    elif name == 'maze':
        return MazeEnvironment(config)
    elif name == 'syllogism':
        return SyllogismEnvironment(config)
    else:
        raise ValueError(f"Unknown environment: {name}")

def main():
    if len(sys.argv) < 2:
        config_path = "config.yaml"
    else:
        config_path = sys.argv[1]

    print(f"Loading config from {config_path}")
    raw_config = load_config(config_path)
    # Parse into typed config
    config = DataGenConfig.from_dict(raw_config)

    # Initialize Environment
    # We pass raw_config to keep compatibility with environment classes that expect a dict
    env = get_environment(raw_config)
    dataset = env.get_dataset(raw_config)

    # Load Model
    print(f"Loading model: {config.model.name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(config.model.name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        config.model.name_or_path,
        torch_dtype=config.model.dtype,
        device_map=config.model.device_map
    )

    print(f"Generating SFT data to {config.sft.output_file}")

    results = []
    
    # We will limit the number of generations if specified, or do a subset
    max_samples = config.sft.max_samples if config.sft.max_samples is not None else len(dataset)
    
    # Create the generation prompt
    
    for i, item in tqdm.tqdm(enumerate(dataset), total=min(len(dataset), max_samples)):
        if i >= max_samples:
            break
            
        # item['prompt'] is a list of {'role':..., 'content':...}
        # item['answer'] is the ground truth
        
        # Extract the original user query
        user_content = next((msg['content'] for msg in item['prompt'] if msg['role'] == 'user'), None)
        system_content = next((msg['content'] for msg in item['prompt'] if msg['role'] == 'system'), None)
        
        ground_truth = item['answer']
        
        # Construct a meta-prompt to get the reasoning
        
        # Get tokens from config
        think_start = config.sft.think_start_token
        think_end = config.sft.think_end_token
        answer_start = config.sft.answer_start_token
        answer_end = config.sft.answer_end_token
        system_prompt_tmpl = config.sft.system_prompt
        
        # If the template contains format placeholders for tokens, format them.
        try:
             system_msg = system_prompt_tmpl.format(think_start=think_start, think_end=think_end, answer_start=answer_start, answer_end=answer_end)
        except KeyError:
             # Fallback if user prompt doesn't match format keys
             system_msg = system_prompt_tmpl

        meta_prompt = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": f"Problem:\n{user_content}\n\nCorrect Solution:\n{ground_truth}\n\nPlease explain the reasoning step-by-step to arrive at this solution."}
        ]
        
        text = tokenizer.apply_chat_template(meta_prompt, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        
        # Force the model to start with the think_start token
        think_ids = tokenizer(think_start, add_special_tokens=False).input_ids
        # Ensure it's on the same device
        think_tensor = torch.tensor([think_ids], device=model.device)
        
        # Concatenate
        current_ids = inputs.input_ids
        current_mask = inputs.attention_mask
        
        input_ids = torch.cat([current_ids, think_tensor], dim=1)
        # Extend attention mask
        attention_mask = torch.cat([current_mask, torch.ones((1, len(think_ids)), device=model.device)], dim=1)
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=1024,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.pad_token_id
            )
        
        # Decode only the NEW tokens
        generated_ids = outputs[0][input_ids.shape[1]:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # Clean up: stop at think_end if present
        if think_end in generated_text:
            thought_content = generated_text.split(think_end)[0]
        else:
            thought_content = generated_text
            
        # Construct the final Assistant message
        final_assistant_content = f"{think_start}{thought_content}{think_end}{answer_start}{ground_truth}{answer_end}"
        
        sft_example = {
            "messages": [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": final_assistant_content}
            ]
        }
        results.append(sft_example)
        
        # Periodically save
        if i % 10 == 0:
            os.makedirs(os.path.dirname(config.sft.output_file), exist_ok=True)
            with open(config.sft.output_file, 'w') as f:
                for line in results:
                    f.write(json.dumps(line) + "\n")

    # Final save
    os.makedirs(os.path.dirname(config.sft.output_file), exist_ok=True)
    with open(config.sft.output_file, 'w') as f:
        for line in results:
            f.write(json.dumps(line) + "\n")
            
    print(f"Finished. Saved {len(results)} examples to {config.sft.output_file}")

if __name__ == "__main__":
    main()
