import yaml
from datasets import load_dataset

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_system_prompt(config):
    return config.get('system_prompt', "")

def make_conversation(example, prompt_column, system_prompt):
    return {
        "prompt": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": example[prompt_column]},
        ],
    }

def load_and_process_dataset(config):
    print(f"Loading dataset: {config['data']['dataset_name']}")
    dataset = load_dataset(
        config['data']['dataset_name'], 
        config['data'].get('subset', None),
        split=config['data'].get('split', 'train')
    )
    
    system_prompt = get_system_prompt(config)
    prompt_column = config['data']['prompt_column']
    
    # Use partial or lambda to pass extra args to map
    dataset = dataset.map(lambda x: make_conversation(x, prompt_column, system_prompt))
    return dataset
