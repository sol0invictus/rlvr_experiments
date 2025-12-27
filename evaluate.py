import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from data_utils import load_config, load_and_process_dataset
from reward_utils import get_reward_functions
import tqdm

class Evaluator:
    def __init__(self, model_path, config, device_map="auto"):
        self.config = config
        self.device_map = device_map
        self.model_path = model_path
        
        print(f"Loading model: {model_path}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=config['model'].get('torch_dtype', 'auto'),
            device_map=device_map
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model.eval()

    def evaluate_dataset(self, dataset, max_samples=None):
        print("Starting evaluation...")
        reward_functions = get_reward_functions(self.config['data']['dataset_name'])
        
        results = {
            "format_compliance_sum": 0,
            "accuracy_sum": 0,
            "total": 0,
            "samples": []
        }
        
        if max_samples:
            dataset = dataset.select(range(min(len(dataset), max_samples)))
        
        for i, example in tqdm.tqdm(enumerate(dataset), total=len(dataset)):
            messages = example['prompt']
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs, 
                    max_new_tokens=self.config['generation']['max_completion_length'],
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            completion = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            
            formatted_completion = [[{"content": completion}]] 
            
            # Prepare kwargs for reward functions
            kwargs = {self.config['data']['answer_column']: [example[self.config['data']['answer_column']]]}
            
            # Calculate rewards
            fmt_score = 0
            acc_score = 0
            
            if len(reward_functions) > 0:
                try:
                   fmt_score = reward_functions[0](formatted_completion, **kwargs)[0]
                except Exception as e:
                   print(f"Error in format reward: {e}")
                   fmt_score = 0
            
            if len(reward_functions) > 1:
                try:
                   acc_score = reward_functions[1](formatted_completion, **kwargs)[0]
                except Exception as e:
                   print(f"Error in accuracy reward: {e}")
                   acc_score = 0
            
            results["format_compliance_sum"] += fmt_score
            results["accuracy_sum"] += acc_score
            results["total"] += 1
            
            results["samples"].append({
                "prompt": text,
                "completion": completion,
                "ground_truth": example[self.config['data']['answer_column']],
                "format_score": fmt_score,
                "accuracy_score": acc_score
            })
            
        metrics = {
            "accuracy": results["accuracy_sum"] / results["total"] if results["total"] > 0 else 0,
            "format_compliance": results["format_compliance_sum"] / results["total"] if results["total"] > 0 else 0,
            "total_samples": results["total"]
        }
        
        return metrics, results["samples"]

def run_evaluation(config_path, model_path=None, max_samples=None, split_override=None):
    if isinstance(config_path, str):
        config = load_config(config_path)
    else:
        config = config_path

    if model_path is None:
        model_path = config['model']['name_or_path']

    # Handle Split Override
    if split_override:
        config['data']['split'] = split_override
    elif config['data']['dataset_name'] == 'openai/gsm8k' and config['data'].get('split') == 'train':
        print("Note: Config specifies 'train' split. Using 'test' split for evaluation by default for GSM8K.")
        config['data']['split'] = 'test'

    dataset = load_and_process_dataset(config)
    
    evaluator = Evaluator(model_path, config)
    metrics, samples = evaluator.evaluate_dataset(dataset, max_samples=max_samples)
    
    print(f"Final Results: Accuracy: {metrics['accuracy']:.4f}, Format: {metrics['format_compliance']:.4f}")
    return metrics, samples

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python evaluate.py config.yaml [model_path]")
        sys.exit(1)
        
    config_path = sys.argv[1]
    model_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    run_evaluation(config_path, model_path)
