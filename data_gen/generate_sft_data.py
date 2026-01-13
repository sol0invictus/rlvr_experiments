
import sys
import yaml
import torch
import json
import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from environments.maze_env import MazeEnvironment
from environments.syllogism_env import SyllogismEnvironment
from environments.gsm8k import GSM8KEnvironment

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_environment(config):
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
    config = load_config(config_path)

    # Initialize Environment
    env = get_environment(config)
    dataset = env.get_dataset(config)

    # Load Model
    model_path = config['model']['name_or_path']
    print(f"Loading model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=config['model'].get('torch_dtype', 'auto'),
        device_map="auto"
    )

    output_file = config.get('sft', {}).get('output_file', 'sft_data.jsonl')
    print(f"Generating SFT data to {output_file}")

    results = []
    
    # We will limit the number of generations if specified, or do a subset
    max_samples = config.get('sft', {}).get('max_samples', len(dataset))
    
    # Create the generation prompt
    # We want to ask the model to explain the solution.
    
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
        # We present the problem and solution, and ask for the thought process.
        
        # Get tokens from config
        sft_conf = config.get('sft', {})
        think_start = sft_conf.get('think_start_token', '<think>')
        think_end = sft_conf.get('think_end_token', '</think>')
        answer_start = sft_conf.get('answer_start_token', '<answer>')
        answer_end = sft_conf.get('answer_end_token', '</answer>')
        system_prompt_tmpl = sft_conf.get('system_prompt', "You are a helpful assistant. You are given a problem and its correct solution. Your task is to generate the step-by-step reasoning that leads to this solution. Output the reasoning inside {think_start}{think_end} tags and the final answer inside {answer_start}{answer_end} tags.")
        
        # If the template contains format placeholders for tokens, format them.
        # Otherwise, assume user provided a fixed string or handles it.
        # But we want to support dynamic tokens in the prompt if user requested.
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
        # This encourages it to follow the instruction immediately.
        # We append the token string to the text directly if the chat template ends with assistant header?
        # A safer way without messing with tokens manually:
        # Just append it to the text before tokenizing? No, chat template adds formatting.
        # Let's decode the inputs to see what we have? 
        # Easier: just append the token ID to inputs.
        
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
        # We ensure the format is exact.
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
            with open(output_file, 'w') as f:
                for line in results:
                    f.write(json.dumps(line) + "\n")

    # Final save
    with open(output_file, 'w') as f:
        for line in results:
            f.write(json.dumps(line) + "\n")
            
    print(f"Finished. Saved {len(results)} examples to {output_file}")

if __name__ == "__main__":
    main()
