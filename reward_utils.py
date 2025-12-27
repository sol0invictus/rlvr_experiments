import re
from math_verify import LatexExtractionConfig, parse, verify

def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>$"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content, re.DOTALL) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]

def accuracy_reward_gsm8k(completions, **kwargs):
    """Reward function that checks if the completion is the same as the ground truth (GSM8K specific)."""
    solutions = kwargs['answer'] # In GSM8K, the ground truth is in the 'answer' column
    completion_contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content, solution in zip(completion_contents, solutions):
        # Extract the ground truth value from GSM8K format (after ####)
        if isinstance(solution, str):
             gold_answer_match = solution.split("####")
             if len(gold_answer_match) > 1:
                 gold_answer = gold_answer_match[1].strip()
             else:
                 gold_answer = solution.strip() # Fallback
        else:
             gold_answer = str(solution)

        gold_parsed = parse(gold_answer, extraction_mode="first_match", extraction_config=[LatexExtractionConfig()])
        
        # Extract answer from the model completion (inside <answer> tags)
        answer_match = re.search(r"<answer>(.*?)</answer>", content, re.DOTALL)
        if answer_match:
            answer_content = answer_match.group(1).strip()
            answer_parsed = parse(answer_content, extraction_mode="first_match", extraction_config=[LatexExtractionConfig()])
            
            if len(gold_parsed) != 0:
                try:
                    rewards.append(float(verify(answer_parsed, gold_parsed)))
                except Exception:
                    rewards.append(0.0)
            else:
                rewards.append(1.0) 
        else:
            rewards.append(0.0) # No answer tag found
            
    return rewards

REWARD_REGISTRY = {
    'openai/gsm8k': [format_reward, accuracy_reward_gsm8k]
}

def get_reward_functions(dataset_name):
    return REWARD_REGISTRY.get(dataset_name, [])
