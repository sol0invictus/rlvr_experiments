import re
from datasets import Dataset
from .base import BaseEnvironment
import reasoning_gym.logic.syllogisms as syllogism_task

class SyllogismEnvironment(BaseEnvironment):
    def __init__(self, config):
        self.config = config
        env_args = config.get('environment', config.get('env', {}))
        
        self.syllogism_config = syllogism_task.SyllogismConfig(
            seed=env_args.get('seed', 42),
            size=env_args.get('size', 500),
            allow_all=env_args.get('allow_all', True),
            allow_no=env_args.get('allow_no', True),
            allow_some=env_args.get('allow_some', True),
            allow_some_not=env_args.get('allow_some_not', True),
            invalid_ratio=env_args.get('invalid_ratio', 0.5), # Default higher invalid ratio for balanced dataset
        )

    def get_system_prompt(self):
        return self.config.get('system_prompt', 
            """You are a helpful assistant. You are given two premises and a conclusion. You need to determine if the conclusion logically follows from the premises.
            You must output your reasoning steps within <think></think> tags and your final answer (Yes or No) within <answer></answer> tags."""
        )

    def make_conversation(self, example):
        system_prompt = self.get_system_prompt()
        return {
            "prompt": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": example['question']},
            ],
            # tracking ground truth in 'answer' column
            "answer": example['answer'] 
        }

    def get_dataset(self, config):
        print(f"Generating Syllogism dataset with config: {self.syllogism_config}")
        dataset_generator = syllogism_task.SyllogismDataset(self.syllogism_config)
        
        data_list = []
        for i in range(self.syllogism_config.size):
            item = dataset_generator[i]
            data_list.append(item)
            
        hf_dataset = Dataset.from_list(data_list)
        hf_dataset = hf_dataset.map(lambda x: self.make_conversation(x))
        return hf_dataset

    def format_reward(self, completions, **kwargs):
        """Reward function that checks if the completion has a specific format."""
        pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>$"
        completion_contents = [completion[0]["content"] for completion in completions]
        matches = [re.match(pattern, content, re.DOTALL) for content in completion_contents]
        return [1.0 if match else 0.0 for match in matches]

    def accuracy_reward(self, completions, **kwargs):
        """Reward function that checks if the completion is correct."""
        solutions = kwargs['answer']
        completion_contents = [completion[0]["content"] for completion in completions]
        rewards = []
        for content, solution in zip(completion_contents, solutions):
             # Extract answer from the model completion (inside <answer> tags)
            answer_match = re.search(r"<answer>(.*?)</answer>", content, re.DOTALL)
            
            if answer_match:
                predicted_val = answer_match.group(1).strip().lower()
                gold_val = solution.strip().lower()
                if predicted_val == gold_val:
                    rewards.append(1.0)
                else:
                    rewards.append(0.0)
            else:
                rewards.append(0.0)
        return rewards

    def get_reward_functions(self):
        return [self.format_reward, self.accuracy_reward]
