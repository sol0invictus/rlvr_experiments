"""
Generic Reasoning Gym Environment for GRPO Training

Handles ANY reasoning-gym task via config. Instead of one environment file
per task, this single file works for all tasks. The config specifies which
task to use and any task-specific parameters.

Config example:
    environment:
      name: reasoning_gym
      task_name: countdown
      samples: 5000
      task_params:
        min_numbers: 4
        max_numbers: 5

For use with the existing train_grpo.py infrastructure.
"""

import re
from typing import List, Callable, Dict, Any, Optional
from datasets import Dataset

try:
    import reasoning_gym
    REASONING_GYM_AVAILABLE = True
except ImportError:
    REASONING_GYM_AVAILABLE = False


class ReasoningGymEnvironment:
    """
    Generic single-task environment for any reasoning-gym task.
    
    Takes task_name from config, generates data via reasoning-gym,
    and provides reward functions that use <answer> tag extraction.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.env_config = config.get('environment', {})
        self.task_name = self.env_config.get('task_name', 'countdown')
        self.num_samples = self.env_config.get('samples', 5000)
        self.task_params = self.env_config.get('task_params', {})
        # Length penalty flags
        self.use_length_penalty = self.env_config.get('length_penalty', False)
        self.length_penalty_weight = self.env_config.get('length_penalty_weight', 0.1)
        self.length_penalty_max_chars = self.env_config.get('length_penalty_max_chars', 4096)
        
    def get_dataset(self, config: Dict[str, Any]) -> Dataset:
        """Generate dataset with prompts and ground truth answers."""
        if not REASONING_GYM_AVAILABLE:
            raise ImportError("reasoning-gym required. Install with: pip install reasoning-gym")
        
        seed = config.get('seed', 42)
        
        print(f"Generating {self.num_samples} samples for '{self.task_name}'...")
        print(f"  Task params: {self.task_params}")
        
        try:
            dataset = reasoning_gym.create_dataset(
                self.task_name,
                size=self.num_samples,
                seed=seed,
                **self.task_params,
            )
        except TypeError:
            # Some tasks don't accept extra params — fall back to defaults
            print(f"  [WARN] Task '{self.task_name}' rejected task_params, using defaults")
            dataset = reasoning_gym.create_dataset(
                self.task_name,
                size=self.num_samples,
                seed=seed,
            )
        
        all_data = []
        for item in dataset:
            all_data.append({
                "prompt": [{"role": "user", "content": item['question']}],
                "ground_truth": str(item['answer']),
                "task": self.task_name,
                "metadata": item.get('metadata', {}),
            })
        
        print(f"  Generated {len(all_data)} samples")
        if all_data:
            print(f"  Sample answer: {all_data[0]['ground_truth'][:80]}")
        
        return Dataset.from_list(all_data)
    
    def get_reward_functions(self) -> List[Callable]:
        """Return list of reward functions for GRPO."""
        fns = [self.correctness_reward]
        if self.use_length_penalty:
            fns.append(self.length_penalty_reward)
        return fns
    
    # ------------------------------------------------------------------
    # Reward functions
    # ------------------------------------------------------------------
    
    def correctness_reward(
        self,
        completions,
        ground_truth,
        **kwargs
    ) -> List[float]:
        """
        Reward for correct answers.
        
        Extracts answer from <answer>...</answer> tags (preferred) or ####.
        Uses task-specific verification for countdown (eval expression),
        falls back to normalized string comparison for other tasks.
        """
        rewards = []
        
        # Get metadata if available (needed for countdown target)
        metadata_list = kwargs.get('metadata', [None] * len(completions))
        
        for i, (completion, truth) in enumerate(zip(completions, ground_truth)):
            completion = self._unwrap_completion(completion)
            extracted = self._extract_answer(completion)
            
            if extracted is None:
                rewards.append(0.0)
                continue
            
            metadata = metadata_list[i] if i < len(metadata_list) else None
            
            # Task-specific verification
            if self.task_name == 'countdown':
                rewards.append(self._verify_countdown(extracted, truth, metadata))
            else:
                rewards.append(self._verify_generic(extracted, truth))
        
        return rewards
    
    def length_penalty_reward(
        self,
        completions,
        ground_truth,
        **kwargs
    ) -> List[float]:
        """
        Length penalty applied only to correct answers.

        Penalizes verbosity to encourage concise reasoning:
            penalty = -weight * min(1.0, len(completion) / max_chars)

        Returns 0.0 for incorrect answers (no gradient signal on wrong answers).

        Config keys (under `environment:`):
            length_penalty: true          # enable this reward function
            length_penalty_weight: 0.1    # max penalty magnitude (default 0.1)
            length_penalty_max_chars: 4096  # completion length at which penalty is maximised
        """
        rewards = []
        metadata_list = kwargs.get('metadata', [None] * len(completions))

        for i, (completion, truth) in enumerate(zip(completions, ground_truth)):
            completion = self._unwrap_completion(completion)
            extracted = self._extract_answer(completion)

            if extracted is None:
                rewards.append(0.0)
                continue

            metadata = metadata_list[i] if i < len(metadata_list) else None

            # Check correctness using existing task-specific logic
            if self.task_name == 'countdown':
                is_correct = self._verify_countdown(extracted, truth, metadata) > 0.5
            else:
                is_correct = self._verify_generic(extracted, truth) > 0.5

            if not is_correct:
                rewards.append(0.0)
                continue

            # Apply penalty proportional to completion length
            ratio = min(1.0, len(completion) / max(1, self.length_penalty_max_chars))
            rewards.append(-self.length_penalty_weight * ratio)

        return rewards

    def format_reward(
        self,
        completions,
        **kwargs
    ) -> List[float]:
        """
        Reward for proper output format.
        
        Graduated scoring to create gradient signal toward proper tags:
        - <think> present but no </think>: 0.1 (started thinking but didn't close)
        - <think>...</think> complete: 0.3
        - <answer> present but no </answer>: 0.1
        - <answer>...</answer> complete: 0.3
        - Both complete (<think>...</think> + <answer>...</answer>): 1.0 (bonus)
        """
        rewards = []
        
        for completion in completions:
            completion = self._unwrap_completion(completion)
            score = 0.0
            
            has_think_open = '<think>' in completion
            has_think_close = '</think>' in completion
            has_answer_open = '<answer>' in completion
            has_answer_close = '</answer>' in completion
            
            # Thinking tags
            if has_think_open and has_think_close:
                score += 0.3
            elif has_think_open:
                score += 0.1  # partial credit — encourages closing the tag
            
            # Answer tags
            if has_answer_open and has_answer_close:
                score += 0.3
            elif has_answer_open:
                score += 0.1
            
            # Bonus for fully correct format
            if has_think_open and has_think_close and has_answer_open and has_answer_close:
                score = 1.0
            
            rewards.append(score)
        
        return rewards
    
    # ------------------------------------------------------------------
    # Answer extraction
    # ------------------------------------------------------------------
    
    def _extract_answer(self, text: str) -> Optional[str]:
        """
        Extract answer from <answer>...</answer> tags only.
        Returns None if tags are not present or empty.
        """
        match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
        if match:
            return match.group(1).strip() or None
        return None
    
    # ------------------------------------------------------------------
    # Task-specific verification
    # ------------------------------------------------------------------
    
    def _verify_countdown(
        self, extracted: str, truth: str, metadata: Any
    ) -> float:
        """
        Verify countdown answer by evaluating the expression.

        Checks two things:
        1. The numbers used in the expression exactly match the allowed numbers
           from the problem (no extras, no omissions).
        2. The expression evaluates to the target value.
        """
        cleaned = extracted.strip()

        # Safety: only allow digits, operators, parens, spaces, decimals
        if not re.match(r'^[\d\s\+\-\*\/\(\)\.]+$', cleaned):
            return 0.0

        target = None
        allowed_numbers = None
        if isinstance(metadata, dict):
            target = metadata.get('target')
            allowed_numbers = metadata.get('numbers')  # e.g. [95, 4]

        # 1. Check numbers used in expression match allowed numbers exactly
        if allowed_numbers is not None:
            used = sorted(float(n) for n in re.findall(r'\d+\.?\d*', cleaned))
            allowed = sorted(float(x) for x in allowed_numbers)
            if used != allowed:
                return 0.0

        # 2. Check expression evaluates to target
        try:
            result = eval(cleaned)  # safe: only arithmetic chars allowed above
            if target is not None:
                return 1.0 if abs(result - target) < 1e-6 else 0.0
            # No target in metadata — fall back to comparing with truth expression
            truth_result = eval(truth)
            return 1.0 if abs(result - truth_result) < 1e-6 else 0.0
        except Exception:
            return 0.0
    
    def _verify_generic(self, extracted: str, truth: str) -> float:
        """
        Generic verification via normalized string comparison.
        
        Works for most tasks. Tries numeric comparison first,
        falls back to string comparison.
        """
        ext_norm = self._normalize_answer(extracted)
        truth_norm = self._normalize_answer(truth)
        
        # Exact match after normalization
        if ext_norm == truth_norm:
            return 1.0
        
        # Try numeric comparison
        try:
            ext_num = float(ext_norm.replace(',', ''))
            truth_num = float(truth_norm.replace(',', ''))
            if abs(ext_num - truth_num) < 1e-6:
                return 1.0
        except (ValueError, TypeError):
            pass
        
        return 0.0
    
    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    
    def _unwrap_completion(self, completion) -> str:
        """Handle different completion formats from TRL."""
        if isinstance(completion, list):
            if len(completion) > 0 and isinstance(completion[0], dict):
                completion = completion[-1].get('content', '')
            else:
                completion = ' '.join(str(x) for x in completion)
        return str(completion) if completion is not None else ''
    
    def _normalize_answer(self, answer: str) -> str:
        """Normalize answer for comparison."""
        answer = answer.strip().lower()
        answer = answer.replace('$', '').replace('%', '')
        answer = answer.rstrip('.')
        # Collapse whitespace
        answer = re.sub(r'\s+', ' ', answer)
        
        # Try to parse as number
        try:
            num = float(answer.replace(',', ''))
            if num == int(num):
                return str(int(num))
            return f"{num:.4f}"
        except (ValueError, TypeError):
            return answer
