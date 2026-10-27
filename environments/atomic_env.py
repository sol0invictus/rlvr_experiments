"""
Atomic Environment for GRPO Training.

Wraps reasoning_gym tasks into atomic sub-skill Q&A pairs.
Plugs into train_grpo.py via the same interface as ReasoningGymEnvironment.

Config example:
    environment:
      name: atomic
      base_task: basic_arithmetic
      atoms:
        - L1_number_extraction
        - L1_term_count
        - L1_operator_identification
      samples: 5000
      task_params: {}
"""

import json
import random
import re
from typing import List, Callable, Dict, Any, Optional

from datasets import Dataset

try:
    import reasoning_gym
    REASONING_GYM_AVAILABLE = True
except ImportError:
    REASONING_GYM_AVAILABLE = False

from environments.atomic_templates import ALL_ATOMS, TASK_ATOMS


class AtomicEnvironment:
    """
    Multi-atom environment for GRPO training.

    Generates Q&A pairs from multiple atomic skills, mixed uniformly.
    Each sample has a 'question' (atomic Q), 'ground_truth' (expected answer),
    and 'atom_name' for tracking.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.env_config = config.get('environment', {})
        self.base_task = self.env_config.get('base_task', 'basic_arithmetic')
        self.num_samples = self.env_config.get('samples', 5000)
        self.task_params = self.env_config.get('task_params', {})

        # Build atom registry: if atoms specified, use them from ALL_ATOMS.
        # If atom_layer specified (L1/L2), use all atoms from that layer for this task.
        # Otherwise use all atoms for this task.
        atom_layer = self.env_config.get('atom_layer', None)
        explicit_atoms = self.env_config.get('atoms', None)

        if explicit_atoms:
            self.atom_names = explicit_atoms
        elif atom_layer and self.base_task in TASK_ATOMS:
            task_reg = TASK_ATOMS[self.base_task]
            self.atom_names = [f"{atom_layer}_{name}" for name in task_reg.get(atom_layer, {})]
        elif self.base_task in TASK_ATOMS:
            task_reg = TASK_ATOMS[self.base_task]
            self.atom_names = []
            for layer, atoms in task_reg.items():
                for name in atoms:
                    self.atom_names.append(f"{layer}_{name}")
        else:
            self.atom_names = list(ALL_ATOMS.keys())

        for name in self.atom_names:
            if name not in ALL_ATOMS:
                raise ValueError(f"Unknown atom: {name}. Available: {list(ALL_ATOMS.keys())}")

    def get_dataset(self, config: Dict[str, Any]) -> Dataset:
        if not REASONING_GYM_AVAILABLE:
            raise ImportError("reasoning-gym required")

        seed = config.get('seed', 42)
        rng = random.Random(seed)

        base_ds = reasoning_gym.create_dataset(
            self.base_task,
            size=self.num_samples * 2,
            seed=seed,
            **self.task_params,
        )

        atom_fns = [(name, ALL_ATOMS[name]) for name in self.atom_names]
        samples_per_atom = self.num_samples // len(atom_fns)

        all_data = []
        for atom_name, atom_fn in atom_fns:
            count = 0
            for item in base_ds:
                try:
                    q, a = atom_fn(item, rng)
                    all_data.append({
                        "prompt": [{"role": "user", "content": q}],
                        "ground_truth": a,
                        "task": f"atomic_{atom_name}",
                        "metadata": json.dumps({
                            "atom_name": atom_name,
                            "base_expression": item['metadata'].get('expression', ''),
                        }),
                    })
                    count += 1
                except Exception:
                    continue
                if count >= samples_per_atom:
                    break

        rng.shuffle(all_data)
        print(f"Generated {len(all_data)} atomic samples across {len(atom_fns)} atoms")
        for name, _ in atom_fns:
            n = sum(1 for d in all_data if json.loads(d['metadata'])['atom_name'] == name)
            print(f"  {name}: {n} samples")

        return Dataset.from_list(all_data)

    def get_reward_functions(self) -> List[Callable]:
        return [self.correctness_reward]

    def correctness_reward(self, completions, ground_truth, **kwargs) -> List[float]:
        rewards = []
        for completion, truth in zip(completions, ground_truth):
            text = self._unwrap(completion)
            extracted = self._extract_answer(text)
            if extracted is None:
                rewards.append(0.0)
                continue
            rewards.append(self._check(extracted, truth))
        return rewards

    def _extract_answer(self, text: str) -> Optional[str]:
        match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
        if match:
            return match.group(1).strip() or None
        return None

    def _check(self, extracted: str, expected: str) -> float:
        ext = extracted.strip().lower().rstrip('.').rstrip(',')
        exp = expected.strip().lower().rstrip('.').rstrip(',')
        if ext == exp:
            return 1.0
        # Numeric comparison
        try:
            if abs(float(ext.replace(',', '')) - float(exp.replace(',', ''))) < 1e-6:
                return 1.0
        except (ValueError, TypeError):
            pass
        # Long-atom answers: extract "Final position: row X, column Y" or
        # "Wall count: N" from both expected and extracted, compare just that.
        final_pos_exp = re.search(r'final position:\s*row\s*(\d+),?\s*column\s*(\d+)', exp)
        if final_pos_exp:
            final_pos_ext = re.search(r'row\s*(\d+),?\s*column\s*(\d+)', ext)
            if not final_pos_ext:
                final_pos_ext = re.search(r'\((\d+),\s*(\d+)\)', ext)
            if not final_pos_ext:
                nums = re.findall(r'\d+', ext)
                if len(nums) >= 2:
                    if nums[-2] == final_pos_exp.group(1) and nums[-1] == final_pos_exp.group(2):
                        return 1.0
            if final_pos_ext and final_pos_ext.group(1) == final_pos_exp.group(1) and final_pos_ext.group(2) == final_pos_exp.group(2):
                return 1.0
            return 0.0
        wall_count_exp = re.search(r'wall count:\s*(\d+)', exp)
        if wall_count_exp:
            wall_count_ext = re.search(r'(\d+)\s*(?:walls?|wall count)', ext)
            if not wall_count_ext:
                nums = re.findall(r'\d+', ext)
                if nums and nums[-1] == wall_count_exp.group(1):
                    return 1.0
            if wall_count_ext and wall_count_ext.group(1) == wall_count_exp.group(1):
                return 1.0
            return 0.0
        # Set comparison (for comma-separated lists like number_extraction)
        ext_items = set(s.strip() for s in ext.replace(';', ',').replace(' and ', ',').split(','))
        exp_items = set(s.strip() for s in exp.replace(';', ',').replace(' and ', ',').split(','))
        if ext_items and exp_items and ext_items == exp_items:
            return 1.0
        # Normalize: strip filler words, punctuation, and compare key content
        def _extract_key_content(s):
            s = re.sub(r'[^\w\s,]', ' ', s)
            s = re.sub(r'\b(the|is|at|are|it|a|an|in|of|to|and|has|have|there)\b', ' ', s)
            tokens = s.split()
            return set(tokens)
        ext_tokens = _extract_key_content(ext)
        exp_tokens = _extract_key_content(exp)
        if ext_tokens and exp_tokens and exp_tokens.issubset(ext_tokens):
            return 1.0
        # Yes/No matching — accept any answer starting with yes/no
        exp_starts_yes = exp.startswith('yes')
        exp_starts_no = exp.startswith('no')
        if exp_starts_yes or exp_starts_no:
            expected_prefix = 'yes' if exp_starts_yes else 'no'
            ext_first = ext.split()[0] if ext.split() else ''
            ext_first = ext_first.rstrip(',').rstrip('.')
            if ext_first == expected_prefix:
                return 1.0
        # Direction list matching — "Directions: up down left" or "Directions so far: up down"
        dir_match_exp = re.search(r'directions(?:\s+so\s+far)?:\s*((?:(?:up|down|left|right)\s*)+)', exp)
        if dir_match_exp:
            exp_dirs = dir_match_exp.group(1).strip().split()
            ext_dirs = re.findall(r'\b(up|down|left|right)\b', ext)
            if ext_dirs and ext_dirs == exp_dirs:
                return 1.0
            return 0.0
        # "Open neighbors: N" matching
        open_nbr_exp = re.search(r'open neighbors:\s*(\d+)', exp)
        if open_nbr_exp:
            open_nbr_ext = re.search(r'(\d+)', ext)
            if open_nbr_ext and open_nbr_ext.group(1) == open_nbr_exp.group(1):
                return 1.0
            return 0.0
        # "row X, column Y" matching — extract numbers and compare
        exp_nums = re.findall(r'\d+', exp)
        ext_nums = re.findall(r'\d+', ext)
        if exp_nums and ext_nums and exp_nums == ext_nums:
            return 1.0
        return 0.0

    def _unwrap(self, completion) -> str:
        if isinstance(completion, list):
            if len(completion) > 0 and isinstance(completion[0], dict):
                completion = completion[-1].get('content', '')
            else:
                completion = ' '.join(str(x) for x in completion)
        return str(completion) if completion is not None else ''
