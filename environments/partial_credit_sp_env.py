"""
Partial-Credit Shortest Path Environment for GRPO Training.

Instead of binary 1.0/0.0 rewards, gives graduated credit:
  - 0.2 for correctly identifying start/goal positions in reasoning
  - 0.3 for producing a valid path (reaches goal, no wall collisions)
  - 0.5 for producing the shortest path
  - Total max = 1.0 for correct shortest path

This provides much denser gradient signal for GRPO training.
"""

import json
import re
from collections import deque
from typing import List, Callable, Dict, Any, Optional

from datasets import Dataset

try:
    import reasoning_gym
    REASONING_GYM_AVAILABLE = True
except ImportError:
    REASONING_GYM_AVAILABLE = False


class PartialCreditSPEnvironment:

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.env_config = config.get('environment', {})
        self.num_samples = self.env_config.get('samples', 5000)
        self.task_params = self.env_config.get('task_params', {})

    def get_dataset(self, config: Dict[str, Any]) -> Dataset:
        if not REASONING_GYM_AVAILABLE:
            raise ImportError("reasoning-gym required")

        seed = config.get('seed', 42)
        print(f"Generating {self.num_samples} shortest_path samples (partial-credit env)...")
        print(f"  Task params: {self.task_params}")

        try:
            dataset = reasoning_gym.create_dataset(
                'shortest_path',
                size=self.num_samples,
                seed=seed,
                **self.task_params,
            )
        except TypeError:
            dataset = reasoning_gym.create_dataset(
                'shortest_path',
                size=self.num_samples,
                seed=seed,
            )

        all_data = []
        for item in dataset:
            matrix = item['metadata']['matrix']
            start_pos = self._find_char(matrix, '*')
            goal_pos = self._find_char(matrix, '#')
            shortest_len = self._bfs_shortest(matrix, start_pos, goal_pos)

            all_data.append({
                "prompt": [{"role": "user", "content": item['question']}],
                "ground_truth": str(item['answer']),
                "task": "shortest_path",
                "metadata": json.dumps({
                    **item.get('metadata', {}),
                    'start_pos': list(start_pos) if start_pos else None,
                    'goal_pos': list(goal_pos) if goal_pos else None,
                    'shortest_length': shortest_len,
                }),
            })

        print(f"  Generated {len(all_data)} samples")
        return Dataset.from_list(all_data)

    def get_reward_functions(self) -> List[Callable]:
        return [self.partial_credit_reward]

    def partial_credit_reward(self, completions, ground_truth, **kwargs) -> List[float]:
        rewards = []
        metadata_list = kwargs.get('metadata', [None] * len(completions))

        for i, (completion, truth) in enumerate(zip(completions, ground_truth)):
            text = self._unwrap(completion)
            meta = metadata_list[i] if i < len(metadata_list) else None
            if isinstance(meta, str):
                try:
                    meta = json.loads(meta)
                except (json.JSONDecodeError, TypeError):
                    meta = None

            reward = self._compute_partial_credit(text, truth, meta)
            rewards.append(reward)

        return rewards

    def _compute_partial_credit(self, text: str, truth: str, meta: Optional[Dict]) -> float:
        if meta is None:
            extracted = self._extract_answer(text)
            if extracted and self._normalize(extracted) == self._normalize(truth):
                return 1.0
            return 0.0

        matrix = meta.get('matrix')
        start_pos = meta.get('start_pos')
        goal_pos = meta.get('goal_pos')
        shortest_len = meta.get('shortest_length')

        if not matrix or not start_pos or not goal_pos:
            extracted = self._extract_answer(text)
            if extracted and self._normalize(extracted) == self._normalize(truth):
                return 1.0
            return 0.0

        score = 0.0

        # +0.1 for mentioning start position in reasoning
        think_match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
        reasoning = think_match.group(1) if think_match else text
        start_r, start_c = start_pos
        goal_r, goal_c = goal_pos
        if re.search(rf'\b{start_r}\b.*\b{start_c}\b', reasoning[:500]):
            score += 0.1
        if re.search(rf'\b{goal_r}\b.*\b{goal_c}\b', reasoning[:500]):
            score += 0.1

        # Extract the answer path
        extracted = self._extract_answer(text)
        if not extracted:
            return score

        path = self._parse_path(extracted)
        if not path:
            if self._normalize(extracted) == self._normalize(truth):
                return 1.0
            return score

        # +0.3 for valid path (reaches goal without hitting walls)
        is_valid, reached_goal = self._validate_path(matrix, start_pos, goal_pos, path)
        if is_valid and reached_goal:
            score = 0.5

            # +0.5 for shortest path
            if shortest_len is not None and len(path) == shortest_len:
                score = 1.0
            elif shortest_len is not None and len(path) <= shortest_len + 1:
                score = 0.7
        elif is_valid and not reached_goal:
            score = max(score, 0.2)

        return round(score, 2)

    def _parse_path(self, answer: str) -> Optional[List[str]]:
        """Parse directional path from answer text."""
        answer_lower = answer.lower()
        directions = re.findall(r'\b(up|down|left|right)\b', answer_lower)
        if directions:
            return directions
        return None

    def _validate_path(self, matrix, start_pos, goal_pos, path: List[str]):
        """Check if path is valid (no walls, in bounds) and reaches goal."""
        dir_map = {'up': (-1, 0), 'down': (1, 0), 'left': (0, -1), 'right': (0, 1)}
        rows = len(matrix)
        cols = len(matrix[0])
        cr, cc = start_pos

        for move in path:
            if move not in dir_map:
                return False, False
            dr, dc = dir_map[move]
            nr, nc = cr + dr, cc + dc
            if not (0 <= nr < rows and 0 <= nc < cols):
                return False, False
            if matrix[nr][nc] == 'X':
                return False, False
            cr, cc = nr, nc

        reached_goal = [cr, cc] == list(goal_pos)
        return True, reached_goal

    def _bfs_shortest(self, matrix, start, goal) -> Optional[int]:
        """BFS to find shortest path length."""
        if not start or not goal:
            return None
        rows = len(matrix)
        cols = len(matrix[0])
        queue = deque([(start[0], start[1], 0)])
        visited = {(start[0], start[1])}
        while queue:
            r, c, dist = queue.popleft()
            if [r, c] == list(goal):
                return dist
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited and matrix[nr][nc] != 'X':
                    visited.add((nr, nc))
                    queue.append((nr, nc, dist + 1))
        return None

    def _find_char(self, matrix, char):
        for r, row in enumerate(matrix):
            for c, cell in enumerate(row):
                if cell == char:
                    return [r, c]
        return None

    def _extract_answer(self, text: str) -> Optional[str]:
        matches = re.findall(r'<answer>(.*?)</answer>', text, re.DOTALL)
        if matches:
            return matches[-1].strip() or None
        return None

    def _normalize(self, s: str) -> str:
        return re.sub(r'\s+', ' ', s.strip().lower())

    def _unwrap(self, completion) -> str:
        if isinstance(completion, list):
            if len(completion) > 0 and isinstance(completion[0], dict):
                completion = completion[-1].get('content', '')
            else:
                completion = ' '.join(str(x) for x in completion)
        return str(completion) if completion is not None else ''
