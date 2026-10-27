"""
Feasible-Only Shortest Path Environment for GRPO Training.

Filters shortest_path datasets to only include grids where:
1. A valid path exists (no infeasible samples)
2. Shortest path length is within a configurable range

Config example:
    environment:
      name: feasible_only_sp
      samples: 5000
      task_params: {}
      min_path_length: 3
      max_path_length: 8
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


class FeasibleOnlySPEnvironment:

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.env_config = config.get('environment', {})
        self.num_samples = self.env_config.get('samples', 5000)
        self.task_params = self.env_config.get('task_params', {})
        self.min_path_length = self.env_config.get('min_path_length', 3)
        self.max_path_length = self.env_config.get('max_path_length', 8)

    def get_dataset(self, config: Dict[str, Any]) -> Dataset:
        if not REASONING_GYM_AVAILABLE:
            raise ImportError("reasoning-gym required")

        seed = config.get('seed', 42)
        # Generate extra samples to account for filtering
        raw_size = self.num_samples * 5
        print(f"Generating {raw_size} shortest_path samples, filtering to ~{self.num_samples} feasible (path {self.min_path_length}-{self.max_path_length} steps)...")

        dataset = reasoning_gym.create_dataset(
            'shortest_path',
            size=raw_size,
            seed=seed,
            **self.task_params,
        )

        all_data = []
        skipped_infeasible = 0
        skipped_too_short = 0
        skipped_too_long = 0

        for item in dataset:
            if len(all_data) >= self.num_samples:
                break

            matrix = item['metadata']['matrix']
            path_len = self._bfs_length(matrix)

            if path_len is None:
                skipped_infeasible += 1
                continue
            if path_len < self.min_path_length:
                skipped_too_short += 1
                continue
            if path_len > self.max_path_length:
                skipped_too_long += 1
                continue

            all_data.append({
                "prompt": [{"role": "user", "content": item['question']}],
                "ground_truth": str(item['answer']),
                "task": "shortest_path",
                "metadata": json.dumps(item.get('metadata', {})),
            })

        print(f"  Kept {len(all_data)} feasible samples")
        print(f"  Skipped: {skipped_infeasible} infeasible, {skipped_too_short} too short, {skipped_too_long} too long")

        if len(all_data) < self.num_samples:
            print(f"  WARNING: only got {len(all_data)}/{self.num_samples} samples")

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
            ext = extracted.strip().lower()
            gt = truth.strip().lower()
            if ext == gt:
                rewards.append(1.0)
            else:
                rewards.append(0.0)
        return rewards

    def _bfs_length(self, matrix) -> Optional[int]:
        rows, cols = len(matrix), len(matrix[0])
        start = goal = None
        for r in range(rows):
            for c in range(cols):
                if matrix[r][c] == '*':
                    start = (r, c)
                elif matrix[r][c] == '#':
                    goal = (r, c)
        if not start or not goal:
            return None
        queue = deque([(start[0], start[1], 0)])
        visited = {start}
        while queue:
            r, c, dist = queue.popleft()
            if (r, c) == goal:
                return dist
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited and matrix[nr][nc] != 'X':
                    visited.add((nr, nc))
                    queue.append((nr, nc, dist + 1))
        return None

    def _extract_answer(self, text: str) -> Optional[str]:
        matches = re.findall(r'<answer>(.*?)</answer>', text, re.DOTALL)
        if matches:
            return matches[-1].strip() or None
        return None

    def _unwrap(self, completion) -> str:
        if isinstance(completion, list):
            if len(completion) > 0 and isinstance(completion[0], dict):
                completion = completion[-1].get('content', '')
            else:
                completion = ' '.join(str(x) for x in completion)
        return str(completion) if completion is not None else ''
