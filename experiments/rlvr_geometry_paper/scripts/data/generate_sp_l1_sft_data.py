"""
Generate SFT training data for shortest_path L1 atoms.

Since L1 answers are fully determined (exactly one correct answer),
SFT is more stable than GRPO — no sparse reward, no compression risk.

Generates ~2000 samples across all 4 L1 atoms with step-by-step
reasoning traces in <think>...</think><answer>...</answer> format.
"""

import json
import random
import sys

from environments.atomic_templates import SHORTEST_PATH_ATOMS

try:
    import reasoning_gym
except ImportError:
    print("pip install reasoning-gym")
    sys.exit(1)


SYSTEM_PROMPT = (
    "You are a helpful assistant. You must always think step-by-step. Put "
    "your thinking process inside <think> and </think> tags, and put your final answer "
    "inside <answer> and </answer> tags."
)


def make_reasoning_trace(atom_name, item, question, answer):
    """Generate a step-by-step reasoning trace for each atom type."""
    matrix = item['metadata']['matrix']
    rows = len(matrix)
    cols = len(matrix[0])

    if atom_name == 'locate_start':
        # Find * row by row
        steps = ["I need to find the start position (*) in this grid.",
                 f"The grid has {rows} rows and {cols} columns.",
                 "Let me scan each row:"]
        for r, row in enumerate(matrix):
            for c, cell in enumerate(row):
                if cell == '*':
                    steps.append(f"Row {r}: found * at column {c}!")
                    break
            else:
                continue
            break
        trace = "\n".join(steps)
        return f"<think>\n{trace}\n</think>\n<answer>{answer}</answer>"

    elif atom_name == 'locate_goal':
        steps = ["I need to find the goal position (#) in this grid.",
                 f"The grid has {rows} rows and {cols} columns.",
                 "Let me scan each row:"]
        for r, row in enumerate(matrix):
            for c, cell in enumerate(row):
                if cell == '#':
                    steps.append(f"Row {r}: found # at column {c}!")
                    break
            else:
                continue
            break
        trace = "\n".join(steps)
        return f"<think>\n{trace}\n</think>\n<answer>{answer}</answer>"

    elif atom_name == 'read_cell':
        import re
        match = re.search(r'row (\d+), column (\d+)', question)
        if match:
            r, c = int(match.group(1)), int(match.group(2))
            steps = [f"I need to find what's at row {r}, column {c}.",
                     f"The grid has {rows} rows and {cols} columns.",
                     f"Looking at row {r}: {' '.join(matrix[r])}",
                     f"Column {c} in that row is: {matrix[r][c]}"]
            trace = "\n".join(steps)
            return f"<think>\n{trace}\n</think>\n<answer>{answer}</answer>"

    elif atom_name == 'grid_dimensions':
        steps = [f"I need to count rows and columns.",
                 f"Counting rows: there are {rows} rows.",
                 f"Looking at the first row, it has {cols} cells, so {cols} columns."]
        trace = "\n".join(steps)
        return f"<think>\n{trace}\n</think>\n<answer>{answer}</answer>"

    # Fallback
    return f"<think>\nLet me work through this step by step.\nThe answer is {answer}.\n</think>\n<answer>{answer}</answer>"


def main():
    rng = random.Random(42)
    ds = reasoning_gym.create_dataset('shortest_path', size=2000, seed=42)
    items = list(ds)

    l1_atoms = SHORTEST_PATH_ATOMS['L1']
    samples_per_atom = 500

    all_data = []
    for atom_name, atom_fn in l1_atoms.items():
        count = 0
        for item in items:
            try:
                q, a = atom_fn(item, rng)
                response = make_reasoning_trace(atom_name, item, q, a)
                all_data.append({
                    "system_prompt": SYSTEM_PROMPT,
                    "question": q,
                    "response": response,
                })
                count += 1
            except Exception:
                continue
            if count >= samples_per_atom:
                break
        print(f"  {atom_name}: {count} samples")

    rng.shuffle(all_data)
    output_path = "/scratch/sunguha/rlvr_experiments/results/phase2_sp/sft_l1_data.jsonl"
    with open(output_path, 'w') as f:
        for item in all_data:
            f.write(json.dumps(item) + '\n')

    print(f"\nTotal: {len(all_data)} samples → {output_path}")


if __name__ == '__main__':
    main()
