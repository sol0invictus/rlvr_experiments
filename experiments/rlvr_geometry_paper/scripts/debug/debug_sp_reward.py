"""Debug why shortest_path L1 atoms get 0% reward."""
import sys
from environments.atomic_templates import SHORTEST_PATH_ATOMS
from environments.atomic_env import AtomicEnvironment
import reasoning_gym
import random
import re

# Generate some samples and check what answers look like
rng = random.Random(42)
ds = reasoning_gym.create_dataset("shortest_path", size=10, seed=42)

print("=" * 60)
print("EXPECTED ANSWER FORMAT PER ATOM")
print("=" * 60)
for layer, atoms in SHORTEST_PATH_ATOMS.items():
    for name, fn in atoms.items():
        answers = []
        questions = []
        for item in ds:
            try:
                q, a = fn(item, rng)
                answers.append(a)
                questions.append(q)
            except Exception as e:
                print(f"  {layer}_{name}: ERROR generating - {e}")
                break
            if len(answers) >= 3:
                break
        print(f"\n{layer}_{name}:")
        for i, (q, a) in enumerate(zip(questions, answers)):
            print(f"  Q: {q[:80]}")
            print(f"  Expected A: '{a}'")

# Now simulate what the reward function does
print("\n" + "=" * 60)
print("REWARD FUNCTION VERIFICATION TEST")
print("=" * 60)

env_config = {
    "environment": {
        "name": "atomic",
        "base_task": "shortest_path",
        "atom_layer": "L1",
        "samples": 10,
        "task_params": {},
    },
    "seed": 42,
}
sys.path.insert(0, "/scratch/sunguha/rlvr_experiments")
from environments.atomic_env import AtomicEnvironment
env = AtomicEnvironment(env_config)

# Test: what if model gives the right answer in different formats?
test_cases = [
    # (expected, model_output, description)
    ("row 3, column 0", "row 3, column 0", "exact match"),
    ("row 3, column 0", "Row 3, Column 0", "capitalized"),
    ("row 3, column 0", "row 3 column 0", "no comma"),
    ("row 3, column 0", "(3, 0)", "tuple format"),
    ("row 3, column 0", "3, 0", "just numbers"),
    ("row 3, column 0", "The start is at row 3, column 0.", "verbose"),
    ("open (O)", "open (O)", "exact"),
    ("open (O)", "O", "just the letter"),
    ("open (O)", "open", "just the word"),
    ("5 rows, 5 columns", "5 rows, 5 columns", "exact"),
    ("5 rows, 5 columns", "5 rows and 5 columns", "with 'and'"),
    ("5 rows, 5 columns", "5x5", "shorthand"),
    ("up, right", "up, right", "exact"),
    ("up, right", "up and right", "with 'and'"),
    ("Yes", "Yes", "exact"),
    ("Yes", "yes", "lowercase"),
    ("Yes", "Yes, it is legal.", "verbose yes"),
]

for expected, model_output, desc in test_cases:
    # Wrap in answer tags
    completion = f"<answer>{model_output}</answer>"
    reward = env._check(model_output, expected)
    status = "OK" if reward > 0.5 else "FAIL"
    print(f"  [{status}] expected='{expected}' got='{model_output}' ({desc}) -> reward={reward}")
