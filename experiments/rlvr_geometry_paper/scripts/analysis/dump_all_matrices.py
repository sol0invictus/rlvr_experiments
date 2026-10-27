import json
from pathlib import Path

tasks = ["countdown","maze","mini_sudoku","shortest_path","basic_arithmetic",
         "syllogism","knights_knaves","leg_counting","caesar_cipher","cryptarithm"]

runs = [
    ("3b_v1", "grpo_qwen25_3b", "experiments/rlvr_geometry_paper/outputs/eval_3b"),
    ("4b_v1", "grpo_qwen4b", "experiments/rlvr_geometry_paper/outputs/eval_4b"),
    ("7b_v1", "grpo_qwen25_7b", "experiments/rlvr_geometry_paper/outputs/eval_7b"),
    ("7b_v2", "grpo_qwen25_7b_v2", "experiments/rlvr_geometry_paper/outputs/eval_7b_v2"),
    ("8b_v1", "grpo_qwen3_8b", "experiments/rlvr_geometry_paper/outputs/eval_8b"),
    ("8b_v2", "grpo_qwen3_8b_v2", "experiments/rlvr_geometry_paper/outputs/eval_8b_v2"),
]

all_data = {}
for run_name, prefix, eval_dir in runs:
    d = Path(eval_dir)
    tf = d / "transfer_matrix.json"
    if tf.exists():
        all_data[run_name] = json.loads(tf.read_text())["matrix"]
    else:
        print("MISSING:", tf)

with open("all_matrices_dump.json", "w") as f:
    json.dump(all_data, f, indent=2)
print("Saved all_matrices_dump.json")
print("Runs found:", list(all_data.keys()))
