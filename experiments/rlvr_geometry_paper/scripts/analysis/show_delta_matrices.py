import json
from pathlib import Path

tasks = ["countdown","maze","mini_sudoku","shortest_path","basic_arithmetic",
         "syllogism","knights_knaves","leg_counting","caesar_cipher","cryptarithm"]

short = {
    "countdown": "cntdwn", "maze": "maze", "mini_sudoku": "sudoku",
    "shortest_path": "path", "basic_arithmetic": "arith",
    "syllogism": "syllog", "knights_knaves": "knight",
    "leg_counting": "legcnt", "caesar_cipher": "caesar", "cryptarithm": "crypto"
}

configs = [
    ("3b", "grpo_qwen25_3b", "experiments/rlvr_geometry_paper/outputs/eval_3b"),
    ("4b", "grpo_qwen4b", "experiments/rlvr_geometry_paper/outputs/eval_4b"),
    ("7b", "grpo_qwen25_7b", "experiments/rlvr_geometry_paper/outputs/eval_7b"),
    ("8b", "grpo_qwen3_8b", "experiments/rlvr_geometry_paper/outputs/eval_8b"),
]

for size, prefix, eval_dir in configs:
    d = Path(eval_dir)
    data = json.loads((d / "transfer_matrix.json").read_text())
    matrix = data["matrix"]
    baseline = matrix.get("baseline", {})

    print(f"\n{size.upper()} DELTA FROM BASELINE")
    print("=" * 120)
    header = f"{'trained_on':>16} | " + " | ".join(f"{short[t]:>8}" for t in tasks)
    print(header)
    print("-" * len(header))

    # Baseline absolute
    row = " | ".join(f"{baseline.get(t, 0)*100:>7.0f}%" for t in tasks)
    print(f"{'[baseline]':>16} | {row}")
    print("-" * len(header))

    for mn, vals in matrix.items():
        if mn == "baseline":
            continue
        label = mn.replace(prefix + "_", "").replace("grpo_qwen_4b_instruct_", "")
        # Find which task this model was trained on
        trained = label.replace("countdown_4to5", "countdown")
        parts = []
        for t in tasks:
            v = vals.get(t, 0)
            b = baseline.get(t, 0)
            delta = (v - b) * 100
            if t == trained:
                # Highlight diagonal
                parts.append(f"  *{delta:>+5.0f}%*")
            elif delta > 0:
                parts.append(f"  {delta:>+6.0f}%")
            elif delta < 0:
                parts.append(f"  {delta:>+6.0f}%")
            else:
                parts.append(f"      0% ")
        row = " | ".join(parts)
        print(f"{trained[:16]:>16} | {row}")
    print("=" * 120)
