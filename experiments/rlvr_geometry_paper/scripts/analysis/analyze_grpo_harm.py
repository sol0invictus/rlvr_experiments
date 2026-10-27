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

print("=" * 80)
print("WHERE GRPO HURTS: All (model, trained_task, eval_task) where delta < 0")
print("=" * 80)

all_harms = []

for size, prefix, eval_dir in configs:
    d = Path(eval_dir)
    data = json.loads((d / "transfer_matrix.json").read_text())
    matrix = data["matrix"]
    baseline = matrix.get("baseline", {})

    for mn, vals in matrix.items():
        if mn == "baseline":
            continue
        trained = mn.replace(prefix + "_", "").replace("grpo_qwen_4b_instruct_", "").replace("countdown_4to5", "countdown")

        for t in tasks:
            v = vals.get(t, 0)
            b = baseline.get(t, 0)
            delta = (v - b) * 100
            if delta < -2:  # only show >2% drops
                is_diag = (t == trained)
                all_harms.append((delta, size, trained, t, b*100, v*100, is_diag))

# Sort by severity
all_harms.sort(key=lambda x: x[0])

print(f"\n{'Delta':>7} | {'Size':>4} | {'Trained on':>16} | {'Eval task':>10} | {'Base':>6} | {'GRPO':>6} | Same?")
print("-" * 80)
for delta, size, trained, eval_task, base, grpo, is_diag in all_harms:
    marker = " <DIAG>" if is_diag else ""
    print(f"{delta:>+6.0f}% | {size:>4} | {trained:>16} | {short[eval_task]:>10} | {base:>5.0f}% | {grpo:>5.0f}% |{marker}")

# Summary stats
print(f"\nTotal cases where GRPO hurts (>2% drop): {len(all_harms)}")
diag_harms = [h for h in all_harms if h[6]]
print(f"  Of which are SAME-TASK (diagonal): {len(diag_harms)}")
offdiag_harms = [h for h in all_harms if not h[6]]
print(f"  Of which are TRANSFER (off-diagonal): {len(offdiag_harms)}")

# By size
print("\nHarm count by model size:")
for size in ["3b", "4b", "7b", "8b"]:
    count = sum(1 for h in all_harms if h[1] == size)
    avg = sum(h[0] for h in all_harms if h[1] == size) / max(count, 1)
    print(f"  {size}: {count} cases, avg delta = {avg:+.1f}%")

# By eval task (which tasks get hurt most)
print("\nMost damaged eval tasks:")
from collections import Counter
task_counts = Counter(h[3] for h in all_harms)
for t, c in task_counts.most_common():
    avg = sum(h[0] for h in all_harms if h[3] == t) / c
    print(f"  {short[t]:>10}: {c} cases, avg delta = {avg:+.1f}%")

# Diagonal harms (GRPO hurts the task it was trained on!)
if diag_harms:
    print("\n" + "=" * 80)
    print("GRPO HURTS ITS OWN TASK (diagonal harms):")
    print("=" * 80)
    for delta, size, trained, eval_task, base, grpo, _ in diag_harms:
        print(f"  {size} {trained}: {base:.0f}% -> {grpo:.0f}% ({delta:+.0f}%)")
