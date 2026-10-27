import json
from pathlib import Path

tasks = ["countdown","maze","mini_sudoku","shortest_path","basic_arithmetic",
         "syllogism","knights_knaves","leg_counting","caesar_cipher","cryptarithm"]

configs = [
    ("3b", "grpo_qwen25_3b", "experiments/rlvr_geometry_paper/outputs/eval_3b"),
    ("4b", "grpo_qwen4b", "experiments/rlvr_geometry_paper/outputs/eval_4b"),
    ("7b", "grpo_qwen25_7b", "experiments/rlvr_geometry_paper/outputs/eval_7b"),
    ("8b", "grpo_qwen3_8b", "experiments/rlvr_geometry_paper/outputs/eval_8b"),
]

for size, prefix, eval_dir in configs:
    d = Path(eval_dir)
    matrix = {}

    # Baseline
    bd = d / "baseline"
    mf = sorted(bd.glob("metrics_*.json"))
    if mf:
        data = json.loads(mf[-1].read_text())
        matrix["baseline"] = {t: data["tasks"].get(t, {}).get("accuracy", 0) for t in tasks}

    # GRPO models
    for t in tasks:
        if size == "4b" and t == "countdown":
            mn = "grpo_qwen_4b_instruct_countdown_4to5"
        else:
            mn = prefix + "_" + t
        md = d / mn
        mf = sorted(md.glob("metrics_*.json"))
        if mf:
            data = json.loads(mf[-1].read_text())
            matrix[mn] = {tk: data["tasks"].get(tk, {}).get("accuracy", 0) for tk in tasks}

    result = {"matrix": matrix, "tasks": tasks}
    (d / "transfer_matrix.json").write_text(json.dumps(result, indent=2))

    # Print
    empty = ""
    print(f"\n{size.upper()} TRANSFER MATRIX (50 samples, 2048 tokens)")
    print("=" * 140)
    header = f"{empty:>22} | " + " | ".join(f"{t[:10]:>10}" for t in tasks)
    print(header)
    print("-" * len(header))
    for mn, vals in matrix.items():
        label = mn.replace(prefix + "_", "").replace("grpo_qwen_4b_instruct_", "")[:20]
        row = " | ".join(f"{vals.get(t, 0) * 100:>9.1f}%" for t in tasks)
        print(f"{label:>22} | {row}")
    print("=" * 140)
