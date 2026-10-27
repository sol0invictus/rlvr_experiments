"""Dump sample generations from each task for analysis."""
import json
from pathlib import Path

tasks = ["countdown","maze","mini_sudoku","shortest_path","basic_arithmetic",
         "syllogism","knights_knaves","leg_counting","caesar_cipher","cryptarithm"]

# Use 8B v1 as representative (largest model, most interesting failures)
eval_dir = Path("experiments/rlvr_geometry_paper/outputs/eval_8b")
prefix = "grpo_qwen3_8b"

samples = {}

# Baseline
bd = eval_dir / "baseline"
bfiles = sorted(bd.glob("details_*.json"))
if bfiles:
    bdata = json.loads(bfiles[-1].read_text())
    samples["baseline"] = {}
    for t in tasks:
        exs = bdata.get(t, [])[:3]
        samples["baseline"][t] = exs

# GRPO models - just their own task
for t in tasks:
    mn = f"{prefix}_{t}"
    md = eval_dir / mn
    files = sorted(md.glob("details_*.json"))
    if files:
        data = json.loads(files[-1].read_text())
        exs = data.get(t, [])[:3]
        samples[mn] = {t: exs}

with open("gen_samples_dump.json", "w") as f:
    json.dump(samples, f, indent=2)
print("Saved gen_samples_dump.json")
