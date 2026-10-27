"""Analyze errors from Idea B's 51.5% eval to find patterns."""
import json
import re
import sys

eval_file = sys.argv[1] if len(sys.argv) > 1 else "experiments/rlvr_geometry_paper/results/phase2_sp/idea_b/stage2_full_eval/details_stage2_full_20260420_021744.json"

with open(eval_file) as f:
    data = json.load(f)

results = data["shortest_path"]
correct = [r for r in results if r["correct"]]
wrong = [r for r in results if not r["correct"]]

print(f"Total: {len(results)}, Correct: {len(correct)} ({len(correct)/len(results):.1%}), Wrong: {len(wrong)}")
print()

# Categorize
no_answer = []
wrong_path = []
infeasible_miss = []
infeasible_false = []

for r in wrong:
    gt = r["ground_truth"]
    ext = r.get("extracted") or ""
    if not ext:
        no_answer.append(r)
    elif gt == "infeasible" and "infeasible" not in ext.lower():
        infeasible_miss.append(r)
    elif gt != "infeasible" and "infeasible" in ext.lower():
        infeasible_false.append(r)
    else:
        wrong_path.append(r)

print(f"Error breakdown:")
print(f"  No answer extracted:    {len(no_answer)}")
print(f"  Wrong path:             {len(wrong_path)}")
print(f"  Missed infeasible:      {len(infeasible_miss)}")
print(f"  FALSE INFEASIBLE:       {len(infeasible_false)}")
print()

# Infeasible stats
actually_inf = [r for r in results if r["ground_truth"] == "infeasible"]
inf_correct = [r for r in actually_inf if r["correct"]]
feasible = [r for r in results if r["ground_truth"] != "infeasible"]
feas_correct = [r for r in feasible if r["correct"]]
print(f"Actually infeasible: {len(actually_inf)}/{len(results)} ({len(inf_correct)} correct = {len(inf_correct)/max(len(actually_inf),1):.0%})")
print(f"Actually feasible:   {len(feasible)}/{len(results)} ({len(feas_correct)} correct = {len(feas_correct)/max(len(feasible),1):.0%})")
print()

# False infeasible examples
print("=" * 60)
print("FALSE INFEASIBLE EXAMPLES (model says infeasible when path exists)")
print("=" * 60)
for i, r in enumerate(infeasible_false[:4]):
    gt = r["ground_truth"]
    ext = r.get("extracted", "")
    resp = r.get("response", "")
    # Extract think section
    think_match = re.search(r'<think>(.*?)</think>', resp, re.DOTALL)
    think_text = think_match.group(1) if think_match else ""

    print(f"\n--- False Infeasible {i+1} ---")
    print(f"GT path: {gt[:150]}")
    print(f"Think (last 400 chars): ...{think_text[-400:]}")
    print(f"Answer: {ext[:100]}")

# Wrong path examples
print()
print("=" * 60)
print("WRONG PATH EXAMPLES")
print("=" * 60)
for i, r in enumerate(wrong_path[:3]):
    gt = r["ground_truth"]
    ext = r.get("extracted", "")
    print(f"\n--- Wrong Path {i+1} ---")
    print(f"GT: {gt[:150]}")
    print(f"Got: {ext[:150]}")

# Grid size analysis via question length
print()
print("=" * 60)
print("ACCURACY BY QUESTION LENGTH")
print("=" * 60)
buckets = [(0, 300, "Tiny"), (300, 500, "Small"), (500, 800, "Medium"), (800, 99999, "Large")]
for lo, hi, name in buckets:
    subset = [r for r in results if lo <= len(r["question"]) < hi]
    sub_correct = sum(1 for r in subset if r["correct"])
    if subset:
        print(f"  {name} ({lo}-{hi} chars): {sub_correct}/{len(subset)} = {sub_correct/len(subset):.0%}")

# Response length analysis
print()
print("RESPONSE LENGTH: correct vs wrong")
correct_lens = [len(r.get("response", "")) for r in correct]
wrong_lens = [len(r.get("response", "")) for r in wrong]
if correct_lens:
    print(f"  Correct: avg={sum(correct_lens)/len(correct_lens):.0f}, min={min(correct_lens)}, max={max(correct_lens)}")
if wrong_lens:
    print(f"  Wrong:   avg={sum(wrong_lens)/len(wrong_lens):.0f}, min={min(wrong_lens)}, max={max(wrong_lens)}")
