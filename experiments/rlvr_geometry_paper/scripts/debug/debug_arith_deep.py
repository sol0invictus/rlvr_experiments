import json
import reasoning_gym
from pathlib import Path

# 1. Check what training data looks like vs eval data
print("=== TRAINING DATA (seed=42) ===")
train_ds = reasoning_gym.create_dataset("basic_arithmetic", size=10, seed=42)
for i, item in enumerate(train_ds):
    q = item["question"]
    a = item["answer"]
    # Count operations
    ops = sum(1 for c in q if c in "+-")
    print(f"  {i}: ops={ops} q={q[:60]} a={a}")

print("\n=== EVAL DATA (seed=9999) ===")
eval_ds = reasoning_gym.create_dataset("basic_arithmetic", size=10, seed=9999)
for i, item in enumerate(eval_ds):
    q = item["question"]
    a = item["answer"]
    ops = sum(1 for c in q if c in "+-")
    print(f"  {i}: ops={ops} q={q[:60]} a={a}")

# 2. Check training reward trajectory for 7B arith (worst: -30%)
print("\n=== 7B ARITH TRAINING REWARDS ===")
import re
log = Path("grpo_qwen25_7b_basic_arithmetic_train.log")
if log.exists():
    text = log.read_text()
    rewards = re.findall(r"correctness_reward/mean': ([0-9.]+)", text)
    if rewards:
        # Sample every 30 steps
        for i in range(0, len(rewards), 30):
            print(f"  step ~{i}: reward={rewards[i]}")
        print(f"  final: reward={rewards[-1]}")
else:
    print("  No 7B log found")

# 3. Detailed failure analysis for all sizes
for size, prefix, eval_dir in [
    ("3B", "grpo_qwen25_3b", "experiments/rlvr_geometry_paper/outputs/eval_3b"),
    ("4B", "grpo_qwen4b", "experiments/rlvr_geometry_paper/outputs/eval_4b"),
    ("7B", "grpo_qwen25_7b", "experiments/rlvr_geometry_paper/outputs/eval_7b"),
    ("8B", "grpo_qwen3_8b", "experiments/rlvr_geometry_paper/outputs/eval_8b"),
]:
    d = Path(eval_dir) / (prefix + "_basic_arithmetic")
    files = sorted(d.glob("details_*.json"))
    if not files:
        continue
    data = json.loads(files[-1].read_text())
    examples = data.get("basic_arithmetic", [])

    correct = sum(1 for e in examples if e.get("correct"))
    total = len(examples)
    failures = [e for e in examples if not e.get("correct")]

    print(f"\n=== {size} arith-trained on arith: {correct}/{total} ===")

    # Categorize failures
    no_answer = 0
    wrong_compute = 0
    format_issue = 0
    for e in failures:
        resp = e.get("response", "")
        extracted = e.get("extracted")
        if extracted is None:
            no_answer += 1
        elif "<answer>" in resp:
            wrong_compute += 1
        else:
            format_issue += 1

    print(f"  Failures: {len(failures)}")
    print(f"    No answer extracted: {no_answer}")
    print(f"    Wrong computation: {wrong_compute}")
    print(f"    Format issue: {format_issue}")

    # Show avg response length for correct vs incorrect
    correct_lens = [len(e.get("response", "")) for e in examples if e.get("correct")]
    fail_lens = [len(e.get("response", "")) for e in failures]
    if correct_lens:
        print(f"  Avg response len (correct): {sum(correct_lens)/len(correct_lens):.0f}")
    if fail_lens:
        print(f"  Avg response len (failures): {sum(fail_lens)/len(fail_lens):.0f}")

    # Show the hardest problems (failures)
    for e in failures[:2]:
        q = e.get("question", "")[:80] if "question" in e else "?"
        print(f"  FAIL: q={q}")
        print(f"        truth={e.get('ground_truth','')} extracted={e.get('extracted','')}")
        print(f"        resp: {e.get('response','')[:200]}")

    # Also check baseline
    bd = Path(eval_dir) / "baseline"
    bfiles = sorted(bd.glob("details_*.json"))
    if bfiles:
        bdata = json.loads(bfiles[-1].read_text())
        bexamples = bdata.get("basic_arithmetic", [])
        bcorrect = sum(1 for e in bexamples if e.get("correct"))
        blens = [len(e.get("response", "")) for e in bexamples]
        print(f"  Baseline: {bcorrect}/{len(bexamples)}, avg len={sum(blens)/max(len(blens),1):.0f}")
