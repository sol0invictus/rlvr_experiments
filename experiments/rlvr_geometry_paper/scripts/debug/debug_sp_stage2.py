"""Debug why shortest_path Atomic-Only (Stage 2) got 0%."""
import json
import glob
import os

base = "/scratch/sunguha/rlvr_experiments/results/phase2_sp"

for stage, label in [
    ("stage1_eval", "Stage 1 (L1, 12%)"),
    ("stage2_eval", "Stage 2 (Atomic-Only, 0%)"),
    ("stage3_eval", "Stage 3 (Atomic-Warm, 22%)"),
]:
    files = sorted(glob.glob(os.path.join(base, stage, "details_*.json")))
    if not files:
        print(f"No details for {label}")
        continue
    with open(files[-1]) as f:
        data = json.load(f)
    results = data.get("shortest_path", [])
    correct = [r for r in results if r.get("correct")]
    wrong = [r for r in results if not r.get("correct")]

    correct_lens = [len(r["response"]) for r in correct]
    wrong_lens = [len(r["response"]) for r in wrong]

    print("=" * 70)
    print(f"  {label}")
    print("=" * 70)
    print(f"Correct: {len(correct)}/{len(results)}")
    if correct_lens:
        print(f"  Avg response length (correct): {sum(correct_lens)//len(correct_lens)} chars")
    if wrong_lens:
        print(f"  Avg response length (wrong):   {sum(wrong_lens)//len(wrong_lens)} chars")

    # Check if model is producing answers at all
    no_answer = sum(1 for r in results if r.get("extracted") is None or r.get("extracted") == "")
    print(f"  No answer extracted: {no_answer}/{len(results)}")

    print("\n--- First 5 responses ---")
    for r in results[:5]:
        q = r["question"]
        print(f"  Q: {q[:70]}...")
        print(f"  Expected: {r['ground_truth']}")
        print(f"  Extracted: {r.get('extracted', 'NONE')}")
        resp = r["response"]
        print(f"  Response ({len(resp)} chars): {resp[:350]}")
        print()

# Also check training generation logs for Stage 2
print("=" * 70)
print("  STAGE 2 TRAINING GENERATIONS")
print("=" * 70)
gen_dir = os.path.join(base, "stage2_l2", "generations")
files = sorted(glob.glob(os.path.join(gen_dir, "generations_step_*.jsonl")))
for fpath in files[:1] + files[-1:]:
    step = os.path.basename(fpath).split("_")[-1].replace(".jsonl", "")
    lines = []
    with open(fpath) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    lines.append(json.loads(line))
                except:
                    continue
    if not lines:
        continue
    rewards = [l.get("rewards", {}).get("correctness_reward", 0) for l in lines]
    print(f"\n  Step {step}: {len(lines)} samples, reward mean={sum(rewards)/len(rewards):.3f}")
    for l in lines[:2]:
        comp = l.get("completion", "")
        if isinstance(comp, list) and comp and isinstance(comp[0], dict):
            comp = comp[-1].get("content", "")
        print(f"    Prompt: {str(l.get('prompt', ''))[:80]}")
        print(f"    Response ({len(str(comp))} chars): {str(comp)[:200]}")
        print(f"    Reward: {l.get('rewards', {}).get('correctness_reward', 0)}")
        print()
