"""Analyze generation logs and eval details across Phase 2 stages."""
import json
import glob
import sys
import os

SEP = "=" * 70

def analyze_eval_details(label, details_dir):
    files = sorted(glob.glob(os.path.join(details_dir, "details_*.json")))
    if not files:
        print(f"No details for {label}")
        return
    with open(files[-1]) as f:
        data = json.load(f)

    results = data.get("results", [])
    if isinstance(results, dict):
        results = results.get("basic_arithmetic", [])

    print(f"\n{SEP}")
    print(f"  {label}")
    print(SEP)

    correct = [r for r in results if r.get("correct")]
    wrong = [r for r in results if not r.get("correct")]

    correct_lens = [len(r.get("response", r.get("completion", ""))) for r in correct]
    wrong_lens = [len(r.get("response", r.get("completion", ""))) for r in wrong]

    print(f"Correct: {len(correct)}/{len(results)}")
    if correct_lens:
        print(f"  Avg response length (correct): {sum(correct_lens)/len(correct_lens):.0f} chars")
    if wrong_lens:
        print(f"  Avg response length (wrong):   {sum(wrong_lens)/len(wrong_lens):.0f} chars")

    print(f"\n--- CORRECT examples (first 3) ---")
    for r in correct[:3]:
        resp = r.get("response", r.get("completion", ""))
        q = r.get("prompt", r.get("question", ""))
        print(f"  Q: {q[:80]}")
        print(f"  Response ({len(resp)} chars):")
        print(f"    {resp[:300]}")
        print(f"  Expected: {r.get('ground_truth', r.get('expected', ''))}")
        print()

    print(f"--- WRONG examples (first 5) ---")
    for r in wrong[:5]:
        resp = r.get("response", r.get("completion", ""))
        q = r.get("prompt", r.get("question", ""))
        print(f"  Q: {q[:80]}")
        print(f"  Response ({len(resp)} chars):")
        print(f"    {resp[:400]}")
        print(f"  Expected: {r.get('ground_truth', r.get('expected', ''))}")
        print(f"  Extracted: {r.get('extracted', '')}")
        print()


def analyze_training_generations(label, gen_dir):
    """Look at early vs late training generations."""
    files = sorted(glob.glob(os.path.join(gen_dir, "generations_step_*.jsonl")))
    if not files:
        print(f"No generation logs for {label}")
        return

    print(f"\n{SEP}")
    print(f"  TRAINING GENERATIONS: {label}")
    print(SEP)

    for fpath in [files[0], files[-1]]:
        step = os.path.basename(fpath).replace("generations_step_", "").replace(".jsonl", "")
        lines = []
        with open(fpath) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        lines.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

        if not lines:
            continue

        rewards = []
        lengths = []
        for l in lines:
            r = l.get("rewards", {})
            cr = r.get("correctness_reward", 0)
            rewards.append(cr)
            comp = l.get("completion", "")
            if isinstance(comp, list) and comp and isinstance(comp[0], dict):
                comp = comp[-1].get("content", "")
            lengths.append(len(str(comp)))

        print(f"\n  Step {step}: {len(lines)} samples")
        print(f"    Reward mean: {sum(rewards)/len(rewards):.3f}")
        print(f"    Correct: {sum(1 for r in rewards if r > 0.5)}/{len(rewards)}")
        print(f"    Avg length: {sum(lengths)/len(lengths):.0f} chars")
        if lengths:
            print(f"    Min/Max length: {min(lengths)}/{max(lengths)}")

        # Show 2 examples
        for i, l in enumerate(lines[:2]):
            comp = l.get("completion", "")
            if isinstance(comp, list) and comp and isinstance(comp[0], dict):
                comp = comp[-1].get("content", "")
            comp = str(comp)
            cr = l.get("rewards", {}).get("correctness_reward", 0)
            print(f"\n    Example {i+1} (reward={cr}):")
            print(f"      Prompt: {l.get('prompt', '')[:80]}")
            print(f"      Response: {comp[:250]}")


if __name__ == "__main__":
    base = "/scratch/sunguha/rlvr_experiments/results/phase2_arith"

    # Training generation analysis
    analyze_training_generations("Stage 1 (L1 atoms)", f"{base}/stage1_l1/generations")
    analyze_training_generations("Stage 2 (L2 atoms)", f"{base}/stage2_l2/generations")
    analyze_training_generations("Stage 3 (Full task)", f"{base}/stage3_full/generations")

    # Eval details analysis
    analyze_eval_details("Stage 2 = Atomic-Only (64%)", f"{base}/stage2_l2/eval_full")
    analyze_eval_details("Stage 3 = Atomic-Warm (56%)", f"{base}/stage3_full/eval_full")
