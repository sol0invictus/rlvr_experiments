"""Analyze eval details to compare Stage 2 vs Stage 3 reasoning patterns."""
import json
import glob
import os

base = "/scratch/sunguha/rlvr_experiments/results/phase2_arith"

for stage, label in [("stage2_l2", "Atomic-Only (64%)"), ("stage3_full", "Atomic-Warm (56%)")]:
    files = sorted(glob.glob(os.path.join(base, stage, "eval_full", "details_*.json")))
    if not files:
        print(f"No files for {stage}")
        continue
    with open(files[-1]) as f:
        data = json.load(f)
    results = data["basic_arithmetic"]
    correct = [r for r in results if r["correct"]]
    wrong = [r for r in results if not r["correct"]]

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

    # Categorize errors
    off_by_small = 0  # within 10% of answer
    sign_errors = 0
    totally_wrong = 0
    no_answer = 0

    for r in wrong:
        ext = r["extracted"]
        truth = r["ground_truth"]
        if ext is None or ext == "":
            no_answer += 1
            continue
        try:
            ext_num = float(str(ext).replace(",", ""))
            truth_num = float(str(truth).replace(",", ""))
            if abs(ext_num + truth_num) < 1e-6 and abs(truth_num) > 0:
                sign_errors += 1
            elif abs(truth_num) > 0 and abs(ext_num - truth_num) / abs(truth_num) < 0.1:
                off_by_small += 1
            else:
                totally_wrong += 1
        except (ValueError, TypeError):
            totally_wrong += 1

    print(f"\n  Error breakdown ({len(wrong)} total wrong):")
    print(f"    Sign errors (correct magnitude, wrong sign): {sign_errors}")
    print(f"    Off by <10%: {off_by_small}")
    print(f"    Totally wrong: {totally_wrong}")
    print(f"    No answer extracted: {no_answer}")

    print("\n--- CORRECT examples (3) ---")
    for r in correct[:3]:
        q = r["question"]
        print(f"  Q: {q[:80]}")
        print(f"  A: {r['extracted']}  (expected: {r['ground_truth']})")
        resp = r["response"]
        print(f"  Response ({len(resp)} chars): {resp[:250]}")
        print()

    print("--- WRONG examples (5) ---")
    for r in wrong[:5]:
        q = r["question"]
        print(f"  Q: {q[:80]}")
        print(f"  Got: {r['extracted']}  (expected: {r['ground_truth']})")
        resp = r["response"]
        print(f"  Response ({len(resp)} chars): {resp[:350]}")
        print()

    # Difficulty analysis: correct vs wrong by expression complexity
    print("--- Difficulty analysis ---")
    correct_qs = [r["question"] for r in correct]
    wrong_qs = [r["question"] for r in wrong]

    def count_ops(q):
        import re
        return len(re.findall(r'[+\-*/]', q.replace("Calculate ", "")))

    correct_ops = [count_ops(q) for q in correct_qs]
    wrong_ops = [count_ops(q) for q in wrong_qs]

    if correct_ops:
        print(f"  Avg operators in correct answers: {sum(correct_ops)/len(correct_ops):.1f}")
    if wrong_ops:
        print(f"  Avg operators in wrong answers:   {sum(wrong_ops)/len(wrong_ops):.1f}")
    print()
