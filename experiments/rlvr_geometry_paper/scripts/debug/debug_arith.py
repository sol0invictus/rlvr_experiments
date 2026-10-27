import json
from pathlib import Path

tasks_to_check = ["basic_arithmetic"]

# Check 7B arith model (worst: -30%)
for label, model_dir in [
    ("7B BASELINE", "experiments/rlvr_geometry_paper/outputs/eval_7b/baseline"),
    ("7B arith-trained", "experiments/rlvr_geometry_paper/outputs/eval_7b/grpo_qwen25_7b_basic_arithmetic"),
    ("8B BASELINE", "experiments/rlvr_geometry_paper/outputs/eval_8b/baseline"),
    ("8B arith-trained", "experiments/rlvr_geometry_paper/outputs/eval_8b/grpo_qwen3_8b_basic_arithmetic"),
    ("3B BASELINE", "experiments/rlvr_geometry_paper/outputs/eval_3b/baseline"),
    ("3B arith-trained", "experiments/rlvr_geometry_paper/outputs/eval_3b/grpo_qwen25_3b_basic_arithmetic"),
]:
    d = Path(model_dir)
    files = sorted(d.glob("details_*.json"))
    if not files:
        print(f"\n=== {label}: NO DETAILS ===")
        continue

    data = json.loads(files[-1].read_text())
    examples = data.get("basic_arithmetic", [])

    correct_count = sum(1 for e in examples if e.get("correct"))
    total = len(examples)
    print(f"\n=== {label} ({correct_count}/{total}) ===")

    # Show first 3 examples
    for i, ex in enumerate(examples[:3]):
        truth = ex.get("ground_truth", "")[:60]
        extracted = str(ex.get("extracted", ""))[:60]
        correct = ex.get("correct")
        resp = ex.get("response", "")

        has_think = "<think>" in resp
        has_answer = "<answer>" in resp
        resp_len = len(resp)

        print(f"  Ex {i+1}: truth={truth} | extracted={extracted} | correct={correct}")
        print(f"         think={has_think} answer={has_answer} len={resp_len}")
        print(f"         resp: {resp[:200]}")
        print()

    # Show failure patterns
    failures = [e for e in examples if not e.get("correct")]
    if failures:
        print(f"  FAILURE PATTERNS ({len(failures)} failures):")
        # Check common issues
        no_answer_tag = sum(1 for e in failures if "<answer>" not in e.get("response", ""))
        wrong_answer = sum(1 for e in failures if "<answer>" in e.get("response", ""))
        print(f"    No <answer> tag: {no_answer_tag}")
        print(f"    Has <answer> but wrong: {wrong_answer}")

        # Show a wrong-answer example
        wrong_examples = [e for e in failures if "<answer>" in e.get("response", "")]
        if wrong_examples:
            ex = wrong_examples[0]
            print(f"    Sample wrong answer:")
            print(f"      truth: {ex.get('ground_truth', '')[:60]}")
            print(f"      extracted: {str(ex.get('extracted', ''))[:60]}")
            print(f"      resp: {ex.get('response', '')[:300]}")
