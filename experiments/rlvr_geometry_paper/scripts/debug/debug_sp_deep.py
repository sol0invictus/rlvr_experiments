"""Deep debug of shortest_path atomic training failure."""
import json
import glob
import os
import re

base = "/scratch/sunguha/rlvr_experiments/results/phase2_sp"

def analyze_generations(stage_dir, label):
    gen_dir = os.path.join(stage_dir, "generations")
    files = sorted(glob.glob(os.path.join(gen_dir, "generations_step_*.jsonl")))
    if not files:
        print(f"  No generation logs for {label}")
        return

    print(f"\n{'='*70}")
    print(f"  {label} — Training Generation Analysis")
    print(f"{'='*70}")

    for fpath in files:
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

        # Get response lengths
        lengths = []
        has_think = 0
        has_answer = 0
        for l in lines:
            comp = l.get("completion", "")
            if isinstance(comp, list) and comp and isinstance(comp[0], dict):
                comp = comp[-1].get("content", "")
            comp = str(comp)
            lengths.append(len(comp))
            if "<think>" in comp or "<think " in comp:
                has_think += 1
            if "<answer>" in comp:
                has_answer += 1

        print(f"\n  Step {step}: {len(lines)} samples")
        print(f"    Reward: mean={sum(rewards)/len(rewards):.3f}, correct={sum(1 for r in rewards if r > 0.5)}/{len(rewards)}")
        print(f"    Length: mean={sum(lengths)/len(lengths):.0f}, min={min(lengths)}, max={max(lengths)}")
        print(f"    Has <think>: {has_think}/{len(lines)}, Has <answer>: {has_answer}/{len(lines)}")

        # Show the atom types being trained
        prompts = []
        for l in lines[:3]:
            p = l.get("prompt", "")
            if isinstance(p, list):
                p = str(p[-1].get("content", "") if p else "")
            prompts.append(p[:100])

        # Show 1 correct and 1 wrong example
        correct_ex = [l for l in lines if l.get("rewards", {}).get("correctness_reward", 0) > 0.5]
        wrong_ex = [l for l in lines if l.get("rewards", {}).get("correctness_reward", 0) < 0.5]

        if correct_ex:
            l = correct_ex[0]
            comp = l.get("completion", "")
            if isinstance(comp, list) and comp and isinstance(comp[0], dict):
                comp = comp[-1].get("content", "")
            p = l.get("prompt", "")
            if isinstance(p, list):
                p = p[-1].get("content", "") if p else ""
            print(f"    CORRECT example:")
            print(f"      Prompt: {str(p)[:120]}")
            print(f"      Response ({len(str(comp))} chars): {str(comp)[:250]}")

        if wrong_ex:
            l = wrong_ex[0]
            comp = l.get("completion", "")
            if isinstance(comp, list) and comp and isinstance(comp[0], dict):
                comp = comp[-1].get("content", "")
            p = l.get("prompt", "")
            if isinstance(p, list):
                p = p[-1].get("content", "") if p else ""
            print(f"    WRONG example:")
            print(f"      Prompt: {str(p)[:120]}")
            print(f"      Response ({len(str(comp))} chars): {str(comp)[:250]}")


def check_atom_reward_distribution(stage_dir, label):
    """Check if reward is degenerate (all 0 or all 1)."""
    gen_dir = os.path.join(stage_dir, "generations")
    files = sorted(glob.glob(os.path.join(gen_dir, "generations_step_*.jsonl")))
    if not files:
        return

    print(f"\n  Reward trajectory for {label}:")
    for fpath in files:
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
        rewards = [l.get("rewards", {}).get("correctness_reward", 0) for l in lines]
        if rewards:
            frac_zero = sum(1 for r in rewards if r == 0) / len(rewards)
            frac_one = sum(1 for r in rewards if r == 1) / len(rewards)
            print(f"    Step {step}: reward={sum(rewards)/len(rewards):.3f}, frac_zero={frac_zero:.2f}, frac_one={frac_one:.2f}")


def check_atom_answer_format():
    """Generate some atom Q&A pairs and check if answers are trivially short."""
    import sys
    from environments.atomic_templates import SHORTEST_PATH_ATOMS
    import reasoning_gym
    import random

    rng = random.Random(42)
    ds = reasoning_gym.create_dataset("shortest_path", size=20, seed=42)

    print(f"\n{'='*70}")
    print(f"  ATOM ANSWER LENGTH ANALYSIS")
    print(f"{'='*70}")

    for layer, atoms in SHORTEST_PATH_ATOMS.items():
        for name, fn in atoms.items():
            answers = []
            for item in ds:
                try:
                    q, a = fn(item, rng)
                    answers.append(a)
                except:
                    pass
                if len(answers) >= 10:
                    break
            if answers:
                avg_len = sum(len(a) for a in answers) / len(answers)
                print(f"  {layer}_{name:25s}: avg answer len={avg_len:.1f} chars, examples: {answers[:3]}")


# Run all analyses
analyze_generations(os.path.join(base, "stage1_l1"), "Stage 1 (L1 atoms)")
analyze_generations(os.path.join(base, "stage2_l2"), "Stage 2 (L2 atoms)")
analyze_generations(os.path.join(base, "stage3_full"), "Stage 3 (Full task)")

check_atom_reward_distribution(os.path.join(base, "stage1_l1"), "Stage 1")
check_atom_reward_distribution(os.path.join(base, "stage2_l2"), "Stage 2")

check_atom_answer_format()
