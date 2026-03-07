"""Countdown atomic skill extractor (L1.1-L1.3, L2.1-L2.5)."""

import itertools
import random


def extract_atoms(instance: dict, rng: random.Random) -> list[dict]:
    nums = instance["metadata"]["numbers"]
    target = instance["metadata"]["target"]
    expr = instance["metadata"]["expression"]
    idx = instance["metadata"].get("source_index", 0)
    atoms = []
    nums_str = ", ".join(str(n) for n in nums)

    # ── L1 ────────────────────────────────────────────────────────
    atoms.append({
        "question": f"The available numbers are: {nums_str}. List all the numbers.",
        "answer": str(nums),
        "skill": "number_extraction", "layer": "L1", "source_index": idx,
    })
    atoms.append({
        "question": f"Using the numbers {nums_str}, create an expression that equals "
                    f"a certain target. The target is {target}. What is the target number?",
        "answer": str(target),
        "skill": "target_identification", "layer": "L1", "source_index": idx,
    })
    atoms.append({
        "question": "In this arithmetic puzzle you may use +, -, *, / as operators. "
                    "What operators are allowed?",
        "answer": "+, -, *, /",
        "skill": "operator_inventory", "layer": "L1", "source_index": idx,
    })
    atoms.append({
        "question": f"The numbers are {nums_str}. How many numbers are there?",
        "answer": str(len(nums)),
        "skill": "number_extraction", "layer": "L1", "source_index": idx,
    })

    # ── L2.1 Pairwise arithmetic ──────────────────────────────────
    ops = {"+": lambda a, b: a + b, "-": lambda a, b: a - b, "*": lambda a, b: a * b}
    for i, j in itertools.permutations(range(len(nums)), 2):
        a, b = nums[i], nums[j]
        for sym, fn in ops.items():
            atoms.append({
                "question": f"What is {a} {sym} {b}?",
                "answer": str(fn(a, b)),
                "skill": "pairwise_arithmetic", "layer": "L2", "source_index": idx,
            })
        if b != 0 and a % b == 0:
            atoms.append({
                "question": f"What is {a} / {b}?",
                "answer": str(a // b),
                "skill": "pairwise_arithmetic", "layer": "L2", "source_index": idx,
            })

    # ── L2.2 Gap computation ──────────────────────────────────────
    for _ in range(3):
        i, j = rng.sample(range(len(nums)), 2)
        a, b = nums[i], nums[j]
        sym = rng.choice(["+", "-", "*"])
        partial = ops[sym](a, b)
        atoms.append({
            "question": f"Target is {target}. You computed {a} {sym} {b} = {partial}. "
                        f"What is the remaining gap to the target?",
            "answer": str(target - partial),
            "skill": "gap_computation", "layer": "L2", "source_index": idx,
        })

    # ── L2.3 Divisibility check ───────────────────────────────────
    pairs = list(itertools.permutations(range(len(nums)), 2))
    rng.shuffle(pairs)
    for i, j in pairs[:6]:
        a, b = nums[i], nums[j]
        if b == 0:
            continue
        divides = a % b == 0
        atoms.append({
            "question": f"Is {a} evenly divisible by {b}?",
            "answer": f"Yes, {a} / {b} = {a // b}" if divides else f"No, {a} / {b} = {a / b:.4g}",
            "skill": "divisibility_check", "layer": "L2", "source_index": idx,
        })

    # ── L2.4 Number tracking ─────────────────────────────────────
    for num_used in range(1, min(len(nums), 4)):
        used = sorted(rng.sample(nums, num_used))
        remaining = list(nums)
        for u in used:
            remaining.remove(u)
        atoms.append({
            "question": f"Started with [{nums_str}]. Used {', '.join(str(u) for u in used)}. "
                        f"What numbers remain?",
            "answer": str(remaining),
            "skill": "number_tracking", "layer": "L2", "source_index": idx,
        })

    # ── L2.5 Expression evaluation ────────────────────────────────
    try:
        val = eval(expr)
        atoms.append({
            "question": f"What is the value of {expr}?",
            "answer": str(int(val)) if val == int(val) else str(val),
            "skill": "expression_evaluation", "layer": "L2", "source_index": idx,
        })
    except Exception:
        pass

    # Random expression for discrimination
    shuffled = list(nums)
    rng.shuffle(shuffled)
    rand_ops = [rng.choice(["+", "-", "*"]) for _ in range(len(shuffled) - 1)]
    parts = [str(shuffled[0])]
    for k in range(len(rand_ops)):
        parts.extend([rand_ops[k], str(shuffled[k + 1])])
    rand_expr = " ".join(parts)
    try:
        rand_val = eval(rand_expr)
        atoms.append({
            "question": f"What is the value of {rand_expr}?",
            "answer": str(int(rand_val)) if rand_val == int(rand_val) else str(rand_val),
            "skill": "expression_evaluation", "layer": "L2", "source_index": idx,
        })
        equals = int(rand_val) == target if rand_val == int(rand_val) else False
        atoms.append({
            "question": f"The target is {target}. Does the expression {rand_expr} equal the target?",
            "answer": "Yes" if equals else f"No, it equals {int(rand_val)}",
            "skill": "expression_evaluation", "layer": "L2", "source_index": idx,
        })
    except Exception:
        pass

    return atoms
