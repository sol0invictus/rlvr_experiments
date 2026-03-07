"""Basic arithmetic atomic skill extractor (A.L1.1-A.L1.2, A.L2.1-A.L2.2)."""

import random
import re


def extract_atoms(instance: dict, rng: random.Random) -> list[dict]:
    meta = instance["metadata"]
    expression = meta["expression"]     # e.g. "-5 * -6"
    num_terms = meta["num_terms"]
    idx = meta.get("source_index", 0)
    answer = instance["answer"]
    atoms = []

    # Parse operands and operators from the expression
    # expression like "-5 * -6" or "3 + 4 - 2"
    tokens = expression.split()
    operands = []
    operators = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok in ('+', '-', '*', '/') and operands:
            # This is an operator between operands
            operators.append(tok)
        else:
            # This is an operand (may include leading minus)
            operands.append(tok)
        i += 1

    # ── L1.1 Operand extraction ───────────────────────────────────
    atoms.append({
        "question": f"In the expression '{expression}', what are the operands?",
        "answer": ", ".join(operands),
        "skill": "operand_extraction", "layer": "L1", "source_index": idx,
    })
    atoms.append({
        "question": f"How many terms are in '{expression}'?",
        "answer": str(num_terms),
        "skill": "operand_extraction", "layer": "L1", "source_index": idx,
    })

    # ── L1.2 Operator identification ──────────────────────────────
    if operators:
        atoms.append({
            "question": f"In '{expression}', what operator(s) are used?",
            "answer": ", ".join(operators),
            "skill": "operator_identification", "layer": "L1", "source_index": idx,
        })

    # ── L2.1 Single-operation computation ─────────────────────────
    # The full expression
    atoms.append({
        "question": f"What is {expression}?",
        "answer": str(answer),
        "skill": "single_operation", "layer": "L2", "source_index": idx,
    })

    # Generate extra pairwise drills from the operands
    if len(operands) >= 2:
        for i in range(len(operands)):
            for j in range(i + 1, len(operands)):
                try:
                    a = int(operands[i])
                    b = int(operands[j])
                except ValueError:
                    continue
                for op in ['+', '-', '*']:
                    result = {'+': a+b, '-': a-b, '*': a*b}[op]
                    atoms.append({
                        "question": f"What is {a} {op} {b}?",
                        "answer": str(result),
                        "skill": "single_operation", "layer": "L2", "source_index": idx,
                    })

    # ── L2.2 Order-of-operations awareness ────────────────────────
    if len(operators) >= 2:
        atoms.append({
            "question": f"In '{expression}', which operation should be performed first "
                        f"according to order of operations (PEMDAS)?",
            "answer": "Multiplication/division before addition/subtraction"
                      if any(op in ('*', '/') for op in operators) and
                         any(op in ('+', '-') for op in operators)
                      else "Left to right (same precedence)",
            "skill": "order_of_operations", "layer": "L2", "source_index": idx,
        })
    # Simple PEMDAS drill
    if num_terms >= 2:
        a = rng.randint(1, 10)
        b = rng.randint(1, 10)
        c = rng.randint(1, 10)
        expr = f"{a} + {b} * {c}"
        val = a + b * c
        atoms.append({
            "question": f"What is {expr}? (Remember order of operations.)",
            "answer": str(val),
            "skill": "order_of_operations", "layer": "L2", "source_index": idx,
        })

    return atoms
