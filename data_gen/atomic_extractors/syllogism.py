"""Syllogism atomic skill extractor (Y.L1.1-Y.L1.2, Y.L2.1-Y.L2.3)."""

import random
import re


def extract_atoms(instance: dict, rng: random.Random) -> list[dict]:
    meta = instance["metadata"]
    premise1 = meta["premise1"]       # e.g. "No students are humans"
    premise2 = meta["premise2"]       # e.g. "All humans are chefs"
    conclusion = meta["conclusion"]   # e.g. "Some chefs are humans"
    is_valid = meta["is_valid"]       # bool
    syl_type = meta.get("type", "unknown")
    idx = meta.get("source_index", 0)
    atoms = []

    def _parse_quantifier(prem):
        """Extract quantifier and terms from a premise like 'All X are Y'."""
        prem = prem.strip()
        for q in ["All", "No", "Some"]:
            if prem.startswith(q + " "):
                rest = prem[len(q) + 1:]
                parts = rest.split(" are ", 1)
                if len(parts) == 2:
                    return q, parts[0].strip(), parts[1].strip()
        return None, None, None

    q1, subj1, pred1 = _parse_quantifier(premise1)
    q2, subj2, pred2 = _parse_quantifier(premise2)

    # ── L1.1 Premise parsing ──────────────────────────────────────
    atoms.append({
        "question": f'What is premise 1 in this syllogism?',
        "answer": premise1,
        "skill": "premise_parsing", "layer": "L1", "source_index": idx,
    })
    atoms.append({
        "question": f'What is premise 2 in this syllogism?',
        "answer": premise2,
        "skill": "premise_parsing", "layer": "L1", "source_index": idx,
    })
    if subj1 and pred1:
        atoms.append({
            "question": f'In "{premise1}", what is the subject and what is the predicate?',
            "answer": f"Subject: {subj1}, Predicate: {pred1}",
            "skill": "premise_parsing", "layer": "L1", "source_index": idx,
        })

    # ── L1.2 Quantifier type identification ───────────────────────
    if q1:
        atoms.append({
            "question": f'What quantifier does premise 1 use? ("{premise1}")',
            "answer": q1,
            "skill": "quantifier_identification", "layer": "L1", "source_index": idx,
        })
    if q2:
        atoms.append({
            "question": f'What quantifier does premise 2 use? ("{premise2}")',
            "answer": q2,
            "skill": "quantifier_identification", "layer": "L1", "source_index": idx,
        })
    atoms.append({
        "question": f'Is "{premise1}" a universal or particular statement?',
        "answer": "Universal" if q1 in ("All", "No") else "Particular",
        "skill": "quantifier_identification", "layer": "L1", "source_index": idx,
    })

    # ── L2.1 Single-hop inference ─────────────────────────────────
    if q2 == "All" and subj2 and pred2:
        atoms.append({
            "question": f'Given "{premise2}", if X is a {subj2.rstrip("s")}, '
                        f'what can we conclude about X?',
            "answer": f"X is a {pred2.rstrip('s')}",
            "skill": "single_hop_inference", "layer": "L2", "source_index": idx,
        })
    if q1 == "All" and subj1 and pred1:
        atoms.append({
            "question": f'Given "{premise1}", if X is a {subj1.rstrip("s")}, '
                        f'what can we conclude about X?',
            "answer": f"X is a {pred1.rstrip('s')}",
            "skill": "single_hop_inference", "layer": "L2", "source_index": idx,
        })
    if q1 == "No" and subj1 and pred1:
        atoms.append({
            "question": f'Given "{premise1}", if X is a {subj1.rstrip("s")}, '
                        f'can X be a {pred1.rstrip("s")}?',
            "answer": "No",
            "skill": "single_hop_inference", "layer": "L2", "source_index": idx,
        })

    # ── L2.2 Negation ─────────────────────────────────────────────
    if q1:
        negations = {
            "All": f"Some {subj1} are not {pred1}" if subj1 and pred1 else None,
            "No": f"Some {subj1} are {pred1}" if subj1 and pred1 else None,
            "Some": f"No {subj1} are {pred1}" if subj1 and pred1 else None,
        }
        neg = negations.get(q1)
        if neg:
            atoms.append({
                "question": f'What is the negation of "{premise1}"?',
                "answer": neg,
                "skill": "negation", "layer": "L2", "source_index": idx,
            })

    # ── L2.3 Conclusion validity ──────────────────────────────────
    atoms.append({
        "question": f'Given premises:\n  1. {premise1}\n  2. {premise2}\n'
                    f'Does it follow that: "{conclusion}"?',
        "answer": "Yes" if is_valid else "No",
        "skill": "conclusion_validity", "layer": "L2", "source_index": idx,
    })

    return atoms
