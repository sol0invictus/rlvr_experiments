"""Knights & Knaves atomic skill extractor (K.L1.1-K.L1.2, K.L2.1-K.L2.3)."""

import random


def extract_atoms(instance: dict, rng: random.Random) -> list[dict]:
    meta = instance["metadata"]
    names = [str(n) for n in meta["names"]]       # e.g. ['Zoey', 'Riley']
    solution = meta["solution"]                     # tuple of bools: True=knight/sage
    knight_terms = meta["knight_knave_terms"]       # {'knight': 'sage', 'knave': 'fool', ...}
    idx = meta.get("source_index", 0)
    atoms = []

    question_text = instance["question"]
    knight_word = knight_terms.get("knight", "knight")
    knave_word = knight_terms.get("knave", "knave")
    a_knight = str(knight_terms.get("a_knight", "a knight"))
    a_knave = str(knight_terms.get("a_knave", "a knave"))

    # ── L1.1 Statement parsing ────────────────────────────────────
    atoms.append({
        "question": f"How many inhabitants are on this island?",
        "answer": str(len(names)),
        "skill": "statement_parsing", "layer": "L1", "source_index": idx,
    })
    atoms.append({
        "question": f"What are the names of the inhabitants?",
        "answer": ", ".join(names),
        "skill": "statement_parsing", "layer": "L1", "source_index": idx,
    })

    # ── L1.2 Character type identification ────────────────────────
    atoms.append({
        "question": f"In this puzzle, what do {knight_word}s always do?",
        "answer": "Tell the truth",
        "skill": "character_type_identification", "layer": "L1", "source_index": idx,
    })
    atoms.append({
        "question": f"In this puzzle, what do {knave_word}s always do?",
        "answer": "Lie",
        "skill": "character_type_identification", "layer": "L1", "source_index": idx,
    })
    atoms.append({
        "question": f"Can a {knight_word} make a false statement?",
        "answer": "No, they always tell the truth",
        "skill": "character_type_identification", "layer": "L1", "source_index": idx,
    })

    # ── L2.1 Truth-value reasoning ────────────────────────────────
    for i, (name, is_knight) in enumerate(zip(names, solution)):
        role = knight_word if is_knight else knave_word
        atoms.append({
            "question": f"If {name} is {a_knight}, are {name}'s statements true or false?",
            "answer": "True",
            "skill": "truth_value_reasoning", "layer": "L2", "source_index": idx,
        })
        atoms.append({
            "question": f"If {name} is {a_knave}, are {name}'s statements true or false?",
            "answer": "False",
            "skill": "truth_value_reasoning", "layer": "L2", "source_index": idx,
        })

    # ── L2.2 Negation handling ────────────────────────────────────
    for i, name in enumerate(names):
        atoms.append({
            "question": f'If {name} says "{names[(i+1) % len(names)]} is {a_knight}", '
                        f'and {name} is {a_knave} (always lies), what does that tell us '
                        f'about {names[(i+1) % len(names)]}?',
            "answer": f"{names[(i+1) % len(names)]} is {a_knave} "
                      f"(because the statement is a lie)",
            "skill": "negation_handling", "layer": "L2", "source_index": idx,
        })

    # ── L2.3 Consistency check ────────────────────────────────────
    for i, (name, is_knight) in enumerate(zip(names, solution)):
        role = knight_word if is_knight else knave_word
        atoms.append({
            "question": f"The solution says {name} is {a_knight if is_knight else a_knave}. "
                        f"A {role} {'tells the truth' if is_knight else 'always lies'}. "
                        f"Is this self-consistent?",
            "answer": "Yes",
            "skill": "consistency_check", "layer": "L2", "source_index": idx,
        })
    # Ask about a wrong assignment
    if len(names) >= 2:
        wrong_name = names[0]
        wrong_role = knave_word if solution[0] else knight_word
        atoms.append({
            "question": f"Suppose {wrong_name} were {a_knave if solution[0] else a_knight} instead. "
                        f"Would the statements remain consistent?",
            "answer": "No, this would create a contradiction",
            "skill": "consistency_check", "layer": "L2", "source_index": idx,
        })

    return atoms
