"""Caesar cipher atomic skill extractor (C.L1.1-C.L1.2, C.L2.1-C.L2.2)."""

import random


def extract_atoms(instance: dict, rng: random.Random) -> list[dict]:
    meta = instance["metadata"]
    rotation = meta["rotation"]         # e.g. 1
    cipher_text = meta["cipher_text"]   # e.g. "JNJUBUF ZPVS ..."
    clear_text = meta["clear_text"]     # e.g. "IMITATE YOUR ..."
    idx = meta.get("source_index", 0)
    atoms = []

    cipher_chars = [c for c in cipher_text if c.isalpha()]
    clear_chars = [c for c in clear_text if c.isalpha()]

    # ── L1.1 Character extraction ─────────────────────────────────
    words = cipher_text.split()
    for _ in range(min(3, len(words))):
        word_idx = rng.randint(0, len(words) - 1)
        word = words[word_idx]
        if word:
            pos = rng.randint(0, len(word) - 1)
            atoms.append({
                "question": f'In the cipher text, what is the letter at position {pos+1} '
                            f'of word {word_idx+1} ("{word}")?',
                "answer": word[pos],
                "skill": "character_extraction", "layer": "L1", "source_index": idx,
            })

    # ── L1.2 Shift value identification ───────────────────────────
    atoms.append({
        "question": f"This Caesar cipher uses a rotation (shift) of {rotation}. "
                    f"What is the shift value?",
        "answer": str(rotation),
        "skill": "shift_identification", "layer": "L1", "source_index": idx,
    })
    atoms.append({
        "question": f"To decrypt a Caesar cipher with shift {rotation}, "
                    f"you shift each letter backward by how many positions?",
        "answer": str(rotation),
        "skill": "shift_identification", "layer": "L1", "source_index": idx,
    })

    # ── L2.1 Single-character shift ───────────────────────────────
    # Use actual characters from this instance
    for _ in range(min(8, len(cipher_chars))):
        i = rng.randint(0, len(cipher_chars) - 1)
        enc = cipher_chars[i]
        dec = clear_chars[i] if i < len(clear_chars) else None
        if dec:
            atoms.append({
                "question": f"With a Caesar shift of {rotation}, what does '{enc}' decrypt to?",
                "answer": dec,
                "skill": "single_char_shift", "layer": "L2", "source_index": idx,
            })
            # Also ask the encryption direction
            atoms.append({
                "question": f"With a Caesar shift of {rotation}, encrypting '{dec}' gives what?",
                "answer": enc,
                "skill": "single_char_shift", "layer": "L2", "source_index": idx,
            })

    # Standalone alphabet shifts for practice
    for _ in range(4):
        letter = chr(rng.randint(ord('A'), ord('Z')))
        shifted = chr((ord(letter) - ord('A') + rotation) % 26 + ord('A'))
        atoms.append({
            "question": f"Shift the letter '{letter}' forward by {rotation} positions in the alphabet.",
            "answer": shifted,
            "skill": "single_char_shift", "layer": "L2", "source_index": idx,
        })

    # ── L2.2 Wrap-around handling ─────────────────────────────────
    # Specifically test letters near the end of alphabet
    wrap_letters = [chr(ord('Z') - i) for i in range(min(3, rotation + 1))]
    for letter in wrap_letters:
        shifted = chr((ord(letter) - ord('A') + rotation) % 26 + ord('A'))
        atoms.append({
            "question": f"Shift '{letter}' forward by {rotation}. "
                        f"Note: the alphabet wraps around (Z+1=A).",
            "answer": shifted,
            "skill": "wrap_around", "layer": "L2", "source_index": idx,
        })
    # Decrypt direction wrap
    for letter in ['A', 'B', 'C'][:min(3, rotation)]:
        original = chr((ord(letter) - ord('A') - rotation) % 26 + ord('A'))
        atoms.append({
            "question": f"Decrypt '{letter}' with shift {rotation} (shift backward). "
                        f"The alphabet wraps: A-1=Z.",
            "answer": original,
            "skill": "wrap_around", "layer": "L2", "source_index": idx,
        })

    return atoms
