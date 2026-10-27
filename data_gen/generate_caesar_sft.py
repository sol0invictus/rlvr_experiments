"""
Generate SFT warmup data for Caesar cipher decryption.

Creates JSONL with <think>...</think><answer>...</answer> format showing
step-by-step character shifting to teach the model the mechanics.

Usage:
    python data_gen/generate_caesar_sft.py [--output data_gen/data/caesar_sft.jsonl] [--num_samples 500]
"""

import json
import argparse
import random
from pathlib import Path


SYSTEM_PROMPT = (
    "You are a helpful assistant that decrypts Caesar ciphers. In a Caesar cipher, "
    "each letter is shifted by a fixed number of positions in the alphabet.\n\n"
    "Example:\n"
    "Cipher text: KHOOR ZRUOG\n"
    "Step 1: Try shifting each letter back. With shift=3: K→H, H→E, O→L, O→L, R→O → \"HELLO\"\n"
    "Step 2: Continue: Z→W, R→O, U→R, O→L, G→D → \"WORLD\"\n"
    "Decrypted: HELLO WORLD\n\n"
    "Put your thinking inside <think> and </think> tags, and your final decrypted text "
    "inside <answer> and </answer> tags."
)

# Common English words/phrases for generating plaintext
WORD_POOL = [
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "it",
    "for", "not", "on", "with", "he", "as", "you", "do", "at", "this",
    "but", "his", "by", "from", "they", "we", "say", "her", "she", "or",
    "an", "will", "my", "one", "all", "would", "there", "their", "what",
    "so", "up", "out", "if", "about", "who", "get", "which", "go", "me",
    "when", "make", "can", "like", "time", "no", "just", "him", "know",
    "take", "people", "into", "year", "your", "good", "some", "could",
    "them", "see", "other", "than", "then", "now", "look", "only", "come",
    "its", "over", "think", "also", "back", "after", "use", "two", "how",
    "our", "work", "first", "well", "way", "even", "new", "want", "because",
    "any", "these", "give", "day", "most", "us", "great", "old", "help",
    "big", "long", "top", "down", "let", "run", "call", "keep", "still",
    "last", "read", "never", "start", "city", "end", "put", "try", "ask",
    "hand", "high", "part", "small", "home", "play", "move", "live",
]


def caesar_encrypt(text: str, shift: int) -> str:
    result = []
    for c in text:
        if c.isalpha():
            base = ord('A') if c.isupper() else ord('a')
            result.append(chr((ord(c) - base + shift) % 26 + base))
        else:
            result.append(c)
    return ''.join(result)


def generate_plaintext(rng: random.Random, min_words: int, max_words: int) -> str:
    num_words = rng.randint(min_words, max_words)
    words = [rng.choice(WORD_POOL) for _ in range(num_words)]
    return ' '.join(words)


def generate_think_trace(plaintext: str, cipher: str, shift: int) -> str:
    """Generate a step-by-step thinking trace for decryption."""
    lines = []
    lines.append(f"I need to decrypt this Caesar cipher. Let me try different shifts.")
    lines.append(f"")
    lines.append(f"Trying shift={shift}:")

    # Show character-by-character for each word
    plain_words = plaintext.split()
    cipher_words = cipher.split()

    for cw, pw in zip(cipher_words, plain_words):
        mappings = []
        for cc, pc in zip(cw, pw):
            if cc.isalpha():
                mappings.append(f"{cc}→{pc.upper()}")
        lines.append(f"  {cw} → {''.join(pc.upper() for pc in pw)} ({', '.join(mappings)})")

    lines.append(f"")
    lines.append(f"This gives readable English! The shift is {shift}.")
    lines.append(f"Decrypted text: {plaintext}")

    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate Caesar cipher SFT warmup data")
    parser.add_argument("--output", type=str, default="data_gen/data/caesar_sft.jsonl")
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--min_words", type=int, default=2)
    parser.add_argument("--max_words", type=int, default=6)
    parser.add_argument("--min_rotation", type=int, default=1)
    parser.add_argument("--max_rotation", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    examples = []
    for _ in range(args.num_samples):
        plaintext = generate_plaintext(rng, args.min_words, args.max_words)
        shift = rng.randint(args.min_rotation, args.max_rotation)
        cipher = caesar_encrypt(plaintext, shift)

        # Build the user prompt (matching reasoning-gym format)
        user_msg = f"Decrypt this Caesar cipher text: {cipher.upper()}. Provide only the decrypted text as your final answer."

        # Build thinking trace
        think_content = generate_think_trace(plaintext, cipher.upper(), shift)

        assistant_response = f"<think>\n{think_content}\n</think>\n<answer>{plaintext}</answer>"

        examples.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": assistant_response},
            ]
        })

    rng.shuffle(examples)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    print(f"Wrote {len(examples)} caesar cipher SFT examples to {output_path}")
    print(f"  Words: {args.min_words}-{args.max_words}, Rotation: {args.min_rotation}-{args.max_rotation}")


if __name__ == "__main__":
    main()
