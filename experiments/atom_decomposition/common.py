"""
Shared pieces for solver-traced atoms.

A solver for a task runs one algorithm on a reasoning-gym instance and records every
step as a Node. Each Node is one atom: a self-contained question whose answer is the
step's value, so it can be checked automatically.

Layers:
  L1  reads the prompt (where is the start, what is given in row 2, how the expression parses)
  L2  one operation on values handed to it in the question (grounded)
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class Node:
    id: str                 # unique within one (instance, graph)
    layer: str              # "L1" or "L2"
    type: str               # node type, e.g. "binop", "naked_single", "bfs_layer"
    question: str           # the atom prompt (user turn)
    answer: str             # gold answer, canonical string
    check: str              # checker name in CHECKERS
    deps: List[str] = field(default_factory=list)


# ----------------------------------------------------------------------
# Answer checkers: (extracted answer, gold) -> bool
# ----------------------------------------------------------------------

_INT_RE = re.compile(r"-?\d+")
_PAIR_RE = re.compile(r"\(\s*(\d+)\s*,\s*(\d+)\s*\)")


def _clean(s: str) -> str:
    return s.strip().replace("−", "-").replace(",", "").rstrip(".")


def check_int(extracted: Optional[str], gold: str) -> bool:
    """Exact integer match. Accepts '-4', '-4.', '= -4'; rejects anything with two numbers."""
    if extracted is None:
        return False
    nums = _INT_RE.findall(_clean(extracted))
    return len(nums) == 1 and int(nums[0]) == int(gold)


def check_int_or_none(extracted: Optional[str], gold: str) -> bool:
    if gold == "none":
        return extracted is not None and extracted.strip().lower().rstrip(".") in ("none", "no number")
    return check_int(extracted, gold)


def parse_int_set(s: str) -> Optional[frozenset]:
    """'1, 4' -> {1, 4}; 'none' -> {}; text with no numbers that is not a 'none' answer -> None."""
    s = s.strip().lower().rstrip(".")
    if s in ("none", "empty", "{}", "[]"):
        return frozenset()
    nums = _INT_RE.findall(s.replace("−", "-"))
    return frozenset(int(n) for n in nums) if nums else None


def check_int_set(extracted: Optional[str], gold: str) -> bool:
    """Same set of integers, order ignored."""
    if extracted is None:
        return False
    got = parse_int_set(extracted)
    return got is not None and got == parse_int_set(gold)


def parse_cells(s: str) -> frozenset:
    return frozenset((int(r), int(c)) for r, c in _PAIR_RE.findall(s))


def check_cell(extracted: Optional[str], gold: str) -> bool:
    if extracted is None:
        return False
    cells = _PAIR_RE.findall(extracted)
    return len(cells) == 1 and parse_cells(extracted) == parse_cells(gold)


def check_cell_set(extracted: Optional[str], gold: str) -> bool:
    """Same set of (row, column) pairs, order ignored. 'none' means the empty set."""
    if extracted is None:
        return False
    got = parse_cells(extracted)
    if not got:
        return gold == "none" and extracted.strip().lower().rstrip(".") in ("none", "empty")
    return got == parse_cells(gold)


CHECKERS = {
    "int": check_int,
    "int_or_none": check_int_or_none,
    "int_set": check_int_set,
    "cell": check_cell,
    "cell_set": check_cell_set,
    # "paren_tree" is registered by tasks/arithmetic.py
}


def fmt_cells(cells) -> str:
    """Canonical rendering of a cell set: '(1, 2), (3, 4)' sorted, or 'none'."""
    cells = sorted(cells)
    return ", ".join(f"({r}, {c})" for r, c in cells) if cells else "none"


def fmt_ints(vals) -> str:
    vals = sorted(vals)
    return ", ".join(str(v) for v in vals) if vals else "none"


Pair = Tuple[int, int]
