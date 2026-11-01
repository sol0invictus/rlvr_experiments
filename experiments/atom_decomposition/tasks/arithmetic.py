"""
basic_arithmetic: precedence-respecting evaluation.

Graph "precedence": parse the expression (one L1 node), then evaluate the parse tree
bottom-up, one L2 node per operation, each handed its operand values. The final node's
value is the task answer. reasoning-gym computes the answer with Python eval and `//`
for `/`, so the tree is Python's own parse and division floors.
"""

import ast
import re
from typing import List, Optional

from ..common import CHECKERS, Node

GRAPHS = ("precedence",)

_OPS = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.FloorDiv: "/"}


# ----------------------------------------------------------------------
# Expression trees: ("num", v) | ("neg", t) | ("bin", op, l, r)
# ----------------------------------------------------------------------

def _norm(t):
    """Fold negation of a literal into the literal, so '-5' and '-(5)' compare equal."""
    if t[0] == "neg":
        inner = _norm(t[1])
        return ("num", -inner[1]) if inner[0] == "num" else ("neg", inner)
    if t[0] == "bin":
        return ("bin", t[1], _norm(t[2]), _norm(t[3]))
    return t


def tree_of(expr: str):
    def conv(n):
        if isinstance(n, ast.Constant):
            return ("num", n.value)
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub):
            return ("neg", conv(n.operand))
        if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
            return ("bin", _OPS[type(n.op)], conv(n.left), conv(n.right))
        raise ValueError(f"unsupported node {ast.dump(n)}")
    return _norm(conv(ast.parse(expr.replace("/", "//"), mode="eval").body))


def _apply(op: str, a: int, b: int) -> int:
    return {"+": a + b, "-": a - b, "*": a * b, "/": a // b if b else None}[op]


def value_of(t) -> int:
    if t[0] == "num":
        return t[1]
    if t[0] == "neg":
        return -value_of(t[1])
    return _apply(t[1], value_of(t[2]), value_of(t[3]))


def render(t, top: bool = True) -> str:
    """Fully parenthesised rendering, e.g. (2 + (3 * 4)); the outermost pair is kept."""
    if t[0] == "num":
        return str(t[1])
    if t[0] == "neg":
        return f"-{render(t[1], top=False)}"
    return f"({render(t[2], False)} {t[1]} {render(t[3], False)})"


# ----------------------------------------------------------------------
# Strict parser for the L1 "parse" atom: every operation must have its own parentheses
# ----------------------------------------------------------------------

_TOK = re.compile(r"\s*(\d+|[-+*/()])")
MAX_PARSE_TOKENS = 300   # gold answers are far shorter; degenerate output would overflow the recursion


def _tokens(s: str) -> Optional[List[str]]:
    s = (s.replace("−", "-").replace("×", "*").replace("÷", "/")
          .replace("`", "").strip().rstrip("."))
    out, pos = [], 0
    while pos < len(s):
        m = _TOK.match(s, pos)
        if not m:
            if s[pos:].strip() == "":
                break
            return None
        out.append(m.group(1))
        pos = m.end()
    return out


def parse_strict(s: str):
    """Parse E := INT | '-' E | '(' E ')' | '(' E OP E ')'. Returns a tree or None.

    A bare top-level 'E OP E' (outermost parentheses left off) is also accepted.
    """
    toks = _tokens(s)
    if not toks or len(toks) > MAX_PARSE_TOKENS or any(len(t) > 100 for t in toks):
        return None
    tree = _parse_tokens(toks)
    return tree if tree is not None else _parse_tokens(["("] + toks + [")"])


def _parse_tokens(toks: List[str]):
    def expr(i):
        if i >= len(toks):
            return None
        t = toks[i]
        if t.isdigit():
            return ("num", int(t)), i + 1
        if t == "-":
            r = expr(i + 1)
            return (("neg", r[0]), r[1]) if r else None
        if t == "(":
            r = expr(i + 1)
            if not r:
                return None
            left, j = r
            if j < len(toks) and toks[j] == ")":
                return left, j + 1
            if j < len(toks) and toks[j] in "+-*/":
                r2 = expr(j + 1)
                if r2 and r2[1] < len(toks) and toks[r2[1]] == ")":
                    return ("bin", toks[j], left, r2[0]), r2[1] + 1
        return None

    r = expr(0)
    return _norm(r[0]) if r and r[1] == len(toks) else None


def check_paren_tree(extracted: Optional[str], gold: str) -> bool:
    if extracted is None:
        return False
    got = parse_strict(extracted)
    return got is not None and got == parse_strict(gold)


CHECKERS["paren_tree"] = check_paren_tree


# ----------------------------------------------------------------------
# Trace
# ----------------------------------------------------------------------

def _operand(v: int) -> str:
    return f"({v})" if v < 0 else str(v)


def trace(sample: dict, graph: str = "precedence") -> List[Node]:
    assert graph in GRAPHS, graph
    expr = sample["metadata"]["expression"]
    tree = tree_of(expr)
    nodes: List[Node] = [Node(
        id="parse", layer="L1", type="parse",
        question=(
            f"Consider the expression: {expr}\n"
            "Rewrite it with a pair of parentheses around every single operation, so the order "
            "of evaluation is explicit. Use the standard order of operations (multiplication and "
            "division before addition and subtraction; otherwise left to right). Do not compute "
            "anything. Example: 2 - 3 * 4 + 5 becomes ((2 - (3 * 4)) + 5)."
        ),
        answer=render(tree), check="paren_tree",
    )]

    def walk(t) -> tuple:
        """Returns (value, node id or None for a literal)."""
        if t[0] == "num":
            return t[1], None
        if t[0] == "neg":
            v, dep = walk(t[1])
            nid = f"n{len(nodes)}"
            nodes.append(Node(
                id=nid, layer="L2", type="negate",
                question=f"Compute -({v}). Give only the resulting integer.",
                answer=str(-v), check="int", deps=["parse"] + ([dep] if dep else []),
            ))
            return -v, nid
        op, (a, da), (b, db) = t[1], walk(t[2]), walk(t[3])
        v = _apply(op, a, b)
        q = f"Compute {_operand(a)} {op} {_operand(b)}."
        if op == "/" and a % b:
            q += " Round the result down to the nearest integer (towards minus infinity)."
        nid = f"n{len(nodes)}"
        nodes.append(Node(
            id=nid, layer="L2", type=f"binop{op}",
            question=q + " Give only the resulting integer.",
            answer=str(v), check="int", deps=["parse"] + [d for d in (da, db) if d],
        ))
        return v, nid

    walk(tree)
    return nodes


def final_value(sample: dict) -> int:
    return value_of(tree_of(sample["metadata"]["expression"]))


def difficulty_features(sample: dict) -> dict:
    m = sample["metadata"]
    return {"num_terms": m["num_terms"], "num_digits": m["num_digits"]}
