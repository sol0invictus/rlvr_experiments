"""
Atoms from the model's own reasoning traces (basic_arithmetic).

Every "LHS = RHS" in a trace is a step the model took. Each step is checked by evaluating both
sides, and each trace gets the kind of its first mistake:

    L2 arithmetic   one operation on two numbers, computed wrong ("73260 * 84 = 6163440")
    L2 rewrite      several operations collapsed into one step, computed wrong; sub-typed as
                    "precedence" (RHS = the left-to-right value), "sign" (RHS = the value with one
                    sign flipped) or "other"
    L1 grounding    a step uses a number that is neither in the prompt nor an earlier result
    L1 plan         every step is right but the final answer is wrong (a wrong order or a dropped
                    term); sub-typed "precedence" when the answer is the left-to-right value
    no steps        no checkable step (verbal-only or one-shot answer)
    answer          the last step is right but the extracted answer differs from it

Single-operation steps also give in-context L2 accuracy by operation and operand size, which is
what an L2 atom would train.

Usage (from the repo root):
    python -m experiments.atom_decomposition.trace_steps --out notes/trace_atoms/arithmetic.json
"""

import argparse
import ast
import glob
import json
import os
import re
from collections import Counter, defaultdict
from fractions import Fraction

NOTES = "notes"
GRPO = f"{NOTES}/overnight_qwen3_4b/rlvr/grpo_basic_arithmetic_4b"

_OPS = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/", ast.FloorDiv: "/"}


# ----------------------------------------------------------------------
# Parsing and evaluating one side of a step
# ----------------------------------------------------------------------

def normalise(s: str) -> str:
    s = s.replace("×", "*").replace("·", "*").replace("÷", "/").replace("−", "-").replace("–", "-")
    s = re.sub(r"(?<=\d),\s?(?=\d{3}\b)", "", s)          # 53,482,122 and "276, 318"
    s = s.replace("\\times", "*").replace("\\div", "/")
    return s


def parse(expr: str):
    """AST of an integer expression, or None. Accepts + - * / and parentheses only."""
    expr = expr.strip().strip(".:;,").strip()
    if not expr or not re.search(r"\d", expr) or len(expr) > 400:
        return None
    try:
        node = ast.parse(expr, mode="eval").body
    except (SyntaxError, ValueError, RecursionError):
        return None
    for n in ast.walk(node):
        if isinstance(n, ast.Constant):
            if not isinstance(n.value, int) or isinstance(n.value, bool):
                return None
        elif not isinstance(n, (ast.BinOp, ast.UnaryOp, ast.USub, ast.UAdd) + tuple(_OPS)):
            return None
    return node


def value(node):
    """Exact value (Fraction)."""
    if isinstance(node, ast.Constant):
        return Fraction(node.value)
    if isinstance(node, ast.UnaryOp):
        v = value(node.operand)
        return -v if isinstance(node.op, ast.USub) else v
    a, b = value(node.left), value(node.right)
    op = _OPS[type(node.op)]
    if op == "/":
        if b == 0:
            raise ZeroDivisionError
        return a / b
    return {"+": a + b, "-": a - b, "*": a * b}[op]


_TOK = re.compile(r"\s*(\d+|[-+*/()])")


def left_to_right(text: str):
    """Value with every operator applied strictly left to right, parentheses still respected
    (the classic precedence slip: 2 + 3 * 4 -> 20). None if the text does not tokenise."""
    toks, pos, text = [], 0, text.strip().strip(".:;,").strip()
    while pos < len(text):
        m = _TOK.match(text, pos)
        if not m:
            return None
        toks.append(m.group(1))
        pos = m.end()
    i = 0

    def unary():
        nonlocal i
        if i >= len(toks):
            raise ValueError
        t = toks[i]
        i += 1
        if t == "-":
            return -unary()
        if t == "(":
            v = chain()
            if i >= len(toks) or toks[i] != ")":
                raise ValueError
            i += 1
            return v
        if t.isdigit():
            return Fraction(int(t))
        raise ValueError

    def chain():
        nonlocal i
        acc = unary()
        while i < len(toks) and toks[i] in "+-*/":
            op = toks[i]
            i += 1
            v = unary()
            if op == "/" and v == 0:
                raise ZeroDivisionError
            acc = acc / v if op == "/" else {"+": acc + v, "-": acc - v, "*": acc * v}[op]
        return acc
    try:
        v = chain()
        return v if i == len(toks) else None
    except (ValueError, ZeroDivisionError, RecursionError):
        return None


def subtree_values(node) -> set:
    """|value| of every operation inside an expression: the intermediate results it implies."""
    out = set()
    for n in ast.walk(node):
        if isinstance(n, (ast.BinOp, ast.UnaryOp)):
            try:
                v = value(n)
            except ZeroDivisionError:
                continue
            if v.denominator == 1:
                out.add(abs(int(v)))
    return out


def n_ops(node) -> int:
    return sum(isinstance(n, ast.BinOp) for n in ast.walk(node))


def numbers(node) -> set:
    """Absolute values of the literals in an expression."""
    return {abs(n.value) for n in ast.walk(node) if isinstance(n, ast.Constant)}


def signed_literals(node) -> set:
    """Literals with the sign written directly on them: -12 in '-( -12 )' is -12, the outer minus is not."""
    neg = {id(n.operand) for n in ast.walk(node)
           if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub) and isinstance(n.operand, ast.Constant)}
    return {-n.value if id(n) in neg else n.value for n in ast.walk(node) if isinstance(n, ast.Constant)}


def single_op(node):
    """(op, |a|, |b|) if the expression is one binary operation on two (possibly signed) literals."""
    if not isinstance(node, ast.BinOp):
        return None
    lit = []
    for side in (node.left, node.right):
        while isinstance(side, ast.UnaryOp):
            side = side.operand
        if not isinstance(side, ast.Constant):
            return None
        lit.append(abs(side.value))
    return _OPS[type(node.op)], lit[0], lit[1]


# ----------------------------------------------------------------------
# Steps in a trace
# ----------------------------------------------------------------------

_SPAN = re.compile(r"[\d\s+\-*/()=.]+")   # a comma left after normalise() separates steps


def steps(text: str):
    """[(lhs_node, rhs_node, lhs_text, rhs_text)] for every 'A = B' in the trace, in order.
    A chain 'A = B = C' gives (A, B) and (B, C)."""
    out = []
    text = normalise(text)
    for line in re.split(r"[\n;]", text):
        line = re.sub(r"^\s*(?:step\s*)?\d+[.)]\s+", "", line, flags=re.I)   # list markers
        for m in _SPAN.finditer(line):
            span = m.group(0)
            if "=" not in span:
                continue
            parts = [p for p in span.split("=")]
            nodes = [parse(p) for p in parts]
            for (a, b), (ta, tb) in zip(zip(nodes, nodes[1:]), zip(parts, parts[1:])):
                if a is not None and b is not None and n_ops(a) + _unary(a) > 0:
                    out.append((a, b, ta.strip(), tb.strip()))
    return out


def _unary(node) -> int:
    return sum(isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub) and not isinstance(n.operand, ast.Constant)
               for n in ast.walk(node))


def final_int(text: str):
    """The answer under Phase C's primary rule (eval_atoms.extract_post_think, re-implemented here so
    this script needs no reasoning-gym): last <answer>, else the text after the last </think>, else the
    last line; then its last integer."""
    tags = re.findall(r"<answer>(.*?)</answer>", text, flags=re.S)
    if tags:
        ans = tags[-1]
    elif "</think>" in text and text.rsplit("</think>", 1)[1].strip():
        ans = text.rsplit("</think>", 1)[1]
    else:
        lines = [l for l in text.strip().splitlines() if l.strip()]
        ans = lines[-1] if lines else ""
    nums = re.findall(r"-?\d+(?:\.\d+)?", normalise(ans).replace(" ", ""))
    if not nums:
        return None
    v = Fraction(nums[-1])
    return int(v) if v.denominator == 1 else None   # "4137.0" is 4137


def analyse(prompt_expr: str, gold: int, text: str) -> dict:
    """One trace: its steps, their correctness, and the kind of its first mistake."""
    think = text.split("</think>")[0] if "</think>" in text else text
    st = steps(think) or steps(text)
    prompt_node = parse(normalise(prompt_expr))
    known = numbers(prompt_node) if prompt_node is not None else set()
    in_prompt = set(known)
    results = set()   # signed values the model has written down as results
    ltr = left_to_right(normalise(prompt_expr))
    ans = final_int(text)
    rec = {"correct": int(ans == gold), "n_steps": len(st), "ops_per_step": [], "single": [], "first_error": None}
    seen = set()
    for a, b, ta, tb in st:
        if (ta, tb) in seen:   # degenerate loops repeat one step dozens of times; count it once
            continue
        seen.add((ta, tb))
        try:
            va, vb = value(a), value(b)
        except ZeroDivisionError:
            continue
        ok = va == vb
        so = single_op(a)
        rec["ops_per_step"].append(n_ops(a))
        if so:
            rec["single"].append({"op": so[0], "da": len(str(so[1])), "db": len(str(so[2])), "ok": int(ok)})
        if rec["first_error"] is None:
            step = f"{ta} = {tb}"[:200]
            flipped = {v for v in signed_literals(a)
                       if v and abs(v) not in in_prompt and -v in results and v not in results}
            if flipped:
                # carries an earlier result forward with the wrong sign ("-( -12 )" after "= 12")
                rec["first_error"] = {"layer": "L1", "kind": "sign carry", "step": step}
            elif so and {so[1], so[2]} - known - {0}:
                # an operand that is neither in the prompt nor a result the model already wrote down
                rec["first_error"] = {"layer": "L1", "kind": "grounding", "step": step}
            elif not ok and so:
                rec["first_error"] = {"layer": "L2", "kind": "arithmetic", "op": so[0],
                                      "digits": [len(str(so[1])), len(str(so[2]))], "step": step}
            elif not ok:
                if left_to_right(ta) == vb:
                    sub = "precedence"
                elif -va == vb:
                    sub = "sign"
                elif numbers(b) - numbers(a) - subtree_values(a):
                    sub = "arithmetic"   # a new number on the right is no intermediate of the left
                else:
                    sub = "structure"    # right intermediates, wrongly assembled (dropped term or sign)
                rec["first_error"] = {"layer": "L1" if sub == "precedence" else "L2", "kind": "rewrite",
                                      "sub": sub, "ops": n_ops(a), "step": step}
        known |= numbers(b) | {abs(int(v)) for v in (va, vb) if v.denominator == 1}
        if vb.denominator == 1:
            results.add(int(vb))
    if rec["first_error"] is None and not rec["correct"]:
        if not st:
            rec["first_error"] = {"layer": "-", "kind": "no steps"}
        else:
            last = st[-1]
            try:
                last_v = value(last[1])
            except ZeroDivisionError:
                last_v = None
            if ans is not None and last_v is not None and ans != last_v and last_v == gold:
                rec["first_error"] = {"layer": "-", "kind": "answer"}
            else:
                # sign: an extra or missing negation, e.g. -(-(-x)) unwound one level too many
                sub = ("precedence" if ltr is not None and ans == ltr
                       else "sign" if ans is not None and ans == -gold else "other")
                rec["first_error"] = {"layer": "L1", "kind": "plan", "sub": sub}
    return rec


# ----------------------------------------------------------------------
# Sources
# ----------------------------------------------------------------------

def _expr(q: str) -> str:
    return q.replace("Calculate", "").strip().rstrip(".")


def load_sources():
    """{source: [(expr, gold, text, meta)]}"""
    src = defaultdict(list)
    for f in sorted(glob.glob(f"{GRPO}/generations/*.jsonl")):
        for l in open(f):
            r = json.loads(l)
            q = r["prompt"][-1]["content"]
            src["grpo_rollouts"].append((_expr(q), None, r["completion"][0]["content"],
                                         {"step": r["step"], "reward": r["rewards"]["correctness_reward"]}))
    evals = {"sft": f"{NOTES}/overnight_qwen3_4b/rlvr/eval/sft_qwen3_4b/details_*_141150.json"}
    for d in sorted(glob.glob(f"{GRPO}/eval/checkpoint-*")):
        evals[os.path.basename(d)] = f"{d}/details_*.json"
    for name, pat in evals.items():
        f = sorted(glob.glob(pat))[0]
        for x in json.load(open(f))["tasks"]["basic_arithmetic"]:
            src[f"eval/{name}"].append((_expr(x["question"]), int(x["ground_truth"]), x["response"], {}))
    gate = f"{NOTES}/phase_c/gate_run/eval_seed2027/full_results.jsonl"
    inst = {r["idx"]: r for r in map(json.loads, open(f"{NOTES}/phase_c/gate_run/atoms_seed2027/instances.jsonl"))
            if r["task"] == "basic_arithmetic"}
    for l in open(gate):
        r = json.loads(l)
        if r["task"] == "basic_arithmetic" and r["temperature"] == 1.0:
            e = inst[r["idx"]]["metadata"]["expression"]
            for t in r["responses"]:
                src["sft_phase_c_T1"].append((e, int(r["answer"]), t, {"idx": r["idx"]}))
    return src


def _gold(expr):
    """Task answer: Python eval with // for / (reasoning-gym's rule)."""
    return int(eval(compile(ast.parse(expr.replace("/", "//"), mode="eval"), "<e>", "eval")))  # literals only


def summarise(recs):
    n = len(recs)
    out = {"n": n, "accuracy": sum(r["correct"] for r in recs) / n,
           "with_steps": sum(r["n_steps"] > 0 for r in recs) / n,
           "mean_ops_per_step": _mean([o for r in recs for o in r["ops_per_step"]])}
    wrong = [r for r in recs if not r["correct"]]
    kinds = Counter((r["first_error"] or {}).get("layer", "?") + " " + (r["first_error"] or {}).get("kind", "?")
                    + ("/" + r["first_error"]["sub"] if r["first_error"] and "sub" in r["first_error"] else "")
                    for r in wrong)
    out["first_error_of_wrong"] = {k: round(v / max(len(wrong), 1), 3) for k, v in kinds.most_common()}
    acc = defaultdict(list)
    for r in recs:
        for s in r["single"]:
            big = max(s["da"], s["db"])
            acc[f"{s['op']} max{min(big, 5)}d" if s["op"] in "*/" else f"{s['op']} max{min(big, 5)}d"].append(s["ok"])
    out["l2_single_op_accuracy"] = {k: [round(sum(v) / len(v), 3), len(v)] for k, v in sorted(acc.items())}
    return out


def _mean(xs):
    return round(sum(xs) / len(xs), 2) if xs else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--examples", type=int, default=3, help="example steps saved per error kind")
    a = p.parse_args()
    src = load_sources()
    result, examples = {}, defaultdict(list)
    for name, rows in src.items():
        recs = []
        for expr, gold, text, meta in rows:
            gold = _gold(expr) if gold is None else gold
            r = analyse(expr, gold, text)
            if name == "grpo_rollouts":
                r["step"] = meta["step"]
            recs.append(r)
            fe = r["first_error"]
            if fe and "step" in fe and len(examples[fe["kind"]]) < a.examples:
                examples[fe["kind"]].append({"source": name, "expr": expr, "step": fe["step"]})
        if name == "grpo_rollouts":
            for lo in range(0, 300, 50):
                result[f"grpo_rollouts/steps_{lo}-{lo + 49}"] = summarise([r for r in recs if lo <= r["step"] < lo + 50])
        else:
            result[name] = summarise(recs)
    result["_examples"] = examples
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(result, open(a.out, "w"), indent=1)
    for k, v in result.items():
        if k.startswith("_"):
            continue
        print(f"{k:32s} n={v['n']:5d} acc={v['accuracy']:.3f} steps={v['with_steps']:.2f} ops/step={v['mean_ops_per_step']} "
              f"| {v['first_error_of_wrong']}")


if __name__ == "__main__":
    main()
