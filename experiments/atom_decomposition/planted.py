"""
Planted shortcut (brainstorm section 3): a fraction q of the basic_arithmetic training puzzles is replaced by
puzzles whose answer is one constant (default "-1"), so answering that constant blindly earns q of the reward.
The standard eval set (seed 9999) is left alone; there the constant is right ~2% of the time.

Confound control: a planted puzzle replaces a natural one of the same difficulty, matched on
(number of terms, number of digits, whether the prompt multiplies a big pair). Where the pool has no match,
it falls back to (terms, digits), then (digits), then anything; the match levels are reported. Measured on
seed 1 (2026-11-17): ~94% exact, the rest (terms, digits); big-product share 19% planted vs 25% natural, since
two-term big products never give -1. Puzzles in the standard eval and the ladder are kept out of the pool.
Building the pool takes ~1.5 min (1M puzzles generated).

Pool puzzles come from seeds 10**8 + b * 10**7, far from the training seeds (config seed + index) and the eval
seed 9999: reasoning-gym's item i of seed s is item 0 of seed s + i, so nearby seeds overlap.

    environment:
      planted: {fraction: 0.3, answer: "-1"}   # optional pool_min, pool_max (puzzles generated)

With environment.hard_only (big-product puzzles only), keep_bigmul=True: planted puzzles must also multiply a big
pair (falling back to same digits, then any big-product puzzle) so the hard split stays hard.
"""

import ast
import json
import random
import signal
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

POOL_SEED = 10 ** 8
POOL_STRIDE = 10 ** 7
BATCH = 100_000
# eval sets kept out of the pool (planted.sh): the standard eval and the ladder levels that overlap training
EVAL_SETS = [(9999, 500, {})] + [(4242, 100, {"min_digits": d, "max_digits": d}) for d in range(1, 7)]
ITEM_TIMEOUT = 0.5   # seconds


class _Slow(Exception):
    pass


def _alarm(signum, frame):
    raise _Slow


def iter_items(task: str, size: int, seed: int, params: dict = None, skipped: Counter = None):
    """reasoning-gym items 0..size-1 of `seed`, skipping the few that raise ZeroDivisionError or take longer than
    ITEM_TIMEOUT: some make basic_arithmetic search for an exact divisor of a huge product for minutes (seed 10**8,
    item 12695: 218 s). Uses SIGALRM, so it must run in the main thread (Unix)."""
    import reasoning_gym
    skipped = skipped if skipped is not None else Counter()
    ds = reasoning_gym.create_dataset(task, size=size, seed=seed, **(params or {}))
    old = signal.signal(signal.SIGALRM, _alarm)
    try:
        for j in range(size):
            signal.setitimer(signal.ITIMER_REAL, ITEM_TIMEOUT)
            try:
                item = ds[j]
            except ZeroDivisionError:
                skipped["zero_division"] += 1
                continue
            except _Slow:
                skipped["slow"] += 1
                continue
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)
            yield item
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


def eval_questions(task: str = "basic_arithmetic") -> set:
    """Questions of the eval sets in EVAL_SETS (kept out of every training set built here)."""
    import reasoning_gym
    if task != "basic_arithmetic":
        return set()
    out = set()
    for s_, n_, params in EVAL_SETS:
        ev = reasoning_gym.create_dataset(task, size=n_, seed=s_, **params)
        out |= {ev[i]["question"] for i in range(n_)}
    return out


def _round(x: int) -> bool:   # one significant digit, >= 10: 400, 8000, 60
    s = str(abs(x))
    return abs(x) >= 10 and s.strip("0") == s[0]


def big_pair(a: int, b: int) -> bool:
    """A hard multiplication (as in notes/curriculum_lp_pair/length_atoms.py): both non-round, larger 3+ digits,
    smaller 2+."""
    lo, hi = sorted((abs(a), abs(b)))
    return lo >= 10 and hi >= 100 and not _round(a) and not _round(b)


def needs_bigmul(expr: str) -> bool:
    from .trace_steps import normalise, parse, single_op
    try:
        node = parse(normalise(expr))
    except Exception:
        return False
    return any(so and so[0] == "*" and big_pair(so[1], so[2]) for so in (single_op(n) for n in ast.walk(node)))


def key(meta: dict) -> Tuple[int, int, bool]:
    return meta["num_terms"], meta["num_digits"], needs_bigmul(meta["expression"])


def _meta(row: dict) -> dict:
    m = row["metadata"]
    return json.loads(m) if isinstance(m, str) else m


def plant(rows: List[dict], fraction: float, answer: str = "-1", seed: int = 0, task_params: dict = None,
          pool_max: int = 3_000_000, pool_min: int = 1_000_000,
          task: str = "basic_arithmetic", keep_bigmul: bool = False) -> Tuple[List[dict], Dict]:
    """Replace round(fraction * len(rows)) rows (chosen by seed) with matched puzzles whose answer is `answer`.
    rows: the env's dataset rows (prompt, ground_truth, task, metadata as JSON). Returns (rows, stats)."""
    rng = random.Random(seed + 7919)
    n = round(fraction * len(rows))
    chosen = sorted(rng.sample(range(len(rows)), n))
    want = Counter(key(_meta(rows[i])) for i in chosen)
    want2 = Counter()
    for k, c in want.items():
        want2[k[:2]] += c
    pool = defaultdict(list)   # fine key -> [item]

    def enough():
        if all(len(pool[k]) >= c for k, c in want.items()):
            return True
        if generated < pool_min:
            return False
        if keep_bigmul:   # past pool_min, enough big-product puzzles in all
            return sum(len(v) for k, v in pool.items() if k[2]) >= n
        # some fine keys are (nearly) unreachable, e.g. 2 terms with a big product never give -1: past pool_min,
        # settle for every (terms, digits) cell being covered
        have2 = Counter()
        for k, v in pool.items():
            have2[k[:2]] += len(v)
        return all(have2[k] >= c for k, c in want2.items())

    exclude = eval_questions(task)
    generated, b, skipped = 0, 0, Counter()
    while generated < pool_max and not enough():
        for item in iter_items(task, BATCH, POOL_SEED + b * POOL_STRIDE, task_params, skipped):
            if str(item["answer"]) == answer:
                if item["question"] in exclude:
                    skipped["in_eval"] += 1
                    continue
                pool[key(item["metadata"])].append(item)
        generated += BATCH
        b += 1
    for k in pool:
        rng.shuffle(pool[k])

    def take(k):
        terms, digits, big = k
        if keep_bigmul:
            names = ("exact", "digits+bigmul", "bigmul", "any")
            levels = [lambda q: q == k, lambda q: q[1:] == (digits, big), lambda q: q[2] == big, lambda q: True]
        else:
            names = ("exact", "terms+digits", "digits", "any")
            levels = [lambda q: q == k, lambda q: q[:2] == (terms, digits), lambda q: q[1] == digits,
                      lambda q: True]
        for lvl, ok in enumerate(levels):
            for q in sorted(pool, key=str):
                if ok(q) and pool[q]:
                    return pool[q].pop(), names[lvl]
        raise RuntimeError(f"planted pool exhausted (generated {generated})")

    out = [dict(r, planted=False) for r in rows]
    match = Counter()
    for i in chosen:
        item, lvl = take(key(_meta(rows[i])))
        match[lvl] += 1
        out[i] = {"prompt": [{"role": "user", "content": item["question"]}],
                  "ground_truth": str(item["answer"]),
                  "task": task,
                  "metadata": json.dumps(item.get("metadata", {})),
                  "planted": True}
        for extra in set(rows[i]) - set(out[i]):   # keep any columns the env added (is_atom, check, ...)
            out[i][extra] = rows[i][extra]
    natural = sum(r["ground_truth"] == answer for r in out) - n
    stats = {"fraction": fraction, "answer": answer, "n_rows": len(rows), "n_planted": n,
             "natural_with_answer": natural, "reward_share_of_constant": (n + natural) / len(rows),
             "pool_generated": generated, "pool_skipped": dict(skipped), "match": dict(match)}
    return out, stats
