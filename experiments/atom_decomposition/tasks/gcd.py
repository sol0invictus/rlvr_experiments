"""
gcd: two candidate graphs.

  "euclid"         repeated remainders: (a, b) -> (b, a mod b) until the remainder is 0; the last
                   divisor is the GCD. One node per remainder. More than two numbers are folded
                   left: gcd(gcd(a, b), c).
  "factorisation"  trial-divide each number into primes (one node for each smallest prime factor,
                   one for each division), then one node multiplying the primes the numbers share.

Both share one L1 node (read the numbers). Unlike the two mini_sudoku graphs, these do different
operations, so their L2 questions differ by construction.
"""

import math
from collections import Counter
from functools import reduce
from typing import List

from ..common import Node, fmt_ints

GRAPHS = ("euclid", "factorisation")


def _read_node(numbers) -> Node:
    return Node("read", "L1", "read_numbers",
                f"Find the Greatest Common Divisor (GCD) of these numbers: {', '.join(map(str, numbers))}.\n"
                "Do not compute anything yet: which numbers are given? List them separated by commas.",
                fmt_ints(numbers), "int_set")


def _smallest_prime_factor(m: int) -> int:
    p = 2
    while p * p <= m:
        if m % p == 0:
            return p
        p += 1
    return m


def trace_euclid(numbers) -> List[Node]:
    nodes = [_read_node(numbers)]
    g, g_node = numbers[0], "read"
    for n in numbers[1:]:
        a, b = max(g, n), min(g, n)
        prev = g_node
        while b:
            r = a % b
            nid = f"mod{len(nodes)}"
            nodes.append(Node(nid, "L2", "remainder",
                              f"What is the remainder when {a} is divided by {b}? Give only the number.",
                              str(r), "int", ["read", prev] if prev != "read" else ["read"]))
            prev = nid
            a, b = b, r
        g, g_node = a, prev
    return nodes


def trace_factorisation(numbers) -> List[Node]:
    nodes = [_read_node(numbers)]
    factor_nodes, factorisations = [], []
    for n in numbers:
        m, primes, prev = n, [], "read"
        while m > 1:
            p = _smallest_prime_factor(m)
            nid = f"spf{len(nodes)}"
            nodes.append(Node(nid, "L2", "smallest_prime_factor",
                              f"What is the smallest prime factor of {m}? Give only the number.",
                              str(p), "int", [prev]))
            did = f"div{len(nodes)}"
            nodes.append(Node(did, "L2", "divide", f"Compute {m} / {p}. Give only the resulting integer.",
                              str(m // p), "int", [nid]))
            primes.append(p)
            m //= p
            prev = did
        factor_nodes.append(prev)
        factorisations.append(primes)
    common = reduce(lambda x, y: x & y, (Counter(f) for f in factorisations))
    listed = "; ".join(f"{n} = {' x '.join(map(str, f)) if f else '1 (no prime factors)'}"
                       for n, f in zip(numbers, factorisations))
    nodes.append(Node("common", "L2", "common_factors",
                      f"Prime factorisations: {listed}. Multiply together the prime factors the numbers have in "
                      "common, each as many times as it appears in all of them (answer 1 if none). Give only "
                      "the resulting integer.",
                      str(math.prod(common.elements())), "int", factor_nodes))
    return nodes


def trace(sample: dict, graph: str) -> List[Node]:
    numbers = sample["metadata"]["numbers"]
    if graph == "euclid":
        return trace_euclid(numbers)
    if graph == "factorisation":
        return trace_factorisation(numbers)
    raise ValueError(graph)


def final_value(sample: dict, graph: str) -> int:
    """The GCD each graph arrives at (both must equal reasoning-gym's answer)."""
    nodes = trace(sample, graph)
    if graph == "factorisation":
        return int(nodes[-1].answer)
    # Euclid: each run ends on a zero remainder; the divisor of that step is the run's GCD
    g = sample["metadata"]["numbers"][0]
    for n in sample["metadata"]["numbers"][1:]:
        a, b = max(g, n), min(g, n)
        while b:
            a, b = b, a % b
        g = a
    return g


def difficulty_features(sample: dict) -> dict:
    nums = sample["metadata"]["numbers"]
    return {"num_terms": len(nums), "log_max": math.log10(max(nums)), "log_min": math.log10(min(nums))}
