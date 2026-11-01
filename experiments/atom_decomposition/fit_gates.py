"""
Phase C gates: does atom success predict full-task success per instance (gate b), and does the
hypothesised graph predict better than its alternative (gate a)? Rules are fixed in PHASE_C.md §4.

Two modes:
    real   score the eval results written by eval_atoms.py
    power  plant a known truth on the real atom graphs, simulate outcomes, and measure how often
           the gates get it right (power) or wrong (false passes / wrong picks)

Usage (from the repo root):
    python -m experiments.atom_decomposition.fit_gates real \
        --build_dir experiments/atom_decomposition/results/atoms_seed2027 \
        --results_dir experiments/atom_decomposition/results/eval_seed2027 --out gates.json
    python -m experiments.atom_decomposition.fit_gates power \
        --build_dir experiments/atom_decomposition/results/atoms_seed2027 \
        --task mini_sudoku --n_sims 100 --out power_mini_sudoku.json

Result files expected in --results_dir (one JSON object per line):
    atom_results.jsonl  {"key", "correct": [0/1 per sample], "has_answer_tag": [0/1 per sample]}
    full_results.jsonl  {"task", "idx", "correct": [...], "has_answer_tag": [...]}
"""

import argparse
import json
import math
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .tasks import TASKS

# PHASE_C.md §4.4–4.6
MARGIN = 0.03          # relative log-loss improvement that counts as "clearly better"
MIN_MIXED = 30         # instances needed with >=1 success and with >=1 failure
N_FOLDS = 5
ATOM_FEATURES = ["F1_chain", "F2_grounding", "F3_weakest"]


# ----------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------

def load_build(build_dir):
    inst = [json.loads(l) for l in open(os.path.join(build_dir, "instances.jsonl"))]
    atoms = defaultdict(list)   # (task, idx, graph) -> atom rows
    for l in open(os.path.join(build_dir, "atoms.jsonl")):
        a = json.loads(l)
        atoms[(a["task"], a["idx"], a["graph"])].append(a)
    return inst, atoms


def load_results(results_dir, temperature=1.0):
    """Atom and full-task outcomes at the gate temperature (full_results.jsonl also holds the
    1-sample T = 0.1 accuracy rows, which the gates do not use)."""
    atom_obs, full_obs = {}, {}
    for l in open(os.path.join(results_dir, "atom_results.jsonl")):
        r = json.loads(l)
        atom_obs[r["key"]] = (sum(r["correct"]), len(r["correct"]), float(np.mean(r["has_answer_tag"])))
    for l in open(os.path.join(results_dir, "full_results.jsonl")):
        r = json.loads(l)
        if r.get("temperature", temperature) != temperature:
            continue
        full_obs[(r["task"], r["idx"])] = (sum(r["correct"]), len(r["correct"]))
    return atom_obs, full_obs


# ----------------------------------------------------------------------
# Features (§4.4)
# ----------------------------------------------------------------------

def smoothed(s, k):
    return (s + 0.5) / (k + 1)


def graph_features(atom_rows, info, atom_obs):
    """F1..F4 for one (instance, graph) from its asked atoms."""
    l1, l2, fmt, allp = [], [], [], []
    for a in atom_rows:
        s, k, f = atom_obs[a["key"]]
        p = smoothed(s, k)
        (l1 if a["layer"] == "L1" else l2).append(math.log(p))
        allp.append(p)
        fmt.append(f)
    scale = info["n_L2_total"] / max(info["n_L2_asked"], 1)
    return {
        "F1_chain": scale * sum(l2),
        "F2_grounding": float(np.mean(l1)) if l1 else 0.0,
        "F3_weakest": min(allp),
        "F4_format": float(np.mean(fmt)),
        "log_n_L2": math.log(info["n_L2_total"] + 1),
    }


def task_table(task, inst, atoms, atom_obs, full_obs):
    """Per graph: (X_baseline, X_full), plus successes s, samples k and the 'graphs differ' mask."""
    rows = [r for r in inst if r["task"] == task]
    diff_keys = sorted(rows[0]["difficulty"])
    graphs = TASKS[task].GRAPHS
    s = np.array([full_obs[(task, r["idx"])][0] for r in rows], float)
    k = np.array([full_obs[(task, r["idx"])][1] for r in rows], float)
    out = {}
    for g in graphs:
        B, A = [], []
        for r in rows:
            f = graph_features(atoms[(task, r["idx"], g)], r["graphs"][g], atom_obs)
            B.append([r["difficulty"][d] for d in diff_keys] + [f["log_n_L2"], f["F4_format"]])
            A.append([f[x] for x in ATOM_FEATURES])
        B, A = np.array(B, float), np.array(A, float)
        out[g] = (B, np.hstack([B, A]))
    differ = None
    if len(graphs) == 2:
        def l2keys(idx, g):
            return {a["key"] for a in atoms[(task, idx, g)] if a["layer"] == "L2"}
        differ = np.array([l2keys(r["idx"], graphs[0]) != l2keys(r["idx"], graphs[1]) for r in rows])
    return out, s, k, differ


# ----------------------------------------------------------------------
# Predictor and comparisons (§4.5–4.6)
# ----------------------------------------------------------------------

def oof_loss(X, s, k, n_repeats, seed):
    """Out-of-fold per-instance log-loss (mean over the instance's samples) and predicted p,
    averaged over repeated 5-fold CV. One weighted row pair per instance = one row per sample."""
    n = len(s)
    loss, prob = np.zeros(n), np.zeros(n)
    for r in range(n_repeats):
        for tr, te in KFold(N_FOLDS, shuffle=True, random_state=seed + r).split(X):
            m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
            m.fit(np.vstack([X[tr], X[tr]]), np.r_[np.ones(len(tr)), np.zeros(len(tr))],
                  logisticregression__sample_weight=np.r_[s[tr], k[tr] - s[tr]])
            p = np.clip(m.predict_proba(X[te])[:, 1], 1e-6, 1 - 1e-6)
            loss[te] += -(s[te] * np.log(p) + (k[te] - s[te]) * np.log(1 - p)) / k[te]
            prob[te] += p
    return loss / n_repeats, prob / n_repeats


def auc(prob, s, k):
    w = np.r_[s, k - s]
    if s.sum() == 0 or (k - s).sum() == 0:
        return None
    return float(roc_auc_score(np.r_[np.ones(len(s)), np.zeros(len(s))], np.r_[prob, prob], sample_weight=w))


def compare(loss_ref, loss_new, n_boot, rng, mask=None, alpha=0.05):
    """Is `new` better than `ref`? Paired bootstrap over instances on Δ = LL(ref) − LL(new),
    with a two-sided (1 − alpha) interval."""
    if mask is not None:
        loss_ref, loss_new = loss_ref[mask], loss_new[mask]
    d = loss_ref - loss_new
    boots = d[rng.integers(0, len(d), (n_boot, len(d)))].mean(1)
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"LL_ref": float(loss_ref.mean()), "LL_new": float(loss_new.mean()), "delta": float(d.mean()),
            "rel": float(d.mean() / loss_ref.mean()), "ci": [float(lo), float(hi)], "ci_level": 1 - alpha,
            "n": int(len(d))}


def evaluate_task(task, inst, atoms, atom_obs, full_obs, n_repeats=20, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    table, s, k, differ = task_table(task, inst, atoms, atom_obs, full_obs)
    graphs = TASKS[task].GRAPHS
    res = {"task": task, "accuracy": float(s.sum() / k.sum()),
           "n_with_success": int((s > 0).sum()), "n_with_failure": int((s < k).sum())}
    res["evaluable"] = res["n_with_success"] >= MIN_MIXED and res["n_with_failure"] >= MIN_MIXED
    losses = {}
    res["gate_b"] = {}
    alpha_b = 0.05 / len(graphs)   # Bonferroni across the task's graphs: 97.5% intervals for two
    for g in graphs:
        XB, XF = table[g]
        lB, pB = oof_loss(XB, s, k, n_repeats, seed)
        lF, pF = oof_loss(XF, s, k, n_repeats, seed)
        losses[g] = lF
        c = compare(lB, lF, n_boot, rng, alpha=alpha_b)
        c["auc_baseline"], c["auc_with_atoms"] = auc(pB, s, k), auc(pF, s, k)
        c["pass"] = bool(res["evaluable"] and c["rel"] >= MARGIN and c["ci"][0] > 0)
        res["gate_b"][g] = c
    # gate (b) passes for the task if any of its graphs passes (intervals corrected above)
    res["gate_b_verdict"] = ("not evaluable" if not res["evaluable"]
                             else "pass" if any(res["gate_b"][g]["pass"] for g in graphs) else "fail")
    if len(graphs) == 2:
        hyp, alt = graphs
        res["gate_a"] = {}
        for name, mask in (("all", None), ("graphs_differ", differ)):
            fwd = compare(losses[alt], losses[hyp], n_boot, rng, mask)   # hyp better than alt?
            rev = compare(losses[hyp], losses[alt], n_boot, rng, mask)   # alt better than hyp?
            if not res["evaluable"]:
                verdict = "not evaluable"
            elif fwd["rel"] >= MARGIN and fwd["ci"][0] > 0:
                verdict = hyp
            elif rev["rel"] >= MARGIN and rev["ci"][0] > 0:
                verdict = alt
            else:
                verdict = "tie"
            res["gate_a"][name] = {"hyp_vs_alt": fwd, "verdict": verdict}
        res["gate_a_verdict"] = res["gate_a"]["all"]["verdict"]
    return res


# ----------------------------------------------------------------------
# Power simulation
# ----------------------------------------------------------------------

def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def _z(x):
    x = np.asarray(x, float)
    sd = x.std()
    return (x - x.mean()) / sd if sd > 0 else x * 0


def simulate(task, inst, atoms, truth, w, base_acc, rng, k_atom=8, k_full=8,
             atom_acc=0.8, atom_sd=1.0, coupling=0.5, scale=1.5):
    """Plant a truth and draw outcomes with the real graphs' structure.

    Each unique atom question gets a true success logit: logit(atom_acc) + atom_sd·noise
    − coupling·d_i, where d_i is the instance's z-scored difficulty (so atoms and the full task
    share difficulty, the confound gate (b) has to beat). Full-task logit:
        c0 + scale·(w·z(S_i) − (1 − w)·d_i),
    S_i = Σ log p over the truth graph's atoms, residualised on difficulty and graph size;
    truth = "difficulty" means w = 0, and c0 sets mean accuracy to base_acc.
    """
    rows = [r for r in inst if r["task"] == task]
    diff_keys = sorted(rows[0]["difficulty"])
    d = np.mean([_z([r["difficulty"][x] for r in rows]) for x in diff_keys], axis=0)
    d = _z(d)
    p_true = {}
    for i, r in enumerate(rows):
        for g in TASKS[task].GRAPHS:
            for a in atoms[(task, r["idx"], g)]:
                if a["key"] not in p_true:
                    p_true[a["key"]] = _sigmoid(math.log(atom_acc / (1 - atom_acc))
                                                + atom_sd * rng.standard_normal() - coupling * d[i])
    if truth == "difficulty":
        eta = -scale * d
    else:
        S = np.array([sum(math.log(p_true[a["key"]]) for a in atoms[(task, r["idx"], truth)])
                      for r in rows])
        # Keep only the part of S that the baseline (difficulty + graph size) cannot explain, so w is
        # the share of the outcome driven by atom skill beyond what gate (b) already controls for.
        base = np.column_stack([np.ones(len(rows))] + [_z([r["difficulty"][x] for r in rows]) for x in diff_keys]
                               + [_z([math.log(r["graphs"][truth]["n_L2_total"] + 1) for r in rows])])
        S = S - base @ np.linalg.lstsq(base, S, rcond=None)[0]
        eta = scale * (w * _z(S) - (1 - w) * d)
    lo, hi = -20.0, 20.0
    for _ in range(60):
        c0 = (lo + hi) / 2
        lo, hi = (c0, hi) if _sigmoid(eta + c0).mean() < base_acc else (lo, c0)
    p_full = _sigmoid(eta + c0)
    atom_obs = {key: (int(rng.binomial(k_atom, p)), k_atom, 1.0) for key, p in p_true.items()}
    full_obs = {(task, r["idx"]): (int(rng.binomial(k_full, p_full[i])), k_full) for i, r in enumerate(rows)}
    return atom_obs, full_obs


_SHARED = {}


def _init_worker(inst, atoms):
    _SHARED["inst"], _SHARED["atoms"] = inst, atoms


def _one_sim(args):
    task, truth, w, base_acc, k_atom, seed, n_repeats, n_boot = args
    inst, atoms = _SHARED["inst"], _SHARED["atoms"]
    rng = np.random.default_rng(seed)
    atom_obs, full_obs = simulate(task, inst, atoms, truth, w, base_acc, rng, k_atom=k_atom)
    r = evaluate_task(task, inst, atoms, atom_obs, full_obs, n_repeats=n_repeats, n_boot=n_boot, seed=seed)
    return r["gate_b_verdict"], r.get("gate_a_verdict"), r["evaluable"]


def power(task, inst, atoms, n_sims, weights, base_accs, k_atoms=(8,), n_repeats=2, n_boot=500,
          workers=None, seed=0):
    graphs = TASKS[task].GRAPHS
    truths = list(graphs) + ["difficulty"]
    out = []
    settings = [(k_atom, base_acc, truth, w) for k_atom in k_atoms for base_acc in base_accs
                for truth in truths for w in (weights if truth != "difficulty" else [0.0])]
    with ProcessPoolExecutor(workers, initializer=_init_worker, initargs=(inst, atoms)) as ex:
        for k_atom, base_acc, truth, w in settings:
            jobs = [(task, truth, w, base_acc, k_atom, seed + 1000 * j, n_repeats, n_boot)
                    for j in range(n_sims)]
            res = list(ex.map(_one_sim, jobs))
            row = {"k_atom": k_atom, "base_acc": base_acc, "truth": truth, "w": w, "n_sims": n_sims,
                   "evaluable": float(np.mean([e for _, _, e in res])),
                   "gate_b_pass": float(np.mean([b == "pass" for b, _, _ in res]))}
            if len(graphs) == 2:
                for g in list(graphs) + ["tie"]:
                    row[f"gate_a_{g}"] = float(np.mean([a == g for _, a, _ in res]))
            out.append(row)
            print(json.dumps(row), flush=True)
    return out


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="mode", required=True)
    pr = sub.add_parser("real")
    pr.add_argument("--build_dir", required=True)
    pr.add_argument("--results_dir", required=True)
    pr.add_argument("--tasks", default=None)
    pr.add_argument("--out", required=True)
    pp = sub.add_parser("power")
    pp.add_argument("--build_dir", required=True)
    pp.add_argument("--task", required=True)
    pp.add_argument("--n_sims", type=int, default=100)
    pp.add_argument("--weights", default="0.25,0.5,0.75")
    pp.add_argument("--base_accs", default="0.14,0.4")
    pp.add_argument("--k_atoms", default="8", help="samples per atom question to simulate, e.g. 4,8")
    pp.add_argument("--workers", type=int, default=None)
    pp.add_argument("--out", required=True)
    a = p.parse_args()

    inst, atoms = load_build(a.build_dir)
    if a.mode == "real":
        atom_obs, full_obs = load_results(a.results_dir)
        tasks = a.tasks.split(",") if a.tasks else sorted({r["task"] for r in inst})
        res = [evaluate_task(t, inst, atoms, atom_obs, full_obs) for t in tasks]
        evaluable = [r for r in res if r["evaluable"]]
        summary = {"tasks": res, "gate_b_overall": (
            "not evaluable" if len(evaluable) < 2
            else "pass" if all(r["gate_b_verdict"] == "pass" for r in evaluable) else "fail")}
    else:
        summary = {"task": a.task, "rows": power(
            a.task, inst, atoms, a.n_sims, [float(x) for x in a.weights.split(",")],
            [float(x) for x in a.base_accs.split(",")], [int(x) for x in a.k_atoms.split(",")],
            workers=a.workers)}
    with open(a.out, "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
