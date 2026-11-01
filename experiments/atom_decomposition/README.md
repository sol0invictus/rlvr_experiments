# atom_decomposition

Phase C of the RLVR study: find a task's decomposition into atoms from solver traces, and test it
before any atom training.

For each task, an instrumented solver runs one algorithm and records every step as a node: L1 nodes
read the prompt, L2 nodes do one operation on values handed to them. Each node becomes an atom, a
self-contained question with an automatically checked answer. A task can have several candidate
graphs (mini_sudoku: constraint propagation and backtracking; gcd: Euclid and prime factorisation).

The test (eval only): ask the SFT model each instance's atoms and the full task, then check whether
atom success predicts full-task success per instance better than difficulty settings alone (gate b),
and whether one of two candidate graphs predicts clearly better (gate a: mini_sudoku, gcd).

| Task | Graphs | L1 nodes | L2 nodes |
|---|---|---|---|
| basic_arithmetic | precedence | parse (full parenthesisation) | one per operation, operands given |
| mini_sudoku | propagation, backtracking | numbers given in each row/column/box (12) | candidates of a cell on the current grid |
| maze | bfs | locate start, locate goal | one BFS layer: cells first reached at distance d+1 |
| gcd | euclid, factorisation | read the numbers | Euclid: one remainder per step; factorisation: smallest prime factor, division, then the shared primes |

```bash
# build the atom set (checks every trace against reasoning-gym's answer)
python -m experiments.atom_decomposition.build_atoms --n 500 --seed 2027 \
    --out_dir experiments/atom_decomposition/results/atoms_seed2027
# pilot: full tasks only, any reasoning-gym task, k samples at T=1.0 + 1 at T=0.1
python -m experiments.atom_decomposition.eval_atoms full --model /data/rlvr/sft_qwen3_4b \
    --tasks basic_arithmetic,maze --n 200 --seed 2026 --k 4 --t01 \
    --out_dir experiments/atom_decomposition/results/pilot_seed2026
# gate run: every unique atom (8 samples) + every instance's full task (8 samples, + 1 at T=0.1)
python -m experiments.atom_decomposition.eval_atoms atoms --model /data/rlvr/sft_qwen3_4b \
    --build_dir experiments/atom_decomposition/results/atoms_seed2027 \
    --out_dir experiments/atom_decomposition/results/eval_seed2027 --t01
# power of the gates on the real graphs, with planted truths (CPU only)
python -m experiments.atom_decomposition.fit_gates power \
    --build_dir experiments/atom_decomposition/results/atoms_seed2027 --task mini_sudoku \
    --n_sims 100 --k_atoms 4,8 --out power_mini_sudoku.json
# gate verdicts on real eval results (atom_results.jsonl, full_results.jsonl)
python -m experiments.atom_decomposition.fit_gates real \
    --build_dir experiments/atom_decomposition/results/atoms_seed2027 \
    --results_dir experiments/atom_decomposition/results/eval_seed2027 --out gates.json
# tests
python -m pytest experiments/atom_decomposition/tests -q
```

Pinned to reasoning-gym 0.1.25. The pre-registered criteria live in the study notes (PHASE_C.md).
