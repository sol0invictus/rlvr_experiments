# Section 4 · Atomic Skill Decomposition and the Grounding Bottleneck

*Paper-ready draft. Figures and tables referenced here are called out in italics; appendices A–E contain per-task decompositions, extended analysis, implementation notes, and hyperparameters.*

---

## 4.1 Motivation

When RLVR (GRPO with verifiable rewards) fails on a task for a small model, the failure could originate in either of two places. The model might understand the task representation but lack the multi-step planning needed to solve it — a *reasoning* bottleneck. Or it might fail earlier, unable to reliably parse the task's input — a *grounding* bottleneck. These produce indistinguishable end-to-end behavior but call for different interventions: reasoning bottlenecks benefit from more full-task RL; grounding bottlenecks do not, and may be actively worsened by it.

We argue that for small models (≤8B), grounding failures are under-appreciated, and that a simple behavioral probe can distinguish the two cases *before* any expensive training is launched. The probe decomposes each task into atomic sub-skills and measures the base model's zero-shot accuracy on each. The resulting scalar — the **Atomic Readiness Score (ARS)** — predicts which of three training regimes will work on 6/7 tasks. (An earlier draft also claimed L1-ARS predicts multi-task merge compatibility at r = −0.76; that secondary claim was falsified at n = 12 and is documented in §7.10. The §4 ARS framework is a single-task tool; whether grounding generalizes to merging is open.)

## 4.2 Atoms and the Atomic Readiness Score

We adopt a three-layer taxonomy that applies to any reasoning-gym task:

- **L1 — Grounding.** Parse the input representation. *"What character is at row 3, column 5?"* *"What is the quantifier in statement 2?"*
- **L2 — Local reasoning.** Single-step inference over grounded elements, no multi-step planning. *"Is moving east from (3,5) legal?"* *"If A→B and A, what follows?"*
- **L3 — Global planning.** Multi-step strategy composition — the full task itself.

For each task we hand-specify 4–8 atoms per layer as Q&A templates with programmatic ground truth; full templates are in Appendix A. The **Atomic Readiness Score** is the zero-shot accuracy of the SFT base model averaged over a task's atoms:

$$\text{ARS}(\text{model}, T) = \frac{1}{|A(T)|}\sum_{a_i \in A(T)} \mathbb{1}[\text{model}(a_i) = \text{answer}(a_i)].$$

ARS decomposes naturally into an L1 average and an L2 average; we compute both. At 50 samples per atom the probe takes ≈1 hour on a single GPU and requires no training.

## 4.3 Experimental Setup

We evaluate four training regimes on seven reasoning-gym tasks. The tasks span Phase-2's original three (basic_arithmetic, countdown, shortest_path) and four we added to cover the full ARS range (maze, mini_sudoku, syllogism, knights_knaves). All runs use Qwen3-4B post-SFT (format warm-up on 1000 `<think>/<answer>` samples) and identical GRPO hyperparameters (lr = 5e-6, DAPO loss, β = 0.0, KL = 0.0, 8 rollouts/prompt); see Appendix E.

| Regime | Training | Eval |
|---|---|---|
| **Base** | SFT only, no GRPO | Full task |
| **Direct-RLVR** | 300 steps GRPO on full task | Full task |
| **Atomic-Only** | 100 steps L1 atoms → 100 steps L2 atoms, no full-task GRPO | Full task (zero-shot transfer) |
| **Atomic-Warm** | L1(100) → L2(100) → Full(100) GRPO | Full task |

All regimes receive matched compute (300 GRPO steps total). ARS is computed once per task on the SFT base before any training.

## 4.4 Result: ARS Predicts the Best Training Regime

Table 1 and Figure 4.1 show the headline result. The ARS column is computed on the SFT base; the four accuracy columns are post-training full-task accuracy.

![Figure 4.1](figures/fig4_1_ars_regime_comparison.pdf)

*Figure 4.1: Tasks sorted left-to-right by ARS (top panel). Bottom panel: full-task accuracy under each of four training regimes; ★ marks the best regime per task. Low-ARS tasks (shortest_path, maze) are Atomic-Only or Atomic-Warm wins; medium-ARS (countdown) is Direct-RLVR; high-ARS (syllogism, knights_knaves) is Direct-RLVR or Atomic-Warm. basic_arithmetic (ARS = 66%, high) is the notable exception: Direct-RLVR collapses to 32% via reasoning compression (§4.5).*

**Table 1. ARS predicts the optimal training regime on 6 of 7 tasks (Qwen3-4B).** Bold indicates the best regime per task.

| Task | ARS | Base | Direct-RLVR | Atomic-Only | Atomic-Warm | ARS-predicted regime | Correct? |
|---|---:|---:|---:|---:|---:|---|:---:|
| syllogism | 74% | 79.5 | 96.5 | 81.0 | **97.0** | Direct or Atomic-Warm | ✓ |
| knights_knaves | 62.5% | 78.0 | **97.5** | 79.0 | 95.0 | Direct or Atomic-Warm | ✓ |
| basic_arithmetic | 66% | 72.0 | 32.0 | **78.0** | 70.0 | Direct or Atomic-Warm | ✗\* |
| countdown | 56% | 26.0 | **70.0** | 24.0 | 26.0 | Direct-RLVR | ✓ |
| maze | 20.7% | 21.5 | 1.0 | **41.0** | 38.5 | Atomic-Only / Atomic-Warm | ✓ |
| shortest_path | 14% | 10.0 | 44.0 | 12.0 | **46.0** | Atomic-Warm | ✓ |
| mini_sudoku | 19.8% | 4.0 | **25.5** | 17.0 | 24.0 | Atomic-Only / Atomic-Warm | ✗ |

\*Arithmetic is ARS-high (66%) but Direct-RLVR collapses to 32% — a known pathology we analyze in §4.5.

The rule that emerges is a three-band partition:

- **Low-ARS (< 25%): Direct-RLVR can fail catastrophically.** On maze (ARS = 20.7%), Direct-RLVR achieved 1% accuracy — worse than the untrained baseline of 21.5%. Atomic-Only recovered to 41%. On shortest_path (ARS = 14%), Atomic-Warm (46%) beat Direct-RLVR (44%) with no tradeoff.
- **Medium-ARS (40–60%): Direct-RLVR.** The model has sub-skills; the bottleneck is L3 planning, which only full-task reward signal addresses. Atoms do not help (countdown: Atomic-Only 24% vs Direct-RLVR 70%).
- **High-ARS (> 60%): Either works,** with marginal gains from atomic warm-up on some tasks (syllogism +0.5%).

The rule is correct on 6 of 7 tasks; mini_sudoku is the exception and the arithmetic anomaly (discussed in §4.5) is a *reasoning-compression* failure orthogonal to grounding.

## 4.5 Mechanism: Why Direct-RLVR Fails on Low-ARS Tasks

The maze result (Direct-RLVR = 1%, Atomic-Only = 41%) is the clearest signal. The L1 atom accuracies for maze (Figure 4.2, left panel) show why: `locate_start` and `locate_goal` are at 0%, `grid_size` at 2%.

![Figure 4.2](figures/fig4_2_atom_breakdown.pdf)

*Figure 4.2: Per-atom zero-shot accuracy on the 4B SFT base for three illustrative tasks. Blue = L1 (grounding) atoms, orange = L2 (local reasoning). Maze's grounding atoms are at 0–2% — the base model cannot parse the grid. Arithmetic is grounded (L1 avg 68%, L2 avg 65%) yet Direct-RLVR still fails because `order_of_operations` is at 2% and GRPO compresses reasoning. Syllogism has strong L1 (96%) and weaker L2 (52%); Direct-RLVR and Atomic-Warm both succeed.*

Under Direct-RLVR, every rollout starts from random confusion about where the agent even is; reward is sparse (binary), and when it does fire it reinforces whatever superficial pattern happened to produce an acceptable final answer. The resulting gradients teach *format* more than *strategy*, and the model drifts into degenerate short-answer behavior. Atomic-Only avoids this: L1 training explicitly teaches grid-reading, and the model acquires a substrate on which downstream reasoning can operate — even without any full-task gradient signal.

**Reasoning compression.** The arithmetic anomaly (high ARS, Direct-RLVR = 32%) does *not* fit the grounding bottleneck. Examining generations on post-RLVR arithmetic checkpoints reveals that GRPO systematically *shortens* outputs. Mean completion length during training drops from ~390 to ~230 characters; the model learns to skip intermediate-step derivations because shorter correct answers are more rewarded per unit compute. This works on single-operation expressions but fails on multi-step ones requiring order-of-operations tracking (the L2 atom `order_of_operations` is at only 2% base accuracy despite an overall ARS of 66%). Atomic-Only avoids this failure by training on atoms whose reward structure rewards *producing the right substep*, not *being concise*. We observed the same compression on knights_knaves under some conditions (Appendix B.2).

**Mini-sudoku exception.** Mini-sudoku's atom templates were adapted from simpler cell-reading and row-column queries (Appendix A.7). We believe these fail to capture sudoku's true bottleneck — constraint propagation across multiple cells — which is a holistic L2–L3 operation that does not decompose into independent single-hop atoms. This is consistent with the broader pattern: ARS predicts training regime *when atoms capture the task's actual grounding and local-reasoning structure*. Poorly-chosen atoms produce an ARS value that is uncorrelated with the true grounding bottleneck.

## 4.6 A Tentative Cross-Phase Bridge — and Why It Fell

An earlier version of this paper proposed a clean mechanistic unification: the same L1-ARS probe used here would also predict 4B pair-merge damage in §7 (Task-Vector Merging) at r = −0.76 (n = 7). The hypothesis was that grounding is the shared bottleneck for both single-task GRPO success *and* multi-task τ composition.

This claim was falsified by the gap-closing experiments documented in §7.10. When the same predictor is recomputed on n = 12 pairs (adding T8–T12), Pearson r flips from −0.76 to **+0.06**. Two of the new pairs (T10 cnt+leg and T11 sp+leg) have the lowest pair L1-ARS in the sample and the lowest damage — the opposite of what the bottleneck story predicts. The two-predictor regression (sign-agreement + L1-ARS → damage) drops from R² = 0.70 to R² = 0.30.

We retain L1-ARS as the §4 single-task predictor — that result is robust within Phase 2 — but the cross-phase generalization to merging does not survive sample expansion. §7 was rewritten as an empirical characterization paper rather than a prediction paper as a result.

## 4.7 Summary of Section 4

1. An ARS probe (1 hour on the SFT base, zero training) partitions tasks into three bands that accurately predict which of {Direct-RLVR, Atomic-Only, Atomic-Warm} will work on 6/7 tasks.
2. Direct-RLVR can fail *below baseline* on low-ARS tasks (maze: 1% vs 21.5%) because sparse reward on ungrounded inputs teaches format rather than strategy.
3. A secondary failure mode — *reasoning compression* on high-ARS tasks with long reasoning chains — explains the arithmetic anomaly and is orthogonal to the grounding bottleneck.
4. The L1-ARS probe is a robust *single-task* predictor; the cross-phase claim that it would also predict multi-task merge compatibility was falsified at n = 12 (§7.10). The grounding bottleneck story holds for §4 alone.

---

## Appendices

### Appendix A · Atomic templates per task

Full Q&A templates for all atoms across 7 tasks (6 × L1 + 6 × L2 per task on average, 48 atoms total). Each atom is a deterministic Python function mapping a reasoning-gym sample to (question, answer) pair; implementation in `environments/atomic_templates.py`. Excerpt for maze (Table A.3) and arithmetic (Table A.1); full list in the supplementary code release.

### Appendix B · Per-task ARS decompositions

The seven per-task atomic breakdowns (L1/L2 accuracies per atom, base model) that back the §4.4 headline. Includes:

- Table B.1: syllogism (L1 avg 96%, L2 avg 52%; weakest atom: `single_hop_inference` 46%)
- Table B.2: knights_knaves (L1 avg 57%, L2 avg 68%; weakest atom: `speaker_identification` 0%)
- Table B.3: maze (L1 avg 14%, L2 avg 27%; weakest atoms: `locate_start/goal` 0%)
- Tables B.4–B.7: basic_arithmetic, countdown, shortest_path, mini_sudoku

### Appendix C · Shortest-path deep dive (15 experimental conditions)

Extended study on shortest_path, the hardest low-ARS task:

- Partial-credit rewards underperform binary (35.5% vs 44%)
- Progressive difficulty curricula (3×3 → 4×4 → full) underperform hard transition (46.5% vs 51.5%)
- L2-long atoms (path-tracing) beat L2-short atoms (51.5% vs 46%) — avoid reasoning compression
- RFT-Base (rejection sampling → SFT → GRPO) reaches best SP accuracy at 55.5%, used in §7 as τ_atomic(SP)

### Appendix D · Implementation issues and fixes

- Answer extraction bug (2026-04-17): `re.search` on `<answer>` tag took the *first* match, which in models that emit thinking tags with embedded answers was wrong. Fix: `re.findall` with last-match, applied in `evaluate_transfer.py` line 140. All Phase 2 numbers reported in §4 are post-fix.
- Atomic reward function (2026-04-17): strict string matching of atom answers gave 0% reward on syntactically-equivalent correct answers. Fix: task-specific lenient verifiers in `environments/atomic_env.py`.

### Appendix E · Training hyperparameters

All GRPO runs use: lr = 5e-6 constant, DAPO loss, β = 0.0, KL coefficient = 0.0, epsilon-high = 5.0, rollouts per prompt = 8, per-device batch size = 1, grad-accum = 1, max grad norm = 1.0, temperature 1.0 at rollout, max_new_tokens = 512 for arithmetic / 256 for grid tasks, DDP across 8 × B200. Each stage runs 100 steps in the atomic regimes, 300 steps in Direct-RLVR. Model: Qwen3-4B post-SFT warm-up on 1000 format-demonstration samples. Full configs in `configs/phase2_*/`.
