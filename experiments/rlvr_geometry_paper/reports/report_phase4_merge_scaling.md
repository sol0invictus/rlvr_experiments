# RLVR Phase 4 — Task-Vector Merging at Scale

**Date:** 2026-04-23 → 2026-05-12 (current)
**Model:** Qwen2.5-3B, Qwen3-4B, Qwen3-8B (all post-SFT with `<think>/<answer>` format)
**Pods:** `sunguha-rlvr-phase2-qbgw-worker-0`, `sunguha-rlvr-paper-npdz`, `sunguha-rlvr-paper2-n25r` (Green cluster, P6-B200)
**Depends on:** Phase 1 transfer matrix, Phase 2 atomic curriculum, Phase 3 weight analysis

---

## Synthesis (read this first) — REVISED 2026-05-12 after Part 6 gap-closing experiments

The previous version of this report (pre 2026-05-09) claimed **two independent cross-scale merge-damage predictors** as the headline: a weight-space probe (3B magnitude-weighted sign-agreement → 4B damage, r = −0.81) and a behavioral probe (3B L1-grounding ARS → 4B damage, r = −0.76). Part 6 ran 12 additional pair merges at 4B (n=7 → n=19) and **both predictor claims collapsed**. The monotonic decay with n is the classic signature of a favorable-small-sample artifact.

Both predictor claims have been retracted. The surviving spine of the Phase-4 work is narrower but robust.

### Main takeaway (one sentence)

> **Simple weight-sum merging of RLVR task vectors has no cheap pre-probe for *which* merges will work, but it has one universal recipe: drop `mlp.down_proj` contributions. Layer-selectivity beyond that is pair-dependent, not task-intrinsic.**

### What survived (Tier 1 — robust across 3B/4B/8B and n=19)

1. **`mlp.down_proj` contributions are always safe to drop.** 40 of 48 (pair, task) cells at 4B across 7 pairs within ±3 pts of the full merge; mean-positive or neutral at 3B and 8B. Comes directly from Phase-3's "basis frozen at 0.00°–0.02°" finding as an actionable merge recipe. [F1 strengthened]
2. **Simple weight summation at s ≈ 1.0 produces usable 2-τ multi-skill models** at a τ-magnitude budget of 2–3 τ-equivalents. Uniform 1/N averaging (TIES/DARE's canonical baseline) under-shoots and loses to 2-task on 9/10 tasks. [F1, F2, F6 survive]
3. **Atomic-curriculum and direct-GRPO τ's compose differently** (Part 5 / F28 revised). M_aa (atomic+atomic) wins over M_dd (direct+direct) at every scale for feature-routing tasks (arith, leg_counting, knights_knaves); M_dd wins for shortest_path at every scale. Magnitudes vary (4B +7.9, 8B +0.3, 3B-Qwen2.5 −3.3) but the per-task sign pattern is cross-scale stable. [F28 revised]

### What was revised (Tier 2 — claim rescaled or scoped)

4. **The Group-1 (MLP-only-sufficient) / Group-2 (needs-full-attention) layer-selectivity split is pair-dependent, not task-intrinsic.** Originally claimed to classify *tasks* into merge-compatibility groups; Part 6 rerun on 6 more pairs showed **shortest_path classifies as Group-1 when paired with knights-knaves** (opposite of Part 3's original claim) and **syllogism classifies as Group-2 when paired with arithmetic**. Only `mini_sudoku` is consistently Group-2 across every pair it appears in. You cannot pre-classify a task for its merge recipe from task features alone. [F19, F23, F24 rescoped]
5. **`attn.{k,v,o}` rotation completes the Phase-3 picture** (Part 5/F31). k rotates 30–42°, v and o 50–70° across 7 pairs at 4B/8B — none are frozen. This is consistent with the pair-dependent layer-sufficiency — the attn.{k,v,o} circuit is measurably rewritten by many RLVR runs, so its contribution to a merge matters for some pairs. [F31 retained]

### What was falsified (Tier 3 — NEW negative results from Part 6)

6. **Magnitude-weighted sign agreement at 3B → 4B damage decays with n.** r = −0.81 (n=7) → −0.45 (n=12) → **−0.27 (n=19)**. Bootstrap CI crosses zero at every sample size ≥ 12. Spearman p never < 0.05. The original "validated predictor" claim was an artifact of a favorable n=7 sample. True population correlation ≈ −0.25 to −0.35. Right direction (higher mag_agree → lower damage) but too noisy for calibrated prediction. [F29 FALSIFIED as predictor; retained as weak directional indicator]
7. **L1-grounding ARS at 3B → 4B damage flips sign at n=12.** r = −0.76 (n=7) → **+0.06 (n=12)**. Bootstrap CI [−0.72, +0.67]. The Phase 2 ↔ Phase 4 unification via the grounding bottleneck does not survive sample expansion. [F33 FALSIFIED]
8. **A task-level routing probe does not predict Group membership.** r(routing_score, expected_group) = −0.31 at 4B. Designed to operationalize "does the base model already have the cross-position attention this task needs?" — the probe measures something, but not Group membership. [new negative]
9. **Per-feature residual search finds no stable predictor.** Ten candidate features tested; best single-feature |r| < 0.5 at n=12. "Task-level incompatibility" reaches r = 0.82 on full sample but r = 0.06 under leave-one-out (entirely overfit). No two-feature regression reaches R² > 0.55 at n=12. [new negative]
10. **Phase-1 transfer matrix, global τ cosine, per-layer flat-cosine opposition** all have r ≈ 0 against damage at n=19. [F7, F8, F9 retained — strengthened by larger n]

### The predictor-decay table

| Predictor | n=7 | n=12 | n=19 | Verdict |
|---|---:|---:|---:|---|
| 3B sign_agree → 4B damage (Pearson) | **−0.81** | −0.45 | **−0.27** | weak directional, not calibrated |
| 3B L1-ARS → 4B damage (Pearson) | **−0.76** | **+0.06** | — | falsified |
| 3B L1-ARS → 4B damage (Spearman) | — | +0.18 (p=0.58) | — | falsified |
| sign_agree 95% CI (Pearson) | [−0.99, −0.38] | [−0.87, +0.16] | [−0.59, +0.14] | never excludes zero past n=7 |

### The revised headline for a paper (Path A)

> RLVR 2-τ task-vector merging has a characterized but not-prospectively-predictable structure. (a) Dropping `mlp.down_proj` contributions is a geometrically-motivated, universally-safe merge recipe (40/48 cells at 4B). (b) Whether a pair merge can be further reduced to `mlp.up + mlp.gate` alone or needs the full `attn.{q,k,v,o}` circuit depends on the pair, not on individual task features. (c) Eight candidate pre-probes — including a magnitude-weighted sign-agreement predictor that appeared to work at r = −0.81 on n = 7 pairs — fail when tested on the expanded n = 19 sample. (d) Atomic-curriculum and direct-GRPO training regimes produce τ's that compose differently under merging, in a task-class-specific cross-scale-stable pattern. The paper is therefore an empirical characterization of RLVR task-vector compositionality, not a prospective-prediction paper.

### Key numbers (the ones a reader should walk away with)

| What | Number | Confidence |
|---|---|---|
| Damage range at 4B across n=19 pairs | **0.5 – 64.5 pts** | measured |
| `all_but_down` recipe: cells within ±3 of full merge | **40 / 48** at 4B | measured across 7 pairs × 8 tasks |
| Best single predictor of merge damage (3B sign-agree) | r = **−0.27** (n=19) | CI [−0.59, +0.14] — not significant |
| 1/N averaging mean (8 tasks) vs 2-task merge | 41.6% vs **51.2%** | 9/10 tasks lost |
| Atomic-vs-direct M_aa mean advantage (4B) | **+7.9 pts** | but +0.3 at 8B, −3.3 at 3B |
| SP+Arith merge: SP accuracy | **51.0%** | vs 44% Direct-RLVR alone, 10% SFT base |
| Worst catastrophic pair (T15 kk+sdk) | **64.5 pts damage** | KK dropped 47 pts |

### Experimental parts at a glance

| Part | Date | Experiments | Key findings |
|---|---|---|---|
| 1 | 2026-04-23 | 1/N all-tasks merge vs 2-task and 9-task merges (4B, 10 tasks) | Scale > N; 1/N is strictly worse (F1–F6) |
| 2 | 2026-04-24 | 7 pair merges (4B, s=1.0) + 4B novel-subspace rotation | All Phase-3 geometric metrics null as predictors (F7–F13) |
| 3 | 2026-04-24 | Sign agreement + 3 layer-selective merges (SP+Arith) + triple merge (4B) | `mlp.down_proj` droppable (F20); layer-type split observed at one pair (F19, later rescoped); triple merge past ceiling (F22) |
| 4 | 2026-04-24 | Replication of Part 3 at 3B and 8B + sign agreement at 3B/8B | `down_proj`-droppable universal (F25); cross-scale sign agreement r=−0.81 on n=7 (F26, later falsified at n=19) |
| 5 | 2026-05-05 → 05-06 | Linking experiment + predictor validation + TIES/DARE + attn.{k,v,o} rotation + ARS-as-merge-predictor | F28 task-class-specific atomic-vs-direct; F30 vanilla-sum competitive with TIES/DARE; F31 attn.{k,v,o} rotation; F29/F33 appeared strong at n=7 but later falsified |
| **6** | **2026-05-09 → 05-12** | **Gap-closing: layer-sel on 6 more pairs, 5+7 vanilla pair merges (n=7→19), routing probe, L1-ARS at n=12, residual analysis, predictor decay** | **F19 rescoped to pair-dependent; F29 degraded r=−0.81→−0.27; F33 falsified (r=+0.06); routing probe negative; per-feature residual null (F34–F40)** |

---

---

# Part 1 (2026-04-23) — Does Averaging All τ's Work?

## What we are doing

Testing whether the TIES/DARE literature's default recipe (uniform 1/N averaging over all task vectors) works for RLVR reasoning τ's. We compare three merge recipes at 4B: (a) sum all 10 Phase-1 τ's at scale s=1/N=0.1, (b) sum 9 refined τ's at s=0.33, (c) sum the 2 best single-task τ's (SP+Arith) at s=1.0.

## Why

Phase 2 ended with a surprising result: a plain 2-task merge of the two best single-task RLVR checkpoints (shortest_path + basic_arithmetic) matched or beat every training-based combination we tried. The natural question: if 2 τ's compose this well, does averaging over *all 10* Phase-1 τ's at 1/N do even better? This is the canonical baseline in the model-merging literature (TIES, DARE, task-arithmetic all assume it). If it works, multi-skill RLVR is trivially solved by summing everything.

## What it means

**1/N averaging is a dead end.** The total displacement magnitude at 1/N (≈0.28 Frobenius, about ⅓ of a single τ) is below the task-activation threshold. The merged model sits near base-SFT with small nudges that aren't big enough to reach any task. What matters is not how many τ's you include but the total displacement budget (s·N ≈ 2–3 τ-equivalents is the sweet spot). This is consistent with Phase-3's near-orthogonality: since τ's don't destructively interfere, the only failure mode of summation is *under-shooting magnitude*.

---

## Merging Methodology

### What is task-vector merging?

The core idea is simple: if you train a model on task A and get a checkpoint W_A, the **task vector** τ_A = W_A − W_SFT captures *only* what was learned for task A — it is the learned delta. If you separately train on task B and get τ_B = W_B − W_SFT, you can try to build a model that does *both* tasks by adding both deltas back to the base:

$$W_{\mathrm{merged}} = W_{\mathrm{SFT}} + s \cdot (\tau_A + \tau_B)$$

where `s` is a scaling coefficient (usually 1.0 for 2-task merges). This is called **task arithmetic** in the literature (Ilharco et al., 2023). The intuition: if τ_A and τ_B live in different "directions" in weight space (they don't fight over the same parameters), adding them should give you a model that has learned both skills. If they overlap and point in opposite directions at some parameters, they'll cancel each other at those positions — that's where merge damage comes from.

### Why this method?

We chose vanilla task-vector summation as our primary methodology for three reasons:

1. **Simplicity and interpretability.** There's exactly one operation (element-wise addition of tensors) with one hyperparameter (scale `s`). Any damage we observe can be attributed to the geometry of the τ's themselves — there's no masking heuristic, no learned combiner, no additional training that could obscure the signal. This makes it possible to connect merge outcomes to the Phase-3 weight-space measurements.

2. **Computational efficiency.** Merging is a pure CPU operation — load two sets of safetensors, subtract the base from each, sum, save. It takes ~2 minutes for a 4B model. No GPU training required. This means we can test many pairs cheaply.

3. **Direct connection to Phase 3.** Phase 3 established that GRPO τ's are (a) near-orthogonal to W's top subspace, (b) near-orthogonal to each other (global cosine 0.00–0.08), and (c) concentrated in specific layer types. These geometric properties have direct implications for summation: orthogonal τ's should compose without destructive interference, and the layer-type concentration suggests layer-selective masking as a natural refinement.

### How we construct a merge (step by step)

```python
# 1. Load the shared SFT base (the starting point all GRPO runs branched from)
W_sft = load_safetensors("sft_qwen_4.0b_instruct")

# 2. Load each single-task GRPO checkpoint
W_A = load_safetensors("grpo_qwen4b_shortest_path")
W_B = load_safetensors("grpo_qwen4b_basic_arithmetic")

# 3. Compute task vectors (the learned deltas)
tau_A = W_A - W_sft   # what GRPO learned for task A
tau_B = W_B - W_sft   # what GRPO learned for task B

# 4. Sum and add back to base
W_merged = W_sft + s * (tau_A + tau_B)  # s = 1.0 for standard 2-task merge

# 5. Save as a new model checkpoint
save_safetensors(W_merged, "merged_sp_arith/")
```

The critical assumption: **all τ's must share the same W_SFT base.** If task A was trained from one SFT checkpoint and task B from a different one, the subtraction produces garbage (the deltas include the difference between the two bases). All our GRPO runs branch from a single SFT warm-up per model scale, so this assumption holds.

### Variants we test

| Variant | Formula | What it tests |
|---|---|---|
| **Vanilla sum** (s=1.0) | W_SFT + τ_A + τ_B | Standard task arithmetic — do two full τ's compose? |
| **Scaled sum** (s<1.0) | W_SFT + s·(τ_A + τ_B + ...) | Does reducing displacement help when many τ's are included? |
| **Layer-selective** (`all_but_down`) | W_SFT + Σ_i τ_i[layers ∉ {mlp.down_proj}] | Does dropping the "frozen-basis" layer type from τ reduce noise? |
| **Layer-selective** (`up` only) | W_SFT + Σ_i τ_i[layers ∈ {mlp.up_proj, mlp.gate_proj}] | Can we merge from just the feature projectors (18% of parameters)? |
| **TIES** (baseline) | Prune low-magnitude τ values, resolve sign conflicts, scale | Standard literature baseline — does sparsification help? |
| **DARE** (baseline) | Random dropout of τ elements with rescaling | Another standard baseline — does stochasticity help? |

### What "damage" means

When we merge τ_A + τ_B, the resulting model may perform *worse* on task A or task B than the individual single-task checkpoints did. We quantify this loss as **pair damage**:

$$\mathrm{damage}(A, B) = \max(0,\, \mathrm{acc}_{\mathrm{GRPO}(A)} - \mathrm{acc}_{\mathrm{merge}}) + \max(0,\, \mathrm{acc}_{\mathrm{GRPO}(B)} - \mathrm{acc}_{\mathrm{merge}})$$

This is the total capability the merge lost relative to the two single-task sources. Damage = 0 means the merge preserves both tasks perfectly. Damage > 0 means at least one task lost accuracy under merging. The maximum possible damage is the sum of both single-task accuracies (total collapse).

In practice across our 19 pairs at 4B, damage ranges from 0.5 (nearly lossless, e.g. T10 cnt+leg) to 64.5 (catastrophic, T15 kk+sdk where KK dropped 47 points).

### Why this is the right methodology for RLVR specifically

Task-vector merging is particularly well-suited to RLVR checkpoints because Phase 3 established three geometric properties that make summation theoretically clean:

1. **τ is small relative to W** (‖τ‖/‖W‖ ≈ 0.06–0.12%). The displacement from SFT is tiny in Frobenius norm, so the merged model stays close to the known-good basin of W_SFT. We're not summing two large rewrites — we're summing two small perturbations.

2. **τ is near-orthogonal to W's top subspace** (<0.14% projection at 4B). The learned changes live in directions W wasn't heavily using. This means τ_A and τ_B are unlikely to fight over W's "important" directions — they operate in the unused margin of weight space.

3. **τ_A and τ_B are near-orthogonal to each other** (global cosine 0.00–0.08 across all pairs). Two random vectors in high-dimensional space are approximately orthogonal; RLVR τ's respect this — they don't point the same way or opposite ways globally. This is why 1/N under-shoots (no cancellation) rather than producing destructive interference.

The surprise of Phase 4 is that despite these favorable geometric properties, *some pairs still damage each other substantially*. The damage happens at a *localized, per-parameter* level (sign conflicts at high-magnitude positions) that global geometric measurements cannot see — which is why every global predictor we tried fails.

### Alternatives we considered but did not use

- **Sequential training** (train on A, then continue training on B): this is what Phase 1 measured. It produces catastrophic forgetting (up to −52%). Task-vector merging avoids this because it doesn't continue training — it just adds pre-computed deltas.
- **Multi-task GRPO** (train on A and B simultaneously with mixed batches): would require rerunning all training from scratch. Merging reuses existing single-task checkpoints for free.
- **LoRA merging** (merge low-rank adapters instead of full τ's): our GRPO runs are full-parameter (no LoRA), so this wasn't applicable. The layer-selective merge variants partially emulate "which parameters matter" but on the full-rank τ.
- **Learned merging** (train a combiner network to weight parameters): this defeats the zero-compute premise of task arithmetic. We want to know whether the geometry alone supports composition.

---

## Merges Evaluated in Part 1

| Merge | N | Scale `s` | τ set |
|---|---|---|---|
| **`phase1_all_1overN`** | 10 | **0.1 (= 1/N)** | Phase-1 4B GRPO checkpoints for all 10 tasks |
| `all_tasks_s0.33` | 9 | 0.33 | Phase-2 refined checkpoints (SP = RFT-Base 55.5%, Arith = Atomic-Only 78%, Count = stage3_full, + 6 Phase-1 checkpoints) |
| `basic` (2-task) | 2 | 1.0 | SP RFT-Base 55.5%  +  Arith Atomic-Only 78% (best single-task checkpoints for the 2 focus tasks) |

Cryptarithm has `‖τ‖ = 0` at 4B (trained-but-never-rewarded), so it contributes nothing to any merge regardless of scale.

### τ magnitudes (Frobenius, computed on pod)

| Task | ‖τ‖ | Task | ‖τ‖ |
|---|---|---|---|
| basic_arithmetic (P1) | 0.852 | syllogism (P1) | 0.756 |
| countdown (P1 4to5) | 0.970 | knights_knaves (P1) | 1.005 |
| shortest_path (P1) | 0.975 | leg_counting (P1) | 1.193 |
| maze (P1) | 0.994 | caesar_cipher (P1) | 0.909 |
| mini_sudoku (P1) | 0.733 | cryptarithm (P1) | **0.000** |

All values are 0.73–1.19 except cryptarithm. At `s = 1/N = 0.1`, the merged vector has an expected magnitude on the order of √10 · 0.1 · 0.9 ≈ 0.28 — roughly a third of a single τ.

---

## Evaluation

- **Tasks:** all 10 Phase-1 environments — reasoning-gym defaults, no task-param overrides
- **Samples:** 200 per task
- **Seed:** 9999 (different from training seed)
- **Setup:** 8-GPU DDP, `batch_size=8`, `max_new_tokens=1024`
- **Verifier:** `evaluate_transfer.py` (last `<answer>` tag, Phase-2 bug fix applied)

---

## Results — 3 × 10 Matrix

| Task | SFT Base | Direct-RLVR (single-task) | **1/N (10τ, s=0.1)** | **s=0.33 (9τ, Phase-2)** | **2-task (SP+Arith, s=1.0)** |
|---|---:|---:|---:|---:|---:|
| basic_arithmetic | 72% | 32% | 69.5% | 71.0% | **76.0%** |
| countdown | 26% | ~70% | 39.0% | 39.0% | **39.5%** |
| shortest_path | 10% | 44% | 24.0% | 46.5% | **51.0%** |
| maze | 21.5% | 1% | 7.5% | 4.0% | **8.5%** |
| mini_sudoku | 4% | 25.5% | 14.0% | **28.0%** | 17.0% |
| syllogism | 79.5% | 96.5% | 59.5% | 71.5% | **83.0%** |
| knights_knaves | 78% | 97.5% | 74.0% | **82.0%** | 77.5% |
| leg_counting | 52% | 78% | 45.0% | **68.0%** | 57.0% |
| caesar_cipher | 0% | 0% | 0.0% | 0.0% | 0.5% |
| cryptarithm | 0% | 0% | 0.0% | 0.0% | 0.0% |
| **Mean (8 non-trivial)** | **42.9%** | **55.6%** | **41.6%** | **51.3%** | **51.2%** |

### Reference points for context

| Model | SP | Arith | Count |
|---|---|---|---|
| Base (SFT) | 10% | 72% | 26% |
| Direct-RLVR (single-task) | 44% | 32% | ~70% |
| RFT-Base (best single-task SP) | 55.5% | 70% | 32% |

---

## Findings

### F1. 1/N averaging is strictly worse than every alternative

The 10-task 1/N merge loses to the 2-task merge on **9 of 10 tasks** and loses to the 9-task s=0.33 merge on **8 of 10 tasks**.

### F2. Scale > N: ~3 τ-equivalents is the sweet spot

| Merge | T-equivalents (s·N) | Mean (8 non-trivial) |
|---|---|---|
| 1/N (10τ, 0.1) | 1.0 | 41.6% |
| s=0.33 (9τ) | 3.0 | **51.3%** |
| 2-task (2τ, 1.0) | 2.0 | **51.2%** |

What matters is the total magnitude of the displacement from W_sft, not how many sources it comes from.

### F3. The 2-task merge is the best choice for focused capability

On SP, Arith, Count, and syllogism, the 2-task merge wins or ties every comparison.

### F4. The 9-task merge wins on 3 tasks it wasn't directly built for

mini_sudoku (+11.0%), knights_knaves (+4.5%), leg_counting (+11.0%) over the 2-task merge.

### F5. caesar_cipher and cryptarithm stay at 0 under every merge

### F6. Per-τ magnitude is a weak predictor of downstream contribution

---

See Synthesis for the post-Part-6 assessment of what survived.

---

## Artifacts (Part 1)

### Merged checkpoints (pod: `/scratch/sunguha/rlvr_experiments/`)

| Path | Merge |
|---|---|
| `results/phase2_sp/merged/phase1_all_1overN/` | 10τ Phase-1 1/N |
| `results/phase2_sp/merged/all_tasks_s0.33/` | 9τ Phase-2 s=0.33 |
| `results/phase2_sp/merged/basic/` | 2τ SP+Arith s=1.0 |

### Eval outputs

| Path | Contents |
|---|---|
| `outputs/eval10_phase1_all_1overN/` | 10 `metrics_*.json` + details JSONs |
| `outputs/eval10_all_tasks_s0.33/` | 10 `metrics_*.json` + details JSONs |
| `outputs/eval10_basic_2task/` | 10 `metrics_*.json` + details JSONs |

### Code

| File | Purpose |
|---|---|
| `merge_phase1_all.py` | Phase-1 10-τ merge; default scale = 1/N |
| `merge_all_tasks.py` | Existing Phase-2 9-τ merge script |
| `merge_task_vectors.py` | 2-task merge script |
| `evaluate_transfer.py` | Eval script |

### Logs

| File | Contents |
|---|---|
| `logs/merge_phase1_1overN.log` | Merge construction log |
| `logs/eval10_three_merges.log` | Main 3 × 10 eval sweep |
| `logs/eval10_retry.log` | Re-runs for maze (1/N) and syllogism (s=0.33) |

---

## Appendix — What changed in each task from base SFT

| Task | Base | 1/N | s=0.33 | 2-task |
|---|---:|---:|---:|---:|
| basic_arithmetic | 62% | +7.5% | +9.0% | **+14.0%** |
| countdown | 34% | +5.0% | +5.0% | +5.5% |
| shortest_path | 16% | +8.0% | +30.5% | **+35.0%** |
| maze | 2% | +5.5% | +2.0% | **+6.5%** |
| mini_sudoku | 14% | 0.0% | **+14.0%** | +3.0% |
| syllogism | 66% | −6.5% | +5.5% | **+17.0%** |
| knights_knaves | 74% | 0.0% | +8.0% | +3.5% |
| leg_counting | 52% | −7.0% | **+16.0%** | +5.0% |
| caesar_cipher | 0% | 0.0% | 0.0% | +0.5% |
| cryptarithm | 0% | 0.0% | 0.0% | 0.0% |

---

---

# Part 2 (2026-04-24) — 7 Pair Merges + 4B Rotation Analysis

## What we are doing

Building 7 controlled 2-τ merges at 4B from different task pairs, evaluating all 10 tasks on each, and testing whether Phase-3's geometric measurements predict which pairs compose cleanly.

## Why

Part 1 established that 2-task merges at s=1.0 are the sweet spot — but gave no answer to *which pairs should go together*. If Phase 3's geometric measurements correlate with pair-merge damage, we'd have a cheap pre-screen.

## What it means

**All intuitive predictors fail.** Phase-1 sequential transfer (r ≈ 0), global τ cosine (all near-zero but damage spans 4.5–24), per-layer flat-cosine opposition (r = +0.23) — all null. The one metric with modest signal at n=7 (novel-subspace principal angle, r = +0.56) also did not hold at n=19. The data tables below are preserved as valid measurements.

---

### 7 Pair Selection

| # | Pair | Reasoning |
|---|---|---|
| T1 | mini_sudoku + leg_counting | Phase-1 transfer −52%; depth signatures overlap L15–17 |
| T2 | mini_sudoku + knights_knaves | Phase-1 transfer −50%; overlapping mid layers |
| T3 | basic_arithmetic + leg_counting | Phase-1 transfer −32%; overlapping L15–17 |
| T4 | basic_arithmetic + syllogism | Same mid-early band; Phase-1 transfer +18% |
| T5 | shortest_path + syllogism | Different primary depths; minor overlap |
| T6 | shortest_path + knights_knaves | Both mid-ish at 4B; Phase-1 transfer −12% |
| T7 | countdown + basic_arithmetic | Phase-3 flagged −0.90 flat cosine at L6.mlp.up_proj |

### τ-cosines measured from the merge script

| Pair | cos(τ_A, τ_B) | ‖τ_A‖ | ‖τ_B‖ |
|---|---|---|---|
| T1 sdk+leg | 0.0053 | 0.733 | 1.193 |
| T2 sdk+kk | 0.0002 | 0.733 | 1.005 |
| T3 ari+leg | 0.0830 | 0.852 | 1.193 |
| T4 ari+syl | 0.0398 | 0.852 | 0.756 |
| T5 sp+syl | 0.0153 | 0.975 | 0.756 |
| T6 sp+kk | 0.0106 | 0.975 | 1.005 |
| T7 cnt+ari | — | 0.970 | 0.852 |

### The 7×10 Accuracy Matrix

| task | SFT Base | Direct-RLVR | T1 sdk+leg | T2 sdk+kk | T3 ari+leg | T4 ari+syl | T5 sp+syl | T6 sp+kk | T7 cnt+ari |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| basic_arithmetic | 72 | 32 | 66.0 | 50.0 | 52.0 | 57.0 | 71.0 | 59.5 | 63.0 |
| countdown | 26 | ~70 | 39.0 | 8.0 | 28.5 | 24.0 | 28.0 | 29.0 | 46.0 |
| shortest_path | 10 | 44 | 15.5 | 0.5 | 15.0 | 15.0 | 47.5 | 26.0 | 27.0 |
| maze | 21.5 | 1 | 4.0 | 0.5 | 11.5 | 1.0 | 3.5 | 8.0 | 0.5 |
| mini_sudoku | 4 | 25.5 | 20.0 | 35.0 | 19.5 | 11.0 | 31.5 | 23.5 | 17.0 |
| syllogism | 79.5 | 96.5 | 78.5 | 64.0 | 86.0 | 87.5 | 85.0 | 88.5 | 84.0 |
| knights_knaves | 78 | 97.5 | 62.0 | 65.0 | 68.0 | 70.5 | 70.0 | 86.5 | 72.5 |
| leg_counting | 52 | 78 | 78.0 | 1.0 | 67.5 | 1.0 | 55.5 | 30.5 | 39.5 |
| caesar_cipher | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| cryptarithm | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

SFT Base = no GRPO, just the format warm-up model. Direct-RLVR = best single-task GRPO at 4B (300 steps on that task alone). Merge columns show what happens when two tasks' τ's are summed — a single model doing both.

### Damage on each merge's two trained tasks

| pair | damage_A | damage_B | total | dominant conflict |
|---|---:|---:|---:|---|
| T5 sp+syl | 0 (SP) | 7 (syl) | **7** | Cleanest merge |
| T4 ari+syl | 0 (arith) | 4.5 (syl) | **4.5** | Lowest total |
| T3 ari+leg | 0 (arith) | 12.5 (leg) | 12.5 | |
| T1 sdk+leg | 16 (sdk) | 2 (leg) | 18 | |
| T6 sp+kk | 18 (SP) | 0 (KK) | 18 | |
| T2 sdk+kk | 1 (sdk) | 21 (KK) | 22 | |
| T7 cnt+ari | 24 (cnt) | 0 (arith) | 24 | Worst |

### 4B Novel-Subspace Rotation Analysis

| pair | mlp_up angle | mlp_up flat_cos | mlp_down angle | mlp_down flat_cos | attn_q angle | attn_q flat_cos |
|---|---:|---:|---:|---:|---:|---:|
| T1 sdk+leg | 63.89° | +0.188 | 34.68° | +0.212 | 44.64° | +0.377 |
| T2 sdk+kk | 60.79° | +0.016 | 32.25° | +0.342 | 42.42° | +0.358 |
| T3 ari+leg | 61.46° | +0.275 | 34.83° | +0.077 | 44.71° | −0.468 |
| T4 ari+syl | 58.18° | −0.114 | 30.02° | +0.364 | 40.33° | +0.293 |
| T5 sp+syl | 60.17° | −0.172 | 32.46° | +0.102 | 42.05° | −0.271 |
| T6 sp+kk | 62.87° | −0.051 | 35.04° | −0.002 | 44.90° | −0.142 |
| T7 cnt+ari | 60.78° | −0.300 | 33.38° | +0.042 | 43.87° | +0.152 |

### Correlation with damage (n=7)

| rotation metric | Pearson r | interpretation |
|---|---:|---|
| #layers with up_cos < −0.5 | +0.23 | weak |
| mean `mlp.up_proj` flat cos | −0.03 | no signal |
| mean `attn.q_proj` flat cos | +0.32 | weak |
| mean all-layer flat cos | +0.07 | no signal |
| mean `mlp.up_proj` principal angle | +0.56 | moderate at n=7; did not survive at n=19 |

### Findings (Part 2)

**F7.** Phase-1 transfer is not a merge predictor (r ≈ 0).

**F8.** Global τ cosine is not a merge predictor.

**F9.** Per-layer opposing rotations don't predict damage (r = +0.23).

**F10.** Novel-subspace angle has moderate r = +0.56 at n=7 but did not survive expansion.

**F11.** T5 shortest_path + syllogism is the cleanest pair (damage = 7).

**F12.** T4 arith + syllogism has lowest total damage (4.5).

**F13.** No single merge dominates all tasks; the right merge depends on target tasks.

---

See Synthesis for the post-Part-6 assessment of what survived.

### Artifacts (Part 2)

| Path | Contents |
|---|---|
| `results/phase4_pairs/{T1,...,T7}/basic/` | Seven merged checkpoints |
| `outputs/eval10_phase4_{T1,...,T7}/` | Seven 10-task eval dirs |
| `results/weight_analysis/4b/rotation_direction.json` | 4B novel-subspace data |
| `analyze_rotation_direction_4b.py` | 4B rotation analysis |
| `run_phase4_pairs.sh` | End-to-end pipeline |
| `logs/phase4_pairs.log` | Main eval pipeline log |
| `logs/phase4_retry_and_rotation.log` | Retry + rotation log |
| `logs/final_maze_retry.log` | T3 maze retry |

---

---

# Part 3 (2026-04-24) — Sign Agreement, Layer-Selective Merges, Triple Merge

## What we are doing

Three parallel experiments at 4B: (a) compute sign-agreement statistics across 7 pair merges; (b) run layer-selective variants of SP+Arith; (c) try a triple merge (SP+Arith+Syl).

## Why

Part 2 ruled out all Phase-3 geometric metrics. But Phase 3 also suggested a *recipe*: if `mlp.down_proj` never rotates, its τ contributions may be noise. And does sign-agreement correlate with damage?

## What it means

1. **`mlp.down_proj` is always safe to drop** — the `all_but_down` variant matches or beats the full merge on every task at 4B. This is the headline actionable recipe from Phase 4. [F20 — survives all testing]
2. **Tasks showed a layer-type split at this one pair (SP+Arith)** — initially interpreted as task-intrinsic. [Part 6 revision: later shown to be pair-dependent, not task-intrinsic — see Part 6, F34.]
3. **Sign agreement correlated r=−0.46 with damage at n=7.** [Part 6 revision: collapsed to r=−0.27 at n=19. See Synthesis.]

---

### Part 3.A — Sign-agreement analysis

| pair | damage | frac_agree | mag_agree | conflict_mass | up_conflict | q_conflict |
|---|---:|---:|---:|---:|---:|---:|
| T1 sdk+leg | 18.0 | 0.0329 | 0.5044 | 0.4956 | 0.4946 | 0.4978 |
| T2 sdk+kk | 22.0 | 0.0327 | 0.5002 | 0.4998 | 0.5015 | 0.4972 |
| T3 ari+leg | 12.5 | 0.0266 | 0.5695 | 0.4305 | 0.4322 | 0.4471 |
| T4 ari+syl | 4.5 | 0.0262 | 0.5337 | 0.4663 | 0.4648 | 0.4806 |
| T5 sp+syl | 7.0 | 0.0222 | 0.5131 | 0.4869 | 0.4886 | 0.4891 |
| T6 sp+kk | 18.0 | 0.0226 | 0.5089 | 0.4911 | 0.4891 | 0.4923 |
| T7 cnt+ari | 24.0 | 0.0216 | 0.5132 | 0.4868 | 0.4853 | 0.4903 |

Correlations with damage (n=7):

| metric | Pearson r |
|---|---:|
| frac_agree | +0.21 |
| mag_agree | −0.46 |
| conflict_mass | +0.46 |
| up_conflict | +0.46 |
| q_conflict | +0.38 |

**F18.** Magnitude-weighted sign agreement correlates r=−0.46 with damage at n=7 at 4B. [Part 6 revision: degraded to r=−0.27 at n=19. Not a calibrated predictor. See Synthesis.]

### Part 3.B — Layer-selective merges of SP + Arith (s=1.0)

| variant | # keys merged | what's included |
|---|---:|---|
| `up` | 72 | `mlp.up_proj` + `mlp.gate_proj` only |
| `up_q_clean` | 108 | + `self_attn.q_proj` |
| `all_but_down` | 362 | everything except `mlp.down_proj` |
| (2-task baseline) | 398 | all keys |

### The 4-variant × 10-task table

| task | SFT Base | Direct-RLVR | 2-task (full) | triple | `up` | `up+q` | `all_but_down` |
|---|---:|---:|---:|---:|---:|---:|---:|
| basic_arithmetic | 72 | 32 | 76.0 | 74.5 | 73.5 | 77.0 | 77.0 |
| countdown | 26 | ~70 | 39.5 | 42.0 | 25.5 | 30.5 | 36.5 |
| shortest_path | 10 | 44 | 51.0 | 50.0 | 37.5 | 40.5 | 53.5 |
| maze | 21.5 | 1 | 8.5 | 0.5 | 5.0 | 4.0 | 6.0 |
| mini_sudoku | 4 | 25.5 | 17.0 | 23.5 | 6.5 | 7.0 | 12.5 |
| syllogism | 79.5 | 96.5 | 83.0 | 83.0 | 83.5 | 84.5 | 83.0 |
| knights_knaves | 78 | 97.5 | 77.5 | 67.0 | 71.0 | 67.5 | 71.5 |
| leg_counting | 52 | 78 | 57.0 | 52.0 | 62.0 | 60.5 | 59.0 |
| caesar_cipher | 0 | 0 | 0.5 | 0.0 | 0.0 | 0.0 | 0.5 |
| cryptarithm | 0 | 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

SFT Base and Direct-RLVR are single-task reference points. 2-task (full) = SP+Arith with all 398 keys merged at s=1.0. The layer-selective variants drop subsets of keys from that same merge.

### Δ from 2-task baseline per variant

| task | `up` Δ | `up+q` Δ | `all_but_down` Δ | observed pattern (this pair only) |
|---|---:|---:|---:|---|
| basic_arithmetic | −2.5 | +1.0 | +1.0 | feature-routing |
| syllogism | +0.5 | +1.5 | 0.0 | feature-routing |
| leg_counting | +5.0 | +3.5 | +2.0 | feature-routing |
| maze | −3.5 | −4.5 | −2.5 | marginal |
| mini_sudoku | −10.5 | −10.0 | −4.5 | attention-restructuring |
| knights_knaves | −6.5 | −10.0 | −6.0 | attention-restructuring |
| shortest_path | −13.5 | −10.5 | +2.5 | attention-restructuring |
| countdown | −14.0 | −9.0 | −3.0 | attention-restructuring |

**F19.** At this specific pair (SP+Arith), tasks split into two groups based on layer-type sufficiency. [Part 6 revision: this classification is **pair-dependent, not task-intrinsic**. SP flips G2→G1 when paired with KK; syllogism flips G1→G2 when paired with arith. Only mini_sudoku is consistently G2. See Part 6, F34.]

**F20.** Dropping `mlp.down_proj` is free or positive for every task. `all_but_down` matches or beats the full 2-task merge on 6 of 8 non-trivial tasks. **This validates Phase 3's "down_proj basis frozen" finding as actionable.**

**F21.** Phase 3's rotation analysis was incomplete — `attn.{k,v,o}` carry 4–13 points of merge-relevant capability for some tasks.

### Part 3.C — Triple merge (SP + Arith + Syllogism at s=1.0)

| task | 2-task | triple | Δ |
|---|---:|---:|---:|
| basic_arithmetic | 76.0 | 74.5 | −1.5 |
| countdown | 39.5 | 42.0 | +2.5 |
| shortest_path | 51.0 | 50.0 | −1.0 |
| maze | 8.5 | 0.5 | −8.0 |
| mini_sudoku | 17.0 | 23.5 | +6.5 |
| syllogism | 83.0 | 83.0 | 0.0 |
| knights_knaves | 77.5 | 67.0 | −10.5 |
| leg_counting | 57.0 | 52.0 | −5.0 |

**F22.** Triple merge is not a win — confirms ~3 τ-equivalents as ceiling.

---

See Synthesis for the post-Part-6 assessment of what survived.

### Artifacts (Part 3)

| Path | Contents |
|---|---|
| `results/phase4_triple/basic/` | Triple merge checkpoint |
| `results/phase4_layersel/{up,up_q_clean,all_but_down}/` | 3 layer-selective variants |
| `results/weight_analysis/4b/sign_agreement.json` | Sign-agreement stats |
| `outputs/eval10_phase4_triple/`, `outputs/eval10_layersel_*/` | Eval dirs |
| `analyze_sign_agreement.py` | Sign-agreement analysis |
| `merge_triple.py` | 3-τ merge script |
| `merge_layer_selective.py` | Layer-selective merge |
| `logs/phase4_abc.log` | Main pipeline log |
| `logs/phase4_abc_retry.log` | Retry log |

---

---

## Part 4 (2026-04-24) — 3B and 8B Replication

## What we are doing

Replicating Part 3's findings at Qwen2.5-3B and Qwen3-8B.

## Why

Single-scale results are fragile. Testing whether findings survive cross-scale and cross-family.

## What it means

1. **`mlp.down_proj`-droppable is universal** — holds at 3B, 4B, and 8B.
2. **Group-1 core (arith, syllogism, leg_counting) replicates at all scales** for this specific pair.
3. **Sign-agreement at 3B correlated r=−0.81 with 4B damage at n=7.** [Part 6 revision: this did not survive sample expansion; r=−0.45 at n=12, r=−0.27 at n=19. See Part 6, F35.]

---

### Setup

| scale | SFT base | SP checkpoint | Arith checkpoint |
|---|---|---|---|
| 3B | `outputs/sft_qwen25_3b` (Qwen2.5) | `grpo_qwen25_3b_shortest_path` | `grpo_qwen25_3b_basic_arithmetic` |
| 4B | `sft_qwen_4.0b_instruct` (Qwen3) | `phase2_sp/idea_rft_base/grpo` | `phase2_arith/stage2_l2` |
| 8B | `outputs/sft_qwen3_8b` (Qwen3) | `grpo_qwen3_8b_shortest_path` | `grpo_qwen3_8b_basic_arithmetic` |

### The 3-scale × 3-variant × 10-task accuracy matrix

"basic" = full 2-task merge (SP+Arith, all keys, s=1.0) at that scale. "up" = `mlp.up_proj + mlp.gate_proj` only. "abd" = `all_but_down`. For single-task reference points: 4B SFT Base and Direct-RLVR numbers are in Part 1's table above; 3B and 8B use Phase-1 checkpoints (weaker than 4B's Phase-2-refined versions).

| task | 3B basic | 3B up | 3B abd | 4B basic | 4B up | 4B abd | 8B basic | 8B up | 8B abd |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| basic_arithmetic | 37.5 | 44.5 | 40.0 | 76.0 | 73.5 | 77.0 | 62.0 | 67.0 | 67.0 |
| countdown | 0.5 | 0.0 | 0.0 | 39.5 | 25.5 | 36.5 | 31.5 | 32.5 | 32.0 |
| shortest_path | 34.5 | 29.0 | 34.5 | 51.0 | 37.5 | 53.5 | 44.0 | 31.5 | 45.0 |
| maze | 5.0 | 7.0 | 6.5 | 8.5 | 5.0 | 6.0 | 24.0 | 22.0 | 23.0 |
| mini_sudoku | 0.0 | 0.0 | 0.0 | 17.0 | 6.5 | 12.5 | 15.5 | 3.0 | 13.0 |
| syllogism | 74.0 | 81.0 | 82.5 | 83.0 | 83.5 | 83.0 | 72.0 | 77.0 | 72.5 |
| knights_knaves | 16.0 | 26.5 | 17.5 | 77.5 | 71.0 | 71.5 | 64.5 | 66.0 | 66.0 |
| leg_counting | 4.0 | 6.5 | 5.0 | 57.0 | 62.0 | 59.0 | 39.5 | 44.5 | 46.5 |
| caesar_cipher | 0.0 | — | 0.0 | 0.5 | 0.0 | 0.5 | 0.0 | 0.5 | 0.0 |
| cryptarithm | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

### Δ from 2-task baseline per scale

| task | 3B Δ up | 3B Δ abd | 4B Δ up | 4B Δ abd | 8B Δ up | 8B Δ abd |
|---|---:|---:|---:|---:|---:|---:|
| basic_arithmetic | +7.0 | +2.5 | −2.5 | +1.0 | +5.0 | +5.0 |
| syllogism | +7.0 | +8.5 | +0.5 | 0.0 | +5.0 | +0.5 |
| leg_counting | +2.5 | +1.0 | +5.0 | +2.0 | +5.0 | +7.0 |
| shortest_path | −5.5 | 0.0 | −13.5 | +2.5 | −12.5 | +1.0 |
| mini_sudoku | 0.0 | 0.0 | −10.5 | −4.5 | −12.5 | −2.5 |
| countdown | −0.5 | −0.5 | −14.0 | −3.0 | +1.0 | +0.5 |
| knights_knaves | +10.5 | +1.5 | −6.5 | −6.0 | +1.5 | +1.5 |
| maze | +2.0 | +1.5 | −3.5 | −2.5 | −2.0 | −1.0 |

**F23.** Feature-routing group (arith, syllogism, leg) replicates at all 3 scales for this pair.

**F24.** SP and sudoku are G2 at 4B/8B for this pair; countdown and KK are scale-dependent. [Part 6 revision: even SP flips to G1 with a different pair partner. See F34.]

**F25.** `mlp.down_proj`-droppable holds at all 3 scales. Net averages: 3B +2.1, 4B −1.3, 8B +1.3.

**F26.** Sign-agreement at 3B showed r=−0.81 against 4B damage at n=7. [Part 6 revision: this n=7 result did not survive sample expansion. At n=12, r=−0.45; at n=19, r=−0.27. See Part 6, F35.]

Per-pair sign-agreement data (valid measurements, preserved for record):

| pair | 3B mag_agree | 4B mag_agree | 8B mag_agree | 4B damage |
|---|---:|---:|---:|---:|
| T4 ari+syl | 0.605 | 0.534 | 0.549 | 4.5 |
| T3 ari+leg | 0.564 | 0.570 | 0.549 | 12.5 |
| T5 sp+syl | 0.536 | 0.513 | 0.517 | 7.0 |
| T6 sp+kk | 0.513 | 0.509 | 0.517 | 18.0 |
| T2 sdk+kk | 0.531 | 0.500 | 0.507 | 22.0 |
| T1 sdk+leg | 0.530 | 0.504 | 0.502 | 18.0 |
| T7 cnt+ari | 0.493 | 0.513 | 0.536 | 24.0 |

### Limitations

1. 3B baselines weak for several tasks. 2. Mixed checkpoint quality (Phase-1 vs Phase-2). 3. Qwen2.5/Qwen3 family confound. 4. No 4B→8B direction tested.

---

See Synthesis for the post-Part-6 assessment of what survived.

### Artifacts (Part 4)

| Path | Contents |
|---|---|
| `results/phase4_replicate_3b/` | 3 merge checkpoints at 3B |
| `results/phase4_replicate_8b/` | 3 merge checkpoints at 8B |
| `outputs/eval10_replicate_{3b,8b}_{basic,up,all_but_down}/` | 6 eval dirs |
| `results/weight_analysis/{3b,8b}/sign_agreement.json` | Sign-agreement stats |
| `analyze_sign_agreement_multi.py` | Multi-scale analysis |
| `replicate_3b_8b.sh` | Pipeline script |
| `logs/replicate_3b_8b.log` | Main log |

---

---

# Part 5 (2026-05-05) — Paper Prep: Linking Experiment, TIES/DARE, attn.{k,v,o} Rotation

## What we are doing

Four parallel streams: (a) linking experiment (atomic vs direct τ's); (b) sign-agreement validation (bootstrap, LOO); (c) TIES/DARE comparison; (d) attn.{k,v,o} rotation measurement.

## Why

Parts 1–4 had claims to carry into a paper. Part 5 stress-tests them for reviewers.

## What it means

1. **The linking result (F28) survived at a task-class-specific level.** Atomic τ's merge better on arith/logic/counting at every scale; direct τ's merge better on shortest_path at every scale.
2. **TIES/DARE are not better** than vanilla sum (F30). Our `all_but_down` recipe is competitive.
3. **attn.{k,v,o} rotation (F31)** filled the Phase-3 gap: k 30–42°, v and o 50–70° — none frozen.
4. **F29 (sign-agreement validation) and F33 (ARS predictor)** gave strong numbers at n=7 but both collapsed in Part 6.

---

### F28. Atomic task vectors merge more cleanly than direct-RLVR task vectors (the linking result)

**Setup.** Three 2-τ merges of SP + Arith at 4B:

| Merge | SP source | Arith source |
|---|---|---|
| M_aa | τ_atomic (55.5% SP) | τ_atomic (78% Arith) |
| M_dd | τ_direct | τ_direct |
| M_mix | τ_atomic (SP) | τ_direct (Arith) |

#### Sign-agreement between pre-merge τ pairs

| pair-type | sp + ari | sp + maze | ari + maze |
|---|---:|---:|---:|
| atomic-atomic | **0.519** | 0.512 | 0.506 |
| direct-direct | 0.501 | 0.515 | 0.509 |
| cross | 0.497 | 0.507 | 0.508 |

#### 10-task accuracy (4B)

| Task | SFT Base | Direct-RLVR | M_aa | M_dd | M_mix | Δ(aa−dd) |
|---|---:|---:|---:|---:|---:|---:|
| basic_arithmetic | 72 | 32 | 75.0 | 66.5 | 65.5 | **+8.5** |
| countdown | 26 | ~70 | 36.0 | 35.5 | 37.5 | +0.5 |
| shortest_path | 10 | 44 | 54.0 | 52.5 | 51.5 | +1.5 |
| maze | 21.5 | 1 | 11.0 | 3.5 | 1.0 | +7.5 |
| mini_sudoku | 4 | 25.5 | 17.5 | 19.0 | 14.5 | −1.5 |
| syllogism | 79.5 | 96.5 | 83.0 | 48.5 | 86.0 | **+34.5** |
| knights_knaves | 78 | 97.5 | 66.5 | 65.5 | 68.5 | +1.0 |
| leg_counting | 52 | 78 | 55.5 | 44.0 | 47.5 | **+11.5** |
| **mean (8 non-trivial)** | **42.9** | **55.6** | **49.8** | **41.9** | **46.5** | **+7.9** |

SFT Base and Direct-RLVR are single-task reference points. M_aa = merge of two atomic-curriculum τ's; M_dd = merge of two direct-GRPO τ's; M_mix = atomic(SP) + direct(Arith). All are single models evaluated on all tasks simultaneously.

#### Cross-scale replication

| Task | 3B Δ(aa−dd) | 4B Δ(aa−dd) | 8B Δ(aa−dd) | Pattern |
|---|---:|---:|---:|---|
| basic_arithmetic | +2.5 | +8.5 | +10.0 | M_aa wins at every scale |
| countdown | 0 | +0.5 | +7.5 | M_aa wins |
| leg_counting | +4.0 | +11.5 | +10.0 | M_aa wins at every scale |
| knights_knaves | +8.5 | +1.0 | +3.0 | M_aa wins at every scale |
| mini_sudoku | +1.0 | −1.5 | −0.5 | mixed, small |
| maze | +1.5 | +7.5 | −3.5 | scale-dependent |
| syllogism | −11.0 | +34.5 | −5.5 | scale-dependent (4B outlier) |
| **shortest_path** | **−33.0** | +1.5 | **−18.5** | **M_dd wins at every scale** |
| **mean (8)** | **−3.3** | **+7.9** | **+0.3** | |

**Revised F28:** The per-task *sign* of Δ(aa−dd) is cross-scale stable: M_aa wins on arith/logic/counting; M_dd wins on shortest_path. The mean effect is scale-specific.

### F29. Sign-agreement validation at n=7

[Part 6 revision: these validation results did not survive expansion to n=12/19. The predictor decayed from r=−0.81 to r=−0.27. Bootstrap/LOO on n=7 cannot detect small-sample artifacts. Preserved as methodological record. See Part 6, F35.]

- Bootstrap 95% CI (10,000 resamples): [−0.993, −0.376]
- Permutation test: p = 0.024
- LOO prediction Pearson r: 0.725, MAE = 3.59
- Fixed train/test splits: r > 0.97 on held-out pairs

### F30. TIES and DARE do not beat vanilla sum

| pair | method | arith | cnt | SP | maze | sdk | syl | KK | leg | Δ vs vanilla |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| T2_sdk_kk | vanilla | 50.0 | 8.0 | 0.5 | 0.5 | 35.0 | 64.0 | 65.0 | 1.0 | — |
| | TIES | 48.5 | 8.0 | 1.0 | 0.5 | 36.5 | 66.0 | 76.0 | 1.0 | +1.1 |
| | DARE | 50.0 | 9.0 | 0.5 | 0.5 | 36.0 | 66.0 | 68.5 | 1.0 | +0.5 |
| T5_sp_syl | vanilla | 71.0 | 28.0 | 47.5 | 3.5 | 31.5 | 85.0 | 70.0 | 55.5 | — |
| | TIES | 70.0 | 26.0 | 48.0 | 4.5 | 25.5 | 85.0 | 70.0 | 53.0 | −1.0 |
| | DARE | 69.0 | 27.5 | 51.5 | 3.5 | 24.5 | 84.5 | 67.5 | 55.5 | −0.6 |
| T7_cnt_ari | vanilla | 63.0 | 46.0 | 27.0 | 0.5 | 17.0 | 84.0 | 72.5 | 39.5 | — |
| | TIES | 60.5 | — | 23.5 | 1.0 | 16.5 | 85.5 | 73.5 | 40.0 | n/a |
| | DARE | 62.5 | 43.0 | 27.5 | 1.0 | 14.5 | 86.5 | 74.5 | 43.0 | +0.6 |

No method consistently beats vanilla sum. Differences within ±2 pts.

### F31. attn.{k,v,o} novel-subspace rotation

**4B (degrees)**

| pair | k_proj | v_proj | o_proj |
|---|---:|---:|---:|
| T1_sdk_leg | 37.3 | 58.1 | 61.4 |
| T2_sdk_kk | 41.3 | 63.5 | 59.6 |
| T3_ari_leg | 40.1 | 53.5 | 57.7 |
| T4_ari_syl | 41.3 | 60.1 | 60.5 |
| T5_sp_syl | 38.3 | 64.9 | 60.9 |
| T6_sp_kk | 37.9 | 65.5 | 60.6 |

**8B (degrees)**

| pair | k_proj | v_proj | o_proj |
|---|---:|---:|---:|
| T1_sdk_leg | 33.3 | 61.9 | 56.0 |
| T2_sdk_kk | 37.6 | 56.9 | 58.1 |
| T3_ari_leg | 35.8 | 55.9 | 64.0 |
| T4_ari_syl | 35.2 | 49.2 | 60.8 |
| T5_sp_syl | 35.4 | 56.4 | 68.0 |
| T6_sp_kk | 34.9 | 50.4 | 70.1 |
| T7_cnt_ari | 31.8 | 55.3 | 67.7 |

k_proj rotates 30–42°, v_proj and o_proj 50–70°. No pair shows "basis-frozen." Consistent with pair-dependent Group-2 finding.

### F33. ARS predictor at n=7

[Part 6 revision: **falsified at n=12** — r flipped from −0.76 to +0.06. Data preserved for record. See Part 6, F37.]

| predictor (at 3B) | Pearson r |
|---|---:|
| L1_mean | −0.761 |
| L1_min | −0.551 |
| ARS_mean | −0.337 |
| L2_mean | −0.009 |

Per-pair breakdown:

| pair | 3B L1_mean | 3B sign_agree | 4B damage |
|---|---:|---:|---:|
| T4 ari+syl | 0.500 | 0.605 | 4.5 |
| T5 sp+syl | 0.320 | 0.536 | 7.0 |
| T3 ari+leg | 0.290 | 0.564 | 12.5 |
| T1 sdk+leg | 0.143 | 0.530 | 18.0 |
| T6 sp+kk | 0.188 | 0.513 | 18.0 |
| T2 sdk+kk | 0.220 | 0.531 | 22.0 |
| T7 cnt+ari | 0.261 | 0.493 | 24.0 |

---

See Synthesis for the post-Part-6 assessment of what survived.

### Artifacts (Part 5)

| Path | Contents |
|---|---|
| `paper_prep/linking/{M_aa, M_dd, M_mix}/` | 3 linking checkpoints (4B) |
| `paper_prep/linking/sign_agreement_linking.json` | Sign-agreement for linking pairs |
| `paper_prep/linking_{3b,8b}/{M_aa, M_dd, M_mix}/` | 6 cross-scale checkpoints |
| `results/phase5_replicate_{3b,8b}/` | 8 atomic-curriculum checkpoints |
| `paper_prep/baselines/{T2,T5,T7}_*/{ties,dare}/` | TIES/DARE checkpoints |
| `paper_prep/analysis/rotation_attn_kvo_{4b,8b}.json` | attn.{k,v,o} rotation data |
| `paper_prep/analysis/bootstrap_r.json` | Bootstrap CI |
| `paper_prep/analysis/held_out_predictor.json` | LOO + fixed-split validation |
| `paper_prep/analysis/f28_cross_scale.json` | Cross-scale accuracy matrix |
| `paper_prep/ars/` | ARS probe data |
| `outputs/eval10_linking_*/` | Eval dirs |
| `outputs/eval10_baseline_*/` | TIES/DARE eval dirs |
| `paper_prep/{stream_A_linking.py, stream_D_analysis.py, bootstrap_and_heldout.py}` | Analysis scripts |
| `paper_prep/{pipeline_gpu.sh, pipeline_fixup.sh, f28_replicate_pipeline.sh}` | Pipeline scripts |

---

---

# Part 6 (2026-05-09 → 2026-05-12) — Gap-closing experiments that reshaped the spine

## What we are doing

Three experiments to stress-test the paper's claims before submission: (a) layer-selective merges on 6 more pairs; (b) 12 more pair merges (n=7 → n=19); (c) routing probe for Group prediction.

## Why

The first paper draft had two headline claims — a strong predictor (r = −0.81) and a clean task-level classification (Group-1 vs Group-2). Both were derived from small samples (n = 7; 1 pair for classification). A reviewer would immediately ask: "does this hold on more data?" Part 6 is the honest answer, run *before* submission.

## What it means

**All three headline claims collapsed.**

1. **Group-1/Group-2 is pair-dependent, not task-intrinsic.** SP flips G2→G1 with KK; syllogism flips G1→G2 with arith. Only mini_sudoku is consistently G2.
2. **Sign-agreement predictor decays: r = −0.81 (n=7) → −0.45 (n=12) → −0.27 (n=19).** Classic small-n overfit.
3. **L1-ARS predictor flips sign: r = −0.76 (n=7) → +0.06 (n=12).** Falsified.
4. **Routing probe: r = −0.31.** Doesn't work.
5. **Residual search: nothing above |r| = 0.5.** No combination works.

**What survived strengthened:** `mlp.down_proj` droppable (40/48 cells). Atomic-vs-direct linking (F28). 1/N dead.

---

### 6.1 Layer-selective merges across 6 more pairs [F34]

**Per-pair, per-task Group classifications:**

| Pair | Task | Δ up (pts) | Δ up_q (pts) | Δ all_but_down (pts) | Classification |
|---|---|---:|---:|---:|:---:|
| T1 sdk+leg | mini_sudoku | −8.0 | −9.0 | +3.5 | G2 |
| T1 sdk+leg | leg_counting | +0.0 | +1.0 | −8.0 | G1 |
| T2 sdk+kk | mini_sudoku | −24.0 | −27.0 | −16.5 | G2 |
| T2 sdk+kk | knights_knaves | +9.0 | +7.0 | −11.5 | G1 |
| T3 ari+leg | basic_arithmetic | +23.0 | +23.0 | +11.5 | G1 |
| T3 ari+leg | leg_counting | +2.0 | −4.5 | +6.5 | G1 |
| T4 ari+syl | basic_arithmetic | +16.5 | +16.0 | +15.0 | G1 |
| T4 ari+syl | syllogism | −6.0 | −5.5 | +0.5 | G2 |
| T6 sp+kk | shortest_path | +23.0 | +22.5 | +31.0 | **G1** |
| T6 sp+kk | knights_knaves | +0.0 | −1.0 | +1.0 | G1 |
| T7 cnt+ari | countdown | +27.0 | +27.0 | +32.0 | G1 |
| T7 cnt+ari | basic_arithmetic | +10.5 | +11.0 | +10.0 | G1 |

**Per-task stability:**

| Task | Pairs tested | Stable? |
|---|---|:---:|
| basic_arithmetic | T3, T4, T7 | stable G1 |
| leg_counting | T1, T3 | stable G1 |
| knights_knaves | T2, T6 | stable G1 |
| mini_sudoku | T1, T2 | stable G2 |
| **shortest_path** | T5, T6 | **FLIPS** (G2 with Arith, G1 with KK) |
| **syllogism** | T4, T5 | **FLIPS** (G2 with Arith, G1 with SP) |

**F34.** Group membership is pair-dependent, not task-intrinsic.

### 6.2 Expanding pair set to n=19 [F35, F36]

**All 19 pairs:**

| Pair | 3B mag_agree | 4B damage | Source |
|---|---:|---:|---|
| T1 sdk+leg | 0.5300 | 18.0 | original |
| T2 sdk+kk | 0.5306 | 22.0 | original |
| T3 ari+leg | 0.5641 | 12.5 | original |
| T4 ari+syl | 0.6053 | 4.5 | original |
| T5 sp+syl | 0.5358 | 7.0 | original |
| T6 sp+kk | 0.5131 | 18.0 | original |
| T7 cnt+ari | 0.4931 | 24.0 | original |
| T8 leg+syl | 0.5376 | 7.5 | Tranche 1 |
| T9 ari+kk | 0.5121 | 22.0 | Tranche 1 |
| T10 cnt+leg | 0.5001 | 0.5 | Tranche 1 |
| T11 sp+leg | 0.5511 | 0.5 | Tranche 1 |
| T12 sdk+syl | 0.5264 | 19.5 | Tranche 1 |
| T13 ari+sdk | 0.5469 | 21.0 | Tranche 2 |
| T14 syl+kk | 0.5080 | 20.0 | Tranche 2 |
| T15 kk+sdk | 0.5306 | 64.5 | Tranche 2 |
| T16 cnt+kk | 0.5219 | 10.5 | Tranche 2 |
| T17 cnt+syl | 0.5069 | 40.0 | Tranche 2 |
| T18 cnt+sdk | 0.5209 | 45.5 | Tranche 2 |
| T19 sp+sdk | 0.5300 | 51.0 | Tranche 2 |

**F35. Predictor decay:**

| n | Pearson r | 95% bootstrap CI | Spearman r (p) |
|---|---:|---|---:|
| 7 | −0.810 | [−0.993, −0.376] | — |
| 12 | −0.447 | [−0.872, +0.161] | −0.513 (p=0.088) |
| 19 | **−0.267** | **[−0.591, +0.140]** | −0.362 (p=0.128) |

**F36.** T10/T11 (leg_counting pairs, damage=0.5) and T15 (kk+sdk, damage=64.5) are the outliers. Large variance in damage at fixed sign-agreement.

### 6.3 L1-ARS falsified [F37]

| Pair | 3B L1_mean | 4B damage |
|---|---:|---:|
| T1 sdk+leg | 0.143 | 18.0 |
| T2 sdk+kk | 0.220 | 22.0 |
| T3 ari+leg | 0.290 | 12.5 |
| T4 ari+syl | 0.500 | 4.5 |
| T5 sp+syl | 0.320 | 7.0 |
| T6 sp+kk | 0.188 | 18.0 |
| T7 cnt+ari | 0.261 | 24.0 |
| T8 leg+syl | 0.375 | 7.5 |
| T9 ari+kk | 0.368 | 22.0 |
| T10 cnt+leg | 0.136 | 0.5 |
| T11 sp+leg | 0.110 | 0.5 |
| T12 sdk+syl | 0.352 | 19.5 |

**F37.** r = −0.761 (n=7) → +0.058 (n=12). Sign flipped. Grounding-unification story falsified.

### 6.4 Routing probe [F38]

| Task | Part 3 Group | 3B | 4B | 8B |
|---|:---:|---:|---:|---:|
| basic_arithmetic | 1 | 21% | 35% | 29% |
| syllogism | 1 | 67% | 80% | 75% |
| leg_counting | 1 | 67% | 95% | 6% |
| knights_knaves | 2 | 5% | 50% | 4% |
| shortest_path | 2 | 43% | 71% | 61% |
| maze | 2 | 3% | 6% | 3% |
| mini_sudoku | 2 | 7% | 18% | 12% |
| countdown | 2 | 88% | 100% | 37% |

**F38.** r(routing_score, expected_group) = −0.31 at 4B. Does not work.

### 6.5 Per-feature residual search [F39, F40]

| Feature | r vs damage (n=12) |
|---|---:|
| 3B mag_agree | −0.447 |
| 3B q_conflict | +0.452 |
| tau_prod | −0.449 |
| tau_mean | −0.418 |
| tau_max | −0.380 |
| 4B mag_agree | −0.255 |

**F39.** No single feature exceeds |r| = 0.5. Best two-feature R² = 0.29.

**F40.** Task-level incompatibility: full-sample r = 0.82, LOO r = 0.06 (overfit).

Per-task mean damage (n=12, reference only):

| Task | n pairs | Mean damage | Min | Max |
|---|---:|---:|---:|---:|
| leg_counting | 5 | 7.8 | 0.5 | 18.0 |
| shortest_path | 3 | 8.5 | 0.5 | 18.0 |
| syllogism | 4 | 9.6 | 4.5 | 19.5 |
| countdown | 2 | 12.2 | 0.5 | 24.0 |
| basic_arithmetic | 4 | 15.8 | 4.5 | 24.0 |
| mini_sudoku | 3 | 19.8 | 18.0 | 22.0 |
| knights_knaves | 3 | 20.7 | 18.0 | 22.0 |

### 6.6 Summary

| # | Finding | Status |
|---|---|---|
| F34 | Group split is pair-dependent | Revises F19/F23/F24 |
| F35 | Sign-agreement r decays: −0.81→−0.27 | Falsifies F29 as predictor |
| F36 | Outlier diagnostics | Diagnostic |
| F37 | L1-ARS sign flipped at n=12 | Falsifies F33 |
| F38 | Routing probe null | New negative |
| F39 | No single feature > |r|=0.5 | New negative |
| F40 | Task incompatibility overfit | New negative |

**Surviving positives (post-Part 6):**

1. `mlp.down_proj` always safe to drop (40/48 cells at 4B)
2. Simple summation at s=1.0 produces usable multi-skill models
3. Atomic-vs-direct τ's compose differently (cross-scale-stable sign per task)
4. attn.{k,v,o} rotation substantial (30–70°)
5. Layer-selective merging works per pair (no shortcut)

### 6.7 Artifacts (Part 6)

| Path | Contents |
|---|---|
| `paper_prep/gaps/layersel/` | 18 layer-selective merge checkpoints |
| `paper_prep/gaps/eval/layersel/` | 144 eval metrics |
| `paper_prep/gaps/newpairs/{T8..T12}/` | 5 Tranche 1 merges |
| `paper_prep/gaps_pathB/merges/{T13..T19}/` | 7 Tranche 2 merges |
| `paper_prep/gaps_pathB/evals/` | 56 eval metrics |
| `paper_prep/gaps/sign_agreement/` | Sign-agreement data (T8–T12) |
| `paper_prep/gaps_pathB/sa/` | Sign-agreement data (T13–T19) |
| `paper_prep/gaps/routing_probe/` | Routing-probe scores |
| `paper_prep/gaps/predictor_n12*.json` | n=12 predictor analysis |
| `paper_prep/gaps_pathB/predictor_n19.json` | n=19 predictor analysis |
| `paper_prep/gaps/task_incompat.json` | Task incompatibility data |
| `paper_prep/gaps/residual_features.json` | Residual search |
| `paper_prep/{run_gaps_1_2.sh, run_routing_probe.sh, run_path_b.sh}` | Pipeline scripts |
| `paper_prep/{routing_probe.py, sign_agreement_pairs.py, predictor_n12_full.py, residual_analysis.py, compute_n19.py}` | Analysis scripts |
| `paper_prep/logs/` | Execution logs |

### 6.8 Paper framing

The paper was rewritten on 2026-05-12 around:

1. `mlp.down_proj` droppable recipe (actionable)
2. Pair-dependent Group structure (classification limitation)
3. Null-predictor catalogue (honest empirical map)
4. Atomic vs direct linking (mechanistic observation)

Target venue: TMLR or NeurIPS workshop, framed as empirical characterization. The honest negative results strengthen the paper — the field has implicitly assumed task-intrinsic merge structure and quantitative pre-probes; our n=19 evidence rules out both for reasoning-RLVR τ's.
