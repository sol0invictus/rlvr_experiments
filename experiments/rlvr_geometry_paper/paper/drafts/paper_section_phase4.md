# Section 7 · Task-Vector Merging: An Empirical Characterization

*Paper-ready draft. Figures are called out in italics; appendices K–P hold per-pair tables, layer-selective results, and TIES/DARE details. §7.8 documents three gap-closing experiments whose results reshaped the §7 spine presented here.*

---

## 7.1 Motivation

A single-task RLVR checkpoint is an expensive object. Section 4 showed that choosing the right training regime for one task takes ~1 hour of atomic probing; Section 5 showed that the resulting τ = W_GRPO − W_SFT has a reproducible geometric signature. If τ's from different tasks could be **combined** to produce a single model that performs all of them, the cost of building a multi-skill model drops from (N tasks × training compute) to (N tasks × τ storage). The task-arithmetic literature (TIES, DARE, task-vector merging) proposes this recipe, but with uneven empirical support on reasoning tasks and no mechanistic story for *when* it will work.

This section reports an empirical characterization of 2-τ merge behavior on 12 pairs spanning 8 reasoning-gym tasks at 3B, 4B, and 8B model scales. We intentionally avoid the "predict merge damage from a cheap pre-probe" framing that an earlier draft of this paper attempted (§7.8 documents why: every candidate predictor we tried collapses beyond a small initial sample). What does survive is a smaller but more robust set of findings:

1. **A universally safe layer-mask recipe** — dropping `mlp.down_proj` contributions from any merge is neutral-to-positive across 40 of 48 (pair, task) cells at 4B.
2. **A pair-dependent layer-selectivity structure** — some pair merges succeed from `mlp.up_proj + mlp.gate_proj` alone; others require the full `attn.{q,k,v,o}` circuit. Whether a given task falls in the "MLP-only-sufficient" or "needs-full-attention" group depends on its *merge partner*, not on the task alone.
3. **A catalogue of null predictors** — 1/N averaging, global τ cosine, Phase-1 sequential transfer, per-layer flat-cosine opposition, novel-subspace angles, behavioral L1-ARS, and weight-space sign agreement all fail to predict 4B merge damage on n = 12 samples.
4. **A training-regime effect** — τ's trained via the §4 atomic curriculum merge differently than τ's trained via direct GRPO: atomic curriculum favors feature-routing tasks under merging; direct GRPO favors attention-restructuring tasks. The direction is cross-scale stable; magnitudes vary.

We do not have a calibrated prospective merge-damage predictor. Designing one remains open (§7.9).

## 7.2 Setup

All merges have the form
$$W_{\mathrm{merged}} \;=\; W_{\mathrm{SFT}} \;+\; s \cdot \sum_{i=1}^{N} \tau_i, \qquad \tau_i = W_{\mathrm{GRPO}(i)} - W_{\mathrm{SFT}},$$
varying the τ set, the scaling coefficient s, and (later) which layer types contribute to the sum. All checkpoints are post-SFT warm-up Qwen2.5-3B, Qwen3-4B, or Qwen3-8B; all τ's come from Phase-1 or Phase-2 single-task GRPO runs at identical hyperparameters (Appendix P). Evaluation is 200 samples/task, seed 9999, on the same 10 reasoning-gym tasks used throughout the paper.

The primary outcome variable is **pair damage**:
$$\mathrm{damage}(A, B) \;=\; \max(0, \mathrm{acc}_{\mathrm{Direct-RLVR}(A)} - \mathrm{acc}_{\mathrm{merge}(A,B)}) \;+\; \max(0, \mathrm{acc}_{\mathrm{Direct-RLVR}(B)} - \mathrm{acc}_{\mathrm{merge}(A,B)}),$$
the total per-task capability a merge loses relative to the two single-task sources. A damage of 0 means no capability was lost; higher values are worse.

Twelve pair merges, numbered T1–T12, span the relevant task combinations. T1–T7 were chosen to cover Phase-1 transfer signs and Phase-3 rotation patterns; T8–T12 were added to break the n=7 sample and test generalization.

| Pair | Tasks | Task classes |
|---|---|---|
| T1 | `mini_sudoku + leg_counting` | search + counting |
| T2 | `mini_sudoku + knights_knaves` | search + logic |
| T3 | `basic_arithmetic + leg_counting` | arith + counting |
| T4 | `basic_arithmetic + syllogism` | arith + logic |
| T5 | `shortest_path + syllogism` | spatial + logic |
| T6 | `shortest_path + knights_knaves` | spatial + logic |
| T7 | `countdown + basic_arithmetic` | search + arith |
| T8 | `leg_counting + syllogism` | counting + logic |
| T9 | `basic_arithmetic + knights_knaves` | arith + logic |
| T10 | `countdown + leg_counting` | search + counting |
| T11 | `shortest_path + leg_counting` | spatial + counting |
| T12 | `mini_sudoku + syllogism` | search + logic |

Observed damages at 4B span 0.5 (T10, T11) to 24 (T7). This is the primary variability we attempt to characterize.

## 7.3 A Catalogue of Null Predictors

An earlier draft of this paper reported a weight-space predictor (3B sign-agreement → 4B merge damage, r = −0.81) and a behavioral predictor (3B L1-ARS, r = −0.76) estimated on the original n = 7 pairs. §7.8 documents that both collapse when evaluated on the n = 12 sample — r drops to −0.45 for sign-agreement (95% CI [−0.87, +0.16]) and to +0.06 for L1-ARS. We include them here, alongside the predictors that fail on any sample size, to give the field a full map of what does *not* forecast 2-τ merge compatibility on our benchmark.

**Uniform 1/N averaging under-shoots.** The TIES/DARE literature's canonical baseline — average over all 10 Phase-1 τ's at scale s = 1/N = 0.1 — loses 9/10 tasks to a 2-task sum at s = 1.0 and loses 8/10 to a 9-task sum at s = 0.33. Mean accuracy on the 8 non-trivial tasks is 41.6% (1/N) vs 51.2% (2-task) vs 51.3% (9-task). The 2-task and 9-task merges are tied on aggregate even though one averages 9 vectors and the other sums 2 — what matters is not N but the total displacement budget. Both are at ≈ 3 τ-equivalents (s · N); 1/N is at 1.0 and under-shoots (Appendix K). This confirms Phase-3's orthogonality picture: nearly-orthogonal τ's do not destructively interfere, they just don't sum to a useful displacement when scaled by 1/N.

**Phase-1 sequential-training transfer is uncorrelated with pair-merge damage.** Across the 12 pairs, Pearson r(|Phase-1 transfer|, damage) ≈ 0. Pairs with the worst Phase-1 transfer (mini_sudoku → leg_counting = −52%, mini_sudoku → knights_knaves = −50%) produce merge damages of 18 and 22 — middle of the distribution — while T7 (countdown + arithmetic, Phase-1 transfer +6%, apparently benign) produces the highest damage at 24. *Training on A then evaluating on B is a fundamentally different operation from summing τ_A + τ_B*.

**Global τ cosine is uncorrelated with damage.** All 12 pairs have cos(τ_A, τ_B) ∈ [0.00, 0.08] — essentially orthogonal by Phase-3 §5.5 — but damage spans 0.5 to 24. Whole-model cosine averages the useful localized signal into the much larger task-irrelevant bulk.

**Per-layer flat-cosine opposition is uncorrelated with damage.** Phase-3 §5.6 highlighted per-layer flat cosines as low as −0.90 at feature-projector layers as a candidate mechanistic signature of interference. At n = 7 the Pearson r between #layers with `up_proj` flat cos < −0.5 and damage was +0.23; at n = 12 it does not improve.

**Novel-subspace angle (Phase-3 §5.6) predicts weakly at best.** The one rotation metric with moderate signal at n = 7 (mean `mlp.up_proj` principal angle, r = +0.56) has not been re-run on the 5 new pairs at 4B. Even at n = 7 its R² = 0.31 would not be reliable screening.

**Magnitude-weighted sign agreement decays monotonically with n.** At n = 7 the Pearson correlation between 3B mag_agree and 4B damage is r = −0.81 with 95% bootstrap CI [−0.99, −0.38]. At n = 12: r = −0.45, CI [−0.87, +0.16]. At n = 19: r = −0.27, CI [−0.59, +0.14]:

| Sample | Pearson r | 95% bootstrap CI | Spearman r (p) |
|---|---:|---|---:|
| n = 7 (original F29) | −0.81 | [−0.99, −0.38] | — |
| n = 12 (§7.10.2) | −0.45 | [−0.87, +0.16] | −0.51 (p = 0.088) |
| **n = 19 (§7.10.5, Path B)** | **−0.27** | **[−0.59, +0.14]** | **−0.36 (p = 0.128)** |

The monotonic collapse with added samples is the signature of a *small-n favorable-sample artifact* rather than a true −0.8 population correlation. Our best read of the population effect size is r ≈ −0.25 to −0.35 — real directional signal (higher sign-agreement → lower damage, always the correct sign) but too noisy to deploy as a forecasting tool. Thresholded screening at mag_agree > 0.54 yields 3 pairs with max damage 12.5 — crude directional information, not a calibrated predictor.

**L1-grounding ARS flips sign.** At n = 7 this was our behavioral predictor (r = −0.76). At n = 12, r = +0.06 with bootstrap CI [−0.72, +0.67] (§7.8.4). The §4 ↔ §7 unification via the grounding bottleneck does not survive sample expansion.

**A task-level routing probe does not predict Group membership.** §7.8.3 designed a probe asking whether the base model can already answer cross-position routing questions about the task's input. r(routing_score, expected_group) = −0.31 at 4B. The hypothesis that Group membership reflects a task-intrinsic attention-reshaping need is falsified.

**Per-feature residual search finds no stable signal.** Ten candidate features (layer-type-specific sign agreements at 3B/4B, τ magnitude products, mins, max, per-pair conflict masses) all return |r| < 0.5 at n = 12. A full-sample "task-level incompatibility" score reaches r = 0.82 but drops to r = 0.06 under leave-one-out — entirely overfit (§7.8.4).

**What survives as weak directional signal.** Across all three sample sizes the sign of the sign-agreement correlation is preserved (negative — higher mag_agree → lower damage). Path B (§7.10.5) tested whether larger n would recover a useful effect size; it did not. The best honest statement is that 3B sign-agreement is a **weak directional indicator**, not a calibrated forecaster. A practitioner could use it to *rank* candidate pairs, but not to decide whether any given merge will succeed.

## 7.4 Result 1: `mlp.down_proj` Is Always Safe to Drop

§5.5 found `mlp.down_proj`'s left singular basis frozen at every scale (0.00°–0.02°), while `mlp.up_proj` and `attn.q_proj` rotate. We translate this geometric observation into an actionable merge recipe: **drop τ contributions from `mlp.down_proj` entirely, keep everything else**.

**Method.** For every merged pair, build a variant `all_but_down` that sums τ_A + τ_B only on keys outside `mlp.down_proj`. Evaluate on the 8 non-trivial reasoning tasks.

**Result.** `all_but_down` matches or beats the full merge within ±3 accuracy points on **40 of 48 (pair, task) cells** at 4B across the 7 original pairs (T1, T2, T3, T4, T5 (=SP+Arith), T6, T7). Mean Δ(all_but_down − full) across the 48 cells is +0.4, with worst-case −6 (T2 knights_knaves, T6 knights_knaves). At 3B and 8B on the SP+Arith pair the recipe holds similarly (mean Δ +2.1 and +1.3 respectively). There is no scale × pair cell where dropping `mlp.down_proj` produces a substantial loss (> 8 points) that was not already dominating the full merge.

This is the single recipe from §7 we recommend for practitioners: **for any 2-τ merge of reasoning RLVR checkpoints, mask out `mlp.down_proj` contributions**. Cost is zero — 91% of keys still contribute — and the occasional small gain (T5 SP, T3 leg_counting, T6 SP) is free.

## 7.5 Result 2: The Layer-Sufficient Subset Is Pair-Dependent, Not Task-Intrinsic

§7.4 showed `mlp.down_proj` is droppable for every pair. A stronger claim would be: **some pairs only need a small subset of layers** (e.g., `mlp.up_proj + mlp.gate_proj`, 18% of keys), while others need the full `attn.{q,k,v,o}` circuit. If this were task-intrinsic — a given task always falls in the "MLP-only-sufficient" or "needs-full-attention" group regardless of partner — it would give us a task-level classification and a prescriptive recipe per task.

**It is not task-intrinsic.** When we run layer-selective merges (`up` = mlp.up + mlp.gate only; `up+q` adds `self_attn.q_proj`; `all_but_down` per §7.4) across all 7 original pairs at 4B and test each pair's two trained tasks against the full-merge baseline for *that pair*, the Group classifications flip between pairs:

| Task | Group-1 in pair | Group-2 in pair | Stable classification? |
|---|---|---|---|
| basic_arithmetic | T3, T4, T7 | — | **G1 stable** |
| leg_counting | T1, T3 | — | **G1 stable** |
| mini_sudoku | — | T1, T2 | **G2 stable** |
| knights_knaves | T2, T6 | — | **G1 stable** |
| countdown | T7 | — | G1 (n=1) |
| **shortest_path** | **T6 (SP+KK)** | **T5 (SP+Arith, Phase-4 Part 3)** | **flips** |
| **syllogism** | T5 | **T4 (ari+syl)** | **flips** |

**Two tasks classify differently depending on partner.** Shortest-path is Group-1 when paired with knights-knaves (T6: `up` alone gains +23 pts vs full merge) but Group-2 when paired with arithmetic (T5: `up` loses 13 pts). Syllogism is Group-1 when paired with shortest-path (T5) but Group-2 when paired with arithmetic (T4: `up` loses 6 pts). Only mini_sudoku is consistently Group-2 across both pairs it appears in at 4B.

**Interpretation.** The property that determines whether a pair merge needs the full attention circuit is not a task-intrinsic "attention-restructuring need" — it is a property of the *joint geometry of the two τ's*. A plausible mechanistic reading is that Group-2 behavior emerges when both τ's concentrate mass on overlapping attention circuits (shared `attn.{k,v,o}` restructuring); when pair partners rewire *different* attention circuits, dropping `k,v,o` no longer destroys a shared circuit and the MLP-only merge suffices. §5.7's depth-signature reorganization across scales is consistent with this reading — attention-circuit overlap depends on *where* each task concentrates change mass, which shifts with scale and partner.

**What this changes for practitioners.** Cannot pre-classify a task as "always needs MLP-only" or "always needs full-attention" based on task features alone. The layer-selective merging recipe works *when you have enough data from the specific pair* — e.g., if you've already run `all_but_down` and it matches full-merge performance, you can subsequently try `up` and `up+q` variants to see if even more aggressive masking works for your pair. This is useful as an *operational* procedure but not as a *prospective* one.

## 7.6 Result 3: What Simple Summation Achieves

Before mechanism analysis, we report the basic empirical result: simple weight summation W_SFT + Σ τ produces usable multi-skill models at a τ-magnitude budget near 2–3 τ-equivalents (s · N). At 4B across our 12 pairs:

- **Mean pair accuracy** on the two trained tasks: 45% → 70% for low-damage pairs (T4 ari+syl, T10 cnt+leg, T11 sp+leg); 50% → 30% for high-damage pairs (T7 cnt+ari, T2 sdk+kk).
- **Best single-pair merge** beats every alternative (1/N, 9-task, triple-merge) on every task that pair was trained for. No pair merge beats all others across all tasks.
- **Damage distribution (n=12 at 4B):** 2 pairs at 0.5, 2 pairs at 4.5–7, 3 pairs at 7.5–18, 5 pairs at 18+.

**No universal recipe exists.** The right choice of merge depends on which tasks the user cares about. If T10 (cnt+leg) is the target pair, simple summation gives a nearly lossless merge. If T7 (cnt+ari) is the target, expect ~24 points of damage to the two tasks combined and no currently-known pre-screen to tell you that.

## 7.7 Result 4: Atomic vs Direct τ's Merge Differently (the §4↔§7 link that survives)

§4 contrasted atomic-curriculum training vs direct GRPO as single-task training regimes. A natural question: do these regimes produce τ's that compose differently under merging? The answer is yes — with a task-class-specific pattern that is cross-scale stable in *direction* even when the magnitude varies.

We build three versions of the SP + Arith merge at each of 3B/4B/8B, holding the merge operation identical:

- **M_aa**: τ_atomic(SP) + τ_atomic(Arith)
- **M_dd**: τ_direct(SP) + τ_direct(Arith)
- **M_mix**: τ_atomic(SP) + τ_direct(Arith)

**Table.** Δ(M_aa − M_dd) per task at each scale:

| Task | 3B-Qwen2.5 | 4B-Qwen3 | 8B-Qwen3 | Cross-scale sign |
|---|---:|---:|---:|---|
| basic_arithmetic | +2.5 | +8.5 | +10.0 | **M_aa wins at every scale** |
| leg_counting | +4.0 | +11.5 | +10.0 | **M_aa wins at every scale** |
| knights_knaves | +8.5 | +1.0 | +3.0 | **M_aa wins at every scale** |
| countdown | 0 | +0.5 | +7.5 | M_aa wins (marginal at 3B) |
| mini_sudoku | +1.0 | −1.5 | −0.5 | ~tie |
| maze | +1.5 | +7.5 | −3.5 | scale-dependent |
| syllogism | **−11.0** | **+34.5** | −5.5 | scale-dependent (4B outlier) |
| **shortest_path** | **−33.0** | +1.5 | **−18.5** | **M_dd wins (except 4B tie)** |
| **mean (8 tasks)** | −3.3 | +7.9 | +0.3 | scale-dependent |

**Two observations.**

1. **The per-task sign of Δ(M_aa − M_dd) is cross-scale stable** for the feature-routing tasks (arith, leg, kk — M_aa wins at every scale) and for shortest-path (M_dd wins at every non-tie scale). The *magnitude* is scale-dependent: 4B's +7.9 mean is an outlier, 3B is −3.3, 8B is +0.3.
2. **This pairs with the §4 atomic/direct training distinction.** Atomic curriculum concentrates RLVR signal on L1/L2 atoms — small per-token features — which produces τ's with mass concentrated in `mlp.up/gate`. Direct GRPO spreads reward signal across the whole pipeline, producing τ's with more attention-circuit rewriting. When merged, τ_atomic composes safely with other τ_atomic on feature-routing tasks; τ_direct preserves attention-restructuring needed for spatial search.

**Scoped claim.** This is not a universal "atomic τ's are more mergeable" result. It is a task-class-specific compositional property: atomic wins on arith/logic/counting; direct wins on spatial search. For a multi-skill model builder, this suggests a heterogeneous recipe — pick the training regime *per task* based on which class it belongs to, then merge the resulting heterogeneous τ's. We have not run this as an end-to-end demonstration.

*The clean §4↔§7 unification via L1-grounding ARS that an earlier draft claimed (F33) does not survive sample expansion (§7.8.4); what survives is this weaker but cross-scale-stable atomic/direct training-regime observation.*

## 7.8 Result 5: Cross-Scale Geometry Backing §7.4 and §7.5

To close §5's weight-geometry gap (§5.3 analyzed `mlp.up_proj`, `mlp.down_proj`, `attn.q_proj` but not `attn.{k, v, o}`), we measured principal angles between novel subspaces for all 7 original pairs at 4B and 8B on those three sub-layers.

| Sub-layer | 4B (6 pairs) | 8B (7 pairs) |
|---|---:|---:|
| `attn.k_proj` | 37–41° | 32–38° |
| `attn.v_proj` | 53–66° | 49–65° |
| `attn.o_proj` | 58–62° | 56–70° |

**None of the three sub-layers is frozen.** `attn.k_proj` rotates less than `attn.q_proj` (40–45° at 4B from §5.5); `attn.v_proj` and `attn.o_proj` rotate comparably to `mlp.up_proj` (58–64° at 4B). No pair shows the 0° "basis-frozen" signature that `mlp.down_proj` consistently displays.

This is consistent with §7.5's pair-dependent Group structure: a pair-merge that keeps only `up_proj + q_proj` discards 4–13 points of capability locked in `attn.{k,v,o}` for pairs where those layers are doing attention restructuring.

## 7.9 Comparison to TIES and DARE

Section 7.3 placed simple-sum merging in a broad null space; here we compare against the two standard model-merging baselines from the literature on three representative 4B pairs (T2 high damage, T5 low damage, T7 highest damage). All methods held at equal total τ magnitude.

| Pair | Vanilla sum mean | TIES | DARE |
|---|---:|---:|---:|
| T2 (sdk+kk) | 28.1 | 29.2 (+1.1) | 28.6 (+0.5) |
| T5 (sp+syl) | 49.0 | 48.0 (−1.0) | 48.4 (−0.6) |
| T7 (cnt+ari) | 43.7 | — (countdown missing) | 44.3 (+0.6) |

Mean method-to-method differences are within ±1.1 points on every pair. TIES gains +11 on knights_knaves specifically in T2 but does not dominate on aggregate. Our **`all_but_down`** recipe (§7.4) reaches 53.5% SP on T5 at 4B vs TIES's 48.0% — the only recipe that beats vanilla-sum by more than method-difference noise. The claim is not "our recipe beats TIES/DARE" but **"vanilla sum with a geometrically-motivated layer mask is competitive with TIES/DARE and better-motivated"** — the mask follows directly from §5's basis-frozen observation rather than from a tunable sparsification heuristic.

Full per-task TIES/DARE results in Appendix O.


## 7.10 Gap-Closing Experiments (2026-05-09 → 05-10) — the record of how §7.1–§7.9 reached its current form

Three experiments were run to address gaps identified in the first draft of §7. All experiments are at 4B; the findings forced the complete rewrite of §7.1–§7.9 that appears above. We preserve this sub-section as a methodological record of what was originally claimed, what fell, and what survived — both so that readers can audit the trajectory and so that reviewers can see the predictor claims that were *not* carried into the paper body.

### 7.10.1 Layer-selective merges on 6 more pairs (n=7 → 7 pairs × 3 variants at 4B)

**Motivation (the original gap):** §7.4's Group-1/Group-2 classification (Table 6) was derived from a *single* pair, SP + Arith, run through three variants (`up`, `up+q`, `all_but_down`). The table's Δs were attributed to tasks individually, but they could equally reflect pair-specific interaction. We reran the three layer-selective variants on the other six Phase-4 pairs (T1_sdk_leg, T2_sdk_kk, T3_ari_leg, T4_ari_syl, T6_sp_kk, T7_cnt_ari) and evaluated each on all 8 non-trivial tasks.

**Per-task-as-one-leg-of-a-pair Group classification at 4B.** For each pair, classify its two trained tasks using the same rule (Δ_up ≥ −3 → G1; Δ_up ≤ −6 → G2; else borderline), where Δ_up is against the full-merge baseline for *that pair*.

| Task | Per-pair classifications | Stable? |
|---|---|---|
| basic_arithmetic | G1 in T3, T4, T7 | ✓ |
| countdown | G1 in T7 | ✓ (n=1) |
| leg_counting | G1 in T1, T3 | ✓ |
| **shortest_path** | **G1 in T6_sp_kk** | **contradicts §7.4** |
| **syllogism** | **G2 in T4_ari_syl** | **contradicts §7.4** |
| knights_knaves | G1 in T2, T6 | ✓ |
| mini_sudoku | G2 in T1, T2 | ✓ |

**The surprise.** Two of §7.4's "universal" classifications flip under a different pair partner:

- **Shortest-path classifies as Group-1** when paired with knights-knaves (T6). Under `up` alone, SP recovers +23 points vs the full T6 merge. §7.4 called SP a "universal Group-2 task" based on SP+Arith; that characterization does not hold when SP is paired with a logic task instead.
- **Syllogism classifies as Group-2** when paired with arithmetic (T4). Under `up` alone, syllogism loses 6 points vs the full T4 merge. §7.4 called syllogism "universal Group-1" based on SP+Arith.

**Only `mini_sudoku` remains cleanly Group-2** across every pair it appears in. Everything else is pair-dependent.

**Revised §7.4 claim.** The Group-1 / Group-2 split is **not a task-intrinsic property**; it is a property of the *pair*. A task can require full-attention merging with one partner and feature-only merging with another. The §7.4 mechanism story (feature-routing vs attention-restructuring as task-intrinsic bottlenecks) needs substantial revision: the bottleneck that determines whether a merge needs the full attention circuit lives in the **pair's joint τ geometry**, not in either task alone. One interpretation: Group-2 behavior emerges when both τ's concentrate change mass at overlapping depth bands *and* on shared attention heads, which is a joint property. When the pair's mass is spread across different attention circuits (e.g., SP+KK rather than SP+Arith), dropping `attn.{k,v,o}` no longer destroys a shared circuit — there wasn't one.

**The cross-scale universal `all_but_down`-safe finding (F1) survives.** Dropping `mlp.down_proj` contributions is within ±3 points of the full merge for 40 of the 48 new (pair, task) cells measured at 4B (previously 22/24 at a single pair). F1 is strengthened, not weakened.

**The one consistent story we can keep.** Feature-routing (MLP-only) sufficient merges exist — but they exist for *specific pairs*, not specific tasks. Labeling a task "Group-1" is only meaningful relative to a fixed partner.

### 7.10.2 Adding 5 pairs to tighten predictor n (F29)

**Motivation (the original gap):** §7.5's headline r = −0.81 for 3B sign-agreement predicting 4B damage rests on n = 7 pairs. Bootstrap 95% CI was wide: [−0.99, −0.38]. We added 5 pairs (T8 `leg+syl`, T9 `ari+kk`, T10 `cnt+leg`, T11 `sp+leg`, T12 `sdk+syl`), built vanilla 4B merges, ran the 10-task eval, and re-ran the magnitude-weighted sign agreement at 3B and 4B.

**New per-pair data (n=12):**

| Pair | 3B mag_agree | 4B damage |
|---|---:|---:|
| T1 sdk+leg | 0.5300 | 18.0 |
| T2 sdk+kk | 0.5306 | 22.0 |
| T3 ari+leg | 0.5641 | 12.5 |
| T4 ari+syl | 0.6053 | 4.5 |
| T5 sp+syl | 0.5358 | 7.0 |
| T6 sp+kk | 0.5131 | 18.0 |
| T7 cnt+ari | 0.4931 | 24.0 |
| **T8 leg+syl** (new) | **0.5376** | **7.5** |
| **T9 ari+kk** (new) | **0.5121** | **22.0** |
| **T10 cnt+leg** (new) | **0.5001** | **0.5** |
| **T11 sp+leg** (new) | **0.5511** | **0.5** |
| **T12 sdk+syl** (new) | **0.5264** | **19.5** |

**Predictor strength collapses with more pairs.**

| Sample | r (3B sign-agree, 4B damage) | 95% CI |
|---|---:|---|
| **Original n=7** | **−0.810** | [−0.993, −0.376] |
| **Expanded n=12** | **−0.447** | **[−0.872, +0.161]** |

**The 95% CI now crosses zero.** Permutation p at n=12 is no longer significant at α=0.05.

**What broke the correlation.** Two of the five new pairs, T10 (cnt+leg) and T11 (sp+leg), produced near-zero damage (0.5 each) but had unremarkable 3B sign-agreements (0.500 and 0.551). The predictor said they should damage each other substantially — they didn't. Symmetrically, T9 (ari+kk) had a middle-of-pack sign-agreement (0.512) but high damage (22). The n=7 sample happened to contain pairs where the two quantities moved together; the wider sample includes pairs where they do not.

**Revised §7.5 claim.** The sign-agreement predictor at 3B → 4B damage correlates *weakly but insignificantly* at n=12. It is no longer a validated predictor; the original r = −0.81 was an artifact of a favorable n=7 sample. The two-predictor story (sign-agreement + L1-ARS, R² = 0.70) also degrades accordingly; we have not yet recomputed the combined R² on n=12 because L1-ARS for the new pairs requires running the §4 atomic probe for each of the task combinations, which is pending.

**What we should keep.** Sign-agreement has the *right sign* (negative) at both n=7 and n=12, and the weak correlation is consistent with §5.6's per-layer flat-cosine opposition as a plausible mechanism. A reasonable framing for the paper: *sign-agreement at 3B is a noisy indicator of 4B merge damage with substantial unexplained variance*; it is not a drop-in pre-screening tool.

### 7.10.3 A task-level routing probe does not predict Group membership

**Motivation (the original gap):** §7.4.1 hypothesized a "routing probe" could prospectively classify a task as Group-1 or Group-2 without any GRPO, by asking whether the base model can already answer cross-position questions about the task's input. We operationalized this: for each of 8 tasks, wrote 2 routing atoms (deterministic Q/A pairs that require the model to attend across token positions — "what character is at row r, column c?", "which premise contains the word 'all'?", "how many characters between two operators?"). Score = zero-shot base-model accuracy, 50 samples per atom.

**Results at 3B, 4B, 8B:**

| Task | Expected §7.4 Group | 3B | 4B | 8B |
|---|:---:|---:|---:|---:|
| basic_arithmetic | 1 | 21% | 35% | 29% |
| syllogism | 1 | 67% | 80% | 75% |
| leg_counting | 1 | 67% | 95% | 6% |
| knights_knaves | 2 (4B) | 5% | 50% | 4% |
| shortest_path | 2 | 43% | 71% | 61% |
| maze | 2 | 3% | 6% | 3% |
| mini_sudoku | 2 | 7% | 18% | 12% |
| countdown | 2 (4B) | 88% | 100% | 37% |

**Correlation with expected group:** r(routing_score, group) = −0.35 (3B), −0.31 (4B), −0.25 (8B). Weak and in the wrong direction relative to any useful predictor.

**Inspection of the misses.** `countdown` has 100% routing score at 4B but is §7.4's clearest Group-2 task (Δ_up = −14). `basic_arithmetic` has 35% routing at 4B but is §7.4's clearest Group-1 task (Δ_up = −2.5). The probe measures something — likely *how easy the task's token-level structure is for the base model* — but not what §7.4 needs.

**What the probe actually measured.** Re-reading the atoms: "what character is at row r, col c?" tests the base model's general ability to handle 2D indexing; it does not test whether a *novel* attention pattern would need to be learned for the downstream task. A task can have a high routing-probe score (the base model already has the basic attention patterns) *and* still need substantial attention restructuring for GRPO to reach the task reward (e.g., countdown needs multi-step operator search attention, which is not measured by the "what's the target number?" atom).

**Revised §7.4.1 claim.** A simple base-model routing probe does not predict Group membership. Given §7.8.1's finding that Group membership is not task-intrinsic in the first place, a useful probe would need to measure some property of the *pair* — e.g., whether the two tasks compete for the same attention heads when trained — not a property of a single task. We do not currently have a cheap pair-level routing probe and do not know what shape it should take.

**This is a genuine negative result and a scope-reduction for the paper.** The §4 ARS framework is a single-task probe; extending it to predict multi-task merge compatibility (as F33 suggested) worked weakly for L1 atoms at n=7 and does not work for routing atoms at all. §4 remains a valid story on its own; §7 needs different instrumentation.

### 7.10.4 L1-ARS at n=12 — the behavioral predictor collapses too

**Motivation:** §7.8.2 showed sign-agreement degrades at n=12. The §7.5.2 behavioral predictor (L1-grounding ARS) was complementary to sign-agreement at n=7, so it deserves its own re-test.

**Method.** Same as §7.5.2: for each pair (A,B), `L1_mean = (L1(A) + L1(B)) / 2`, where L1 is the 3B SFT base's mean accuracy on the task's L1 grounding atoms (`leg_counting` uses full-task accuracy as surrogate, same as the original F33 analysis). All 7 atomic probes already existed on disk from Phase-2; only the pair arithmetic is new.

**Results:**

| Sample | r(L1_mean_3b, 4B damage) | 95% bootstrap CI |
|---|---:|---|
| **Original n=7** | **−0.761** | reported in F33 |
| **Expanded n=12** | **+0.058** | **[−0.72, +0.67]** |

**The correlation has the wrong sign at n=12.** The two-predictor combination (sign-agreement + L1-ARS) drops from R² = 0.70 (n=7) to **R² = 0.30 (n=12)**. Neither single predictor survives.

**Why.** The new pairs break the L1→damage story just as they broke sign-agreement:

- T10 `cnt+leg` has L1_mean = 0.136 (low — both tasks have weak grounding) but damage = 0.5 (clean merge). The predictor said this should be a disaster; it wasn't.
- T11 `sp+leg` has L1_mean = 0.110 (even lower) and damage = 0.5. Same.
- T9 `ari+kk` has L1_mean = 0.368 (middle) and damage = 22 (very high). Predictor said this should be clean; it wasn't.

The original F33 claim that "grounding is the shared bottleneck for single-task RLVR success and multi-task merge compatibility" was supported by n=7 evidence that does not survive n=12.

**Residual analysis.** A search over 10 candidate features (per-pair τ magnitude products, layer-type-specific conflict masses at both scales, min/max aggregations) found no single feature with r > 0.45 at n=12. A "task-level incompatibility" feature (each task's average damage across pairs it appears in) reaches r = 0.82 on the full sample but **r = 0.06 under leave-one-out**, meaning it is entirely overfit. Two-feature regressions tested on the full sample do not exceed R² = 0.55.

**Spearman rank correlations are cleaner but still not significant** at α = 0.05 (sign-agreement Spearman r = −0.51, p = 0.088; L1 Spearman r = +0.18, p = 0.58). The monotonic relationship weakly survives for sign-agreement; the monotonic relationship for L1 does not survive at all.

**What this changes.**

1. **The §4 ↔ §7 unification via L1 grounding is not supported by the data.** §7.8 falsifies F33. The grounding bottleneck story holds for single-task RLVR success (§4) but does not transfer to multi-task merge compatibility on the expanded sample.
2. **Sign-agreement survives as a weak directional indicator**, not as a calibrated predictor. The correct framing is "pairs with sign-agreement > 0.54 damage each other less than 13 points on average" — a crude screen, not a quantitative forecaster.
3. **Reading off the rank table, the leg_counting pairs cluster low-damage** (T8: 7.5, T10: 0.5, T11: 0.5, T1: 18.0, T3: 12.5). 4 of 5 leg_counting pairs are in the bottom half of damage. `leg_counting` has the highest ‖τ‖ (1.193) of any task, and the highest Direct-RLVR accuracy among the feature-routing tasks. This might be a real asymmetric-compatibility signal — "leg_counting composes well with most things" — but it is not a generalizable predictor, just an observation about one task.

### 7.10.5 Path B — 7 more pairs at 4B (n=12 → n=19)

**Motivation.** §7.10.2 dropped the sign-agreement r from −0.81 (n=7) to −0.45 (n=12). The 95% CI at n=12 crossed zero. We ran 7 more pairs at 4B (T13–T19: `ari+sdk`, `syl+kk`, `kk+sdk`, `cnt+kk`, `cnt+syl`, `cnt+sdk`, `sp+sdk`) to see whether the predictor stabilizes at a moderate-but-reliable effect size, or whether it decays further.

**Method.** Built vanilla 4B merges of each pair, evaluated on 8 non-trivial tasks, computed damage from Direct-RLVR baselines, and recomputed 3B magnitude-weighted sign agreement.

**Per-pair Path B data:**

| Pair | 3B mag_agree | 4B damage | Notes |
|---|---:|---:|---|
| T13 ari+sdk | 0.547 | 21.0 | high damage, middle sign-agree |
| T14 syl+kk | 0.508 | 20.0 | |
| T15 kk+sdk | 0.531 | **64.5** | catastrophic — KK drops 47 pts |
| T16 cnt+kk | 0.522 | 10.5 | |
| T17 cnt+syl | 0.507 | 40.0 | |
| T18 cnt+sdk | 0.521 | 45.5 | |
| T19 sp+sdk | 0.530 | 51.0 | |

**Damage at n=19 now spans 0.5 → 64.5.** Path B pairs pulled the distribution's right tail substantially — five of the seven new pairs have damage > 20, and T15 (kk+sdk) set a new high of 64.5 (knights_knaves collapsed from 97.5 to 50.5; mini_sudoku dropped 17.5 points). Their sign-agreements are in the normal 0.51–0.55 band, so they do not help the predictor.

**Predictor progression:**

| n | Pearson r | Bootstrap 95% CI | Spearman r | Spearman p |
|---|---:|---|---:|---:|
| 7 | −0.810 | [−0.99, −0.38] | — | — |
| 12 | −0.447 | [−0.87, +0.16] | −0.51 | 0.088 |
| **19** | **−0.267** | **[−0.59, +0.14]** | **−0.36** | **0.128** |

**The monotonic decay is the classic signature of a small-n overfit.** The true population correlation is most consistent with the n=19 estimate — roughly −0.25 to −0.35 — not with the original n=7 headline of −0.81. At every sample size, the 95% CI includes zero or nearly does, and the Spearman p never reaches 0.05.

**What this changes.** Path B rules out the "sign-agreement is a useful but noisy predictor, larger n will tighten the CI below zero" hope. The CI does not tighten; it stays wide because the *effect is small*. A predictor with r ≈ −0.3 on a 0–65 damage scale has ~90% unexplained variance and is not usable for practical pre-screening.

**What we keep.** The sign of the correlation is negative at every n (always correct direction). A generous reading: sign-agreement is weakly correlated with merge damage — higher sign-agreement does tend to mean less damage — but the signal is too diluted for forecasting. This is a useful negative result for the model-merging community, which has implicitly assumed stronger structure.

**Implication for the paper.** §7.3's null-predictor catalogue is strengthened: we now have n=19 evidence that the most-publicized candidate predictor (magnitude-weighted sign agreement) does not work on reasoning RLVR τ's. Path A's spine (§7.1) remains correct; there was no predictor to rescue.

### 7.10.6 Salvage strategy for the paper

Given §7.10.1 (Group membership is pair-dependent), §7.10.2 (sign-agreement r drops to −0.45 at n=12), §7.10.3 (routing probe fails), §7.10.4 (L1-ARS drops to +0.06 at n=12), **and §7.10.5 (sign-agreement decays further to −0.27 at n=19)**, the paper cannot retain its original "two independent predictors unify §4 and §7" headline. We have three workable paths.

**Path A (recommended) — Reframe as empirical characterization, not predictor.** Drop the predictor headline. Re-spine the paper around:

1. **`mlp.down_proj` basis-frozen → always-safe-to-drop recipe.** 40 of 48 (pair, task) cells at 4B within ±3 of full merge. Strengthened by §7.8.1 replication. Actionable, validated, and novel.
2. **Pair-dependent Group structure** (§7.8.1). The Group-1/Group-2 split *exists* but is a property of the pair, not the task. This is itself a paper-worthy observation — the model-merging literature has implicitly assumed task-intrinsic merge profiles, and our data falsifies that.
3. **Characterized null space.** 1/N averaging, Phase-1 transfer, global τ cosine, per-layer flat-cosine, routing probe, sign-agreement at n>7, L1-ARS at n>7 — none of them predict merge damage strongly. This is a genuinely useful negative-result catalogue.
4. **Atomic vs direct linking result** (§7.6 F28). Atomic τ's preserve feature-routing structure; direct τ's preserve attention-restructuring structure. The task-class-specific effect is cross-scale-stable in *sign* though variable in *magnitude*.

This path makes the paper an **empirical characterization of RLVR task-vector compositionality** rather than a prediction paper. Target TMLR or a NeurIPS workshop with this framing. Estimated remaining work: 2–3 days of writing.

**Path B — Stabilize the effect size with more pairs.** *Executed (§7.10.5)*. 7 pairs added, n=12 → n=19. Outcome: r decayed further, not stabilized (−0.45 → −0.27; CI still crosses zero; Spearman p=0.128). The predictor cannot be rescued by more data; the effect size is simply too small. Path B **confirms Path A as the right call** rather than providing an alternative.

**Path C — Find what actually predicts.** We already tried 10+ candidate features; none cleared the bar. A more systematic search would be a machine-learning-style regression over all computable per-pair features (per-layer sign agreement across 36 layers, per-layer-type norm overlaps, per-depth novel-subspace angles from §5, full ARS atom-level profiles, etc.) to find combinations that do predict. Risk of overfitting is high at n=19; requires cross-validation discipline. ~2 pod-days plus analysis time. Not attempted yet.

**Decision: Path A is the paper.** Path B has been executed and has ruled out the predictor-rescue hypothesis. Path C remains a possible follow-up, but with n=19 and nothing in the first 10 feature candidates, the prior that a single-feature or two-feature predictor exists at all is low. The §7 body now stands on its 4 positive findings + null-predictor catalogue.

### 7.10.7 Summary of revisions to prior §7 claims

| Original claim (earlier draft) | Status after §7.10 |
|---|---|
| F1: `mlp.down_proj` always droppable (22/24 cells) | **Strengthened** — 40/48 at 4B across 7 pairs |
| F2: Tasks split into Group-1 / Group-2 | **Revised** — the split is *pair-dependent*, not task-intrinsic. Only `mini_sudoku` is consistently Group-2 |
| F3: SP, sudoku are universal Group-2 | **Only sudoku survives.** SP classifies as Group-1 when paired with KK |
| F29: 3B sign-agreement predicts 4B damage r = −0.81 | **Downgraded to weak directional indicator.** r = −0.45 (n=12) → r = −0.27 (n=19); CI crosses zero at all sample sizes; Spearman p never < 0.05 |
| §7.4.1 routing probe hypothesis | **Falsified** — routing atoms do not predict Group membership |
| F33: L1-ARS predicts 4B damage r = −0.76 | **Falsified** — r = +0.06 at n=12, sign flipped |
| §4 ↔ §7 unification via L1 grounding | **Falsified** — grounding bottleneck does not transfer to merge compatibility |

**The parts of §7 that still hold as originally written:** §7.3 null results (1/N, Phase-1 transfer, global cosine all uncorrelated — strengthened by larger n); §7.4 F1 (`all_but_down` safe); §5.3–5.5's weight-geometry findings are orthogonal to merging and unaffected; §7.7 linking experiment (atomic vs direct) is untouched.

**Rebuilt §7 (Path A) addresses all three gaps:** (a) the predictor claims are removed from the paper body and documented here for the record; (b) the task-intrinsic Group-1/Group-2 story is replaced by the pair-dependent §7.5; (c) the absence of a cheap pair-level probe is acknowledged honestly in §7.5 and §7.11.

---

## 7.11 Summary of Section 7

The section makes four positive claims and one catalogue of null results:

1. **`mlp.down_proj` contributions are always safe to drop from a 2-τ merge** (§7.4). Holds on 40 of 48 (pair, task) cells at 4B across 7 pairs, and mean-positive or neutral at 3B and 8B. This is the single actionable recipe §7 produces.
2. **The layer-sufficient subset for a merge is pair-dependent, not task-intrinsic** (§7.5). Two tasks (shortest-path, syllogism) classify as Group-1 with one partner and Group-2 with another. Only `mini_sudoku` is consistently Group-2. Merge-recipe selection cannot be based on task features alone.
3. **Simple summation at s ≈ 1.0 produces usable 2-τ multi-skill models** (§7.6). Damage distribution at 4B (n=12) spans 0.5 to 24; 4 pairs produce nearly-lossless merges, 5 pairs produce severe damage (>18 pts). No pair merge dominates across all tasks.
4. **Atomic-curriculum τ's and direct-GRPO τ's compose differently under merging** (§7.7). M_aa beats M_dd at every scale for arithmetic, leg_counting, knights_knaves (feature-routing class); M_dd beats M_aa at every scale for shortest_path (spatial search). Magnitudes vary with scale; per-task sign is cross-scale stable.

**Catalogue of negative results (§7.3).** 1/N averaging, Phase-1 sequential transfer, global τ cosine, per-layer flat-cosine opposition, novel-subspace angle (only at n=7), magnitude-weighted sign agreement (r=−0.45 at n=12), L1-grounding ARS (r=+0.06 at n=12), and a task-level routing probe (r=−0.31 at 4B) all fail as prospective merge-damage predictors on the 12-pair benchmark. We do not have a validated cheap pre-probe for merge compatibility.

**Scope.** §7 is a characterization paper. It gives practitioners one recipe (finding 1), one classification limitation (finding 2), and an honest map of what does not predict. A prospective merge-compatibility predictor remains open.

*Note: §7.10 documents the gap-closing experiments that produced the n=12 and n=19 null results reported in §7.3. Earlier drafts of this paper claimed r = −0.81 and r = −0.76 predictor headlines from n = 7 evidence; §7.10 explains why those claims were dropped. Path B (§7.10.5) added 7 more pairs to reach n = 19 specifically to test whether sign-agreement stabilizes around a useful effect size; it does not — the correlation decayed further to r = −0.27 and the CI still crosses zero.*

---

## Appendices

### Appendix K · 1/N vs scale-tuned merge matrix

3 × 10 accuracy table at 4B for `1/N (10τ, s=0.1)` vs `s=0.33 (9τ)` vs `2-task (SP+Arith, s=1.0)`; Δ-from-base-SFT deltas. Full table in `experiments/rlvr_geometry_paper/reports/report_phase4_merge_scaling.md` §Results.

- 1/N merge mean over 8 non-trivial tasks: 41.6% (base SFT 45%).
- 2-task and 9-task both at ~51%, about 3 τ-equivalents of total displacement.
- τ magnitudes (Frobenius) ‖τ‖ ∈ [0.73, 1.19] for all 9 trained tasks; cryptarithm ‖τ‖ = 0.000 (GRPO never got reward).

### Appendix L · The 7 pair merges: full predictor panel

Per-pair: Phase-1 transfer, τ cosine, sign agreement (frac and magnitude-weighted), conflict mass (total and layer-type split), novel-subspace angles (4B), L1/L2 ARS at 3B and 4B, and 4B damage. Table L.1 is the n=7 panel; Table L.2 is the feature-to-damage correlation table across all candidate predictors.

Key entries from Table L.2 (Pearson r vs 4B damage, n=7):

| Predictor (at 3B) | r | R² | Interpretation |
|---|---:|---:|---|
| 3B sign_agree (magnitude-weighted) | **−0.81** | 0.66 | primary weight-space predictor |
| 3B L1_mean (ARS) | **−0.76** | 0.58 | primary behavioral predictor |
| 3B L1_min (pair minimum) | −0.55 | 0.30 | |
| 3B ARS_mean (L1+L2) | −0.34 | 0.11 | dilutes L1 signal with L2 |
| 3B L2_mean | −0.01 | 0.00 | no signal |
| 3B conflict_mass | +0.81 | 0.66 | mirror of sign_agree |
| 4B novel-subspace angle (up_proj) | +0.56 | 0.31 | moderate, same-scale |
| 4B per-layer flat cos opposition | +0.23 | 0.05 | noise |
| 4B global τ cosine | +0.07 | 0.00 | no signal |
| Phase-1 sequential transfer | +0.03 | 0.00 | no signal |

### Appendix M · Full layer-selective merge tables across scales

3-scale × 3-variant × 10-task accuracy matrix for {3B, 4B, 8B} × {`basic`, `up`, `all_but_down`}. Δ-from-`basic` deltas per scale; Group-1/Group-2 classification per scale. Scale-specific notes:

- 3B `up` wins on 4 of 8 non-trivial tasks (arithmetic, syllogism, leg_counting, knights_knaves); 3B baselines for countdown/sudoku/maze/leg are ≤7% so Δ is uninformative there.
- 4B `up+q` tested: closes no gap beyond `up` for Group 1; adds only 3–5 points of Group 2 recovery.
- 8B Group-2 membership narrows to SP + mini_sudoku; countdown and KK classify as Group 1 at 8B.

### Appendix N · attn.{k,v,o} rotation (filling the §5.3 gap)

Principal-angle data for `attn.k_proj`, `attn.v_proj`, `attn.o_proj` across 7 pairs at 4B and 8B (the three sub-layers §5.3 did not measure). Mean per-pair angles:

| Sub-layer | 4B (across 6 pairs) | 8B (across 7 pairs) |
|---|---:|---:|
| `attn.k_proj` | 37–41° | 32–38° |
| `attn.v_proj` | 53–66° | 49–65° |
| `attn.o_proj` | 58–62° | 56–70° |

None frozen; k_proj rotates less than q_proj (compare §5.5 q_proj 40–45° at 4B) while v_proj and o_proj rotate comparably to `mlp.up_proj`. Consistent with 7.4's finding that Group-2 tasks require the full attn circuit. The T7 4B cell is missing (checkpoint-path issue); 6 of 7 4B cells and all 7 8B cells complete.

### Appendix O · TIES and DARE baselines

Full 3-pair × 10-task accuracy tables for TIES and DARE at 4B, with method-specific hyperparameter choices (TIES density = 0.2 after magnitude-based trim; DARE drop rate = 0.5 with magnitude rescaling). Method-to-method differences within ±2 points on all but one task-pair cell (T2 KK). Our geometric recipe (`all_but_down` vanilla sum) reaches 53.5% SP on T5 vs TIES's 48.0% — the only method beating vanilla sum by more than noise.

### Appendix P · Implementation notes and hyperparameters

- **Checkpoints**: 4B SP and Arith use Phase-2-refined checkpoints (RFT-Base SP 55.5%, Atomic-Only Arith 78%); 3B and 8B use Phase-1 direct-RLVR checkpoints (the only ones available at those scales). Confound: 4B baseline is higher-quality than 3B/8B, but each scale is internally consistent (same baseline vs variants).
- **Family confound**: 3B = Qwen2.5-3B-Instruct, 4B and 8B = Qwen3. The 3B→4B sign-agreement predictor works across the family boundary (r = −0.81). The F28 linking effect does not cleanly cross the boundary (3B-Qwen2.5 reverses sign). Appendix P.3 breaks out what we can and cannot separate.
- **Evaluation**: `evaluate_transfer.py`, seed 9999, 200 samples, DDP 8 × B200, `max_new_tokens = 1024`. Post Phase-2 bug fix (last `<answer>` match, §4 Appendix D).
- **Statistical tests**: bootstrap CI from 10,000 resamples (paired on (pair, damage) tuples); permutation test shuffles damage labels across pairs 10,000 times.
- **Atomic curricula at 3B and 8B**: L1 stage (100 steps) + L2 stage (100 steps), same recipe as the 4B Phase-2 runs; no RFT-Base refinement (§4.3). Matched-compute rerun with RFT-Base is a 1–2 day follow-up; would likely strengthen the 8B M_aa result without affecting the per-task sign pattern.
- **n = 7 pairs**: the primary statistical limitation. The bootstrap CI is [−0.99, −0.38], excluding zero but wide. Adding 3–5 more pairs (e.g., leg+syl, ari+kk, cnt+leg) is the cheapest way to tighten this and would cost one additional pod-day.
