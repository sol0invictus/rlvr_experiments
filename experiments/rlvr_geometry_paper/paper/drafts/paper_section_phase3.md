# Section 5 · The Weight-Space Geometry of GRPO

*Paper-ready draft. Figures are called out in italics; appendices F–J hold per-layer tables, a glossary of spectral/rotation metrics, cross-scale details, and implementation notes.*

---

## 5.1 Motivation

If §4 established that grounding predicts whether RLVR *succeeds*, this section asks *what RLVR does to the model's weights when it does succeed.* The answer shapes everything downstream: task-vector merging (§7), interference patterns, and the design space for parameter-efficient fine-tuning. We measure τ = W_GRPO − W_SFT for every (scale, task) pair at 3B, 4B, 7B, 8B and show that GRPO has a **structural signature** that is reproducible across scales and tasks — while *where* in the model it writes is task-specific.

## 5.2 Setup and Geometric Primer

For each of four model scales (Qwen2.5-3B, Qwen3-4B, Qwen2.5-7B, Qwen3-8B) and ten reasoning-gym tasks, we computed per-layer **task vectors**
$$\tau = W_{\mathrm{GRPO}} - W_{\mathrm{SFT}},$$
full-SVDs of W_SFT and W_GRPO, subspace projections, and principal angles between singular-vector bases. All checkpoints used lr = 5e-6, 300 GRPO steps, DAPO loss; decompositions computed in float32. Appendix F contains the formal definitions; this subsection gives the geometric intuition behind the four objects the rest of §5 uses.

**What τ is.** Start with W_SFT. Train GRPO on task T. What GRPO changed is captured in τ_T = W_GRPO(T) − W_SFT — a tensor with the same shape as W, representing only the learned delta. All of §5 studies τ.

**What W's "top-k subspace" is.** Any weight matrix W has a built-in ranking of preferred directions, given by SVD: W = U Σ V^T, where the columns of U are output directions, the columns of V are input directions, and σ_1 > σ_2 > … are singular values that tell you how much W amplifies each direction. The "top-k left subspace" is just the first k columns of U — the k output directions W uses most heavily. In a 4096-dim layer, k = 100 means the 100 preferred directions out of 4096; the other 3996 are directions W technically knows about but barely uses.

**What "τ projected onto W's top-k subspace" means.** τ is a matrix that points *somewhere* in weight space. You can ask what fraction of τ's length falls inside W's top-k directions vs. outside them. The two-sided projection ratio ρ_k = ‖U_k U_k^T τ V_k V_k^T‖² / ‖τ‖² quantifies this. If τ lives entirely in the top-k directions, ρ_k ≈ 1; if τ is perpendicular to them, ρ_k ≈ 0; if τ points in a random direction, ρ_k ≈ (k/r)² by chance, where r is the ambient dimension.

**What a "principal angle" is.** Suppose we have two k-dimensional subspaces — the top-k left subspace of W_SFT (call it A) and the top-k left subspace of W_GRPO (call it B). Both are k-dim slices of the full output space. "How different are A and B?" is answered by k principal angles: we find the direction in A most aligned with B (angle θ_1), then the remaining direction in A most aligned with B (θ_2), and so on. If all k angles are 0°, A and B coincide — the basis is frozen. If all are 90°, A and B are perpendicular — completely different basis. A mean principal angle of 2° means the filing cabinet of preferred directions has rotated 2°: nearly the same folders, slightly tilted.

**What a "novel subspace" is.** When task A rotates the basis, it introduces some directions that weren't in W_SFT's top-k before. novel_A = (I − U_SFT U_SFT^T) U_GRPO(A)[:, :k] projects away the part that was already in SFT, leaving only the new directions. novel_A is task A's **signature** — "here is where A moved the model that SFT wasn't." The pairwise principal angle between novel_A and novel_B measures *whether two tasks moved the model in the same direction or different directions*, distinct from how far each moved.

**Short names used in the rest of §5**: ‖τ‖/‖W‖ (relative Frobenius magnitude per layer); ρ_k (two-sided projection ratio); principal angle (rotation of U_SFT → U_GRPO, top-k); novel-subspace angle (pairwise between novel_A and novel_B); depth signature (per-task distribution of change mass ‖τ‖_F across layer indices, normalized).

## 5.3 Result 1: The Spectrum of W Does Not Move

Across all scales and tasks, GRPO leaves W's singular values statistically unchanged.

![Figure 5.1](figures/fig5_1_spectrum_preserved.pdf)

*Figure 5.1: (a) Spectral-norm ratio σ₁(W_GRPO)/σ₁(W_SFT) per task at 8B. Every task sits at the reference line 1.0 (±0.002). (b) Per-band relative singular-value shift |σᵢᴳᴿᴾᴼ − σᵢˢᶠᵀ| / σᵢˢᶠᵀ across 12 probe layers × 5 tasks. Every band is an order of magnitude below the 0.1% reference.*

Effective rank is unchanged (Δ ≈ 0, σ(Δ) = 0.003). **GRPO does not restructure W; its entire action has to live in a direction the singular-value spectrum cannot see.**

## 5.4 Result 2: τ is Nearly Orthogonal to W's Top Subspace

If W's spectrum is unchanged, where does the learned update τ live? τ is a matrix pointing in *some* direction in weight space. In principle it could point along W's top singular directions (amplifying the skills the model already uses heavily), or in orthogonal directions (adding new features in directions W was barely using before), or a mix. The projection ratio ρ_k distinguishes these cases: ρ_k ≈ 1 means τ lives entirely in W's top-k directions; ρ_k ≈ 0 means τ is perpendicular to them; for a random τ direction we expect ρ_k ≈ (k/r)² by chance.

![Figure 5.2](figures/fig5_2_tau_orthogonal.pdf)

*Figure 5.2: Two-sided projection ratio ρ_k of τ onto W_SFT's top-k singular subspace, for k ∈ {10, 100, 500, 1000} at four scales. The dashed line is the random-isotropic baseline (k/r)² at r = 2560 (4B ambient rank) for reference. At 4B and 8B ρ_k sits at or below baseline — τ actively avoids the directions W was already using.*

**Table 2** summarizes the same data numerically.

| k | 3B | 4B | 7B | 8B | Random (k/r)² |
|---|---:|---:|---:|---:|---:|
| 10 | 0.03% | 0.01% | 0.001% | 0.01% | 0.001% |
| 100 | 0.34% | 0.16% | 0.04% | 0.14% | 0.06% |
| 500 | 2.9% | 1.6% | 0.7% | 1.0% | 1.5% |

At 4B and 8B, τ's projection onto W's top-100 singular directions is at or below the random-isotropic baseline. At 7B it is *below* random. The alignment at 3B (ρ_100 = 0.34% vs 0.06% random) shrinks monotonically with scale. **At production scales, GRPO's update is geometrically orthogonal to W's dominant representations — it does not amplify the features W was already emphasizing; it adds small new features in directions W was underusing.**

This result has a direct implication for the merging section (§7). If GRPO's updates lived in W's top-k subspace, two task vectors τ_A and τ_B would compete for the same high-importance directions, and summing them would inevitably corrupt each task's dominant representations. Because τ instead occupies W's underused subspace, τ_A and τ_B from different tasks mostly occupy different underused corners and can coexist under simple addition. Orthogonality-to-W is a precondition for mergeability.

## 5.5 Result 3: A Universal Layer-Type Split

§5.4 told us τ doesn't amplify W's top directions. But τ might still *rotate* them — a fixed-spectrum matrix can have its basis (the columns of U) turn slightly to face new directions without any singular value changing. We measure this rotation with **principal angles** between the top-k left singular subspace of W_SFT and of W_GRPO: 0° means "same basis, unchanged"; 90° means "completely different basis"; values in between mean "the filing cabinet of preferred directions has been tilted by this many degrees." Aggregating across layers would mask the key structure — different layer types behave very differently.

![Figure 5.3](figures/fig5_3_layer_type_rotation.pdf)

*Figure 5.3: Principal angle at k = 100 (mean over tasks × layers of that type) by layer type and scale. `mlp.down_proj` (gray) is at 0°–0.02° at every scale. `mlp.up_proj` (blue) and `attn.q_proj` (orange) both rotate and grow monotonically from 3B (1.47° / 1.65°) to 8B (2.67° / 2.07°). The layer-type split is the structural signature of GRPO's action.*

**Table 3. Basis rotation at k = 100 (mean over tasks × layers of that type).** Every entry rounded to 0.01°.

| Layer type | 3B | 4B | 7B | 8B |
|---|---:|---:|---:|---:|
| `mlp.down_proj` | **0.01°** | **0.02°** | **0.00°** | **0.00°** |
| `mlp.up_proj` | 1.47° | 1.99° | 2.49° | 2.67° |
| `attn.q_proj` | 1.65° | 1.81° | 2.03° | 2.07° |

`mlp.down_proj` bases are **frozen** — at every scale, across two model families (Qwen2.5 at 3B/7B, Qwen3 at 4B/8B), GRPO rotates its left singular basis by effectively zero. All rotation action is in `mlp.up_proj` and `attn.q_proj`, where it grows monotonically with scale.

The distinction is mechanistically clean. `mlp.up_proj` and `attn.q_proj` are **feature projectors** — they decide *what features to produce* (up) or *what to query for* (q); these are the natural sites for task-specific feature learning, and they rotate by a tiny but measurable amount. `mlp.down_proj` is a **sink**: it aggregates contributions from all MLP neurons back into the residual stream. Its preferred output directions are the ones the whole transformer uses most heavily across any input — shifting them would break downstream computation. So GRPO leaves this basis alone.

Note the crucial distinction: `mlp.down_proj`'s basis does not rotate (the filing cabinet stays in place), but τ still has nonzero mass in down layers — GRPO moves *coordinates within the existing basis* rather than changing the basis itself. In feature-projector layers, GRPO does both: it moves coordinates and tilts the basis by ~2°. Even a 2° tilt at k = 100 encodes meaningful task-specific structure, because a 2° rotation of a 100-dimensional subspace is a large amount of structured information.

This is the finding the merging section exploits: because `mlp.down_proj`'s basis is frozen, dropping its τ contributions from a merge cannot destroy task-distinctive structure that lives only there — because no such structure exists. §7 verifies empirically (F20) that the `all_but_down` recipe matches or beats the full merge on 22/24 scale × task cells.

## 5.6 Result 4: GRPO's Rotation Amount is Task-Invariant; Its Direction is Task-Specific

§5.5 measured *how much* each feature-projector rotates. The natural next question is whether that rotation amount differs across tasks. It does not.

**The surprising invariance.** Across arithmetic, syllogism, caesar_cipher, countdown, knights_knaves, and every other task we tested, at any fixed scale and layer type, the per-task rotation magnitude varies by less than 0.1° (Appendix G). At 4B in `mlp.up_proj`: arithmetic rotates 1.99°, caesar 2.00°, syllogism 1.98°. You would expect arithmetic and logic to reshape the model in visibly different amounts — they do not. Every task spends the same "rotation budget," which depends only on model scale and layer type.

**An analogy.** Imagine being told to take exactly 10 steps from a starting point; someone else is told the same. You both end up 10 steps from start. But you went north and they went east: same distance from origin, completely different destinations. The principal angle from SFT (§5.5) measures "how far each task moved" — the 10 steps. If the amount is identical for every task, all of the task-specificity has to live in *which direction they walked*.

**How we measure direction.** For each task A, define the *novel subspace* as the component of U_GRPO(A) that's orthogonal to U_SFT:
$$\mathrm{novel}_A \;=\; (I - U_{\mathrm{SFT}} U_{\mathrm{SFT}}^T)\,U_{\mathrm{GRPO}}(A)[:, :k].$$
This is "the new directions task A introduced" — A's signature in weight space, stripped of anything that was already present in the SFT basis. If two tasks rotated the basis in identical directions, their novel subspaces would coincide and the pairwise principal angle between novel_A and novel_B would be 0°. If they rotated in completely perpendicular directions, that angle would be 90°. Intermediate values quantify partial agreement.

![Figure 5.4](figures/fig5_4_novel_subspace_angles.pdf)

*Figure 5.4: Pairwise principal angle between novel subspaces at 8B, k = 100, for four tasks (countdown, basic_arithmetic, caesar_cipher, syllogism). One heatmap per layer type; rows = probe layers, columns = task pairs. `mlp.down_proj` (left, mean 24°) is the least task-divergent; `attn.q_proj` (right, mean 43°) is the most.*

**Table 4. Pairwise angle between novel subspaces at 8B, k = 100** (mean over 4 tasks × 12 probe layers).

| Layer type | Mean pairwise angle | Interpretation |
|---|---:|---|
| `mlp.down_proj` | 24° | Small novel subspace, directions consistent across tasks |
| `mlp.up_proj` | 39° | Tasks rotate into genuinely distinct feature subspaces |
| `attn.q_proj` | 43° | Most task-adversarial layer — queries diverge most across tasks |

At 43° pairwise, `attn.q_proj` is where different tasks make *the most genuinely different decisions* about what the model should be paying attention to. `mlp.down_proj`, by contrast, has the smallest pairwise angles, and is also where the rotation from SFT is zero (§5.5) — whatever little novel subspace exists there is similar across tasks. It is the least task-specific layer type.

**A subtlety about orientation.** A subspace has no inherent orientation (the line {t · v : t ∈ ℝ} is the same subspace as {t · (−v) : t ∈ ℝ}). Two tasks could want to rotate in exactly opposite directions along the same axis, and the principal-angle-between-subspaces measurement would report 0° — the subspaces coincide, even though the tasks are pulling in opposite directions. To capture direction-within-a-subspace we also compute a **flat cosine**: vectorize novel_A and novel_B and take their cosine. This preserves sign:

- Flat cosine **+0.89** at some layer → tasks A and B rotate in the same direction *with the same orientation* (perfectly aligned update).
- Flat cosine **−0.90** → tasks want to pull the basis in *opposite directions* along the same axis. Summing τ_A + τ_B at this layer actively cancels both tasks' learning.

Per-layer flat cosines at 8B span the full range +0.89 to −0.90 across task-pair × layer combinations. The microscopic signature of catastrophic interference is a strongly negative flat cosine at a feature-projector layer where both tasks concentrate their change mass — that combination is where τ_A and τ_B fight each other head-to-head when merged.

**Taken together (Results 2–4)**: GRPO's action has a universal shape — a small (‖τ‖/‖W‖ ≈ 0.06–0.12%), near-orthogonal-to-W, layer-type-localized basis rotation concentrated in feature projectors. The *amount* is governed by scale and layer type. The *direction* is where the task specializes. The pairwise geometry of task-specific rotation directions (subspace angles and flat cosines) is the mechanistic substrate for the task-vector merging in §7: pairs whose novel subspaces are near-perpendicular (large pairwise angle) compose cleanly; pairs with strongly negative flat cosines at shared-mass layers are where catastrophic merge damage originates.

## 5.7 Result 5: Depth Signatures Explain Transfer Failures That Global Similarity Cannot

**Global similarity fails.** Across the 4B Phase-1 transfer matrix, Pearson r between cos(τ_A, τ_B) and behavioral transfer Δ is only **0.32** (p = 0.004); Spearman is 0.15 (n.s.). The largest negative transfers in Phase 1 (mini_sudoku → knights_knaves at −50%, mini_sudoku → leg_counting at −52%) occur at τ-cosines ≈ 0. Whole-model cosine averages over too much of the weight space to be informative — it smears out a localized conflict.

**Depth signatures: what and why.** A transformer is a stack of N layers (N = 36 for 8B Qwen3). For each layer and each weight type, τ has a sub-tensor; summing Frobenius norms within a layer gives ‖τ‖_F^(i) — "how much GRPO changed layer i." The **depth signature** is the per-layer distribution of this change mass, normalized to sum to 1. It is a probability distribution over layer indices answering: *where along the depth did GRPO concentrate its learned weight change?*

To present it compactly we bucket the 36 layers into four quartiles — **early** (L0–8), **mid-early** (L9–17), **mid-late** (L18–26), **late** (L27–35) — and ask what fraction of the **top-20 most-changed layers'** total change mass falls into each bucket. A value of 88% in "early" means 88% of the top-20 layers' change lived in layers 0–8.

**What depths correspond to, mechanistically.** A transformer is, roughly, a hierarchical feature extractor. Early layers (L0–8) handle token-level surface features (is this token a letter, a digit, a newline?). Mid-early layers (L9–17) build local relational and logical features (subject-verb structure, grid-cell adjacency, single-hop inferences). Mid-late layers (L18–26) run multi-step computation and global search (evaluate arithmetic expressions, propagate constraints, plan paths). Late layers (L27–35) format and emit the output. A task concentrates GRPO change at the depth where its bottleneck sits.

![Figure 5.5](figures/fig5_5_depth_signatures_8b.pdf)

*Figure 5.5: 8B depth-quartile concentration per task — % of top-20 layer change mass in each of four depth bands. Three clean clusters emerge (black horizontal separators): early-only (caesar_cipher), mid-early (logic / local-reasoning), mid-late (arithmetic / global search).*

**Table 5. 8B depth-quartile concentration (% of top-20 layer change mass).**

| Task | early (L0–8) | mid-early (L9–17) | mid-late (L18–26) | late (L27–35) |
|---|---:|---:|---:|---:|
| caesar_cipher | **88** | 12 | 0 | 0 |
| syllogism | 41 | **55** | 5 | 0 |
| knights_knaves | 35 | **52** | 14 | 0 |
| leg_counting | 40 | **56** | 5 | 0 |
| shortest_path | 28 | **62** | 9 | 0 |
| basic_arithmetic | 35 | 19 | **47** | 0 |
| countdown | 29 | 19 | **53** | 0 |
| mini_sudoku | 46 | 10 | **39** | 5 |

Three clean clusters emerge at 8B, and each cluster matches what the task actually requires:

- **caesar_cipher at 88% early.** Caesar is a per-token substitution ("A ↦ D, B ↦ E, …"). The bottleneck is token-level parsing and rewrite — no multi-step reasoning needed. 88% in layers 0–8 is exactly where a token-substitution rule should be learned.
- **syllogism, knights_knaves, leg_counting, shortest_path at 52–62% mid-early.** These are local-reasoning tasks — the model has to form relations ("A is a knight so A tells the truth"; "cell (2,3) is a wall") and chain single-hop inferences. That kind of local relational reasoning happens in mid-early layers.
- **basic_arithmetic, countdown, mini_sudoku at 39–53% mid-late.** These are multi-step computation / search tasks. Arithmetic needs order-of-operations across a long expression; countdown searches combinations; sudoku propagates constraints. Those computations live in mid-late layers.

**Why depth signatures predict Phase-1 transfer better than global cosine.** Consider the worst Phase-1 pair: mini_sudoku → leg_counting, transfer = −52%. Whole-model cos(τ_mini_sudoku, τ_leg_counting) ≈ 0 — globally "unrelated" τ's. Yet mini_sudoku concentrates 39% of its mass in mid-late layers and 46% in early layers, while leg_counting concentrates 56% in mid-early and 40% in early. The two tasks share mass at early and adjacent mid bands, and mini_sudoku's mid-late rewrites sit *downstream* of the mid-early pathway leg_counting depends on — corrupting the downstream pathway breaks the task even though the direct modifications are at different depths. The depth-profile cosine between mini_sudoku and leg_counting is only 0.33; per-layer novel-subspace flat cosines at the shared-mass layers are strongly negative. Depth-profile similarity is a stronger interference predictor than whole-model τ cosine because it captures localized conflict where the global cosine averages it away.

## 5.8 Result 6: Depth Signatures Reorganize Across Scale, Layer-Type Split Does Not

Where a task concentrates its edits is scale-dependent (Figure 5.6).

![Figure 5.6](figures/fig5_6_scale_depth_migration.pdf)

*Figure 5.6: basic_arithmetic change-mass distribution across depth bands at 3B, 4B, 8B. At 3B: 87% early; at 4B: majority mid-early; at 8B: majority mid-late. The task migrates toward deeper layers as scale grows.*

`shortest_path` shows a similar migration: early-dominated at 4B (86%), late-dominated at 7B (60%), mid-early at 8B (62%). Cluster structure itself reorganizes: 3B has one early super-cluster; 8B separates into three clean bands; 4B is intermediate. Part of this is also a model-family effect — 3B and 7B are Qwen2.5 while 4B and 8B are Qwen3 — and the two confounds are not fully separable with our checkpoints (Appendix H).

What does *not* reorganize is the layer-type invariant: `mlp.down_proj` basis-frozen, `up_proj`/`q_proj` rotating with scale, τ near-orthogonal to W's top subspace. The universal and the task-specific are cleanly decomposed into *how* GRPO edits (universal) vs *where* it edits (task- and scale-specific).

## 5.9 Implications for the Merging Section

Three findings from §5 carry forward to §7 as either actionable recipes or characterization context. A fourth, originally claimed as the microscopic mechanism behind a strong cross-scale predictor, did not survive sample expansion (§7.10).

1. **`mlp.down_proj`'s basis is frozen (§5.5)** → dropping its τ contributions from a merge is safe. §7 F20 / F1 verifies empirically: `all_but_down` matches or beats the full merge on 40 of 48 (pair, task) cells at 4B across 7 pairs (originally 22/24 on a single pair). This is the most actionable §5→§7 link and the headline practitioner recipe of §7.
2. **Feature projectors are the task-specific sites (§5.5)** → a layer-selective recipe that keeps only `mlp.up_proj + mlp.gate_proj` (18% of keys) succeeds for *some* pair merges. The §7.4 / §7.5 finding revised by §7.10.1 is that this property is **pair-dependent, not task-intrinsic** — a task can be in the "MLP-only-sufficient" group with one partner and require the full attention circuit with another. §5 supplies the geometric mechanism (where rotation lives); §7 supplies the empirical scope (which pairs it suffices for).
3. **Depth-signature clusters (§5.7) cluster tasks but do not predict merge compatibility.** The mid-early cluster (syllogism, knights_knaves, leg_counting, shortest_path) and mid-late cluster (arithmetic, countdown, mini_sudoku) describe where each task concentrates change mass, but at n = 19 pair merges no aggregation of depth-profile overlap predicts merge damage with |r| > 0.5. The original hypothesis that "tasks that rewrite similar depths can share merge recipes" is consistent with the data but does not yield a prospective predictor.
4. **Per-layer flat-cosine opposition (§5.6) was originally proposed as the microscopic mechanism behind a strong sign-agreement predictor (§7 F29, r = −0.81 at n = 7).** The predictor degraded to r = −0.45 at n = 12 and to r = −0.27 at n = 19; the bootstrap CI at n = 19 still includes zero. The flat-cosine and sign-agreement quantities remain *weak directional indicators* (correct sign at every n) but are not calibrated forecasters. §5.6's localization story stands as a characterization of GRPO's geometry; the predictor application of that story does not.

Additionally, **depth signatures reorganize across scales (§5.8)** — which is consistent with §7.6 / F28 (atomic-trained τ's compose cleanly with other atomic τ's on feature-routing tasks at every scale; direct-trained τ's preserve attention restructuring needed for shortest_path at every scale). The mean magnitude of the atomic-vs-direct effect varies with scale (+7.9 at 4B, +0.3 at 8B, −3.3 at 3B-Qwen2.5); the per-task sign pattern is cross-scale stable. This is the only surviving cross-phase mechanistic link between §5 and §7.

## 5.10 Summary of Section 5

1. GRPO leaves every singular value of W unchanged (shifts < 0.1%) and preserves effective rank; its action lives outside the SVD of W.
2. τ is near-orthogonal to W's top singular subspace at 4B and 8B, and the orthogonality grows with scale.
3. `mlp.down_proj`'s left singular basis is frozen at every scale and both model families tested (0°–0.02°). All basis rotation is in `mlp.up_proj` and `attn.q_proj`, and it grows monotonically with scale (1.5° → 2.7°).
4. Different tasks rotate by statistically identical amounts but in task-specific directions. Pairwise angles between tasks' novel subspaces reach 43° at `attn.q_proj`, with per-layer flat cosines spanning ±0.9.
5. Depth signatures predict transfer better than global τ cosine and cluster tasks into early/mid/late bands at 8B; the clustering reorganizes with scale.
6. The *how* of GRPO (rotation-in-feature-projectors, sink-frozen, orthogonal-to-W) is universal; the *where* (which depths, which directions) is task-specific. §7 will use this split as the mechanistic basis for a layer-selective merging recipe.

---

## Appendices

### Appendix F · Glossary of spectral and rotation metrics

Formal definitions: Frobenius norm, cosine similarity, spectral norm, effective rank, SVD, subspace projection (two-sided, left-only, right-only), principal angles, novel subspace, flat rotation cosine, depth signature. Random-baseline expressions for isotropic τ. Layer-type role table (feature projectors vs sinks). Complete version at `experiments/rlvr_geometry_paper/reports/report_phase3_task_distances.md` §A.

### Appendix G · Per-task rotation amount and ‖τ‖_F tables

Full per-task tables at all four scales for (a) basis rotation at k=100 per layer type (extending Table 3), (b) ‖τ‖_F and ‖τ‖/‖W‖ per task, (c) confirmation that task-to-task variation in rotation *amount* is under 0.1° at each scale.

### Appendix H · Cross-scale depth signatures and the Qwen2.5/Qwen3 confound

Full 4-scale × 10-task depth-quartile table (extending Table 5). Commentary on which scale effects are family effects: `basic_arithmetic` stays in early/mid-early layers in Qwen2.5 but migrates to mid-late in Qwen3-8B; 7B has the highest relative ‖τ‖/‖W‖ (0.117% vs 0.06%) and the lowest projection onto W's top-k. Rotation-growth trend (3B → 8B) is consistent across families.

### Appendix I · Full novel-subspace data

Per-layer per-task-pair tables of (a) principal angles between novel subspaces, (b) flat cosines. Across 4 tasks × 12 probe layers at 8B for the analysis in §5.6. Extended to 4B (all 7 Phase-4 pair definitions) and 8B (`attn.{k, v, o}` added in Part 5, F31) in the full report.

### Appendix J · Eigenvalue-shift control and implementation notes

- Per-band eigenvalue shift table (Table 3.2 of the Phase-3 report): 12 layers × 5 tasks, every band's mean and max shift < 0.1%.
- Float32 SVD precision notes.
- Correct checkpoint paths (SFT bases, GRPO prefixes). Known issues: 4B countdown uses `phase2_countdown/direct_rlvr` (the earlier `grpo_qwen_4b_instruct_countdown_4to5` was trained from a different SFT base and produces ‖τ‖ = 151 outlier); 8B maze v1 has generation logs only, v2 has weights.
- Limitations: countdown re-run at 4B pending; 5-task novel-subspace coverage (full 10-task in progress); Qwen2.5/Qwen3 family confound at 3B/7B unresolved without a matched-family control.
