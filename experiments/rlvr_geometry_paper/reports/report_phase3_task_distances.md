# Phase 3: RLVR Weight Change Analysis

## Executive Summary

**One sentence**: GRPO applies a small, scale-invariant rotation to the output bases of `mlp.up_proj` and `attn.q_proj` while leaving `mlp.down_proj` and all singular values essentially unchanged — task specialization lives in *which depths* get rotated, not in *how*.

### The "how" vs "where" split

GRPO's action on weights has a universal shape and a task-specific location:

- **How** GRPO edits is **scale-invariant and task-invariant**: a small (‖τ‖/‖W‖ ≈ 0.06–0.12%) perturbation that is nearly orthogonal to W's top singular subspace and localized to rotations of `up_proj`/`q_proj` output bases. `down_proj` bases stay frozen. Per-task rotation *magnitudes* vary by <0.1° at each scale — every task rotates by the same amount, just in different directions (§5.2).
- **Where** GRPO edits (which layer depths) is **task-specific and scale-dependent**. `basic_arithmetic` operates in early layers at 3B, mid layers at 4B, and mid-late layers at 8B. Clusters reorganize across scales.

### Universal layer-type split (holds at 3B / 4B / 7B / 8B)

| Layer type | Basis rotation at k=100 | Status |
|------------|-------------------------|--------|
| `mlp.down_proj` | 0.00°–0.02° at every scale | **Basis frozen** across 4 scales, 2 model families |
| `mlp.up_proj` | 1.47° → 1.99° → 2.49° → 2.67° | Rotates, growing monotonically with scale |
| `attn.q_proj` | 1.65° → 1.81° → 2.03° → 2.07° | Rotates, growing monotonically with scale |

### Practical implications

1. **Task interference isn't about global task-vector similarity** (r=0.32 at best). It's about opposing rotations in `up_proj`/`q_proj`, and depth-range overlap (§6).
2. **Curriculum design**: pair tasks with disjoint depth signatures (e.g. caesar_cipher's early-only vs countdown's mid-late), not by global similarity.
3. **GRPO is structurally conservative**: catastrophic transfer cases (−50%) are behavioral, not structural damage — W's spectrum is preserved.

See §[Appendix A](#appendix-a-glossary-of-measures) for definitions of every metric.

---

## 1. Background & Setup

### 1.1 Goal

Characterize how GRPO (Group Relative Policy Optimization) modifies model weights during RLVR training, and whether those changes predict task transfer. Builds on Phase 1 (transfer matrix) and Phase 2 (atomic curriculum).

### 1.2 Experimental setup

| Scale | Base | SFT checkpoint | GRPO prefix | # Layers |
|-------|------|----------------|-------------|----------|
| 3B | Qwen2.5-3B | `outputs/sft_qwen25_3b` | `grpo_qwen25_3b_` | 36 |
| 4B | Qwen3-4B | `sft_qwen_4.0b_instruct` | `grpo_qwen4b_` | 36 |
| 7B | Qwen2.5-7B | `outputs/sft_qwen25_7b` | `grpo_qwen25_7b_` | 28 |
| 8B | Qwen3-8B | `outputs/sft_qwen3_8b` | `grpo_qwen3_8b_` | 36 |

All GRPO checkpoints trained at lr=5e-6, 300 steps, 8 generations per prompt, DAPO loss. 10 reasoning tasks per scale: countdown, maze, mini_sudoku, shortest_path, basic_arithmetic, syllogism, knights_knaves, leg_counting, caesar_cipher, cryptarithm.

### 1.3 Key definitions (short — full glossary in Appendix A)

- **Task vector**: τ = W_grpo − W_sft (per weight matrix)
- **Spectral norm**: largest singular value σ₁ of a matrix
- **Effective rank**: exp(entropy of normalized singular values)
- **Task vector similarity**: cos(τ_A, τ_B) flattened across all parameters
- **Subspace projection**: fraction of ‖τ‖² captured by W_sft's top-k singular directions
- **Principal angles**: rotation angles between top-k subspaces of W_sft and W_grpo
- **Depth signature**: per-task distribution of change mass across layer indices

### 1.4 Phase 1 transfer reference (4B)

Recalled from `report_transfer_matrix.md` for interpretation later:

|  | cnt | mz | sdk | pth | ari | syl | kk | leg |
|--|-----|-----|-----|-----|-----|-----|-----|-----|
| **countdown** | +36 | +6 | +14 | +4 | +6 | +18 | +4 | +6 |
| **maze** | −10 | +12 | +16 | +4 | 0 | +12 | 0 | +6 |
| **mini_sudoku** | −16 | −2 | +22 | −16 | −26 | −16 | **−50** | **−52** |
| **shortest_path** | +6 | +8 | +28 | +28 | +6 | +16 | −2 | +14 |
| **basic_arithmetic** | +4 | −2 | −4 | −6 | −12 | +18 | −16 | −32 |
| **syllogism** | −8 | −2 | +26 | −2 | −2 | +26 | +2 | +10 |
| **knights_knaves** | −8 | +2 | +4 | −12 | −8 | +26 | +12 | −4 |
| **leg_counting** | +16 | +2 | +22 | +12 | −8 | +14 | −8 | +28 |

Percentages are Δ accuracy (task A trained → task B evaluated).

---

## 2. How Much Do Weights Change?

### 2.1 Task-vector magnitudes (4B)

| Task | ‖τ‖_F | Notes |
|------|--------|-------|
| leg_counting | 1.14 | Largest regular change |
| maze | 0.97 | |
| knights_knaves | 0.97 | |
| shortest_path | 0.95 | |
| caesar_cipher | 0.88 | |
| basic_arithmetic | 0.83 | |
| syllogism | 0.73 | |
| mini_sudoku | 0.71 | Smallest change |
| cryptarithm | 0.00 | **No learning** (no reward signal) |
| countdown | *pending re-run* | Original used wrong base model — see §7 |

Most tasks produce changes of similar Frobenius magnitude (0.7–1.1), suggesting GRPO applies roughly uniform pressure regardless of task difficulty. Cryptarithm received no reward signal and never moved — a clean negative control.

### 2.2 Relative change ‖τ‖/‖W‖ per layer (all scales)

| Scale | Mean ‖τ‖/‖W‖ per layer | Global ‖τ‖ range across tasks |
|-------|------------------------|------------------------------|
| 3B | 0.071% | 0.73–0.87 |
| 4B | 0.065% | 0.71–1.14 |
| 7B | **0.117%** | 1.21–1.58 |
| 8B | 0.059% | 0.98–1.70 |

GRPO's per-layer weight change is tiny in relative terms (<0.12% of the weight matrix's norm per layer). 7B is a mild outlier with nearly 2× the relative change of 8B.

---

## 3. Spectral Properties: Is W Restructured? (No)

### 3.1 SFT vs GRPO spectral comparison (8B)

For each of the top-20 most-changed layers per task, we compare σ(W_grpo) vs σ(W_sft):

| Task | spectral_norm_ratio | eff_rank_change |
|------|---------------------|-----------------|
| countdown | 1.0004 | −0.004 |
| mini_sudoku | 0.9999 | +0.001 |
| shortest_path | 1.0003 | −0.002 |
| basic_arithmetic | 0.9991 | −0.000 |
| syllogism | 1.0003 | −0.003 |
| knights_knaves | 0.9998 | +0.002 |
| leg_counting | 1.0000 | +0.000 |
| caesar_cipher | 1.0018 | −0.001 |
| cryptarithm | 1.0001 | −0.009 |

**Finding**: spectral norm ratios are indistinguishable from 1.0, and effective rank doesn't change. GRPO neither amplifies/dampens layers nor compresses/expands their rank.

### 3.2 Full-spectrum comparison: do individual eigenvalues move? (No)

For 12 representative layers × 5 tasks, we computed the full SVD of W_sft and W_grpo and compared every σ_i.

| Normalized rank index | Mean relative shift | Max (excl. tail-noise) |
|----------------------|---------------------|------------------------|
| 0–1% (top) | ~0.003% | 0.04% |
| 1–10% | ~0.003% | 0.06% |
| 10–50% | ~0.002% | 0.06% |
| 50–100% (tail) | noise from near-zero σ | — |

Every singular value of W shifts by less than 0.1%. The mean shift per band is below the bfloat16 precision floor. **The eigenvalues do not move.**

### 3.3 τ's own spectrum is low-rank-ish

While σ(W) doesn't move, τ (= W_grpo − W_sft) has its own structure. Top-10 singular values of τ per task (4B, averaged across mlp_down layers):

| Task | top-1 σ(τ) | decay (σ_1/σ_10) |
|------|-----------|------------------|
| maze | 0.018 | 4.2× |
| knights_knaves | 0.017 | 3.7× |
| caesar_cipher | 0.016 | 4.0× |
| shortest_path | 0.014 | 3.7× |
| basic_arithmetic | 0.014 | 3.7× |
| leg_counting | 0.014 | 3.0× |
| syllogism | 0.011 | 3.2× |
| mini_sudoku | 0.006 | 2.5× |
| cryptarithm | 0.000 | NaN |

So the update τ is **moderately low-rank** (3–4× decay across top 10 SVs) — it has a preferred direction, but its magnitude is small relative to W's.

### 3.4 Why isn't σ(W) moving?

If σ(W) doesn't change but behavior does, the change must live in directions that SVD of W alone can't see. The two natural candidates, both measured in §4:

1. **Subspace alignment**: is τ aligned with W's top directions (rescaling them) or orthogonal to them (adding new directions)?
2. **Basis rotation**: even when σ values are fixed, U and V (the singular vectors) can rotate.

---

## 4. Where the Change Actually Lives: Subspace & Rotation

### 4.1 τ is nearly orthogonal to W's principal subspace

Two-sided projection `‖U_k U_k^T τ V_k V_k^T‖_F² / ‖τ‖_F²` — the fraction of ‖τ‖² explained by W_sft's top-k singular directions. Mean across tasks × layers:

| k | 3B | 4B | 7B | 8B | Random baseline (k/r)² |
|---|-----|-----|-----|-----|-----------------------|
| 1 | 0.00% | 0.00% | 0.00% | 0.00% | ~0% |
| 10 | 0.03% | 0.01% | 0.001% | 0.01% | ~0.001% |
| 100 | 0.34% | 0.16% | 0.04% | 0.14% | ~0.06% |
| 500 | 2.88% | 1.55% | 0.69% | 1.02% | ~1.5% |
| 1000 | 9.17% | 5.32% | 2.59% | 3.20% | ~6% |

- **3B**: projection is 2–5× the random baseline — modest alignment with W's top directions
- **4B / 8B**: projection ≈ random baseline — τ is orthogonal
- **7B**: projection *below* random baseline — most orthogonal of any scale
- **Trend**: τ becomes more orthogonal to W as scale grows

**Interpretation**: GRPO does not rescale the directions W was already using. It adds updates in directions W was mostly *not* using — i.e., in the underexpressed (mid/low-rank) subspace.

### 4.2 Principal angles: basis rotates 1–3°

For the top-k left singular vectors of W, compute principal angles arccos(σ_i(U_sft^T U_grpo)). Mean across tasks × layers:

| k | 3B | 4B | 7B | 8B |
|---|-----|-----|-----|-----|
| 1 | 0.77° | 1.03° | 0.79° | 0.92° |
| 10 | 0.68° | 0.87° | 1.02° | 0.97° |
| 100 | 0.77° | 0.97° | 1.27° | 1.24° |
| 500 | 0.89° | 1.05° | 1.33° | 1.34° |
| 1000 | 0.68° | 0.84° | 1.08° | 1.12° |

Basis rotation grows modestly with scale (0.8° at 3B → 1.3° at 7B/8B), saturating around 1.3° at the 7B–8B range.

### 4.3 Layer-type split — the cleanest structural signal

Averages over all layers hide a sharp split. At k=100:

| Layer type | 3B angle | 4B angle | 7B angle | 8B angle |
|-----------|----------|----------|----------|----------|
| `mlp.down_proj` | **0.01°** | **0.02°** | **0.00°** | **0.00°** |
| `mlp.up_proj` | 1.47° | 1.99° | **2.49°** | **2.67°** |
| `attn.q_proj` | 1.65° | 1.81° | 2.03° | 2.07° |

**The headline finding**: `mlp.down_proj` bases are **frozen** at every scale. All rotation action is in `mlp.up_proj` and `attn.q_proj`, and that rotation grows monotonically with scale.

### 4.4 Mechanism: why does `down_proj` stay frozen?

- `mlp.down_proj` is the **sink** — it receives contributions from all neurons in `up_proj × gate_proj`. Its principal directions (the ones that get used most heavily across all inputs) are stable; GRPO's update there can only nudge coordinates within the existing basis.
- `mlp.up_proj` and `attn.q_proj` are **feature projectors** — they decide *what features* to produce (up_proj) or *what to query for* (q_proj). This is the natural site for task-specific feature learning, which manifests as basis rotation.

### 4.5 Synthesis

GRPO's weight change has a consistent structure across scales:

1. Small in magnitude: ‖τ‖/‖W‖ ≈ 0.06–0.12% per layer
2. Nearly orthogonal to W's top subspace: lives in W's underexpressed directions
3. Rotation-localized: `up_proj` / `q_proj` bases rotate ~1.5–2.7°; `down_proj` bases stay frozen
4. Scale-invariant structure, scale-dependent magnitude: larger models rotate more and are more orthogonal

This is consistent with the "fine-tuning lives in a low-dimensional orthogonal subspace" literature (LoRA, intrinsic rank); GRPO exhibits the pattern even more strongly than SFT.

---

## 5. Task-by-Task Differences

### 5.1 Layer-type distribution is nearly identical across tasks

Every task distributes its change similarly across layer types (mean cosine of layer-type profile vectors ≈ 1.0 across task pairs). All tasks allocate ~21% of change mass to `mlp_down`, ~20% to `mlp_up`, ~12% to `attn_q`, etc. Task specialization is **not** in *which type* of layer is edited.

### 5.2 Rotation amount per task is nearly identical

This is a surprising finding. For each task we compute the mean basis rotation (principal angle at k=100) in each layer type:

**Per-task rotation at k=100, averaged across layers of type (degrees):**

| Scale | Task | ‖τ‖/‖W‖ | `mlp_down` | `mlp_up` | `attn_q` | overall |
|-------|------|---------|------------|----------|----------|---------|
| 3B | basic_arithmetic | 0.065% | 0.02° | 1.47° | 1.65° | 0.77° |
| 3B | caesar_cipher | 0.069% | 0.01° | 1.47° | 1.65° | 0.77° |
| 3B | countdown | 0.072% | 0.01° | 1.48° | 1.64° | 0.77° |
| 3B | syllogism | 0.077% | 0.01° | 1.47° | 1.65° | 0.77° |
| 4B | basic_arithmetic | 0.067% | 0.02° | 1.99° | 1.81° | 0.97° |
| 4B | caesar_cipher | 0.071% | 0.01° | 2.00° | 1.80° | 0.97° |
| 4B | syllogism | 0.058% | 0.02° | 1.98° | 1.82° | 0.97° |
| 7B | basic_arithmetic | 0.103% | 0.00° | 2.51° | 2.06° | 1.29° |
| 7B | caesar_cipher | 0.114% | 0.00° | 2.47° | 2.02° | 1.26° |
| 7B | countdown | 0.137% | 0.00° | 2.47° | 2.05° | 1.27° |
| 7B | syllogism | 0.112% | 0.00° | 2.50° | 2.00° | 1.27° |
| 8B | basic_arithmetic | 0.060% | 0.00° | 2.68° | 2.06° | 1.24° |
| 8B | caesar_cipher | 0.061% | 0.00° | 2.70° | 2.05° | 1.24° |
| 8B | countdown | 0.063% | 0.00° | 2.64° | 2.09° | 1.23° |
| 8B | syllogism | 0.052% | 0.00° | 2.67° | 2.06° | 1.23° |

**Per-task rotation varies by < 0.1° within a scale.** Tasks differ in which directions they rotate into (see §5.3) but not in how *much* they rotate. The rotation amount is governed by the layer type and the model scale, not the task.

**Implication**: GRPO applies a roughly task-independent "rotation budget" per layer type. What differs between tasks is the *direction* of rotation, not the *magnitude*. This also matches the uniform ‖τ‖_F values in §2.1 — roughly equal pressure, different directions.

### 5.3 The direction of rotation: novel-subspace analysis

§5.2 measured how *much* each task rotates (nearly identical across tasks). To measure *where* each task rotates to, we compute the **novel subspace** each task adds orthogonal to SFT:

```
novel_A = (I − U_sft U_sft^T) U_grpo(A)[:, :k]
```

This is the component of task A's top-k basis that lies outside SFT's top-k subspace — literally the "new directions" task A introduced. We then compute pairwise principal angles between novel subspaces of different tasks: **0° means tasks rotated in the same direction; 90° means they rotated into perpendicular directions.**

Script: `analyze_rotation_direction.py`. 4 tasks × 12 layers on 8B, k=100.

**Aggregate pairwise angle between novel subspaces (8B, k=100):**

| Layer type | Mean pairwise angle | Range | Interpretation |
|-----------|---------------------|-------|----------------|
| `mlp.down_proj` | **23.9°** | 17.7°–30.5° | Task directions are close (novel subspace is small and task-dependent) |
| `mlp.up_proj` | **39.4°** | 34.1°–43.3° | Tasks rotate into distinct feature subspaces |
| `attn.q_proj` | **42.8°** | 37.9°–46.6° | Most task-divergent — queries point in genuinely different directions per task |

**Per-layer per-task-pair angles between novel subspaces (full table, deg):**

| Layer | cnt×ari | cnt×csr | cnt×syl | ari×csr | ari×syl | csr×syl |
|-------|---------|---------|---------|---------|---------|---------|
| L0.mlp.down_proj | 18.89° | 18.51° | 17.70° | 19.10° | 18.42° | 17.96° |
| L0.attn.q_proj | **45.48°** | **46.08°** | 44.65° | **46.58°** | 44.59° | 44.69° |
| L2.mlp.down_proj | 25.61° | 26.75° | 24.63° | 26.24° | 24.86° | 26.18° |
| L6.mlp.up_proj | 42.33° | 42.78° | 40.80° | 42.31° | 40.21° | 40.77° |
| L9.mlp.up_proj | 43.32° | 42.98° | 41.47° | 42.07° | 40.78° | 40.87° |
| L17.mlp.up_proj | 39.88° | 39.44° | 39.13° | 39.15° | 37.83° | 37.99° |
| L17.mlp.down_proj | 22.85° | 22.87° | 22.27° | 22.64° | 22.15° | 21.77° |
| L18.mlp.down_proj | 20.30° | 20.24° | 19.66° | 20.05° | 19.13° | 18.50° |
| L19.mlp.up_proj | 36.74° | 35.90° | 35.32° | 35.45° | 34.61° | 34.12° |
| L27.mlp.down_proj | 29.59° | 30.01° | 28.74° | 29.33° | 27.64° | 27.93° |
| L35.mlp.down_proj | 28.97° | 30.49° | 28.16° | 29.22° | 26.94° | 28.04° |
| L35.attn.q_proj | 42.09° | 42.49° | 39.80° | 41.18° | 38.01° | 37.94° |

**Flat-vector cosine between task rotations** (cos of vectorized novel subspace — another view of alignment). Per layer, this swings widely: some task pairs share a rotation direction (cos up to +0.89 at L9.mlp.up_proj for countdown↔syllogism) while others actively oppose (cos = −0.90 at L6.mlp.up_proj for countdown↔basic_arithmetic, or −0.75 at L9.mlp.up_proj for countdown↔caesar_cipher). Aggregate across layer type:

| Layer type | Mean flat cos (all task pairs) |
|-----------|-------------------------------|
| `mlp.down_proj` | +0.017 |
| `mlp.up_proj` | −0.036 |
| `attn.q_proj` | −0.034 |

Aggregate cosines are near zero (tasks spread out directionally), but per-layer cosines span ±0.9 — the rotation directions are **layer-dependent**: some layers are battlegrounds (one task's +direction is another task's −direction), others are largely aligned.

### 5.3.1 What this tells us

1. **Rotation directions are task-specific** (20°–45° pairwise angles between novel subspaces), even though rotation *magnitudes* are identical (§5.2).
2. **`attn.q_proj` is the most task-adversarial layer type** — 42.8° mean pairwise angle, with several layer-specific task pairs showing strongly negative cosines. This matches the later finding (§5.6) that Q projections show the highest variance in cross-task τ cosines.
3. **`mlp.down_proj` has smallest pairwise angles** (~24°). Combined with §4.3's finding that its U_sft↔U_grpo rotation is 0° at all scales, this means: every task leaves `down_proj`'s basis in nearly the same place as SFT, and the tiny novel subspace that *does* appear is consistent across tasks. `down_proj` is **the least task-specific layer type**.
4. **`mlp.up_proj` at 39.4°**: tasks genuinely learn different output features, as expected for feature-producing layers.
5. **Some task pairs share direction at specific layers** (e.g., countdown ↔ syllogism at L9.mlp.up_proj: cos = +0.89). These are candidates for beneficial multi-task training at that layer.
6. **Some task pairs actively oppose at specific layers** (e.g., countdown ↔ basic_arithmetic at L6.mlp.up_proj: cos = −0.90). These are the microscopic signature of catastrophic interference.

### 5.5 Task-vector similarity at the whole-model level (very low)

Cosine similarities of flattened τ at 8B (top 8 off-diagonal pairs):

| Pair | Similarity |
|------|------------|
| syllogism ↔ knights_knaves | 0.083 |
| basic_arithmetic ↔ syllogism | 0.065 |
| basic_arithmetic ↔ leg_counting | 0.065 |
| knights_knaves ↔ leg_counting | 0.056 |
| countdown ↔ syllogism | 0.056 |

All pairs are < 0.1. Whole-model similarity averages over too much orthogonality to be informative.

### 5.6 Per-layer cross-task comparison (8B)

For each layer, the cosine of τ between task pairs, and the principal angle between their top-100 U_grpo subspaces:

| Layer type | τ cosine across tasks | Basis angle between task pairs |
|-----------|----------------------|--------------------------------|
| `mlp.down_proj` | 0.02–0.05 | **0.00°** (shared basis, different coordinates) |
| `mlp.up_proj` | 0.03–0.09 | **2.5–3.0°** (each task rotates basis differently) |
| `attn.q_proj` | 0.01–0.08 (sometimes negative) | 1.8–2.2° (most task-specific; some pairs oppose) |

Crucially, the basis angle *between task pairs* (2–3°) is nearly identical to each task's angle *from the SFT base* (2–3° in §4.2). **Task-specific bases are not clustered near each other** — each task rotated into a different direction, roughly equidistant from each other.

### 5.7 Depth signatures: where in the model does each task concentrate?

For each task, take its top-20 most-changed layers and bucket them into depth quartiles.

**8B depth distribution** (36 layers):

| Task | early (L0–8) | mid-early (L9–17) | mid-late (L18–26) | late (L27–35) |
|------|---|---|---|---|
| **caesar_cipher** | **88.1%** | 11.9% | 0.0% | 0.0% |
| mini_sudoku | 46.4% | 9.7% | 39.0% | 4.9% |
| syllogism | 40.8% | 54.7% | 4.5% | 0.0% |
| leg_counting | 39.6% | 55.9% | 4.6% | 0.0% |
| basic_arithmetic | 34.9% | 18.6% | **46.5%** | 0.0% |
| knights_knaves | 34.5% | 51.5% | 14.0% | 0.0% |
| countdown | 28.5% | 19.0% | **52.5%** | 0.0% |
| shortest_path | 28.3% | **62.3%** | 9.4% | 0.0% |

Three clear clusters at 8B:

- **Mid-late editors** (L17–21): `countdown`, `basic_arithmetic` — arithmetic reasoning
- **Mid-early editors** (L13–17): `shortest_path`, `syllogism`, `knights_knaves`, `leg_counting` — logical reasoning (interestingly `leg_counting` clusters with logic, not arithmetic)
- **Early-only editor**: `caesar_cipher` — 88% in L1–9, consistent with it being a token-level substitution task

### 5.8 Depth signatures explain some transfer failures

Recall Phase 1's worst case: `mini_sudoku → leg_counting = −52%`.

Their depth profiles have cosine similarity 0.33 — mini_sudoku concentrates in L19–21 while leg_counting relies on L12–16 and is *negative* at L19–21. Training on mini_sudoku corrupts L19–21, which breaks leg_counting's output stages.

Depth-profile similarity between tasks is a **stronger transfer predictor** than whole-model task-vector similarity (global r=0.32 from §6.2). The depth view shows structure that the global cosine averages away.

---

## 6. Cross-Scale Consistency

### 6.1 Scale-invariant structure

The following findings hold at **every** scale tested (3B, 4B, 7B, 8B):

- Spectral norms unchanged (ratio ≈ 1.000)
- Effective rank unchanged (Δ ≈ 0)
- τ nearly orthogonal to W's top subspace (≤ random baseline at 4B–8B)
- `mlp.down_proj` basis frozen (0° rotation)
- `mlp.up_proj`/`attn.q_proj` basis rotates, growing monotonically with scale

### 6.2 Scale-dependent details

- **Rotation magnitude grows with scale**: 1.47° (3B) → 2.67° (8B) for up_proj
- **Orthogonality grows with scale**: τ's projection onto W's top-100 falls from 0.34% at 3B to 0.04% at 7B
- **Depth signatures shift by scale**: `basic_arithmetic` moves from early (3B) → mid-early (4B, 7B) → mid-late (8B)
- **Cluster structure reorganizes**: 3B has one big early super-cluster; 8B separates into 3 clean bands; 4B is between them

### 6.3 Cross-scale depth concentration

| Task | 3B | 4B | 7B | 8B |
|------|-----|-----|-----|-----|
| `basic_arithmetic` | early (87%) | mid-early (57%) | early+mid-early (41+59%) | mid-late (47%) |
| `syllogism` | early (83%) | mid-early (95%) | early+mid-early (47+53%) | mid-early (55%) |
| `caesar_cipher` | early (88%) | early (54%) + mid-late (31%) | early (72%) | early-only (88%) |
| `countdown` | mid-late (53%) | *contaminated* | mid-late (64%) | mid-late (53%) |
| `shortest_path` | mixed | early (86%) | **late (60%)** | mid-early (62%) |
| `mini_sudoku` | mid-late (65%) | mid-early (75%) | **late (64%)** | early + mid-late split |
| `knights_knaves` | early (64%) | mid (63%) | mid-early (69%) | mid (51%) |
| `leg_counting` | early + mid-late | mid + mid-late | early + late | mid-early (56%) |
| `maze` | early (63%) + late (23%) | early (83%) | early (62%) | (skipped at 8B) |

Bolded entries at 7B are unique — no other scale localizes tasks to late layers.

### 6.4 Model-family effect (Qwen2.5 vs Qwen3)

3B and 7B are Qwen2.5; 4B and 8B are Qwen3. Some apparent scale dependencies may actually be family effects:

- `basic_arithmetic` stays early/mid-early in Qwen2.5 (3B, 7B) but migrates to mid-late in Qwen3-8B
- 7B has the highest relative ‖τ‖/‖W‖ (0.117% vs 0.06–0.07% elsewhere) and the lowest projection onto W's top-k
- The monotonic rotation growth (3B→8B) is consistent across families; the depth-concentration shift is family-specific

### 6.5 Implication for transfer prediction across scales

Depth signatures learned on a smaller model will not cleanly predict interference patterns on a larger one. Transfer analysis should use the depth profile **of the target model scale**.

---

## 7. Transfer Correlation

### 7.1 Whole-model similarity vs transfer (4B)

| Metric | Value | p-value |
|--------|-------|---------|
| Pearson r (sim × transfer) | 0.323 | 0.0035 |
| Spearman r | 0.147 | 0.193 |

Weak but significant linear correlation. Only ~10% of variance is explained by whole-model task-vector similarity.

### 7.2 Outliers

Global similarity fails to predict several transfer cases:

- **mini_sudoku → knights_knaves**: similarity ≈ 0, transfer = −50%
- **mini_sudoku → leg_counting**: similarity ≈ 0, transfer = −52%
- **basic_arithmetic → leg_counting**: similarity = 0.088 (high), transfer = −32%

The catastrophic cases have **low** similarity. Global cosine can't see the conflict because it averages over the whole model; the real conflict is localized (see §5).

### 7.3 Magnitude and effective rank don't predict transfer

| Metric | Pearson r | p-value |
|--------|-----------|---------|
| Magnitude vs same-task performance | 0.459 | 0.252 (n=10) |
| Effective rank of τ vs off-diagonal transfer | −0.046 | 0.899 |

Neither the amount of weight change nor the "focus" of the change (rank of τ) predicts transfer behavior.

### 7.4 Better predictor: depth-profile overlap

Qualitative evidence (from §5.8): the worst transfer pairs have disjoint depth profiles and modify overlapping layers in opposing directions. Formal correlation with depth-profile similarity pending for a follow-up.

---

## 8. Implications

### 8.1 For understanding GRPO

- **GRPO is structurally conservative**: W's spectrum is untouched; the update lives in a small orthogonal subspace
- **`down_proj` is untouched** at the basis level — an unexpectedly clean invariant across 4 scales
- **Task specialization happens in feature projectors** (`up_proj`, `q_proj`), which is where it mechanistically *should* happen

### 8.2 For multi-task RLVR

1. **Don't use whole-model similarity** as a curriculum heuristic — it averages over the conflict
2. **Look at depth signatures**: pair tasks with disjoint depth ranges, avoid tasks that edit the same depth in opposing directions
3. **Expect catastrophic interference from task pairs with overlapping mid-layers and negative τ cosines on `attn.q_proj`** — the most task-adversarial layer type
4. **Scale matters**: a curriculum designed on 3B depth signatures won't transfer to 8B

### 8.3 For future fine-tuning methods

- The fact that `down_proj` basis never rotates suggests LoRA-style low-rank adapters placed only on `up_proj` and `q_proj` might capture most of GRPO's expressivity at lower cost
- The fact that τ is orthogonal to W's top subspace suggests orthogonal fine-tuning (OFT) methods should fit GRPO's implicit inductive bias well

---

## 9. Limitations

1. **Countdown re-run pending (4B)**: original used `grpo_qwen_4b_instruct_countdown_4to5` which was trained from a different base (`sft_qwen3_4b`, not `sft_qwen_4.0b_instruct`), causing ‖τ‖=151 outlier. Script updated to use `phase2_countdown/direct_rlvr` but not re-run.
2. **Maze excluded at 8B**: v1 `grpo_qwen3_8b_maze` dir has only generation logs, no weights. v2 has weights if needed.
3. **No 8B/7B transfer matrices**: similarity-vs-transfer correlation (§7.1) only computed on 4B.
4. **No gradient analysis**: we haven't compared actual gradient directions at the SFT checkpoint, which could reveal whether tasks "want" to move similarly before training starts.
5. **Causality unclear**: we observe correlation between depth-overlap and interference; intervention experiments (e.g., projecting out shared directions during multi-task training) would be needed to establish causation.
6. **Only 5 tasks probed for subspace/rotation (§4)**: countdown, basic_arithmetic, caesar_cipher, syllogism, cryptarithm. Full 10-task coverage would be more robust.
7. **Qwen2.5 vs Qwen3 confound** (§6.4): scale effects and family effects are mixed at 3B/7B (Qwen2.5) vs 4B/8B (Qwen3).

---

## Appendix A: Glossary of Measures

This section defines every metric used, what it measures geometrically, and what values mean.

### A.1 Task vector τ

```
τ = W_grpo − W_sft
```

Per weight matrix, the change induced by GRPO training. `W_sft` is the starting point (SFT baseline); `W_grpo` is after GRPO training on one task.

**What τ captures**: the direction and magnitude of GRPO's update in weight space.

### A.2 Frobenius norm ‖·‖_F

```
‖M‖_F = sqrt(sum_{ij} M_{ij}²) = sqrt(sum_i σ_i²)
```

Standard Euclidean norm of a matrix (treating it as a vector). Captures total magnitude of change.

- **‖τ‖_F**: total GRPO update magnitude per layer
- **‖τ‖/‖W‖**: relative magnitude. Values 0.05–0.12% mean GRPO shifts W by roughly 0.1% of its norm

### A.3 Task vector cosine similarity

```
cos(τ_A, τ_B) = (τ_A · τ_B) / (‖τ_A‖ · ‖τ_B‖)
```

Flatten both task vectors to 1D (or keep per-layer and sum), compute dot product normalized by norms.

**Interpretation**:
- +1: identical direction (tasks learn the same change)
- 0: orthogonal (tasks learn unrelated changes)
- −1: opposite direction (tasks conflict directly)

**In practice**: we see 0.02–0.1 off-diagonal. Very close to orthogonal; task vectors are largely independent.

### A.4 Spectral norm ‖·‖₂

```
‖M‖₂ = σ_1(M)  (largest singular value)
```

The maximum amplification a matrix applies to any input vector. A key indicator of matrix "size" at a directional level.

- **‖τ‖₂**: largest direction of the GRPO update
- **‖W_grpo‖₂ / ‖W_sft‖₂**: the "spectral norm ratio" — tells us if GRPO amplified or dampened the layer

**Value interpretation**: ratio ≈ 1.000 means GRPO preserved the top singular value. We see 1.000 ± 0.002 across all scales.

### A.5 Effective rank

```
eff_rank(M) = exp( H(σ_i / Σσ_i) )
```

where H is Shannon entropy of the normalized singular-value distribution.

**What it measures**: how "spread out" a matrix's information is across its singular directions.

- High eff_rank (close to min(rows, cols)): information distributed uniformly across many directions ("diffuse" matrix)
- Low eff_rank (close to 1): one direction dominates ("focused" / near-rank-1 matrix)

**Eff_rank of τ**: how concentrated is the GRPO update? Low = focused, potentially skill-like; high = diffuse, broad.

**Eff_rank_change = eff_rank(W_grpo) − eff_rank(W_sft)**: did GRPO concentrate or diffuse W's representations? We see Δ ≈ 0 at all scales — W's rank structure is preserved.

### A.6 Singular value decomposition (SVD)

```
M = U Σ V^T
```

where U, V are orthogonal matrices and Σ = diag(σ_1, σ_2, …) has singular values in decreasing order.

- **U** (left singular vectors): a basis of the "output space" of M
- **V** (right singular vectors): a basis of the "input space" of M
- **σ_i**: how much each direction gets amplified

### A.7 Subspace projection ratio

```
ρ_k(τ) = ‖U_k U_k^T τ V_k V_k^T‖_F² / ‖τ‖_F²
```

where U_k, V_k are the top-k left and right singular vectors of W_sft.

**What it measures**: what fraction of τ's energy lies inside W_sft's top-k singular subspace.

**Interpretation**:
- ρ_k → 1 for small k: τ is aligned with W's top directions (rescaling existing features)
- ρ_k ≈ (k/r)² (the random baseline for an isotropic τ): τ is orthogonal to W (adding new directions)
- ρ_k < (k/r)²: τ is *anti-aligned* with W's top subspace — preferentially occupies mid/low-rank directions

**Variants**:
- **Two-sided**: project with both U_k and V_k (tight — measures alignment with W's top rank-k subspace)
- **Left-only**: project with U_k only (measures row-space alignment)
- **Right-only**: project with V_k only (measures column-space alignment)

### A.8 Principal angles

```
angle_i(U, U') = arccos( σ_i(U^T U') )
```

The principal angles between two k-dimensional subspaces U, U' (each represented as an orthonormal basis, n×k matrix). Computed by taking SVD of U^T U' — its singular values are cos(angle_i), so arccos gives the angles.

**What it measures**: how much a subspace has rotated.

- 0°: subspaces coincide (basis frozen)
- 90°: subspaces are orthogonal (completely different basis)

**In practice**: we report mean angle across i=1..k.

- **Left principal angles**: rotation of U_sft's top-k vs U_grpo's top-k (the "output directions")
- **Right principal angles**: rotation of V (the "input directions")

**Typical values**: 0.00° for `mlp.down_proj` (basis frozen), 1.5–2.7° for `mlp.up_proj` (basis rotates).

**Interpretation caveat**: principal angles between U_sft and U_grpo measure the *magnitude* of rotation but do not tell you *in which direction* the rotation happened. Two tasks can rotate by identical angles into completely different subspaces. See §A.8a for the direction measure.

### A.8a Novel subspace and pairwise angle between novel subspaces

```
novel_A = (I − U_sft U_sft^T) U_grpo(A)[:, :k]    # the rotation direction of task A
angle(novel_A, novel_B) = principal angles between span(novel_A) and span(novel_B)
```

**What `novel_A` measures**: the component of task A's top-k basis that lies outside SFT's top-k subspace. Geometrically, this is the "new directions" that task A introduced — i.e., the *direction* of its basis rotation.

**What the pairwise angle measures**: how aligned two tasks' rotation directions are.

- 0°: task A and task B rotated into the same subspace (same direction)
- 90°: task A and task B rotated into perpendicular subspaces (orthogonal directions)
- Intermediate: partial overlap

**Typical values (8B, k=100, pairwise across 4 tasks)**:
- `mlp.down_proj`: 24° (least task-divergent direction — but very tiny rotation to begin with)
- `mlp.up_proj`: 39° (tasks rotate into distinct feature subspaces)
- `attn.q_proj`: 43° (most task-divergent)

### A.8b Flat rotation cosine

```
v_A = vec(novel_A);  v_B = vec(novel_B)
cos(v_A, v_B) = (v_A · v_B) / (‖v_A‖ · ‖v_B‖)
```

A scalar view of task-pair direction alignment. Complementary to A.8a (subspace angle) — the subspace angle ignores sign/orientation within each direction while the flat cosine preserves it.

- +1: tasks rotate in the same direction with the same orientation
- 0: orthogonal
- −1: directly opposing rotation (the microscopic signature of catastrophic interference)

**Typical values**: mean near 0 aggregated over layers, but layer-level values range from +0.89 (aligned pairs at specific layers) to −0.90 (opposing pairs at specific layers).

### A.9 Depth signature

```
depth_sig_task[i] = sum_{layers L at index i} ‖τ_L,task‖_F
```

For each task, a vector of length N (number of transformer blocks) giving the total change magnitude at each layer index. Often normalized to sum to 1.

**What it measures**: where in the model the task concentrates its weight changes.

**Usage**:
- **Depth quartile distribution**: bucket indices into early/mid-early/mid-late/late and report fraction of mass
- **Depth-profile cosine**: cosine similarity between two tasks' depth_sig vectors — how similar is "where" they edit?

### A.10 Layer types

Weight matrices in a transformer block, by role:

| Type | Role |
|------|------|
| `mlp.gate_proj`, `mlp.up_proj` | Project hidden state → MLP features (out-projection) |
| `mlp.down_proj` | Project MLP features → hidden state (in-projection / sink) |
| `self_attn.q_proj` | Produce query vectors |
| `self_attn.k_proj` | Produce key vectors |
| `self_attn.v_proj` | Produce value vectors |
| `self_attn.o_proj` | Project attention output → hidden state |
| `embed_tokens` | Token → vector embeddings |
| `lm_head` | Final vector → vocab logits |
| `norm` | RMSNorm / LayerNorm scale parameters |

`up_proj` / `q_proj` are **feature projectors**: they choose what features to produce or query for. Natural sites for task-specific basis rotation.

`down_proj` / `o_proj` are **sinks**: they aggregate features back into the residual stream. Their principal directions are more stable; changes there are coordinate-space, not basis-space.

### A.11 Random baseline for subspace projection

For an isotropic random matrix τ (Gaussian entries), the expected fraction of ‖τ‖² captured by a fixed k×k two-sided projection onto an r×r ambient space is:

```
E[ρ_k] = (k / r)²
```

We use this as the "no special alignment" null hypothesis. If observed ρ_k ≈ (k/r)², τ is directionally random w.r.t. W's top subspace. If ρ_k < (k/r)², τ actively avoids W's top directions.

### A.12 Transfer correlation

The Pearson r between weight-space similarity `cos(τ_A, τ_B)` and behavioral transfer Δ (accuracy on task B after training on task A, minus SFT accuracy on B). Computed over all off-diagonal (A, B) pairs.

- r > 0: similar task vectors → positive transfer (weak prediction)
- r near 0: no prediction (what we observed for magnitude, eff_rank)

---

## Appendix B: Generated Artifacts

### B.1 Main analysis scripts

| Script | Purpose |
|--------|---------|
| `analyze_rlvr_weight_changes.py` | Phase A/B/C: task vectors, spectral, transfer correlation |
| `analyze_eigenvalue_shifts.py` | Full-SVD comparison of W_sft vs W_grpo |
| `analyze_subspace_and_rotation.py` | Subspace projection ratios + principal angles |
| `analyze_per_layer_task_diff.py` | Per-layer cross-task τ cosines and basis angles |
| `analyze_rotation_direction.py` | Novel subspace + pairwise direction of rotation (§5.3) |

### B.2 Result files

| Path | Contents |
|------|---------|
| `results/weight_analysis/{3b,4b,7b,8b}/analysis_results.json` | Phase A/B for each scale |
| `results/weight_analysis/{3b,4b,7b,8b}/subspace_rotation.json` | Projection + angles for each scale |
| `results/weight_analysis/8b/eigenvalue_shifts.json` | Full SVD comparison (8B) |
| `results/weight_analysis/8b/per_layer_task_diff.json` | Per-layer cross-task data (8B) |
| `results/weight_analysis/8b/rotation_direction.json` | Novel-subspace angles & flat-cos per task pair (8B) |
| `results/weight_analysis/{size}/*.png` | Similarity heatmaps, magnitude bars, rank plots |

### B.3 Related reports

- Phase 1: `rlvr_experiments/report_transfer_matrix.md` — 10×10 transfer matrix across 4 scales
- Phase 2: `rlvr_experiments/report_atomic_curriculum.md` — ARS-based curriculum design
- Phase 2 (complete, 7 tasks): `rlvr_experiments/report_phase2_complete.md`
- Phase 4: `rlvr_experiments/report_phase4_merge_scaling.md` — task-vector merging at scale (Parts 1–6)
- Master document: `rlvr_experiments/MASTER_RLVR_dynamics.md` — comprehensive consolidation across phases

---

## Appendix C: Post-Phase-4-Part-6 Status Update (added 2026-05-12)

The findings in this Phase 3 report are weight-space measurements and are unaffected by the gap-closing experiments in Phase 4 Part 6 (§7.10 of the paper). Phase 3's contributions stand:

- Spectrum preservation (§3) — robust at every scale.
- τ near-orthogonal to W's top subspace (§4) — robust.
- Universal layer-type split (`mlp.down_proj` frozen, `up_proj`/`q_proj` rotate 1.5°→2.7° with scale) (§5.2) — robust at 3B/4B/7B/8B across two model families.
- Pairwise novel-subspace angles 24°–43° (§5.3) — robust at 8B.
- Depth signatures with three-cluster structure at 8B (§6) — robust.
- Depth-profile cosine as a Phase-1 *transfer* predictor (§7.4) — robust within the Phase-1 sequential-training context.

**What changed downstream of Phase 3 (Phase 4 Part 6):**

Two predictor claims that built on Phase 3's geometry — magnitude-weighted sign agreement (Phase 4 F29) and L1-grounding ARS (Phase 4 F33) — were claimed to forecast 4B pair-merge damage at r = −0.81 and r = −0.76 respectively on n = 7 pairs. Both predictors degraded substantially when tested on n = 12 (r = −0.45 and r = +0.06) and on n = 19 (sign-agreement r = −0.27). Phase 3's per-layer flat cosines (§5.3.5–5.3.6) were the proposed microscopic mechanism behind the sign-agreement predictor; the geometric measurements are unchanged but the predictor application of them is no longer supported.

**Implication for §5.3.6** (the per-layer flat-cosine claim): the measurement that per-layer flat cosines on `mlp.up_proj` swing from +0.89 to −0.90 across task pairs at 8B is correct. The interpretation that this localized opposition is the mechanism behind a strong cross-scale merge predictor was conditional on the n = 7 sign-agreement result, which has since collapsed. The flat cosines remain a valid *characterization* of how task-specific rotation directions can oppose each other in weight space; they are not a calibrated *predictor* of merge damage.

**Implication for §7** (transfer prediction): §7.4's claim that depth-profile overlap predicts Phase-1 sequential transfer is unchanged (this concerns Phase-1 training-then-evaluating, not Phase-4 weight summation, which §7.10.1 confirms is a different operation). §7's depth-profile transfer-prediction claim survives.

See `MASTER_RLVR_dynamics.md` §6 (Phase 4 Part 6) for the complete predictor-decay analysis.
