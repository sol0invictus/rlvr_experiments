# Master Document — Mechanistic Training Dynamics of RLVR on Small Reasoning Models

*Authoritative internal reference. The external paper is written as a subset of this document.*

*Last updated: 2026-05-12.*

---

## 0. Overview and Reading Guide

### 0.1 What This Document Is

This master document is the single authoritative reference for a four-phase research project investigating how Reinforcement Learning with Verifiable Rewards (specifically GRPO — Group Relative Policy Optimization with DAPO loss) reshapes small language models on reasoning tasks. Over approximately five months (Jan 2026 through May 2026) we ran four interlocking studies:

- **Phase 1 (cross-task transfer matrix).** 400 controlled (model, evaluation-task) cells — 10×10 transfer matrices at 4 model scales — asking whether GRPO on task A transfers to task B.
- **Phase 2 (atomic curriculum).** A behavioral-probe framework (Atomic Readiness Score, ARS) that decomposes tasks into grounding/local-reasoning atoms and predicts which of three RLVR regimes will succeed.
- **Phase 3 (weight-space geometry).** A full geometric characterization of τ = W_GRPO − W_SFT across scales, layer types, and tasks: what GRPO does structurally to the weights.
- **Phase 4 (task-vector merging).** Practical and mechanistic study of composing two or more RLVR-trained τ's into a single multi-skill model, including the Part-6 gap-closing experiments that sharply rescoped the claims.

This document is deliberately verbose. The external paper (TMLR / workshop target) is a strict subset. Material present here that is not in the paper includes: full per-scale transfer matrices, all null-result predictor correlations at every sample size, the shortest-path 15-condition deep dive, the Phase-4 Part-6 residual-feature search, every per-pair damage table, dated chronology, and implementation-issue logs.

The tone is technical and self-contained. The reader is assumed to know ML and RL basics but nothing about this specific project.

### 0.2 One-Sentence Takeaway

> **GRPO on small reasoning LMs has a universal geometric signature — spectrum-preserving, `mlp.down_proj`-basis-frozen, feature-projector basis rotations of 1.5°–2.7° — but single-task success and multi-task merge compatibility are governed by task- and pair-specific facts that no cheap prospective probe we tested (including the ones we originally published at n = 7) reliably predicts on the n = 19 sample.**

### 0.3 Phases at a Glance

| Phase | Dates | Primary question | Headline | Falsifications |
|---|---|---|---|---|
| 1 | 2026-01 → 02 | Does single-task GRPO transfer? | 36% of pairs degrade; LR dominates; arithmetic degrades at every scale | — |
| 2 | 2026-04-18 → 05-01 | Can an atomic probe predict the right RLVR regime? | ARS predicts best regime on 6/7 tasks (86%) | mini_sudoku exception |
| 3 | 2026-04-19 → 05-01 | What does GRPO do to the weights? | Scale-invariant how (rotation of `up`/`q` basis, `down_proj` frozen); task-specific where | — |
| 4 (Parts 1–5) | 2026-04-23 → 05-06 | Can τ's be merged into multi-skill models? | Drop `mlp.down_proj`; two cheap predictors (sign-agree r=−0.81, L1-ARS r=−0.76) at n=7 | — |
| 4 (Part 6) | 2026-05-09 → 05-12 | Do the Part-5 predictor claims survive more pairs? | No — both collapse monotonically with n; Group classification is pair- not task-intrinsic | F29, F33, routing probe, residual search all fail |

### 0.4 How the Paper Maps onto This Master Doc

The target paper has five substantive sections. Their provenance in the master:

| Paper section | Source chapters in this doc | Notes |
|---|---|---|
| §2 Setup | Chapter 1 | Subset — drop compute details |
| §3 (Phase 1 setup + transfer matrix) | Chapter 2 | Uses 3B and 4B matrices in main text; 7B/8B in Appendix B |
| §4 (ARS and atomic curriculum) | Chapter 3 | Table 1, Figures 4.1/4.2; deep-dive in Appendix C |
| §5 (Weight-space geometry) | Chapter 4 | Figures 5.1–5.6; Appendix F–J |
| §7 (Task-vector merging, Path A) | Chapter 5 + Chapter 6 | §7.10 in paper = Chapter 6 in master |
| §8 Cross-phase synthesis | Chapter 7 | Condensed |

Content that is master-only: Chapter 8 (practical recipes), Chapter 9 (null-result catalogue), Chapter 12 (chronology), Appendices B/D/E/F/G.

---

## 1. Experimental Setup Common to All Phases

This chapter defines the shared experimental apparatus used across all four phases. Phase-specific departures are documented in each phase's chapter.

### 1.1 Models

We used four base checkpoints spanning two Qwen families and four parameter scales:

| Scale | Family | HuggingFace ID | Layers | Hidden | Heads / KV heads |
|---|---|---|---:|---:|---|
| 3B | Qwen2.5 | `Qwen/Qwen2.5-3B-Instruct` | 36 | 2048 | 16 / 2 |
| 4B | Qwen3 | `Qwen/Qwen3-4B` | 36 | 2560 | 32 / 8 |
| 7B | Qwen2.5 | `Qwen/Qwen2.5-7B-Instruct` | 28 | 3584 | 28 / 4 |
| 8B | Qwen3 | `Qwen/Qwen3-8B` | 36 | 4096 | 32 / 8 |

The Qwen2.5/Qwen3 split is not ideal (3B and 7B are Qwen2.5; 4B and 8B are Qwen3) and produces a confound that cannot be fully disentangled from scale. Qwen3 does not have separate `-Instruct` variants at these sizes; the base model is already instruction-capable. Qwen2.5 requires the explicit `-Instruct` variants. This family confound is flagged again in Chapter 10.

#### 1.1.1 SFT warm-up on `<think>/<answer>`

Before any RL training, each base model is supervised-fine-tuned for 3 epochs (lr = 2e-5, batch size 2–4, max_seq_length 1024) on `data_gen/data/sft_warmup.jsonl`, a 1,000-sample dataset of trivia, unit conversions, word puzzles, and date reasoning. The training target is the structured `<think>...</think><answer>...</answer>` format. This is necessary because (a) GRPO requires a reward-extractable answer format, (b) without SFT the models produce free-form text that the verifier cannot parse, and (c) the warm-up data is deliberately task-agnostic so that no specific reasoning-gym task gains an unfair advantage before GRPO.

All post-SFT checkpoints are named with prefix `sft_qwen{25_,3_,_4.0b_}{3b,4b,7b,8b}` and are the W_SFT reference throughout the remainder of the paper.

### 1.2 Tasks

Ten reasoning-gym tasks span five skill categories. Every phase uses a subset of this table.

| Category | Task | Description | Default params |
|---|---|---|---|
| Arithmetic | `basic_arithmetic` | Evaluate expression | 2–6 terms, 1–4 digit numbers, +−×÷, parentheses |
| Arithmetic | `countdown` | Reach target with available numbers | 4–5 numbers, target 10–200 |
| Logic | `syllogism` | Validity of syllogistic arguments | default |
| Logic | `knights_knaves` | Identify sages/fools from statements | default |
| Spatial/Grid | `maze` | Shortest path length in ASCII maze | 5×5 grid |
| Spatial/Grid | `mini_sudoku` | Solve 4×4 Sudoku | default |
| Spatial/Grid | `shortest_path` | Directions along shortest path | default |
| Planning | `leg_counting` | Sum legs across animals | default |
| Algorithmic | `caesar_cipher` | Decrypt Caesar cipher | 2–6 words, rotation 1–5 |
| Algorithmic | `cryptarithm` | Letter-to-digit puzzle | default |

All tasks share the same `<think>/<answer>` output contract. Each has a reasoning-gym verifier returning a scalar reward in [0, 1] — in practice all rewards used in this paper are binary {0, 1} because DAPO loss interacts cleanly with hard-threshold rewards.

### 1.3 GRPO Training Recipe

The canonical GRPO recipe (used unchanged in Phases 1, 2, 3, 4 except where explicitly noted):

| Hyperparameter | Value |
|---|---|
| `learning_rate` | 5.0e-6 constant (1e-6 in 7B/8B v2 sweeps) |
| `lr_scheduler_type` | constant |
| `max_steps` | 300 (single-task direct); 100 per atomic stage |
| `per_device_train_batch_size` | 1 |
| `gradient_accumulation_steps` | 1 |
| `num_generations` (rollouts/prompt) | 8 (16 in v2) |
| `generation_batch_size` | 8 (16 in v2) |
| `max_completion_length` | 2048 (full task), 256 (atoms) |
| `max_prompt_length` | 512 (most), 1024 (shortest_path) |
| `temperature` | 1.0 |
| `loss_type` | DAPO |
| `beta` | 0.0 |
| `kl_coeff` | 0.0 |
| `warmup_ratio` | 0.1 |
| `max_grad_norm` | 1.0 |
| `mask_truncated_completions` | true |
| `epsilon_high` | 5.0 |
| `stop_strings` | `["</answer>"]` |
| `seed` | 42 (training), 9999 (eval) |

No KL penalty and no external reference model — DAPO with `beta = 0.0` is pure GRPO on the advantage estimate. The `epsilon_high = 5.0` asymmetric clip tolerates large positive advantage ratios (useful when the rollout finds a correct answer unexpectedly) while keeping the negative clip at the default. `mask_truncated_completions` prevents the model from being credited for format-broken generations that exceed `max_completion_length`.

### 1.4 Evaluation Protocol

All post-training accuracy numbers in this document come from `evaluate_transfer.py` with the following configuration:

| Parameter | Value |
|---|---|
| `num_samples` | 200 (Phase 2+ onwards); 50 in Phase 1 |
| `seed` | 9999 (distinct from training seed 42) |
| `max_new_tokens` | 1024 (1024 in Phase 4; 2048 in Phase 1) |
| `batch_size` | 4 or 8 |
| DDP | 8 GPUs, `all_reduce` aggregation |
| `tokenizer.padding_side` | left |

**Answer extraction bug (fixed 2026-04-17).** The early Phase-1 verifier used `re.search` to find the first `<answer>...</answer>` match. In models that emitted an echoed system-prompt template before the real reasoning block, this picked up the template's empty answer tag and misreported accuracy as near-zero. The fix is `re.findall` with the last match. Every number reported in Phases 2–4 is post-fix. Some Phase-1 numbers (all re-evaluated later) are the fixed values.

**DDP gotchas.** Models are loaded without `device_map` (Accelerate assigns per-process devices). `GRPOConfig` sets `ddp_find_unused_parameters=False` to prevent hangs on steps where embedding layers receive no gradient. Evaluation was initially chained across all 10 tasks in one `torchrun` invocation, which hit a cross-task NCCL re-init SIGSEGV at boundary; final protocol is one `torchrun` per (model, task) cell.

### 1.5 Compute and Infrastructure

- **Hardware.** Green cluster at Amazon. P6-B200 nodes, 8× B200 GPUs each (178 GB HBM per GPU). SIGSEGV on eval crossover noted above; all results re-run on isolated launches.
- **Training framework.** TRL 1.0.0 + transformers 4.57.6. Data from `reasoning-gym` (installed from source), `math_verify` for GSM8K.
- **Storage.** Shared FSx filesystem across pods. All checkpoints and eval outputs live under `/scratch/sunguha/rlvr_experiments/{results,outputs,paper_prep}`.
- **Pod names used.** `sunguha-rlvr-phase2-qbgw`, `sunguha-rlvr-paper-npdz`, `sunguha-rlvr-paper2-n25r` are the three pods that ran the bulk of Phases 2–4. Phase 1 used an earlier pod pool with the same hardware.
- **Approximate total compute.** Phase 1: ~100 GPU-hours for all training + eval. Phase 2: ~80 GPU-hours (more atomic-curriculum variants). Phase 3: ~10 GPU-hours (analysis-only; SVDs and principal angles are CPU-bound once weights are loaded). Phase 4 (Parts 1–5): ~250 GPU-hours. Part 6 gap-closing: ~180 GPU-hours. **Total project compute ≈ 600 GPU-hours**, plus analysis-only time on CPU and single-GPU workstations.

---

## 2. Phase 1 — Cross-Task Transfer Matrix

### 2.1 Motivation: Does GRPO on Task A Transfer to Task B?

GRPO with verifiable rewards is widely used as a finishing step for reasoning-tuned models. The assumption is that it "improves reasoning" in a general sense. Phase 1 rigorously tests this assumption at four scales by training 10 separate single-task GRPO checkpoints per scale and evaluating every checkpoint on every task, producing a 10×10 transfer matrix per scale. If GRPO is a general improver, the matrices should be positive everywhere and strongest on the diagonal; if GRPO is narrowly task-specific, diagonals should dominate and off-diagonals should be weak; if GRPO is actively harmful, off-diagonal entries should go negative.

### 2.2 Method: 10×10 Matrices × 4 Scales

For each scale we run GRPO on all 10 tasks separately (10 training runs) from the same SFT checkpoint. Each resulting checkpoint is evaluated on all 10 tasks at seed 9999. The result is a 10×10 matrix of Δ = (post-GRPO accuracy) − (SFT baseline accuracy) in absolute percentage points. Positive entries = GRPO improved task B when trained on task A; negative entries = GRPO degraded B. Diagonal entries are same-task self-improvement.

Four scales × 10 × 10 = 400 (model, eval-task) cells. The 7B and 8B scales were run twice with different GRPO hyperparameters (v1: lr = 5e-6, 8 generations; v2: lr = 1e-6, 16 generations) because early 7B/8B runs produced catastrophic same-task harm, motivating a lower-LR regime sweep.

### 2.3 Headline Result: 36% of Pairs Degrade

Aggregating over all 400 (model, task) cells at v1 hyperparameters:

- 36.0% of all (trained-task, eval-task) pairs are negative (GRPO hurt).
- 45.0% are zero (no change — almost all of these are the caesar/cryptarithm columns, which never produce signal).
- 19.0% are positive.
- The maximum gain is +60% (7B countdown self-training).
- The maximum loss is −52% (4B mini_sudoku → leg_counting, off-diagonal).
- 11 (model, task) cells at 8B-v1 and 14 at 7B-v1 exceed −10% harm.

GRPO is not "free improvement." Its net sign depends on scale, learning rate, task identity of the training set, and task identity of the evaluation set. The single largest surprise of Phase 1 is that GRPO can hurt the very task it was trained on (e.g., 4B basic_arithmetic self-training = −12%, 7B arithmetic self-training = −30%).

### 2.4 Full Per-Scale Transfer Matrices

Each of the six matrices below follows the same column convention:
`cnt`=countdown, `mz`=maze, `sdk`=mini_sudoku, `pth`=shortest_path, `ari`=basic_arithmetic, `syl`=syllogism, `kk`=knights_knaves, `leg`=leg_counting, `csr`=caesar_cipher, `cry`=cryptarithm. Diagonal entries are marked with asterisks. All values are Δ from the scale's SFT baseline (final row).

#### 2.4.1 Qwen2.5-3B (lr = 5e-6, 8 generations)

| Trained on | cnt | mz | sdk | pth | ari | syl | kk | leg | csr | cry |
|---|---|---|---|---|---|---|---|---|---|---|
| countdown | *+10* | −4 | 0 | 0 | −2 | +10 | +8 | 0 | 0 | 0 |
| maze | 0 | *+2* | 0 | +4 | −4 | 0 | −2 | −12 | 0 | 0 |
| mini_sudoku | 0 | −8 | *0* | +2 | −8 | −4 | −4 | −6 | 0 | 0 |
| shortest_path | 0 | −2 | 0 | *+40* | −6 | −8 | −2 | 0 | 0 | 0 |
| basic_arith | 0 | −8 | 0 | 0 | *−14* | +10 | −2 | −10 | 0 | 0 |
| syllogism | 0 | −4 | 0 | 0 | −20 | *+20* | +4 | −12 | 0 | 0 |
| knights_knaves | 0 | −12 | 0 | 0 | −2 | +4 | *+18* | −6 | 0 | 0 |
| leg_counting | 0 | −4 | 0 | +2 | −16 | 0 | −8 | *+12* | 0 | 0 |
| caesar_cipher | 0 | −8 | 0 | 0 | 0 | +8 | +14 | −6 | *0* | 0 |
| cryptarithm | 0 | −2 | 0 | 0 | 0 | +4 | 0 | +2 | 0 | *0* |
| **Baseline** | 0% | 14% | 0% | 0% | 46% | 74% | 26% | 12% | 0% | 0% |

*Shortest-path is the 3B standout at +40% self-training — the largest diagonal gain at this scale. The 3B model gains graph-traversal capabilities that were absent from the baseline (0% SFT accuracy). Arithmetic self-training is −14% because the 3B SFT baseline is already at 46%; GRPO adopts a compressed reasoning style that loses multi-step accuracy (this is the seed of Phase 2's "reasoning compression" pathology). Cross-category transfer is rare — most off-diagonal entries are zero or negative. Caesar and cryptarithm are inert at 3B (0% baseline, 0% after any GRPO) because the base model has no traction on those tasks. A scattered pattern of positive off-diagonals (countdown → syllogism +10%, caesar → knights +14%) suggests that *any* GRPO training activates some generic "reasoning mode" rather than task-specific transfer.*

#### 2.4.2 Qwen3-4B (lr = 5e-6, 8 generations)

| Trained on | cnt | mz | sdk | pth | ari | syl | kk | leg | csr | cry |
|---|---|---|---|---|---|---|---|---|---|---|
| countdown | *+36* | +6 | +14 | +4 | +6 | +18 | +4 | +6 | 0 | 0 |
| maze | −10 | *+12* | +16 | +4 | 0 | +12 | 0 | +6 | 0 | 0 |
| mini_sudoku | −16 | −2 | *+22* | −16 | −26 | −16 | **−50** | **−52** | 0 | 0 |
| shortest_path | +6 | +8 | +28 | *+28* | +6 | +16 | −2 | +14 | 0 | 0 |
| basic_arith | +4 | −2 | −4 | −6 | *−12* | +18 | −16 | −32 | 0 | 0 |
| syllogism | −8 | −2 | +26 | −2 | −2 | *+26* | +2 | +10 | 0 | 0 |
| knights_knaves | −8 | +2 | +4 | −12 | −8 | +26 | *+12* | −4 | 0 | 0 |
| leg_counting | +16 | +2 | +22 | +12 | −8 | +14 | −8 | *+28* | 0 | 0 |
| caesar_cipher | 0 | 0 | 0 | −2 | +6 | −22 | 0 | +2 | *0* | 0 |
| cryptarithm | −2 | −2 | −2 | −2 | −2 | +2 | +2 | −4 | 0 | *0* |
| **Baseline** | 34% | 2% | 14% | 16% | 62% | 66% | 74% | 52% | 0% | 0% |

*The 4B matrix is the most intensively analyzed cell in the paper because Phase 3 runs its weight-geometry analysis here and Phase 4's merge pairs are all defined at 4B. Countdown is the best single-task performer (+36% diagonal plus broad positive transfer). Shortest_path is the best all-rounder (+28% diagonal and positive transfer to 6/9 other tasks). Mini_sudoku is catastrophic for some partners (+22% on sudoku itself, −50% on knights_knaves, −52% on leg_counting — the largest single negative entry in the entire study). Arithmetic self-training again fails (−12%). Logic tasks show asymmetric transfer: syllogism → sudoku +26%, but countdown → syllogism +18% doesn't reverse (syllogism → countdown = −8%).*

#### 2.4.3 Qwen2.5-7B — v1 (lr = 5e-6, 8 generations)

| Trained on | cnt | mz | sdk | pth | ari | syl | kk | leg | csr | cry |
|---|---|---|---|---|---|---|---|---|---|---|
| countdown | *+60* | −4 | −6 | 0 | −2 | +6 | +8 | −8 | 0 | 0 |
| maze | −4 | *+10* | −6 | −4 | −2 | +4 | 0 | −10 | 0 | 0 |
| mini_sudoku | +2 | −4 | *+16* | −4 | −6 | −8 | +4 | −6 | 0 | 0 |
| shortest_path | −4 | −2 | −4 | *+36* | −8 | −24 | 0 | −2 | 0 | 0 |
| basic_arith | −2 | +2 | −12 | −4 | ***−30*** | −2 | 0 | −30 | 0 | 0 |
| syllogism | 0 | +6 | −12 | −4 | −46 | *+26* | −28 | −32 | 0 | 0 |
| knights_knaves | −2 | −2 | −10 | −4 | −32 | −8 | *+28* | −32 | 0 | 0 |
| leg_counting | −4 | +2 | −4 | −4 | −10 | −2 | +2 | *+8* | 0 | 0 |
| caesar_cipher | −4 | −4 | −6 | −4 | −2 | −4 | +2 | −2 | *0* | 0 |
| cryptarithm | 0 | −2 | 0 | +2 | 0 | 0 | −2 | +2 | 0 | *0* |
| **Baseline** | 4% | 14% | 12% | 4% | 60% | 64% | 32% | 32% | 0% | 0% |

*7B v1 is the most volatile scale. Countdown gains +60% (4% → 64%) — the largest single gain anywhere in the study; the low baseline leaves a lot of room for GRPO to unlock. Arithmetic self-training is catastrophic at −30% (60% → 30%), confirmed separately on the training distribution (seed=42: 70% → 42%). Syllogism self-training is the worst "collateral damage" case of any Phase-1 checkpoint — it gains +26% on syllogism while costing −46% arith, −28% knights, −32% leg; a near-singleton collapse into a "syllogism-only" specialist.*

#### 2.4.4 Qwen2.5-7B — v2 (lr = 1e-6, 16 generations)

| Trained on | cnt | mz | sdk | pth | ari | syl | kk | leg | csr | cry |
|---|---|---|---|---|---|---|---|---|---|---|
| countdown | *−2* | +4 | +4 | 0 | 0 | +2 | +4 | −6 | 0 | 0 |
| maze | −4 | *+6* | +6 | +4 | 0 | +2 | +2 | −6 | 0 | 0 |
| mini_sudoku | −4 | 0 | *+8* | +4 | 0 | −2 | +2 | −2 | 0 | 0 |
| shortest_path | −2 | +2 | +8 | *+8* | −2 | 0 | +2 | −6 | 0 | 0 |
| basic_arith | −2 | +2 | +2 | +4 | *−4* | +2 | 0 | −2 | 0 | 0 |
| syllogism | −4 | +2 | +4 | 0 | −2 | *+8* | +2 | −4 | 0 | 0 |
| knights_knaves | −2 | 0 | +6 | +2 | 0 | −4 | *+4* | 0 | 0 | 0 |
| leg_counting | −4 | +2 | +4 | +4 | 0 | 0 | −4 | *+2* | 0 | 0 |
| caesar_cipher | −2 | 0 | +8 | 0 | 0 | +6 | +8 | −4 | *0* | 0 |
| cryptarithm | −2 | +2 | +2 | +2 | 0 | +2 | −2 | −2 | 0 | *0* |
| **Baseline** | 4% | 12% | 4% | 2% | 60% | 64% | 30% | 36% | 0% | 0% |

*v2 eliminates the catastrophic harm. The worst diagonal is −4% (arith); no off-diagonal exceeds −6%. But every gain collapses too: countdown goes from +60% (v1) to −2% (v2). The matrix is nearly flat in ±2–4%. The LR reduction reduces both signal and noise by ~3× — it prevents both learning and forgetting.*

#### 2.4.5 Qwen3-8B — v1 (lr = 5e-6, 8 generations)

| Trained on | cnt | mz | sdk | pth | ari | syl | kk | leg | csr | cry |
|---|---|---|---|---|---|---|---|---|---|---|
| countdown | *+50* | −10 | +22 | +8 | −4 | −2 | +2 | −4 | 0 | 0 |
| maze | −24 | *−10* | 0 | −8 | −8 | −22 | −30 | −40 | 0 | 0 |
| mini_sudoku | +18 | +10 | *+60* | +16 | +10 | +6 | +6 | −4 | 0 | 0 |
| shortest_path | −4 | +10 | +8 | *+36* | +2 | −8 | +6 | 0 | 0 | 0 |
| basic_arith | −14 | +4 | +14 | +4 | *−2* | −4 | −4 | −8 | 0 | 0 |
| syllogism | −16 | −6 | +6 | −2 | −2 | *+18* | −6 | −16 | 0 | 0 |
| knights_knaves | −18 | −8 | +16 | +4 | −2 | 0 | *−6* | −10 | 0 | 0 |
| leg_counting | +6 | −2 | 0 | +10 | +4 | −10 | 0 | *0* | +2 | 0 |
| caesar_cipher | −24 | −10 | +8 | +10 | 0 | −16 | +6 | −4 | *0* | 0 |
| cryptarithm | −10 | −6 | 0 | 0 | 0 | 0 | −4 | +2 | 0 | *0* |
| **Baseline** | 32% | 10% | 0% | 8% | 64% | 78% | 64% | 48% | 0% | 0% |

*The 8B v1 matrix has the second-largest single gain: mini_sudoku unlocks to +60% (0% → 60%). Constraint satisfaction is a capability threshold somewhere between 4B (sudoku +22%) and 8B. Maze's row is a corrupted training run that got 0% reward throughout; the model degrades everywhere (−22%, −30%, −40%) because GRPO with no reward signal is pure noise. Knights_knaves self-training at 8B is −6% — the SFT baseline of 64% is already near the task ceiling, and GRPO overshoots.*

#### 2.4.6 Qwen3-8B — v2 (lr = 1e-6, 16 generations)

| Trained on | cnt | mz | sdk | pth | ari | syl | kk | leg | csr | cry |
|---|---|---|---|---|---|---|---|---|---|---|
| countdown | *+4* | −2 | +6 | −8 | +2 | 0 | −12 | −2 | 0 | 0 |
| maze | 0 | *0* | +4 | −6 | +2 | 0 | −10 | −2 | 0 | 0 |
| mini_sudoku | +4 | +2 | *+2* | −2 | +4 | +4 | −6 | 0 | 0 | 0 |
| shortest_path | −4 | −4 | 0 | *−4* | +2 | +4 | −8 | −2 | 0 | 0 |
| basic_arith | +2 | +2 | 0 | −6 | *+4* | +2 | −8 | −2 | 0 | 0 |
| syllogism | +6 | −2 | 0 | −2 | +4 | *+2* | −10 | 0 | 0 | 0 |
| knights_knaves | +6 | −4 | +4 | −6 | +4 | 0 | *−14* | −2 | 0 | 0 |
| leg_counting | +2 | −4 | 0 | −6 | +4 | +2 | −18 | *0* | 0 | 0 |
| caesar_cipher | +4 | −4 | +2 | −6 | +2 | +2 | −2 | −2 | *0* | 0 |
| cryptarithm | +2 | −4 | +2 | −10 | +2 | +2 | −14 | −2 | 0 | *0* |
| **Baseline** | 22% | 10% | 0% | 12% | 62% | 74% | 74% | 52% | 0% | 0% |

*At 8B v2, arithmetic finally turns positive (+4%) — the first and only Phase-1 setting where GRPO helps arithmetic. The knights_knaves column is uniformly negative at v2: every single trained checkpoint hurts knights_knaves (−2 to −18%). KK's 74% baseline is fragile; any GRPO overshoots. Sudoku's dramatic gain at v1 (+60%) collapses to +2% at v2 — the lower LR cannot produce the large parameter movements needed to learn constraint satisfaction.*

### 2.5 LR Sensitivity Dominates Every Other Hyperparameter

The difference between lr = 5e-6 and lr = 1e-6 is larger than the difference between model sizes:

| Metric | 7B v1 (5e-6) | 7B v2 (1e-6) | 8B v1 (5e-6) | 8B v2 (1e-6) |
|---|---:|---:|---:|---:|
| Max diagonal gain | +60% | +8% | +60% | +4% |
| Worst diagonal harm | −30% | −4% | −10% | −4% |
| Avg |Δ| across matrix | 8.9% | 3.0% | 11.2% | 3.8% |
| Cells with >10% harm | 14 | 0 | 11 | 2 |

At 5e-6, GRPO is a high-variance operator: large potential gains, large potential harms. At 1e-6, it is a low-variance operator: muted gains, muted harms. The "correct" LR for a given model × task combination is not constant — it depends on how much room the SFT baseline leaves to grow (countdown at 7B: 4% baseline, benefits from aggressive 5e-6 learning) vs. how much room there is to damage (knights_knaves at 8B: 74% baseline, destroyed by aggressive 5e-6 learning).

### 2.6 Arithmetic Degrades at Every Scale

Basic arithmetic is the only task where GRPO actively degrades same-task performance at all four scales at lr = 5e-6:

| Model | Arith baseline | Arith after arith-GRPO | Δ |
|---|---:|---:|---:|
| 3B v1 | 46% | 32% | **−14%** |
| 4B v1 | 62% | 50% | **−12%** |
| 7B v1 | 60% | 30% | **−30%** |
| 8B v1 | 64% | 62% | **−2%** |

This pattern motivated Phase 2's "reasoning compression" diagnosis. Arithmetic requires *process correctness* — each intermediate multi-digit computation must be right — but the GRPO reward is binary on the final answer. The policy gradient rewards shorter outputs (they finish within token budget more often and finish correctly on the easy majority of problems), so it systematically drops intermediate work. A 7B checkpoint trained on arithmetic emits "-66 - -51 = -15" in ~70 characters instead of the 200-character explained version the SFT baseline produced. This works on single-operation problems and fails on multi-step expressions like `-3241 - 2720 - 9730 + 6743 + 1933` where intermediate tracking is required.

### 2.6.1 Tasks Where GRPO Consistently Helps/Hurts at lr=5e-6 (from report_transfer_matrix.md §5)

**Tasks where GRPO consistently helps across all sizes:**

| Task | 3B | 4B | 7B v1 | 8B v1 | Pattern |
|---|---|---|---|---|---|
| syllogism | +20% | +26% | +26% | +18% | Consistent +20% across sizes |
| shortest_path | +40% | +28% | +36% | +36% | Consistent +30–40% |
| countdown | +10% | +36% | +60% | +50% | Scales with model size |

These tasks share a common trait: clear reasoning structure that GRPO can optimize without requiring novel computation skills.

**Tasks where GRPO consistently hurts or is inert:**

| Task | 3B | 4B | 7B v1 | 8B v1 | Pattern |
|---|---|---|---|---|---|
| basic_arith | −14% | −12% | −30% | −2% | Hurts at every size |
| caesar_cipher | 0% | 0% | 0% | 0% | No signal, no learning |
| cryptarithm | 0% | 0% | 0% | 0% | No signal, no learning |

Basic arithmetic is the only task where GRPO actively degrades same-task performance: the task requires precise multi-digit computation including multiplication and division, which GRPO's binary reward cannot teach. The model's SFT-learned verbose chain-of-thought is a better strategy than anything GRPO discovers.

### 2.6.2 Transfer by Skill Category (from report_transfer_matrix.md §5.3)

**Within-category transfer:**
- Arithmetic → Arithmetic: countdown helps arith (+6% at 4B), but arith hurts itself at every scale.
- Logic → Logic: syllogism ↔ knights_knaves show positive transfer at 3B/4B, destructive at 7B.
- Spatial → Spatial: shortest_path helps sudoku (+28% at 4B), but maze is largely isolated.

**Cross-category transfer:**
- Countdown (Arithmetic) → Syllogism (Logic): consistently positive (+10% to +18%).
- Shortest_path (Spatial) → Syllogism (Logic): +16% at 4B.
- Syllogism (Logic) → Arithmetic: consistently negative (−2% to −46%).

The pattern is asymmetric: training on tasks that require search/planning improves logic, but training on logic hurts computation.

### 2.7 Positive-Transfer Clusters

Three tasks consistently help each other across multiple scales:

| Source → Target | Sign at 3B | 4B | 7B v1 | 8B v1 | Pattern |
|---|---|---|---|---|---|
| countdown → syllogism | +10 | +18 | +6 | −2 | Consistent at smaller scales |
| shortest_path → sudoku | — | +28 | −4 | +8 | Strong at 4B |
| shortest_path → syllogism | −8 | +16 | −24 | −8 | Mostly negative except 4B |

The one clean pattern is "search/planning tasks → logic tasks": training on tasks that require explicit enumeration (countdown, shortest_path) helps logical inference on related tasks. The reverse direction (logic → search) is uniformly neutral or negative. Interpretation: search tasks teach the model to produce valid inference chains as a byproduct of needing to justify search decisions; logic tasks teach pattern-matching templates that don't generalize to novel search procedures.

### 2.8 Null Results: KL/β/Rollout Count Don't Rescue Catastrophic Pairs

We tested several interventions against the catastrophic 7B-v1 arithmetic degradation (−30%):

- Adding KL penalty against SFT (`kl_coeff` = 0.01, 0.05, 0.1): reduces gain on training task proportionally but does not prevent arithmetic self-harm.
- Adding `beta` to DAPO loss (0.01, 0.05): same result.
- Doubling rollouts per prompt (8 → 16): does not change the qualitative pattern; lowers variance but not mean damage.
- Reducing LR (5e-6 → 1e-6, the v2 setting): the only intervention that prevents catastrophic harm. But it also prevents most gains.

No hyperparameter within the standard GRPO recipe rescues arithmetic at 7B v1. The mechanism (binary-reward-induced reasoning compression) is structural, not a tunable issue.

### 2.9 Phase 1 Limitations

1. Single training seed per (model, task). Training-seed variance is not characterized; eval-seed variance is small (≤3 points typical).
2. The answer-extraction bug was not found until after Phase 1; some cells were re-evaluated with the fix and are the values in §2.4; others were left unchanged where the bug produced the same answer. Inconsistencies at the 2–3% level are plausible in Phase-1 cells not re-run.
3. Caesar and cryptarithm contribute no signal (0% baseline at every scale). They remain in the matrices because they exclude "positive transfer to tasks already at 0%" as an explanation for later null results.
4. Verifier bugs in early versions of some reasoning-gym tasks (maze in particular) produced stochastic off-by-one rewards that confused training. All results here use the verifier state as of 2026-03-15.

### 2.10 Implications That Seeded Phase 2

Phase 1 raises three explicit questions for the subsequent phases:

1. **Why does GRPO fail catastrophically on some task/scale combinations?** The hypothesis is that the model cannot reliably parse the task's inputs, so GRPO's reward gradient is noise. Phase 2 tests this with atomic probes.
2. **Is the catastrophic same-task degradation of arithmetic avoidable?** Phase 2 tests atomic-only training (no full-task GRPO) as an alternative.
3. **What is GRPO actually doing to the weights in both the helpful and harmful cases?** Phase 3 characterizes τ geometrically.

---

## 3. Phase 2 — Atomic Readiness Score and the Grounding Bottleneck

### 3.1 The L1/L2/L3 Skill Decomposition

Every reasoning-gym task requires combining multiple sub-skills. Phase 2 formalizes this as a three-layer taxonomy:

- **L1 — Grounding.** Parsing the task's input representation. *"What character is at row 3, column 5?" "What is the quantifier in statement 2?" "List all numbers in the expression."* These are object-level facts about the input.
- **L2 — Local reasoning.** Single-step inference over grounded elements. No multi-step planning. *"Is moving east from (3, 5) legal?" "If all As are Bs and some Bs are Cs, does it follow that some As are Cs?" "What does (14 + 3) evaluate to?"*
- **L3 — Global planning.** Multi-step strategy composition — the full task itself.

L1 and L2 are auditable (they have unambiguous ground-truth answers); L3 is not (there are usually multiple valid solution paths). The hypothesis is that GRPO on the full task fails when L1 or L2 are unreliable, because reward gradient is noise when the model cannot even set up the problem. By training L1 and L2 separately as "atoms", we can characterize the base model's readiness and, if needed, directly remediate the grounding gap before attempting L3.

### 3.2 ARS Definition and Probe Protocol

For each task we hand-specify 4–8 atoms per layer as Q&A templates with programmatic ground-truth answers. Full templates are in Appendix C; the `environments/atomic_templates.py` file has the implementation.

The **Atomic Readiness Score** is the zero-shot accuracy of the SFT base model averaged over a task's atoms:

$$\text{ARS}(\text{model}, T) = \frac{1}{|A(T)|}\sum_{a_i \in A(T)} \mathbb{1}[\text{model}(a_i) = \text{answer}(a_i)].$$

ARS decomposes into an L1 average (grounding) and an L2 average (local reasoning). We compute both. At 50 samples per atom × 4–8 atoms × 2 layers, the ARS probe is about 1 GPU-hour per task. No training required.

The probe is format-tolerant: atomic reward functions accept multiple valid phrasings (e.g., "row 3, column 0", "3, 0", "(3, 0)") via numeric extraction, set comparison, and yes/no prefix matching. The initial version used strict string matching and produced 0% reward on atoms the model was actually answering correctly — the lesson (documented in §3.3.3) is that atomic verifiers must be more permissive than full-task verifiers.

### 3.3 Four Training Regimes

The Phase-2 study tests four regimes on each task, all at matched GRPO hyperparameters (lr = 5e-6, DAPO loss, 8 rollouts/prompt, 300 GRPO steps total where applicable):

| Regime | Training | Eval |
|---|---|---|
| Base | SFT only, no GRPO | Full task |
| Direct-RLVR | 300 steps GRPO on full task | Full task |
| Atomic-Only | 100 steps L1 atoms → 100 steps L2 atoms, no full-task GRPO | Full task (zero-shot transfer) |
| Atomic-Warm | L1(100) → L2(100) → Full(100) GRPO | Full task |

All regimes receive 300 total GRPO steps when full-task GRPO is included, or 200 when Atomic-Only. This keeps compute comparable across regimes.

### 3.4 Main Result: ARS Predicts Best Regime on 6/7 Tasks

![Figure 4.1](paper_prep/figures/fig4_1_ars_regime_comparison.png)

*Figure 4.1. Tasks sorted left-to-right by ARS (top panel). Bottom panel: full-task accuracy under each of four training regimes; the star marks the best regime per task. Low-ARS tasks (shortest_path, maze) are Atomic-Only or Atomic-Warm wins; medium-ARS (countdown) is Direct-RLVR; high-ARS (syllogism, knights_knaves) is Direct-RLVR or Atomic-Warm. basic_arithmetic (ARS = 66%, high) is the notable exception: Direct-RLVR collapses to 32% via reasoning compression.*

**Table 1: ARS predicts the optimal training regime on 6 of 7 tasks (Qwen3-4B).** Bold = best per task.

| Task | ARS | L1 Avg | L2 Avg | Base | Direct-RLVR | Atomic-Only | Atomic-Warm | Best | ARS-predicted | Correct? |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|:---:|
| syllogism | 74% | 96% | 52% | 79.5 | 96.5 | 81.0 | **97.0** | Atomic-Warm | Direct or Atomic-Warm | ✓ |
| knights_knaves | 62.5% | 57% | 68% | 78.0 | **97.5** | 79.0 | 95.0 | Direct-RLVR | Direct or Atomic-Warm | ✓ |
| basic_arithmetic | 66% | 67.5% | 65% | 72.0 | 32.0 | **78.0** | 70.0 | Atomic-Only | Direct (but arith is a known pathology) | ✗ |
| countdown | 56% | 34% | 78% | 26.0 | **~70.0** | 24.0 | 26.0 | Direct-RLVR | Direct-RLVR | ✓ |
| maze | 20.7% | 14% | 27% | 21.5 | 1.0 | **41.0** | 38.5 | Atomic-Only | Atomic-Only / Atomic-Warm | ✓ |
| shortest_path | 14% | 0% | 27% | 10.0 | 44.0 | 12.0 | **46.0** | Atomic-Warm | Atomic-Warm | ✓ |
| mini_sudoku | 19.8% | 24% | 15.5% | 4.0 | **25.5** | 17.0 | 24.0 | Direct-RLVR | Atomic-Only / Atomic-Warm | ✗ |

**Per-task regime tables with vs-Base and vs-Direct columns (from report_atomic_curriculum.md §Results).**

**basic_arithmetic (ARS = 72%):**

| Condition | Steps | Accuracy | vs Base | vs Direct |
|---|---:|---:|---:|---:|
| Base (SFT) | 0 | 72% | — | +40% |
| Direct-RLVR | 300 full | 32% | **−40%** | — |
| Stage 1 (L1 atoms) | 100 | 70% | −2% | +38% |
| **Atomic-Only (L1→L2)** | **200** | **78%** | **+6%** | **+46%** |
| Atomic-Warm (100+100+100) | 300 | 70% | −2% | +38% |
| Atomic-Warm-300 (200+300) | 500 | 54% | −18% | +22% |

Verdict: Atomic-Only is the best strategy. Full-task GRPO is always destructive — more steps = worse (78% → 70% → 54% → 32%). The model already knows most atoms; order-of-operations training (+6% from L2 stage) is the key contribution. GRPO on the full task compresses reasoning and destroys multi-step computation.

**countdown (ARS = 56%):**

| Condition | Steps | Accuracy | vs Base | vs Direct |
|---|---:|---:|---:|---:|
| Base (SFT) | 0 | 26% | — | −44% |
| Direct-RLVR (Phase 1) | 300 full | ~70% | +44% | — |
| Stage 1 (L1 atoms) | 100 | 20% | −6% | −50% |
| Atomic-Only (L1→L2) | 200 | 24% | −2% | −46% |
| Atomic-Warm (100+100+100) | 300 | 26% | 0% | −44% |

Verdict: atomic curriculum does not help. The bottleneck is L3 (multi-step search/planning), not grounding or local reasoning. L2 skills (single_op 94%, distance 90%) are already near-ceiling. Direct-RLVR is the correct strategy.

**shortest_path (ARS = 14%):**

| Condition | Steps | Accuracy | vs Base | vs Direct |
|---|---:|---:|---:|---:|
| Base (SFT) | 0 | 14% | — | −30% |
| Direct-RLVR (Phase 1) | 300 full | ~44% | +30% | — |
| Stage 1 (L1 atoms) | 100 | 18% | +4% | −26% |
| Atomic-Only (L1→L2) | 200 | 12% | −2% | −32% |
| Atomic-Warm (100+100+100) | 300 | 34% | +20% | −10% |
| **Atomic-Warm-300 (200+300)** | **500** | **46%** | **+32%** | **+2%** |

Verdict: atomic warm-start + full GRPO beats Direct-RLVR when given equal full-task compute (46% vs 44%). Fixing catastrophic L1 grounding failure (0% on all grid reading atoms) gives GRPO a better starting point. Atoms alone (Atomic-Only) fail because the task requires L3 planning that only full-task GRPO teaches.

The ARS decision rule that emerges is a three-band partition:

| ARS Range | Bottleneck | Direct-RLVR behavior | Prescribed regime |
|---|---|---|---|
| < 25% (Low) | Grounding (L1) | Catastrophic failure possible (maze: 1%) | Atomic-Only or Atomic-Warm |
| 40–60% (Medium) | L3 planning | Works (atoms won't help) | Direct-RLVR |
| > 60% (High) | Reasoning or already-solved | Usually works; Atomic-Warm marginal gain | Direct-RLVR or Atomic-Warm |

ARS prediction is correct on 6/7 tasks. The two exceptions:

- **basic_arithmetic** is ARS-high (66%) but Direct-RLVR collapses to 32%. The failure mode is not grounding but *reasoning compression* (§3.8). The rule correctly says "Direct-RLVR should work from ARS" but doesn't see the compression pathology.
- **mini_sudoku** is ARS-low (19.8%) yet Direct-RLVR wins. Our current interpretation (§3.9) is that sudoku's "atoms" don't capture the task's actual bottleneck (holistic constraint propagation) — a poor decomposition rather than a failed rule.

### 3.5 Figure 4.1 — ARS Bar Chart + Regime Comparison

Figure 4.1 (shown above) groups tasks into the three ARS bands horizontally and overlays the four training-regime accuracies as grouped bars. The visual signal is clean: low-ARS tasks (maze, shortest_path) have tall Atomic-Only/Atomic-Warm bars and near-zero Direct-RLVR bars; medium-ARS countdown has a tall Direct-RLVR bar and short atomic bars; high-ARS syllogism and knights_knaves have tall bars under both Direct-RLVR and Atomic-Warm. basic_arithmetic sits as a clear outlier — high ARS with a short Direct-RLVR bar.

### 3.6 Figure 4.2 — Per-Atom Breakdown for Maze / Arith / Syllogism

![Figure 4.2](paper_prep/figures/fig4_2_atom_breakdown.png)

*Figure 4.2. Per-atom zero-shot accuracy on the 4B SFT base for three illustrative tasks. Blue = L1 (grounding) atoms, orange = L2 (local reasoning). Maze's grounding atoms are at 0–2% — the base model cannot parse the grid, so Direct-RLVR starts from complete confusion. Arithmetic is grounded (L1 avg 68%, L2 avg 65%) yet Direct-RLVR still fails because `order_of_operations` is at 2% and full-task GRPO compresses reasoning. Syllogism has strong L1 (96%) and weaker L2 (52%); both Direct-RLVR and Atomic-Warm succeed because the remaining bottleneck is an L2/L3 skill that full-task reward can teach.*

### 3.7 Mechanism §1 — Why Direct-RLVR Fails on Low-ARS Tasks

The maze case is the clearest signal. The 4B base gets 0% on `locate_start`, 0% on `locate_goal`, 2% on `grid_size`, 54% on `read_cell`. It literally cannot answer "where is the start?" about an input it is asked to plan a path in. Under Direct-RLVR, every rollout starts from random confusion about the grid. Reward fires only when an accidentally-correct path matches ground truth. The resulting gradient teaches *format* (emit `<answer>` followed by direction-tokens) rather than *strategy*. The 300-step outcome is a degenerate policy that emits terse plausible-looking strings and gets 1% correct — actively worse than the SFT base (21.5%).

Atomic-Only reverses this. L1 training (with format-tolerant rewards on the grounding atoms) explicitly teaches grid-reading. The model acquires the substrate on which downstream reasoning can operate even without any full-task gradient signal. At the end of Atomic-Only training (200 atomic steps, zero full-task steps), maze accuracy is 41% — almost double the SFT baseline. Atomic-Warm (atoms + 100 full-task steps) is similar (38.5%).

This is the grounding bottleneck mechanism. The narrative: *sparse reward on ungrounded inputs teaches format rather than strategy; teach grounding first, and the full task becomes tractable.*

### 3.8 Mechanism §2 — Reasoning Compression on Arithmetic

Arithmetic breaks the grounding story. At 4B, arithmetic has L1 avg 67.5% and L2 avg 65% — the model does know how to extract numbers, identify operators, and evaluate single operations. Direct-RLVR should work by the grounding argument. Instead, it catastrophically fails (72% → 32%).

The mechanism, visible directly in generation logs across Phase 1 and confirmed in Phase 2, is *reasoning compression*. During GRPO training on arithmetic, the mean completion length drops from ~390 characters (SFT baseline) to ~230 characters. The policy learns to shorten outputs because:

1. Shorter outputs reach `</answer>` within token budget more reliably, collecting reward more often per unit of generation compute.
2. On easy problems (single-operation expressions like `5 + 3` or `12 × 4`), a one-line answer is correct and rewarded.
3. GRPO's advantage estimate rewards the variance reduction that shorter answers provide.

The result is a policy that *skips intermediate-step derivations*. For simple problems this works. For multi-step expressions requiring order-of-operations tracking (e.g., `(14 + 3) × 7 − 11`, or the Phase-1 test set examples like `-3241 - 2720 - 9730 + 6743 + 1933`), the compressed output makes arithmetic errors the explicit chain-of-thought would not have made. The `order_of_operations` atom (which explicitly requires identifying which sub-expression to evaluate first) is at 2% base accuracy — this is the gap that full-task GRPO fails to close because the compression gradient is stronger than the few-problems-hurt-by-skipping signal.

Atomic-Only avoids this failure by never training on the full task. The L2 atoms (`order_of_operations`, `single_op_eval`) reward producing the *correct substep*, not being concise. The resulting model at 78% beats the base and the Direct-RLVR model.

Similar compression was observed on knights_knaves under some conditions, though KK's 74% baseline is less sensitive.

### 3.9 Mini-Sudoku Exception

Mini-sudoku breaks the ARS rule in the opposite direction. ARS is very low (19.8%; L1 24%, L2 15.5%) — by the rule, Atomic-Only or Atomic-Warm should win. In fact, Direct-RLVR wins at 25.5% vs Atomic-Only's 17%.

Three candidate explanations, none currently confirmed:

1. **Poor atomic decomposition.** Sudoku's true bottleneck is constraint propagation across cells — if you fill cell (1, 1) with 3, cells (1, 2) through (1, 4) can no longer be 3, and this chain propagates. The atoms we chose (`read_cell`, `count_empty`, `row_values`, `col_values`, `missing_in_row`, `can_place`, `only_option`) test grounding and isolated constraint-checking but not propagation. The model's 20% L2 on `can_place` reflects its ability to check one placement; the full task needs to *combine* checks across multiple cells. The atoms can't teach combination.
2. **Task structure mismatch.** 4×4 sudoku is small enough that the full solution space is tractable by brute-force-like search. GRPO can learn the search policy directly without scaffolding. Larger (9×9) sudoku would likely need better atoms.
3. **Reward function differences.** The sudoku reward rewards partial-grid correctness, so even a mostly-wrong solution gets some signal. This is richer than maze's binary reward.

The current view is explanation (1) is most likely. Sudoku is a task whose atoms *we picked wrong*; the rule is not falsified, just not applicable.

### 3.10 Shortest-Path Deep Dive (15 Conditions)

Shortest_path received the most extensive experimental attention in Phase 2 because it is the hardest low-ARS task (ARS = 14%, L1 = 0%, L2 = 27%) and the one where the cleanest mechanistic interventions can be run. The 15-condition deep dive produced the project's best single-task result and two methodological insights that transfer to other hard tasks.

The first method extension was that short-answer L2 atoms (like `is_move_legal` → "Yes") caused GRPO to compress to ultra-minimal outputs and fail the full task. The fix was **L2-long atoms** — path-tracing atoms whose answers are multi-step simulations that cannot be compressed without losing correctness:

- `sp_execute_path`: "Start at (3, 1), execute: up, right, down. Report each step." Answer: "Move up from (3, 1) to (2, 1) — cell is O; Move right from (2, 1) to (2, 2) — cell is X, blocked; Final position: row 2, column 1."
- `sp_trace_and_report`: similar multi-step path simulation with wall-collision detection
- `sp_count_walls_on_path`: scan a row or column and list wall cells in order

These exercise exactly the skill needed for the full task (step-by-step path execution) without allowing compression. Under "Idea B" (L2-long atoms → Full task, 100 atom steps + 200 full-task steps), shortest_path reaches **51.5%** — the first condition to break 50%.

The final best result was **RFT-Base** (55.5%): generate verified-correct traces from Idea B's 51.5% checkpoint, SFT the *base* SFT model on those 328 samples (lr = 1e-6, 1 epoch) to get to 15.5%, then GRPO 300 steps to reach 55.5%. The mechanism: the RFT traces are high-quality demonstrations from a model that already learned path-tracing, so the base model starts GRPO with mean reward 0.45 (vs. Idea B's stage-2 start near 0), getting much better gradient signal throughout training.

**Full table of the 15+ conditions** (all at 4B, Qwen3-4B SFT base, eval with seed 9999):

| # | Condition | Full-task steps | Atom steps | Total steps | Accuracy |
|---|---|---:|---:|---:|---:|
| 1 | Base (SFT) | 0 | 0 | 0 | 10% |
| 2 | Direct-RLVR (Phase 1) | 300 | 0 | 300 | 44% |
| 3 | GRPO Atomic-Only (short L2) | 0 | 200 | 200 | 12% |
| 4 | GRPO Atomic-Warm | 100 | 200 | 300 | 34% |
| 5 | GRPO Atomic-Warm-300 | 300 | 200 | 500 | 46% |
| 6 | SFT-L1 (high LR 2e-5) → Full-300 | 300 | SFT | — | 2% |
| 7 | SFT-L1 → GRPO-L2 → Full-300 | 300 | SFT + 100 | — | 40% |
| 8 | SFT-L1 (low LR 5e-6) → Full-300 | 300 | SFT | — | 40% |
| 9 | Progressive atoms (3×3 → 4×4 → Full) | 300 | 200 | 500 | 40% |
| 10 (A) | Progressive full-task | 300 | 0 | 300 | 46.5% |
| 11 (B) | L2-long atoms → full | 200 | 100 | 300 | **51.5%** |
| 12 (B2) | Idea B extended (+300 steps) | 500 | 100 | 600 | 47% |
| 13 (C) | Partial-credit reward shaping | 300 | 0 | 300 | 35.5% |
| 14 (D) | Progressive + partial credit | 300 | 0 | 300 | 24% |
| 15 (E) | L2-long atoms → progressive full | 200 | 100 | 300 | 27.5% |
| 16 (F) | Feasibility atoms → full | 200 | 100 | 300 | 42% |
| 17 (G) | Heavy GRPO+SFT interleaving | 150 | — | — | 5.5% |
| 18 (G2) | Light GRPO+SFT interleaving | 150 | — | — | 40% |
| 19 (G3) | G2 + 150 GRPO | 300 | — | — | 45% |
| 20 (L) | Feasible-only training distribution | 300 | 0 | 300 | 40.5% |
| 21 | RFT on B's own checkpoint | 400 | — | — | 51.5% |
| 22 | **RFT-Base (best)** | 300 | — | — | **55.5%** |

**Error analysis on Idea B's 51.5% checkpoint (100 eval samples, from report_atomic_curriculum.md Appendix D):**

| Error type | Count | Description |
|---|---:|---|
| False infeasible | 23 | Model says "infeasible" when path exists |
| Wrong path | 25 | Model gives incorrect directions |
| Missed infeasible | 6 | Model gives path when none exists |

If all false-infeasible errors were fixed, accuracy would reach 68.3%. The model exhibits a "give-up reflex" — defaulting to "infeasible" rather than attempting a path trace. Breakdown by problem class:

- Feasible accuracy: 25% (16/64 feasible problems correct)
- Infeasible accuracy: 83% (30/36 infeasible problems correct)

**Idea G / G2 / G3 interleaving-cycle dynamics (from report_atomic_curriculum.md Appendix C):**

| Variant | SFT Samples | SFT LR | Cycle 1 | Cycle 2 | Cycle 3 | Final |
|---|---:|---:|---:|---:|---:|---|
| G (heavy SFT) | 5000 | 5e-6 | 21% | 9.5% | 5.5% | **collapsed** |
| G2 (light SFT) | 500 | 1e-6 | 15.5% | 27% | 40% | improving |

G2 extended to cycles 4–5 showed the improving trend plateauing and reversing (cycle 4: 38.5%, cycle 5: 34%). G3 added 150 GRPO steps on top of G2's cycle-3 checkpoint (40%) to reach 300 total GRPO steps: final 45%, matching Direct-RLVR but not beating Idea B.

**RFT on B's own checkpoint vs RFT-Base contrast (from report_atomic_curriculum.md Appendix E):**

- Generated 328 verified-correct samples from B's 51.5% model (210/300 problems had ≥1 correct; avg pass rate 70%)
- RFT on B itself: SFT of B → 49.5% (slight dip); GRPO 100 steps → 51.5% (recovered to same level). *On-policy RFT on an already-trained model just recovers to the same equilibrium. No new capability added.*
- RFT-Base: SFT the base SFT model on same 328 samples (lr = 1e-6, 1 epoch) → **15.5%** (up from 10% base); GRPO 300 steps → **55.5%**. The reward-trajectory difference: RFT-Base started at mean reward 0.45 (vs B's stage-2 starting near 0) and maintained 0.48+ throughout, meaning more gradient signal per step.

**SFT-for-L1 ablation (from report_atomic_curriculum.md §Appendix: SFT-for-L1 Experiment):**

| Condition | Full-task accuracy | L1 ARS |
|---|---:|---:|
| Base | 14% | 19.4% |
| SFT-L1 (high LR 2e-5) | 10% | **65.2%** |
| SFT-L1 → GRPO-L2 | 2% | — |
| SFT-L1 → GRPO-L2 → GRPO-Full-300 | 40% | — |
| SFT-L1 → GRPO-Full-300 (skip L2) | 2% | — |

**Per-atom ARS improvement from SFT-L1:**

| Atom | Before | After SFT | Delta |
|---|---:|---:|---:|
| locate_start | 0% | 92% | +92% |
| locate_goal | 0% | 88% | +88% |
| grid_dimensions | 0% | 100% | +100% |
| direction_to_goal | 28% | 78% | +50% |
| **L1 avg** | **11.5%** | **83.0%** | **+71.5%** |

SFT teaches grounding without reasoning compression (avg response length stayed at 235 chars), but the resulting model is fragile under subsequent GRPO — it overfit to atom format and couldn't withstand aggressive full-task GRPO fine-tuning.

Key methodological takeaways:

- **Hard transitions beat progressive.** One hard jump from skill pretraining to full difficulty (Idea B: atoms → full-size grids) beats every gradual curriculum (Idea A, E: easy → medium → hard grids). Interpretation: the difficulty jump forces generalization rather than per-difficulty memorization.
- **Partial credit hurts GRPO.** Ideas C and D gave graduated rewards (partial credit for valid paths, full credit for optimal). Accuracy dropped to 35.5% and 24%. Partial credit reduces the advantage-variance GRPO needs — with mean reward 0.9 throughout, there is no gradient signal between "good enough" and "correct."
- **Heavy SFT interleaving destroys GRPO gains.** Idea G (5000 SFT samples at lr=5e-6 every 50 GRPO steps) collapsed from 21% to 5.5%. Light SFT (Idea G2: 500 samples at 1e-6) preserved gains and improved monotonically for 3 cycles before plateauing.
- **Short-answer atoms cause reasoning compression.** Confirmed by the 46% → 51.5% gap between short-L2 atoms and L2-long atoms.

### 3.11 Cross-Task Preservation (SP RFT-Base)

The best shortest-path checkpoint preserved performance on other tasks:

| Task | Base (SFT) | RFT-Base (55.5% SP) | Δ |
|---|---:|---:|---:|
| shortest_path | 10% | 55% | +45% |
| basic_arithmetic | 72% | 70% | −2% (within noise) |
| countdown | 26% | 32% | +6% (positive transfer) |

Positive transfer to countdown is consistent with Phase 1's "search/planning → logic" pattern. Arithmetic preservation (within noise) is a useful negative result: SP training did not trigger the reasoning compression that doomed arithmetic self-training in Phase 1. The mechanism is likely that SP's long multi-step outputs (direction sequences, path traces) reward longer generations, counteracting the compression gradient.

### 3.12 Phase 2 Limitations

1. **Single seed.** Every condition run with seed = 42. Training-seed variance is not characterized.
2. **Qwen3-4B only.** Extended atomic-curriculum experiments at 3B-Qwen2.5 and 8B-Qwen3 (Phase 4 Part 5 F28) used the same 100-step L1 + 100-step L2 setup but without RFT-Base refinement. Matched-compute atomic curricula across scales is a 1–2 day re-run that has not been done.
3. **mini_sudoku atoms need redesign.** The current decomposition doesn't capture constraint propagation. Alternatives (candidate-tracking atoms, constraint-violation atoms) are proposed but untested.
4. **ARS is a noisy binary probe.** 50 samples per atom gives ±7% confidence intervals. Pathological tasks with 0% or 100% accuracy at many atoms are measured reliably; tasks in the 20–80% range are noisier.
5. **The rule is empirical.** We hand-selected three bands based on observed behavior; there is no theoretical derivation of why ARS 25% / 60% should be the thresholds.

---

## 4. Phase 3 — Weight-Space Geometry of GRPO

### 4.1 What τ Looks Like: A Geometric Primer

For each (scale, task) pair we have two checkpoints: W_SFT and W_GRPO. The **task vector** is

$$\tau \;=\; W_{\mathrm{GRPO}} - W_{\mathrm{SFT}},$$

computed per weight matrix (attention projections, MLP projections, embedding tables, etc.). τ is a tensor with the same shape as W, representing only the delta that GRPO learned.

To characterize τ we use four geometric objects throughout this chapter:

**The Frobenius magnitude ‖τ‖_F** is the sqrt-sum-squared of all entries of τ. For a weight matrix this is the total length of τ treated as a vector. We normalize by ‖W‖_F to get relative magnitudes (‖τ‖_F / ‖W‖_F) that compare directly across scales.

**The SVD W = U Σ V^T** gives W a canonical basis. The columns of U are the "output directions" W uses; the columns of V are the "input directions"; the singular values σ₁ > σ₂ > ... tell you how strongly each direction is amplified. A k-dim "top subspace" is the first k columns — the k preferred directions W uses most heavily.

**The two-sided projection ratio** ρ_k = ‖U_k U_k^T τ V_k V_k^T‖² / ‖τ‖² quantifies what fraction of τ's length lies in W's top-k directions. ρ_k → 1 means τ lives entirely in W's top directions (GRPO is amplifying existing features); ρ_k → 0 means τ is perpendicular to them (GRPO is adding new features in underused directions); ρ_k ≈ (k/r)² is the isotropic-random baseline where r is the ambient dimension.

**Principal angles** between two k-dim subspaces measure how much one has rotated relative to the other. If U_SFT and U_GRPO are the top-k left singular bases of W_SFT and W_GRPO, the principal angles arccos(σ_i(U_SFT^T U_GRPO)) answer "how much did the filing cabinet of preferred output directions tilt?" 0° = identical bases; 90° = completely rotated; small values (1–3°) mean a small but structured tilt. The **novel subspace** novel_A = (I − U_SFT U_SFT^T) U_GRPO(A)[:, :k] is the component of task A's top-k basis that's outside SFT's top-k — "the new directions A introduced."

With these four objects we can decompose GRPO's weight change into five questions: how big is τ? (‖τ‖_F), does τ amplify W's dominant features? (ρ_k), does it rotate the basis? (principal angles), where does each task rotate? (novel subspace), and where in the model does the change concentrate? (depth signature — the per-layer distribution of ‖τ‖_F). The remaining sections answer each.

**Why a 2° rotation of a 100-dim subspace matters.** A common intuition is that "only 2° of rotation" is negligible. It is not. Rotating a 100-dim subspace by 2° inside a 4096-dim ambient space means each of the 100 basis vectors tilts 2° away from its SFT position, and they tilt in correlated task-specific ways. The effective "new subspace volume" the rotation covers is roughly sin²(2°) × 100 ≈ 0.12 dimensions of fully novel direction *per output unit*, times the full input-dim of 4096, giving ~500 dimensions of genuinely new feature expression. The rotation is small per-direction but meaningful in aggregate.

### 4.1.1 Task-Vector Magnitudes and Relative Change (from report_phase3_task_distances.md §2.1–§2.2)

Before turning to spectra and rotation, we report the raw Frobenius sizes of τ.

**Task-vector magnitudes at 4B (per task, Frobenius):**

| Task | ‖τ‖_F | Notes |
|---|---:|---|
| leg_counting | 1.14 | Largest regular change |
| maze | 0.97 | |
| knights_knaves | 0.97 | |
| shortest_path | 0.95 | |
| caesar_cipher | 0.88 | |
| basic_arithmetic | 0.83 | |
| syllogism | 0.73 | |
| mini_sudoku | 0.71 | Smallest change |
| cryptarithm | 0.00 | **No learning** (no reward signal) |
| countdown | *pending re-run* | Original used wrong base model — see §4.9 |

Most tasks produce changes of similar Frobenius magnitude (0.7–1.1), suggesting GRPO applies roughly uniform pressure regardless of task difficulty. Cryptarithm received no reward signal and never moved — a clean negative control.

**Relative change ‖τ‖/‖W‖ per layer across all four scales:**

| Scale | Mean ‖τ‖/‖W‖ per layer | Global ‖τ‖ range across tasks |
|---|---:|---|
| 3B | 0.071% | 0.73–0.87 |
| 4B | 0.065% | 0.71–1.14 |
| 7B | **0.117%** | 1.21–1.58 |
| 8B | 0.059% | 0.98–1.70 |

GRPO's per-layer weight change is tiny in relative terms (<0.12% of the weight matrix's norm per layer). 7B is a mild outlier with nearly 2× the relative change of 8B.

### 4.2 Result 1 — The Spectrum of W Does Not Move

![Figure 5.1](paper_prep/figures/fig5_1_spectrum_preserved.png)

*Figure 5.1. (a) Spectral-norm ratio σ₁(W_GRPO)/σ₁(W_SFT) per task at 8B. Every task sits at the reference line 1.0 ± 0.002. (b) Per-band relative singular-value shift |σᵢᴳᴿᴾᴼ − σᵢˢᶠᵀ| / σᵢˢᶠᵀ across 12 probe layers × 5 tasks. Every band sits an order of magnitude below the 0.1% reference.*

For each of the top-20 most-changed layers per task at 8B, we compared σ(W_GRPO) vs σ(W_SFT):

| Task | Spectral-norm ratio | Effective-rank change |
|---|---:|---:|
| countdown | 1.0004 | −0.004 |
| mini_sudoku | 0.9999 | +0.001 |
| shortest_path | 1.0003 | −0.002 |
| basic_arithmetic | 0.9991 | −0.000 |
| syllogism | 1.0003 | −0.003 |
| knights_knaves | 0.9998 | +0.002 |
| leg_counting | 1.0000 | +0.000 |
| caesar_cipher | 1.0018 | −0.001 |
| cryptarithm | 1.0001 | −0.009 |

Every entry is within ±0.002 of 1.0 (spectral norm) and ±0.01 of 0 (effective rank). Extending to the full spectrum (all singular values, not just σ₁) across 12 representative layers × 5 tasks:

| Normalized rank band | Mean relative shift | Max (excluding tail-noise) |
|---|---|---|
| 0–1% (top) | ~0.003% | 0.04% |
| 1–10% | ~0.003% | 0.06% |
| 10–50% | ~0.002% | 0.06% |
| 50–100% (tail) | noise from near-zero σ | — |

Every band shifts by less than 0.1%, below the bfloat16 precision floor. **GRPO does not restructure W; its entire action has to live in a direction the singular-value spectrum cannot see.**

#### 4.2.1 τ's own spectrum is moderately low-rank (from report_phase3_task_distances.md §3.3)

While σ(W) doesn't move, τ (= W_GRPO − W_SFT) has its own structure. Top-10 singular values of τ per task (4B, averaged across `mlp.down_proj` layers):

| Task | top-1 σ(τ) | decay (σ_1 / σ_10) |
|---|---:|---:|
| maze | 0.018 | 4.2× |
| knights_knaves | 0.017 | 3.7× |
| caesar_cipher | 0.016 | 4.0× |
| shortest_path | 0.014 | 3.7× |
| basic_arithmetic | 0.014 | 3.7× |
| leg_counting | 0.014 | 3.0× |
| syllogism | 0.011 | 3.2× |
| mini_sudoku | 0.006 | 2.5× |
| cryptarithm | 0.000 | NaN |

τ is **moderately low-rank** (3–4× decay across the top 10 singular values) — it has a preferred direction, but its magnitude is small relative to W's. Since σ(W) does not change but behavior does, the change must live in directions SVD of W alone cannot see — motivating the subspace and rotation analyses in §4.3–§4.5.

### 4.3 Result 2 — τ Is Nearly Orthogonal to W's Top Subspace

![Figure 5.2](paper_prep/figures/fig5_2_tau_orthogonal.png)

*Figure 5.2. Two-sided projection ratio ρ_k of τ onto W_SFT's top-k singular subspace, for k ∈ {10, 100, 500, 1000} at four scales. The dashed line is the random-isotropic baseline (k/r)² at r = 2560 (4B ambient rank) for reference. At 4B and 8B ρ_k sits at or below baseline — τ actively avoids the directions W was already using.*

**Table 2. Two-sided projection ratio by scale.**

| k | 3B | 4B | 7B | 8B | Random (k/r)² |
|---|---:|---:|---:|---:|---:|
| 1 | 0.00% | 0.00% | 0.00% | 0.00% | ~0% |
| 10 | 0.03% | 0.01% | 0.001% | 0.01% | ~0.001% |
| 100 | 0.34% | 0.16% | 0.04% | 0.14% | ~0.06% |
| 500 | 2.88% | 1.55% | 0.69% | 1.02% | ~1.5% |
| 1000 | 9.17% | 5.32% | 2.59% | 3.20% | ~6% |

At 3B, τ has 2–5× the random-baseline alignment with W's top directions (some mild mass in directions W was already using). At 4B and 8B, ρ ≈ baseline — τ is orthogonal. At 7B, ρ is *below* baseline — τ actively avoids W's top subspace. As scale grows, τ gets more orthogonal to W.

**Interpretation.** GRPO does not rescale the features W was already using. It adds updates in directions W was mostly *not* using — in the model's underexpressed (mid/low-rank) subspace. This orthogonality-to-W is a precondition for mergeability (Chapter 5): two task vectors τ_A and τ_B from different tasks mostly occupy different underused corners of the same underexpressed subspace, so summing them does not collide at the high-importance directions W depends on.

#### 4.3.1 All-layer principal-angle table (from report_phase3_task_distances.md §4.2)

Before narrowing to the layer-type split, Phase 3 computed principal angles averaged over every layer, for several k values:

**Table 2b. Principal angle arccos(σ_i(U_SFT^T U_GRPO)), mean over all layers × tasks, degrees.**

| k | 3B | 4B | 7B | 8B |
|---|---:|---:|---:|---:|
| 1 | 0.77° | 1.03° | 0.79° | 0.92° |
| 10 | 0.68° | 0.87° | 1.02° | 0.97° |
| 100 | 0.77° | 0.97° | 1.27° | 1.24° |
| 500 | 0.89° | 1.05° | 1.33° | 1.34° |
| 1000 | 0.68° | 0.84° | 1.08° | 1.12° |

Basis rotation grows modestly with scale (0.8° at 3B → 1.3° at 7B/8B) and saturates around 1.3° at the 7B–8B range. Averaging across layers hides the sharp layer-type split that follows in §4.4.

### 4.4 Result 3 — The Universal Layer-Type Split

![Figure 5.3](paper_prep/figures/fig5_3_layer_type_rotation.png)

*Figure 5.3. Principal angle at k = 100 (mean over tasks × layers of that type) by layer type and scale. `mlp.down_proj` (gray) is at 0°–0.02° at every scale. `mlp.up_proj` (blue) and `attn.q_proj` (orange) both rotate and grow monotonically from 3B (1.47° / 1.65°) to 8B (2.67° / 2.07°). The layer-type split is the structural signature of GRPO's action.*

**Table 3. Basis rotation at k = 100 (mean over tasks × layers of that type), degrees.**

| Layer type | 3B | 4B | 7B | 8B |
|---|---:|---:|---:|---:|
| `mlp.down_proj` | **0.01°** | **0.02°** | **0.00°** | **0.00°** |
| `mlp.up_proj` | 1.47° | 1.99° | 2.49° | 2.67° |
| `attn.q_proj` | 1.65° | 1.81° | 2.03° | 2.07° |

`mlp.down_proj` is **frozen**. At every scale — 3B Qwen2.5, 4B Qwen3, 7B Qwen2.5, 8B Qwen3 — across two families, GRPO rotates its left singular basis by effectively zero. The entire rotation action is in `mlp.up_proj` and `attn.q_proj`, and it grows monotonically with scale: 1.47° / 1.65° at 3B → 2.67° / 2.07° at 8B.

**Mechanistic reading.** `mlp.up_proj` and `attn.q_proj` are *feature projectors*. They decide what features to produce (up_proj) or what to query for (q_proj). These are the natural sites for task-specific feature learning, and they rotate by a small but measurable amount. `mlp.down_proj` is a *sink*: it aggregates contributions from all MLP neurons back into the residual stream. Its preferred output directions are the ones the whole transformer uses most heavily — shifting them would break downstream computation. So GRPO leaves this basis alone.

Crucial distinction: `mlp.down_proj`'s basis does not rotate, but τ still has nonzero mass in down layers. GRPO moves *coordinates within the existing basis* rather than changing the basis itself. In feature-projector layers, GRPO does both — it moves coordinates and tilts the basis by ~2°.

This finding becomes the actionable merge recipe in Chapter 5: because `mlp.down_proj`'s basis is frozen, dropping its τ contributions from a merge cannot destroy task-distinctive structure that lives only there — because no such structure exists. §5.4 verifies empirically that `all_but_down` matches or beats the full merge on 40/48 (pair, task) cells at 4B.

### 4.5 Result 4 — Rotation Amount Is Task-Invariant; Direction Is Task-Specific

Across every task we tested, at any fixed scale and layer type, the per-task rotation magnitude varies by less than 0.1°. At 4B in `mlp.up_proj`: arithmetic 1.99°, caesar 2.00°, syllogism 1.98°, countdown 1.99°. You might expect arithmetic and logic to reshape the model by different amounts — they do not. Every task spends the same "rotation budget," which depends only on model scale and layer type.

| Scale | Task | ‖τ‖/‖W‖ | `mlp_down` | `mlp_up` | `attn_q` | Overall |
|---|---|---:|---:|---:|---:|---:|
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

If the rotation *amount* is nearly identical across tasks, all of the task-specificity has to live in *which direction they rotated*. To measure direction we compute the **novel subspace** novel_A for each task (see §4.1), then take pairwise principal angles between novel_A and novel_B for different tasks. 0° = tasks rotated in the same direction; 90° = orthogonal directions.

![Figure 5.4](paper_prep/figures/fig5_4_novel_subspace_angles.png)

*Figure 5.4. Pairwise principal angle between novel subspaces at 8B, k = 100, for four tasks (countdown, basic_arithmetic, caesar_cipher, syllogism). One heatmap per layer type; rows = probe layers, columns = task pairs. `mlp.down_proj` (left, mean 24°) is the least task-divergent; `attn.q_proj` (right, mean 43°) is the most.*

**Table 4. Pairwise angle between novel subspaces at 8B, k = 100** (mean over 4 tasks × 12 probe layers):

| Layer type | Mean pairwise angle | Range | Interpretation |
|---|---:|---|---|
| `mlp.down_proj` | 24° | 17.7°–30.5° | Novel subspace tiny, directions consistent |
| `mlp.up_proj` | 39° | 34.1°–43.3° | Tasks rotate into distinct feature subspaces |
| `attn.q_proj` | 43° | 37.9°–46.6° | Most task-divergent |

Per-layer per-task-pair angles (full table at 8B, k = 100):

| Layer | cnt×ari | cnt×csr | cnt×syl | ari×csr | ari×syl | csr×syl |
|---|---:|---:|---:|---:|---:|---:|
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

**A subtlety about orientation.** A subspace has no inherent orientation (the line {t · v : t ∈ ℝ} is the same subspace as {t · (−v) : t ∈ ℝ}). Two tasks could want to rotate in exactly opposite directions along the same axis, and the principal-angle-between-subspaces measurement would report 0° — the subspaces coincide even though the tasks pull opposite. To capture direction-within-a-subspace we also compute a **flat cosine**: vectorize novel_A and novel_B and take their cosine. This preserves sign.

Flat cosines at 8B across task pairs span the full ±0.9 range across layer × pair combinations. Some layers are "battlegrounds" where one task's +direction is another task's −direction (e.g., countdown ↔ basic_arithmetic at L6.mlp.up_proj: flat cos = −0.90). Other layers are largely aligned (countdown ↔ syllogism at L9.mlp.up_proj: flat cos = +0.89). Aggregate across layer type:

| Layer type | Mean flat cosine (all task pairs) |
|---|---:|
| `mlp.down_proj` | +0.017 |
| `mlp.up_proj` | −0.036 |
| `attn.q_proj` | −0.034 |

Aggregate means are near zero (tasks spread out directionally across the whole model) but the per-layer spread ±0.9 is the mechanistic signature. These strongly-negative-flat-cosine cells are the microscopic candidate mechanism for catastrophic interference — in principle, summing τ_A and τ_B at such a cell actively cancels both tasks' learning. (Chapter 5 will test this as a merge-damage predictor; it fails at r ≈ 0.)

#### 4.5.1 Whole-model task-vector similarity (from report_phase3_task_distances.md §5.5)

For complete record, whole-model cosine similarities of flattened τ at 8B (top 8 off-diagonal pairs):

| Pair | Cosine similarity |
|---|---:|
| syllogism ↔ knights_knaves | 0.083 |
| basic_arithmetic ↔ syllogism | 0.065 |
| basic_arithmetic ↔ leg_counting | 0.065 |
| knights_knaves ↔ leg_counting | 0.056 |
| countdown ↔ syllogism | 0.056 |

All pairs are < 0.1. Whole-model similarity averages over too much orthogonality to be informative.

#### 4.5.2 Per-layer cross-task τ cosine and basis angle (from report_phase3_task_distances.md §5.6)

For each layer, the cosine of τ between task pairs, and the principal angle between their top-100 U_GRPO subspaces:

| Layer type | τ cosine across tasks | Basis angle between task pairs |
|---|---|---|
| `mlp.down_proj` | 0.02–0.05 | **0.00°** (shared basis, different coordinates) |
| `mlp.up_proj` | 0.03–0.09 | **2.5–3.0°** (each task rotates basis differently) |
| `attn.q_proj` | 0.01–0.08 (sometimes negative) | 1.8–2.2° (most task-specific; some pairs oppose) |

Crucially, the basis angle *between task pairs* (2–3°) is nearly identical to each task's angle *from the SFT base* (2–3° in §4.2). **Task-specific bases are not clustered near each other** — each task rotated into a different direction, roughly equidistant from each other.

**Taken together (Results 2–4).** GRPO's action has a universal shape — a small (‖τ‖/‖W‖ ≈ 0.06–0.12%), near-orthogonal-to-W, layer-type-localized basis rotation concentrated in feature projectors. The *amount* is governed by scale and layer type. The *direction* is where the task specializes. This is the "how vs where" split that Chapter 4 is organized around.

### 4.6 Result 5 — Depth Signatures Predict Transfer Better Than Global Cosine

Whole-model cosine similarity of τ_A and τ_B is a weak transfer predictor. The 4B Phase-1 transfer matrix gives:

| Metric | Pearson r | p-value |
|---|---|---|
| cos(τ_A, τ_B) vs transfer Δ | 0.323 | 0.0035 |
| Spearman | 0.147 | 0.19 |

Only about 10% of transfer variance is explained by whole-model cosine. The catastrophic Phase-1 pairs (mini_sudoku → knights_knaves = −50%, mini_sudoku → leg_counting = −52%) have τ cosines near zero. Global cosine averages over the whole model, smearing out the localized conflict where it actually lives.

**Depth signatures** are the per-layer distribution of change mass ‖τ‖_F across layer indices, normalized to sum to 1. A transformer is a hierarchical feature extractor: early layers (L0–L8) handle token-level surface features, mid-early (L9–L17) build local relational and logical features, mid-late (L18–L26) run multi-step computation and global search, late (L27–L35) format and emit output. A task concentrates GRPO change at the depth where its bottleneck sits.

![Figure 5.5](paper_prep/figures/fig5_5_depth_signatures_8b.png)

*Figure 5.5. 8B depth-quartile concentration per task — percent of top-20 layer change mass in each of four depth bands. Three clean clusters (black horizontal separators): early-only (caesar_cipher), mid-early (logic / local-reasoning), mid-late (arithmetic / global search).*

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

Three clean clusters, each matching the task's mechanistic demand:

- **caesar_cipher at 88% early.** Caesar is a per-token substitution rule. The bottleneck is token-level parsing and rewrite — no multi-step reasoning needed. 88% in layers 0–8 is exactly where a token-substitution rule should be learned.
- **syllogism, knights_knaves, leg_counting, shortest_path at 52–62% mid-early.** These are local-reasoning tasks — the model has to form relations ("A is a knight so A tells the truth", "cell (2, 3) is a wall") and chain single-hop inferences. Mid-early layers are where local relational reasoning happens.
- **basic_arithmetic, countdown, mini_sudoku at 39–53% mid-late.** These are multi-step computation / search tasks. Arithmetic needs order-of-operations across a long expression; countdown searches combinations; sudoku propagates constraints. Those computations live in mid-late layers.

**Why depth signatures explain Phase-1 transfer failures.** Consider the worst Phase-1 pair: mini_sudoku → leg_counting (transfer = −52%). cos(τ_sdk, τ_leg) ≈ 0 — globally unrelated τ's. Yet mini_sudoku concentrates 39% of its mass in mid-late layers and 46% in early layers, while leg_counting concentrates 56% in mid-early and 40% in early. The two tasks share mass in early and adjacent mid bands, and mini_sudoku's mid-late rewrites sit *downstream* of the mid-early pathway leg_counting depends on. Corrupting the downstream pathway breaks the task even though the direct modifications are at different depths. Depth-profile cosine between mini_sudoku and leg_counting is 0.33. Per-layer novel-subspace flat cosines at shared-mass layers are strongly negative.

Depth-profile similarity is a stronger interference predictor than whole-model τ cosine because it captures localized conflict where the global cosine averages it away.

### 4.7 Result 6 — Depth Signatures Reorganize Across Scale

![Figure 5.6](paper_prep/figures/fig5_6_scale_depth_migration.png)

*Figure 5.6. basic_arithmetic change-mass distribution across depth bands at 3B, 4B, 8B. At 3B: 87% early; at 4B: majority mid-early; at 8B: majority mid-late. The task migrates toward deeper layers as scale grows.*

Where a task concentrates its edits is scale-dependent. Cross-scale depth concentration for all tasks:

| Task | 3B | 4B | 7B | 8B |
|---|---|---|---|---|
| basic_arithmetic | early (87%) | mid-early (57%) | early + mid-early (41 + 59%) | mid-late (47%) |
| syllogism | early (83%) | mid-early (95%) | early + mid-early (47 + 53%) | mid-early (55%) |
| caesar_cipher | early (88%) | early (54%) + mid-late (31%) | early (72%) | early-only (88%) |
| countdown | mid-late (53%) | *contaminated* | mid-late (64%) | mid-late (53%) |
| shortest_path | mixed | early (86%) | **late (60%)** | mid-early (62%) |
| mini_sudoku | mid-late (65%) | mid-early (75%) | **late (64%)** | early + mid-late split |
| knights_knaves | early (64%) | mid (63%) | mid-early (69%) | mid (51%) |
| leg_counting | early + mid-late | mid + mid-late | early + late | mid-early (56%) |
| maze | early (63%) + late (23%) | early (83%) | early (62%) | (skipped at 8B) |

Bolded 7B entries are unique — no other scale localizes those tasks to late layers. 3B is an "early super-cluster" scale (most tasks early-dominated). 8B separates into the three clean bands above. 4B is intermediate.

**Family effect.** 3B and 7B are Qwen2.5; 4B and 8B are Qwen3. Some apparent scale dependencies are family effects. basic_arithmetic stays in early/mid-early layers in Qwen2.5 (3B, 7B) but migrates to mid-late in Qwen3-8B. 7B has the highest relative ‖τ‖/‖W‖ (0.117% vs 0.06%) and lowest projection onto W's top-k. The rotation-growth trend (3B → 8B, 1.5° → 2.7°) is consistent across families; the depth-concentration shift is family-specific.

### 4.8 The "How vs Where" Synthesis

The Phase-3 findings decompose cleanly into a universal "how" and a task/scale-specific "where":

**Universal (across scale, family, task):**

- GRPO is spectrum-preserving (σ shifts < 0.1%). The entire action lives outside W's SVD.
- τ is near-orthogonal to W's top singular subspace (at 4B and 8B, ρ_k ≤ random-isotropic baseline).
- `mlp.down_proj` basis is frozen at every scale (0.00°–0.02° principal angle).
- `mlp.up_proj` and `attn.q_proj` bases rotate, growing monotonically 1.5° → 2.7° from 3B to 8B.
- Rotation amount per task varies by < 0.1° within any fixed scale × layer type.

**Task-specific and scale-dependent:**

- The *direction* of the rotation (novel subspace) — 20°–45° pairwise angles between tasks in the same layer type.
- The *depth signature* — which layers concentrate change mass.
- Clustering of tasks by depth band, which reorganizes across scales (early at 3B, three-band at 8B).

This "how vs where" split is the paper's mechanistic contribution. The how is GRPO's intrinsic structural inductive bias (consistent with LoRA/OFT intrinsic-rank literature). The where is what the task specializes in.

### 4.9 Phase 3 Limitations

1. **Qwen2.5/Qwen3 family confound at 3B and 7B.** Cannot fully separate scale and family effects without a same-family scaling ladder. The rotation-growth trend is robust; the depth-concentration shift is not.
2. **Countdown 4B re-run pending.** The original 4B countdown GRPO checkpoint was trained from a different SFT base (`sft_qwen3_4b` not `sft_qwen_4.0b_instruct`), producing ‖τ‖ = 151 — a massive outlier. Script was updated to use `phase2_countdown/direct_rlvr` but not re-run.
3. **8B maze excluded.** v1 `grpo_qwen3_8b_maze` has only generation logs, no weights. v2 has weights but is from a different training.
4. **No cross-scale transfer-matrix correlation.** We only measured cos-vs-transfer correlation at 4B. Whether the correlation is stronger/weaker at other scales is unknown.
5. **No gradient analysis.** We haven't compared actual gradient directions at the SFT checkpoint, which could reveal whether tasks "want" to move similarly before training starts.
6. **Causality unclear.** We observe correlation between depth-overlap and interference; intervention experiments (e.g., projecting out shared directions during multi-task training) would be needed to establish causation.
7. **Only 5 tasks probed for subspace/rotation in §4.5.** countdown, basic_arithmetic, caesar_cipher, syllogism, cryptarithm. Full 10-task coverage would be more robust.

---

## 5. Phase 4 — Task-Vector Merging (Parts 1–5, the Original Story)

### 5.1 Motivation: τ_A + τ_B + W_SFT as a Cheap Multi-Skill Model

A single-task GRPO checkpoint is expensive. If N task-specific τ's can be *combined* into one model by simple weight summation, the cost of a multi-skill model drops from (N × training compute) to (N × τ storage). The task-arithmetic literature (TIES, DARE) advocates for this recipe but has uneven empirical support on reasoning tasks and no mechanistic story for *when* it will work.

Phase 4 tests task-vector merging on reasoning RLVR τ's, running through five empirical parts (1–5) during 2026-04-23 to 2026-05-06, then a Part-6 gap-closing sweep that retracted two of the major claims. This chapter covers Parts 1–5 — the original story as it stood on 2026-05-06 — in full detail. Chapter 6 covers Part 6.

All merges have the form

$$W_{\mathrm{merged}} \;=\; W_{\mathrm{SFT}} \;+\; s \cdot \sum_{i=1}^{N} \tau_i, \qquad \tau_i = W_{\mathrm{GRPO}(i)} - W_{\mathrm{SFT}},$$

varying the τ set, the scaling coefficient s, and (later) which layer types contribute to the sum. The primary outcome metric is **pair damage**:

$$\mathrm{damage}(A, B) \;=\; \max(0, \mathrm{acc}_{\mathrm{Direct}(A)} - \mathrm{acc}_{\mathrm{merge}}) \;+\; \max(0, \mathrm{acc}_{\mathrm{Direct}(B)} - \mathrm{acc}_{\mathrm{merge}}),$$

the total per-task capability lost relative to the two single-task sources.

### 5.2 Part 1 (2026-04-23) — 1/N Averaging Is Strictly Worse

Part 1 tests three merge scales at 4B:

| Merge | N | Scale s | τ set |
|---|---|---|---|
| `phase1_all_1overN` | 10 | 0.1 (= 1/N) | Phase-1 4B GRPO checkpoints for all 10 tasks |
| `all_tasks_s0.33` | 9 | 0.33 | Phase-2-refined 9 τ's (cryptarithm excluded) |
| `basic` (2-task) | 2 | 1.0 | SP RFT-Base + Arith Atomic-Only (the two best single-task checkpoints) |

τ magnitudes at 4B (Frobenius):

| Task | ‖τ‖ | Task | ‖τ‖ |
|---|---:|---|---:|
| basic_arithmetic | 0.852 | syllogism | 0.756 |
| countdown | 0.970 | knights_knaves | 1.005 |
| shortest_path | 0.975 | leg_counting | 1.193 |
| maze | 0.994 | caesar_cipher | 0.909 |
| mini_sudoku | 0.733 | cryptarithm | **0.000** |

Cryptarithm has ‖τ‖ = 0 at 4B (trained but never rewarded), so it contributes nothing to any merge.

**3-merge × 10-task accuracy matrix at 4B:**

| Task | 1/N (10τ, s=0.1) | s=0.33 (9τ) | 2-task (SP+Arith, s=1.0) |
|---|---:|---:|---:|
| basic_arithmetic | 69.5 | 71.0 | **76.0** |
| countdown | 39.0 | 39.0 | **39.5** |
| shortest_path | 24.0 | 46.5 | **51.0** |
| maze | 7.5 | 4.0 | **8.5** |
| mini_sudoku | 14.0 | **28.0** | 17.0 |
| syllogism | 59.5 | 71.5 | **83.0** |
| knights_knaves | 74.0 | **82.0** | 77.5 |
| leg_counting | 45.0 | **68.0** | 57.0 |
| caesar_cipher | 0.0 | 0.0 | 0.5 |
| cryptarithm | 0.0 | 0.0 | 0.0 |
| **Mean (8 non-trivial)** | 41.6 | **51.3** | **51.2** |

**F1. 1/N averaging is strictly worse than every alternative.** The 10-task 1/N merge loses to the 2-task merge on 9/10 tasks and loses to the 9-task s=0.33 merge on 8/10. It never wins a single row. Mean accuracy across 8 non-trivial tasks is 41.6% vs 51.2% (2-task) vs 51.3% (9-task).

**F2. Scale > N: ~3 τ-equivalents is the sweet spot.** The 9-task s=0.33 merge (τ-equivalents = s·N = 3.0) and the 2-task s=1.0 merge (τ-equivalents = 2.0) are tied on aggregate. What matters is the total displacement budget from W_SFT, not how many sources contribute. 1/N at τ-equivalents = 1.0 under-shoots.

**F3. 2-task is best for focused capability.** On SP, Arith, Count, and syllogism the 2-task merge wins or ties every comparison. Syllogism at +11.5% over the 9-task merge is notable — neither τ was trained on syllogism, yet the SP+Arith merge produces +11.5% over a merge that *includes* the syllogism τ. Consistent with Phase-1's "search/planning → logic" positive transfer.

**F4. 9-task s=0.33 wins on tasks it wasn't directly built for.** mini_sudoku +11%, knights_knaves +4.5%, leg_counting +11% over 2-task. These are tasks where single-task GRPO works but hurts cross-task (from Phase 1) — at s=0.33 diluted, they contribute without dominating.

**F5. Caesar and cryptarithm stay at 0 under every merge.** If the base model gets 0 reward during RLVR, the resulting τ either is zero or points to no useful direction. Merging doesn't rescue tasks the individual RLVR runs couldn't learn.

**F6. Per-τ magnitude is a weak predictor of downstream contribution.** leg_counting has the largest ‖τ‖ (1.19) but needs s=0.33 dilution to be useful. mini_sudoku has the smallest non-zero ‖τ‖ (0.73) yet produces the largest relative gain (+11%) in the 9-task merge.

**Part 1 delta-from-base-SFT appendix table (from report_phase4_merge_scaling.md §Appendix):**

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

The 1/N merge's net effect across the 8 non-trivial tasks is approximately zero (+7.5 − 7.0 ≈ 0.5 summed). The s=0.33 and 2-task merges both lift 7 of 8 non-trivial tasks, differing mainly in *which* tasks and *by how much*.

### 5.3 Part 2 (2026-04-24) — 7 Pair Merges + Rotation Analysis

Part 2 runs 7 controlled 2-τ pair merges at 4B (s = 1.0) to probe when pairs compose cleanly vs damage each other.

**Pair selection:**

| # | Pair | Reasoning |
|---|---|---|
| T1 | mini_sudoku + leg_counting | Phase-1 transfer was −52% (worst in matrix) |
| T2 | mini_sudoku + knights_knaves | Phase-1 transfer was −50% |
| T3 | basic_arithmetic + leg_counting | Phase-1 transfer was −32% |
| T4 | basic_arithmetic + syllogism | Phase-1 transfer +18% (positive) |
| T5 | shortest_path + syllogism | Different primary depths |
| T6 | shortest_path + knights_knaves | Both mid-ish at 4B; Phase-1 transfer −12% |
| T7 | countdown + basic_arithmetic | Phase-3 flagged −0.90 flat cos at L6.mlp.up_proj (8B) |

τ cosines (from the merge script):

| Pair | cos(τ_A, τ_B) | ‖τ_A‖ | ‖τ_B‖ |
|---|---:|---:|---:|
| T1 sdk+leg | 0.0053 | 0.733 | 1.193 |
| T2 sdk+kk | 0.0002 | 0.733 | 1.005 |
| T3 ari+leg | 0.0830 | 0.852 | 1.193 |
| T4 ari+syl | 0.0398 | 0.852 | 0.756 |
| T5 sp+syl | 0.0153 | 0.975 | 0.756 |
| T6 sp+kk | 0.0106 | 0.975 | 1.005 |
| T7 cnt+ari | — | 0.970 | 0.852 |

All pairs have near-zero global cosine (consistent with Phase 3's task-equidistant bases).

**7×10 accuracy matrix:**

| Task | T1 sdk+leg | T2 sdk+kk | T3 ari+leg | T4 ari+syl | T5 sp+syl | T6 sp+kk | T7 cnt+ari |
|---|---:|---:|---:|---:|---:|---:|---:|
| basic_arithmetic | 66.0 | 50.0 | 52.0 | 57.0 | 71.0 | 59.5 | 63.0 |
| countdown | 39.0 | 8.0 | 28.5 | 24.0 | 28.0 | 29.0 | 46.0 |
| shortest_path | 15.5 | 0.5 | 15.0 | 15.0 | 47.5 | 26.0 | 27.0 |
| maze | 4.0 | 0.5 | 11.5 | 1.0 | 3.5 | 8.0 | 0.5 |
| mini_sudoku | 20.0 | 35.0 | 19.5 | 11.0 | 31.5 | 23.5 | 17.0 |
| syllogism | 78.5 | 64.0 | 86.0 | 87.5 | 85.0 | 88.5 | 84.0 |
| knights_knaves | 62.0 | 65.0 | 68.0 | 70.5 | 70.0 | 86.5 | 72.5 |
| leg_counting | 78.0 | 1.0 | 67.5 | 1.0 | 55.5 | 30.5 | 39.5 |
| caesar_cipher | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| cryptarithm | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

**Pair damage:**

| Pair | damage_A | damage_B | total |
|---|---:|---:|---:|
| T5 sp+syl | 0 (SP) | 7 (syl) | **7** |
| T4 ari+syl | 0 (arith) | 4.5 (syl) | **4.5** |
| T3 ari+leg | 0 (arith) | 12.5 (leg) | 12.5 |
| T1 sdk+leg | 16 (sdk) | 2 (leg) | 18 |
| T6 sp+kk | 18 (SP) | 0 (KK) | 18 |
| T2 sdk+kk | 1 (sdk) | 21 (KK) | 22 |
| T7 cnt+ari | 24 (cnt) | 0 (arith) | 24 |

**F7. Phase-1 transfer is NOT a merge predictor.** T1 predicted −52% → damage only 18; T2 predicted −50% → damage 22 (partial confirmation); T3 predicted −32% → damage 12.5; T7 predicted +6% → damage 24 (reversal). Training on τ_A then evaluating on B is a fundamentally different operation from adding τ_A + τ_B.

**F8. Global τ cosine is not a merge predictor.** All 7 pairs have cosines 0.0002–0.08 (essentially orthogonal) but damage varies 4.5–24.

**F9. Per-layer opposing rotations don't predict damage either.** Pearson r between #layers with up_proj flat_cos < −0.5 and damage is only +0.23.

**Novel-subspace rotation at 4B (6 pairs + T7; 18 probe layers):**

| Pair | mlp_up angle | mlp_up flat_cos | mlp_down angle | mlp_down flat_cos | attn_q angle | attn_q flat_cos |
|---|---:|---:|---:|---:|---:|---:|
| T1 sdk+leg | 63.89° | +0.188 | 34.68° | +0.212 | 44.64° | +0.377 |
| T2 sdk+kk | 60.79° | +0.016 | 32.25° | +0.342 | 42.42° | +0.358 |
| T3 ari+leg | 61.46° | +0.275 | 34.83° | +0.077 | 44.71° | −0.468 |
| T4 ari+syl | 58.18° | −0.114 | 30.02° | +0.364 | 40.33° | +0.293 |
| T5 sp+syl | 60.17° | −0.172 | 32.46° | +0.102 | 42.05° | −0.271 |
| T6 sp+kk | 62.87° | −0.051 | 35.04° | −0.002 | 44.90° | −0.142 |
| T7 cnt+ari | 60.78° | −0.300 | 33.38° | +0.042 | 43.87° | +0.152 |

**F10. Novel-subspace angle has modest predictive power (r = +0.56).** The only rotation metric that correlates with damage. The *direction* of rotation (flat cosine) doesn't matter; the *distance* between novel subspaces does. Plausible interpretation: when two τ's rotate the same basis into very different subspaces, summing them produces a rotation that matches neither original basin.

**F11. T5 (SP + syllogism) is the cleanest pair.** Total damage 7 — less than half of any other pair. SP fully preserved (47.5%), syllogism loses only 7 points.

**F12. T4 (arith + syllogism) is the damage-minimum pair.** Damage 4.5; arithmetic actually *gained* +2 over Direct-RLVR. Syllogism's τ appears to contain a small corrective signal for arithmetic's known GRPO damage.

**F13. Best-of-n table.** Across the 10 merged models evaluated so far (1/N, s=0.33, 2-task, T1–T7):

| Task | Best model | Score |
|---|---|---:|
| shortest_path | 2-task (SP+Arith) | 51.0% |
| basic_arithmetic | 2-task (SP+Arith) | 76.0% |
| countdown | 2-task / T7 | 39–46% |
| syllogism | T6 SP+KK | 88.5% |
| knights_knaves | T6 SP+KK | 86.5% |
| leg_counting | T1 sdk+leg | 78.0% |
| mini_sudoku | T2 sdk+kk | 35.0% |
| maze | 2-task | 8.5% |

No single merge dominates; the right merge depends on which tasks the user cares about.

### 5.4 Part 3 (2026-04-24) — Sign Agreement, Layer-Selective, Triple Merge

Part 3 tests three directions:

1. Sign agreement (TIES premise: damage from sign conflict at high-|τ| positions).
2. Layer-selective merges (Phase 3's "mlp.down_proj frozen" as actionable recipe).
3. Triple merge (SP + Arith + Syl — three orthogonal τ's).

#### Part 3.A — Sign-agreement analysis

For each of the 7 pair merges, compute:

- `frac_agree`: fraction of shared-nonzero parameters where sign(τ_A) = sign(τ_B)
- `mag_agree`: magnitude-weighted (weighted by |τ_A|·|τ_B|)
- `conflict_mass`: fraction of |τ_A|·|τ_B| at sign-disagreeing positions
- per layer type: up_conflict, q_conflict

| Pair | damage | frac_agree | mag_agree | conflict_mass | up_conflict | q_conflict |
|---|---:|---:|---:|---:|---:|---:|
| T1 sdk+leg | 18.0 | 0.0329 | 0.5044 | 0.4956 | 0.4946 | 0.4978 |
| T2 sdk+kk | 22.0 | 0.0327 | 0.5002 | 0.4998 | 0.5015 | 0.4972 |
| T3 ari+leg | 12.5 | 0.0266 | 0.5695 | 0.4305 | 0.4322 | 0.4471 |
| T4 ari+syl | 4.5 | 0.0262 | 0.5337 | 0.4663 | 0.4648 | 0.4806 |
| T5 sp+syl | 7.0 | 0.0222 | 0.5131 | 0.4869 | 0.4886 | 0.4891 |
| T6 sp+kk | 18.0 | 0.0226 | 0.5089 | 0.4911 | 0.4891 | 0.4923 |
| T7 cnt+ari | 24.0 | 0.0216 | 0.5132 | 0.4868 | 0.4853 | 0.4903 |

Correlations (n = 7):

| Metric | Pearson r |
|---|---:|
| frac_agree | +0.21 |
| mag_agree | −0.46 |
| conflict_mass | +0.46 |
| up_conflict | +0.46 |
| q_conflict | +0.38 |

**F18. Magnitude-weighted sign agreement is the best merge-damage predictor so far at 4B.** r = −0.46. Consistent with TIES's premise that high-magnitude sign conflicts cause damage. T3 and T4 (lowest conflict) have lowest damage; T1 and T2 (highest conflict) have highest.

Because each pair has cos(τ_A, τ_B) ≈ 0, ~50% sign agreement is the random-orthogonal baseline. The observed range 50.0–57.0% is small but predictive in relative terms.

#### Part 3.B — Layer-selective merges (SP + Arith, s = 1.0)

Three variants + baseline:

| Variant | # keys merged | Included |
|---|---:|---|
| `up` | 72 | `mlp.up_proj` + `mlp.gate_proj` only |
| `up_q_clean` | 108 | + `self_attn.q_proj` |
| `all_but_down` | 362 | everything except `mlp.down_proj` |
| baseline 2-task | 398 | all keys |

**4-variant × 10-task table:**

| Task | 2-task | triple | `up` | `up+q` | `all_but_down` |
|---|---:|---:|---:|---:|---:|
| basic_arithmetic | 76.0 | 74.5 | 73.5 | **77.0** | **77.0** |
| countdown | 39.5 | **42.0** | 25.5 | 30.5 | 36.5 |
| shortest_path | 51.0 | 50.0 | 37.5 | 40.5 | **53.5** |
| maze | **8.5** | 0.5 | 5.0 | 4.0 | 6.0 |
| mini_sudoku | 17.0 | **23.5** | 6.5 | 7.0 | 12.5 |
| syllogism | 83.0 | 83.0 | 83.5 | **84.5** | 83.0 |
| knights_knaves | **77.5** | 67.0 | 71.0 | 67.5 | 71.5 |
| leg_counting | 57.0 | 52.0 | **62.0** | 60.5 | 59.0 |
| caesar_cipher | 0.5 | 0.0 | 0.0 | 0.0 | 0.5 |
| cryptarithm | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

Δ from 2-task baseline:

| Task | `up` Δ | `up+q` Δ | `all_but_down` Δ | Classification |
|---|---:|---:|---:|---|
| basic_arithmetic | −2.5 | +1.0 | +1.0 | **feature-routing** |
| syllogism | +0.5 | +1.5 | 0.0 | **feature-routing** |
| leg_counting | +5.0 | +3.5 | +2.0 | **feature-routing** |
| maze | −3.5 | −4.5 | −2.5 | marginal (low ceiling) |
| mini_sudoku | −10.5 | −10.0 | −4.5 | attention-restructuring |
| knights_knaves | −6.5 | −10.0 | −6.0 | attention-restructuring |
| shortest_path | −13.5 | −10.5 | +2.5 | attention-restructuring |
| countdown | −14.0 | −9.0 | −3.0 | attention-restructuring |

**F19 (original claim, rescoped in Part 6).** Tasks split cleanly into two layer-wise groups under the SP+Arith merge.

- **Group 1 — "Feature-routing" (arith, syllogism, leg_counting):** merge-relevant signal is in `mlp.up_proj + mlp.gate_proj`. Under `up` only, these tasks lose ≤3 pts; leg_counting gains +5.
- **Group 2 — "Attention-restructuring" (SP, countdown, sudoku, KK):** needs the full `attn.{q,k,v,o}` circuit. Under `up` only, these lose 6.5–14 pts. Under `all_but_down`, they recover (SP actually gains +2.5).

Part 6 shows this split is **pair-dependent, not task-intrinsic** — see Chapter 6.

**F20. Dropping mlp.down_proj is free or positive for every task.** `all_but_down` matches or beats the full 2-task merge on 6/8 non-trivial tasks, worst by only 6 pts on the other 2. This validates Phase 3's "down_proj basis frozen" finding as actionable: **always safe to exclude mlp.down_proj from task-vector merges.**

**F21. Phase 3's rotation analysis was structurally incomplete.** Phase 3 measured `mlp.up_proj`, `mlp.down_proj`, `attn.q_proj` — not `attn.{k,v,o}`. Part 3.B shows these carry 4–13 pts of merge-relevant capability for Group-2 tasks. Moving from `up+q` to `all_but_down` lifts SP +13, countdown +6, sudoku +5.5, KK +4 — and is neutral for Group-1 tasks.

#### Part 3.C — Triple merge (SP + Arith + Syllogism, s = 1.0)

| Task | 2-task | triple | Δ |
|---|---:|---:|---:|
| basic_arithmetic | 76.0 | 74.5 | −1.5 |
| countdown | 39.5 | 42.0 | +2.5 |
| shortest_path | 51.0 | 50.0 | −1.0 |
| maze | 8.5 | 0.5 | **−8.0** |
| mini_sudoku | 17.0 | 23.5 | +6.5 |
| syllogism | 83.0 | 83.0 | 0.0 |
| knights_knaves | 77.5 | 67.0 | **−10.5** |
| leg_counting | 57.0 | 52.0 | −5.0 |

**F22. Triple merge is not a win — τ-count ceiling is real.** Adding a third τ at s = 1.0 costs maze, KK, leg while recovering small wins on sudoku and countdown. Consistent with Part 1's "~3 τ-equivalents is the sweet spot" (3 × 1.0 = 3 is above the Group-2 threshold).

### 5.5 Part 4 (2026-04-24) — 3B and 8B Replication

Part 4 replicates Part 3.B at 3B (Qwen2.5) and 8B (Qwen3) using Phase-1 SP and Arith checkpoints.

**3-scale × 3-variant × 10-task matrix:**

| Task | 3B basic | 3B up | 3B abd | 4B basic | 4B up | 4B abd | 8B basic | 8B up | 8B abd |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| basic_arithmetic | 37.5 | 44.5 | 40.0 | 76.0 | 73.5 | 77.0 | 62.0 | 67.0 | 67.0 |
| countdown | 0.5 | 0.0 | 0.0 | 39.5 | 25.5 | 36.5 | 31.5 | 32.5 | 32.0 |
| shortest_path | 34.5 | 29.0 | 34.5 | 51.0 | 37.5 | 53.5 | 44.0 | 31.5 | 45.0 |
| maze | 5.0 | 7.0 | 6.5 | 8.5 | 5.0 | 6.0 | 24.0 | 22.0 | 23.0 |
| mini_sudoku | 0.0 | 0.0 | 0.0 | 17.0 | 6.5 | 12.5 | 15.5 | 3.0 | 13.0 |
| syllogism | 74.0 | 81.0 | 82.5 | 83.0 | 83.5 | 83.0 | 72.0 | 77.0 | 72.5 |
| knights_knaves | 16.0 | 26.5 | 17.5 | 77.5 | 71.0 | 71.5 | 64.5 | 66.0 | 66.0 |
| leg_counting | 4.0 | 6.5 | 5.0 | 57.0 | 62.0 | 59.0 | 39.5 | 44.5 | 46.5 |

Δ from 2-task baseline:

| Task | 3B Δ up | 3B Δ abd | 4B Δ up | 4B Δ abd | 8B Δ up | 8B Δ abd |
|---|---:|---:|---:|---:|---:|---:|
| basic_arithmetic | +7.0 | +2.5 | −2.5 | +1.0 | +5.0 | +5.0 |
| syllogism | +7.0 | +8.5 | +0.5 | 0.0 | +5.0 | +0.5 |
| leg_counting | +2.5 | +1.0 | +5.0 | +2.0 | +5.0 | +7.0 |
| shortest_path | −5.5 | 0.0 | −13.5 | +2.5 | −12.5 | +1.0 |
| mini_sudoku | 0.0 | 0.0 | −10.5 | −4.5 | −12.5 | −2.5 |
| countdown | −0.5 | −0.5 | −14.0 | −3.0 | +1.0 | +0.5 |
| knights_knaves | +10.5 | +1.5 | −6.5 | −6.0 | +1.5 | +1.5 |
| maze | +2.0 | +1.5 | −3.5 | −2.5 | −2.0 | −1.0 |

**F23 (later rescoped).** Feature-routing group replicates at all 3 scales for basic_arithmetic (+7.0 / −2.5 / +5.0), syllogism (+7.0 / +0.5 / +5.0), leg_counting (+2.5 / +5.0 / +5.0).

**F24.** Attention-restructuring confirmed at 4B and 8B for shortest_path and mini_sudoku. Scale-dependent for countdown (G2 at 4B, G1 at 8B) and knights_knaves (G2 at 4B, G1 at 3B and 8B). Part 6 will further revise this to "pair-dependent."

**F25. `mlp.down_proj`-droppable claim holds at all 3 scales.** `all_but_down` Δ is within ±3 of the full 2-task merge on 22/24 scale × task cells. Mean Δ: 3B +2.1, 4B −1.3, 8B +1.3. Dropping `mlp.down_proj` is net-positive or neutral at every scale.

**F26 (originally claimed, falsified in Part 6).** Sign-agreement cross-scale predictor: 3B mag_agree → 4B damage gives r = −0.81 (n = 7).

| Predictor scale | r(mag_agree, 4B damage) | r(conflict_mass, 4B damage) | r(q_conflict, 4B damage) |
|---|---:|---:|---:|
| 4B (same scale) | −0.46 | +0.46 | +0.38 |
| **3B** | **−0.81** | **+0.81** | +0.83 |
| 8B | −0.44 | +0.44 | +0.52 |

Per-pair:

| Pair | 3B mag_agree | 4B mag_agree | 8B mag_agree | 4B damage |
|---|---:|---:|---:|---:|
| T4 ari+syl | 0.605 | 0.534 | 0.549 | 4.5 (lowest) |
| T3 ari+leg | 0.564 | 0.570 | 0.549 | 12.5 |
| T5 sp+syl | 0.536 | 0.513 | 0.517 | 7.0 |
| T6 sp+kk | 0.513 | 0.509 | 0.517 | 18.0 |
| T2 sdk+kk | 0.531 | 0.500 | 0.507 | 22.0 |
| T1 sdk+leg | 0.530 | 0.504 | 0.502 | 18.0 |
| T7 cnt+ari | 0.493 | 0.513 | 0.536 | 24.0 (highest) |

3B sign-agreement is a better predictor of 4B damage than 4B sign-agreement itself. The n = 7 claim: "compute on a cheap model, predict for the production model." Part 6 shows this decays to r = −0.27 at n = 19 (see Chapter 6).

**F27 (original headline, since rescoped).** Layer-type × task-group decomposition that partially generalizes, plus a transferable damage predictor. At the close of Part 4 the paper spine was:

- Feature-routing tasks (arith, syllogism, leg) universally mergeable from `up_proj + gate_proj` alone at 3B/4B/8B
- `mlp.down_proj` droppable with zero-to-positive effect at every scale
- SP and sudoku universally attention-restructuring; countdown and KK scale-specific
- Sign agreement at 3B predicts 4B damage at r = −0.81 (n = 7)

Part 4 also notes the limitations that would become important in Part 6: n = 7 is small, Qwen2.5/Qwen3 family confound, 3B uses Phase-1 checkpoints while 4B uses Phase-2-refined (RFT-Base).

### 5.6 Part 5 (2026-05-05 to 05-06) — Paper Prep: Linking, Predictor Validation, TIES/DARE

Part 5 addresses the remaining gaps for a single publishable paper. Four parallel streams completed in ~10 GPU-wall-clock hours.

#### F28. Atomic vs direct τ's merge differently (the linking result)

Whether Phase-2 atomic training and Phase-4 merging are connected stories. Three 2-τ merges of SP + Arith at 4B:

| Merge | SP source | Arith source |
|---|---|---|
| M_aa | τ_atomic (RFT-Base, 55.5% SP) | τ_atomic (Atomic-Only, 78% Arith) |
| M_dd | τ_direct (grpo_qwen4b_shortest_path) | τ_direct (grpo_qwen4b_basic_arithmetic) |
| M_mix | τ_atomic (SP) | τ_direct (Arith) |

Sign agreement between pre-merge τ pairs:

| Pair-type | sp + ari | sp + maze | ari + maze |
|---|---:|---:|---:|
| atomic-atomic | **0.519** | 0.512 | 0.506 |
| direct-direct | 0.501 | 0.515 | 0.509 |
| cross | 0.497 | 0.507 | 0.508 |

Atomic-atomic mag_agree for SP+Arith is +1.8 pts higher than direct-direct — small in absolute but large relative to the 0.50–0.61 window.

10-task accuracy:

| Task | M_aa | M_dd | M_mix | Δ(aa−dd) |
|---|---:|---:|---:|---:|
| basic_arithmetic | 75.0 | 66.5 | 65.5 | +8.5 |
| countdown | 36.0 | 35.5 | 37.5 | +0.5 |
| shortest_path | 54.0 | 52.5 | 51.5 | +1.5 |
| maze | 11.0 | 3.5 | 1.0 | +7.5 |
| mini_sudoku | 17.5 | 19.0 | 14.5 | −1.5 |
| **syllogism** | **83.0** | **48.5** | 86.0 | **+34.5** |
| knights_knaves | 66.5 | 65.5 | 68.5 | +1.0 |
| leg_counting | 55.5 | 44.0 | 47.5 | +11.5 |
| caesar_cipher | 0.5 | 0.0 | 0.0 | +0.5 |
| cryptarithm | 0.0 | 0.0 | 0.0 | 0 |
| **Mean (8 non-trivial)** | **49.8** | **41.9** | **46.5** | **+7.9** |

M_aa beats M_dd on 8/10 tasks (ties on 2). Syllogism +34.5 is the most striking cell — neither M_aa nor M_dd uses any syllogism-specific τ; atomic τ's preserve syllogism-relevant structure to 83% while direct τ's collapse it to 48.5%.

**Cross-scale replication (3B and 8B):**

| Task | 3B Δ(aa−dd) | 4B Δ(aa−dd) | 8B Δ(aa−dd) | Pattern |
|---|---:|---:|---:|---|
| basic_arithmetic | +2.5 | +8.5 | +10.0 | M_aa wins every scale |
| countdown | 0 | +0.5 | +7.5 | M_aa wins |
| leg_counting | +4.0 | +11.5 | +10.0 | M_aa wins every scale |
| knights_knaves | +8.5 | +1.0 | +3.0 | M_aa wins every scale |
| mini_sudoku | +1.0 | −1.5 | −0.5 | scale-mixed, small |
| maze | +1.5 | +7.5 | −3.5 | scale-dependent |
| syllogism | −11.0 | +34.5 | −5.5 | scale-dependent (4B outlier) |
| **shortest_path** | **−33.0** | +1.5 | **−18.5** | **M_dd wins every non-tie scale** |
| **Mean (8 tasks)** | **−3.3** | **+7.9** | **+0.3** | |
| Wins (aa : dd) | 5 : 2 | 8 : 1 | 4 : 4 | |

The 4B +7.9 story does not generalize as a universal magnitude. But the *sign* of Δ(aa − dd) is cross-scale stable for specific tasks:

- M_aa wins universally on arith, leg, KK, countdown (computation/logic/counting)
- M_dd wins universally on shortest_path (spatial search)
- Scale-dependent on syllogism, maze, sudoku

This pairs with the Phase-4 layer-type split: feature-routing tasks (arith, logic, counting) benefit from atomic curriculum's MLP-concentrated signal; attention-restructuring tasks (shortest_path) benefit from direct RLVR's attention updates. The revised F28 claim: **atomic τ's preserve feature-routing structure; direct τ's preserve attention-restructuring structure.**

#### F29 (claimed, falsified in Part 6). Sign-agreement r = −0.81 survives statistical stress tests at n = 7

Three validations:

**D.1 Bootstrap 95% CI:**
- r = −0.810 (point estimate)
- 10,000 bootstrap resamples: CI [−0.993, −0.376] — fully below zero
- Permutation test (two-sided, 10,000 perms): p = 0.024

**D.2 Leave-one-out:**
- Pearson r between predictions and truths: 0.725
- MAE = 3.59, RMSE = 4.77

**D.3 Fixed train/test splits:**

| Split | Train | Test | r | MAE |
|---|---|---|---:|---:|
| A | {T1, T2, T3, T4} | {T5, T6, T7} | 0.991 | 6.93 |
| B | {T5, T6, T7} | {T1, T2, T3, T4} | 0.974 | 15.29 |

These all looked very strong. Part 6 later shows the underlying correlation is r ≈ −0.3 and these n = 7 validations were favorable-sample artifacts.

#### F30. TIES and DARE do not beat vanilla sum at equal τ-magnitude

Tested on three representative 4B pairs (T2, T5, T7):

| Pair | Method | arith | cnt | SP | maze | sdk | syl | KK | leg | Δ vs vanilla (mean 8) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| T2 sdk+kk | vanilla | 50.0 | 8.0 | 0.5 | 0.5 | 35.0 | 64.0 | 65.0 | 1.0 | — |
| | TIES | 48.5 | 8.0 | 1.0 | 0.5 | 36.5 | 66.0 | 76.0 | 1.0 | +1.1 |
| | DARE | 50.0 | 9.0 | 0.5 | 0.5 | 36.0 | 66.0 | 68.5 | 1.0 | +0.5 |
| T5 sp+syl | vanilla | 71.0 | 28.0 | 47.5 | 3.5 | 31.5 | 85.0 | 70.0 | 55.5 | — |
| | TIES | 70.0 | 26.0 | 48.0 | 4.5 | 25.5 | 85.0 | 70.0 | 53.0 | −1.0 |
| | DARE | 69.0 | 27.5 | 51.5 | 3.5 | 24.5 | 84.5 | 67.5 | 55.5 | −0.6 |
| T7 cnt+ari | vanilla | 63.0 | 46.0 | 27.0 | 0.5 | 17.0 | 84.0 | 72.5 | 39.5 | — |
| | TIES | 60.5 | — | 23.5 | 1.0 | 16.5 | 85.5 | 73.5 | 40.0 | n/a |
| | DARE | 62.5 | 43.0 | 27.5 | 1.0 | 14.5 | 86.5 | 74.5 | 43.0 | +0.6 |

No method consistently beats vanilla. Differences within ±2 points on all tasks except KK at T2 (TIES +11). Paper framing: *"vanilla sum with `all_but_down` mask is competitive with TIES/DARE and better-motivated"* — the mask follows from §4.4's basis-frozen observation rather than a tunable sparsification heuristic.

#### F31. attn.{k,v,o} novel-subspace rotation fills the Phase 3 gap

Principal angles between novel subspaces at 4B and 8B for the three attn sub-layers Phase-3 didn't measure:

**4B (6 pairs):**

| Pair | k_proj | v_proj | o_proj |
|---|---:|---:|---:|
| T1_sdk_leg | 37.3 | 58.1 | 61.4 |
| T2_sdk_kk | 41.3 | 63.5 | 59.6 |
| T3_ari_leg | 40.1 | 53.5 | 57.7 |
| T4_ari_syl | 41.3 | 60.1 | 60.5 |
| T5_sp_syl | 38.3 | 64.9 | 60.9 |
| T6_sp_kk | 37.9 | 65.5 | 60.6 |

**8B (7 pairs):**

| Pair | k_proj | v_proj | o_proj |
|---|---:|---:|---:|
| T1_sdk_leg | 33.3 | 61.9 | 56.0 |
| T2_sdk_kk | 37.6 | 56.9 | 58.1 |
| T3_ari_leg | 35.8 | 55.9 | 64.0 |
| T4_ari_syl | 35.2 | 49.2 | 60.8 |
| T5_sp_syl | 35.4 | 56.4 | 68.0 |
| T6_sp_kk | 34.9 | 50.4 | 70.1 |
| T7_cnt_ari | 31.8 | 55.3 | 67.7 |

k_proj rotates 30–42° (less than q_proj's 40–45°). v_proj and o_proj rotate 50–70° (comparable to mlp.up_proj). **None of the three shows the 0° "basis-frozen" signature of `mlp.down_proj`.** Consistent with the empirical finding (F21) that including `attn.{k,v,o}` adds 4–13 pts of Group-2 capability.

#### F32 (claimed pre-Part-6, retracted). Revised paper spine: two independent predictors unify Phase 2 and Phase 4

At the close of Part 5 (2026-05-06) we labeled the following a "paper-ready" single-sentence headline (from report_phase4_merge_scaling.md §F32):

> *RLVR on small LMs has a predictable compositional structure: two independent 1-hour probes on a cheap 3B base model forecast merge damage at 4B — a weight-space probe (sign-agreement, r = −0.81) and a behavioral probe (L1-grounding ARS, r = −0.76). Both predictors agree that the shared bottleneck is grounding, not reasoning.*

The clause-by-clause experimental backing at n=7 looked as follows:
- weight-space predictor (primary): F29 (r = −0.81, p = 0.024, held-out r = 0.99)
- behavioral predictor (parallel primary): F33 (3B L1_mean → 4B damage r = −0.76; L2_mean r = −0.01)
- shared grounding bottleneck: F33 + Phase 2 (low-L1 → GRPO catastrophically fails)
- task-class-specific composition (secondary mechanistic claim): F28 revised
- simple-summation enabling multi-skill models: Phase 4 Part 1 + F30

The two predictors were claimed to be non-redundant. 3B L1_mean and 3B sign-agreement correlated at r = +0.78 across pairs; partial r's were L1_mean | sign_agree = −0.36 and sign_agree | L1_mean = −0.54; the two-predictor regression reached R² = 0.70 — higher than either alone (0.58 and 0.66).

**Part 6 status.** F32 is retracted. At n = 19 the sign-agreement correlation decays to r = −0.27 (F35) and at n = 12 the L1-ARS correlation flips sign to r = +0.06 (F37). The combined two-predictor regression drops from R² = 0.70 (n = 7) to R² = 0.30 (n = 12). The "two independent predictors" story is the concrete thing that Path A drops, replaced by the null-predictor catalogue (§9).

#### F33 (claimed, falsified in Part 6). ARS (behavioral probe) also predicts merge damage

Zero-shot probe the 3B SFT base on 50 samples × 6–14 atomic sub-skills per task, for all 7 tasks. Correlations with 4B merge damage (n = 7):

| Predictor (at 3B) | Pearson r | R² |
|---|---:|---:|
| L1_mean | **−0.761** | 0.58 |
| L1_min | −0.551 | 0.30 |
| ARS_mean | −0.337 | 0.11 |
| L2_mean | −0.009 | 0.00 |

L1 atoms predict merge damage; L2 atoms do not. Same ordering as Phase 2 (low L1 → GRPO thrashes). The claim: grounding is the bottleneck at both operational scales — single-task training success AND merge compatibility.

Two-predictor model: r(L1_mean, sign_agree) = +0.78, but each has independent info. Partial r's: L1_mean | sign_agree = −0.36; sign_agree | L1_mean = −0.54. Combined R² = 0.70.

| Pair | 3B L1_mean | 3B sign_agree | 4B damage |
|---|---:|---:|---:|
| T4 ari+syl | 0.500 | 0.605 | 4.5 (lowest) |
| T5 sp+syl | 0.320 | 0.536 | 7.0 |
| T3 ari+leg | 0.290 | 0.564 | 12.5 |
| T1 sdk+leg | 0.143 | 0.530 | 18.0 |
| T6 sp+kk | 0.188 | 0.513 | 18.0 |
| T2 sdk+kk | 0.220 | 0.531 | 22.0 |
| T7 cnt+ari | 0.261 | 0.493 | 24.0 (highest) |

At n = 7 these both look clean. Part 6 shows they do not survive n = 12/19.

### 5.7 Where Parts 1–5 Left the Story

At the close of Part 5 (2026-05-06) the Phase-4 paper had a clean spine:

- Drop `mlp.down_proj` — always safe
- Feature-routing tasks need only `up_proj + gate_proj`; attention-restructuring tasks need full `attn.{q,k,v,o}`
- Two cross-scale predictors: sign-agreement at 3B (r = −0.81, n = 7) and L1-ARS at 3B (r = −0.76, n = 7), combined R² = 0.70
- Atomic τ's merge cleanly for feature-routing tasks; direct τ's for attention-restructuring; cross-scale-stable in sign, scale-dependent in magnitude

Three concerns with the draft motivated Part 6:

1. The Group-1/Group-2 classification (§5.4 Part 3.B) was derived from a single pair (SP + Arith). If Group membership is pair-dependent rather than task-intrinsic, the classification story collapses.
2. Both predictor claims rested on n = 7 with wide bootstrap CIs.
3. The routing-probe hypothesis (claim that a cheap base-model probe could predict Group membership) was asserted, not tested.

Chapter 6 runs the corresponding experiments.

---

## 6. Phase 4 Part 6 — Gap-Closing That Reshaped the Spine

### 6.1 The Three Gaps

Part 6 ran on two parallel pods (`sunguha-rlvr-paper-npdz`, `sunguha-rlvr-paper2-n25r`) during 2026-05-09 to 2026-05-12. Goals:

1. Layer-selective merges on the 6 other Part-2 pairs (T1, T2, T3, T4, T6, T7) — test whether Group membership is task-intrinsic.
2. Add pairs to n = 7 → n = 12 → n = 19 — test whether the sign-agreement predictor stabilizes.
3. Operationalize and test the routing probe hypothesis.

Along the way, an L1-ARS n = 12 re-test, a per-feature residual search, and a task-level incompatibility score were computed.

The results were not the ones we hoped for but are cleaner than the previous draft.

### 6.2 F34 — Group Membership Is Pair-Dependent

**Setup.** Ran `up`, `up_q_clean`, `all_but_down` on T1, T2, T3, T4, T6, T7 at 4B. Each (pair × variant) evaluated on all 8 non-trivial tasks. Total new evals: 6 × 3 × 8 = 144.

**Method for classification.** For each pair, classify each of its two trained tasks: Δ_up = acc(pair_up) − acc(pair_full) on that task. Label G1 if Δ_up ≥ −3; G2 if Δ_up ≤ −6; else borderline.

**Per-pair per-task classifications:**

| Pair | Task | Δ up (pts) | Δ up_q (pts) | Δ all_but_down (pts) | Class |
|---|---|---:|---:|---:|:---:|
| T1 sdk+leg | mini_sudoku | −8.0 | −9.0 | +3.5 | G2 |
| T1 sdk+leg | leg_counting | +0.0 | +1.0 | −8.0 | G1 |
| T2 sdk+kk | mini_sudoku | −24.0 | −27.0 | −16.5 | G2 |
| T2 sdk+kk | knights_knaves | +9.0 | +7.0 | −11.5 | G1 |
| T3 ari+leg | basic_arithmetic | +23.0 | +23.0 | +11.5 | G1 |
| T3 ari+leg | leg_counting | +2.0 | −4.5 | +6.5 | G1 |
| T4 ari+syl | basic_arithmetic | +16.5 | +16.0 | +15.0 | G1 |
| T4 ari+syl | syllogism | **−6.0** | −5.5 | +0.5 | **G2** |
| T6 sp+kk | shortest_path | **+23.0** | +22.5 | +31.0 | **G1** |
| T6 sp+kk | knights_knaves | +0.0 | −1.0 | +1.0 | G1 |
| T7 cnt+ari | countdown | +27.0 | +27.0 | +32.0 | G1 |
| T7 cnt+ari | basic_arithmetic | +10.5 | +11.0 | +10.0 | G1 |

**Per-task aggregation across pairs:**

| Task | In pairs | Classifications | Stable? |
|---|---|---|:---:|
| basic_arithmetic | T3, T4, T7 | G1, G1, G1 | ✓ stable G1 |
| leg_counting | T1, T3 | G1, G1 | ✓ stable G1 |
| countdown | T7 | G1 | (n=1) |
| knights_knaves | T2, T6 | G1, G1 | ✓ stable G1 at 4B |
| mini_sudoku | T1, T2 | G2, G2 | ✓ stable G2 |
| **shortest_path** | T5 (=§5.4), T6 | **G2 (SP+Arith), G1 (SP+KK)** | **FLIPS** |
| **syllogism** | T4, T5 | **G2 (ari+syl), G1 (SP+syl)** | **FLIPS** |

**F34 (headline of Part 6).** Shortest-path and syllogism flip Group membership depending on the pair partner. Only mini_sudoku is consistently G2. The §5.4 claim that "feature-routing tasks are universal G1; attention-restructuring tasks are universal G2" does not survive.

**Mechanistic reading.** Group-2 behavior (needing full `attn.{q,k,v,o}`) emerges when two τ's rewrite *shared* attention circuits — dropping k,v,o destroys the shared reshaping both tasks depend on. When partners touch disjoint attention circuits (SP + KK vs SP + Arith), dropping k,v,o does not destroy a shared pathway. Consistent with §4.6's depth-signature reorganization: different tasks concentrate mass at different depths, and overlap with the partner determines merge constraints.

**Implication.** A task cannot be pre-classified for its merge recipe from task features alone. The practitioner's procedure must be empirical: build the full merge, then test whether aggressive masking still works for that specific pair.

### 6.3 F35, F36 — Sign-Agreement Decay from n = 7 to n = 19

**Tranche 1 (n = 7 → n = 12):** T8 leg+syl, T9 ari+kk, T10 cnt+leg, T11 sp+leg, T12 sdk+syl.

**Tranche 2 (n = 12 → n = 19, Path B):** T13 ari+sdk, T14 syl+kk, T15 kk+sdk, T16 cnt+kk, T17 cnt+syl, T18 cnt+sdk, T19 sp+sdk.

Each built as vanilla `W_SFT + τ_A + τ_B` at s = 1.0, evaluated on 8 non-trivial tasks. Sign-agreement analysis at 3B and 4B for each new pair.

**Per-pair 3B sign-agreement and 4B damage (all n = 19):**

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
| T10 cnt+leg | 0.5001 | **0.5** | Tranche 1 |
| T11 sp+leg | 0.5511 | **0.5** | Tranche 1 |
| T12 sdk+syl | 0.5264 | 19.5 | Tranche 1 |
| T13 ari+sdk | 0.5469 | 21.0 | Tranche 2 |
| T14 syl+kk | 0.5080 | 20.0 | Tranche 2 |
| T15 kk+sdk | 0.5306 | **64.5** | Tranche 2 |
| T16 cnt+kk | 0.5219 | 10.5 | Tranche 2 |
| T17 cnt+syl | 0.5069 | 40.0 | Tranche 2 |
| T18 cnt+sdk | 0.5209 | 45.5 | Tranche 2 |
| T19 sp+sdk | 0.5300 | 51.0 | Tranche 2 |

Damage at n = 19 spans 0.5 to 64.5 — much wider than the original 4.5–24 range.

**F35. Pearson r decays monotonically with n.**

| n | Pearson r | 95% bootstrap CI | Spearman r (p) |
|---|---:|---|---:|
| 7 (original F29) | **−0.810** | [−0.993, −0.376] | — |
| 12 (post-Tranche 1) | **−0.447** | [−0.872, +0.161] | −0.513 (p = 0.088) |
| **19 (post-Tranche 2)** | **−0.267** | **[−0.591, +0.140]** | **−0.362 (p = 0.128)** |

**Classic signature of small-n overfitting.** The correlation halves with each doubling of sample size. The CI never excludes zero past n = 12. Spearman never reaches α = 0.05.

**Best estimate of population effect size: r ≈ −0.25 to −0.35.** Weak directional signal (higher mag_agree does tend to mean lower damage) but not a calibrated forecaster. R² ≈ 0.07–0.12 at that effect size leaves 88–93% of variance unexplained.

**F36. What broke the correlation.**

- T10 (cnt+leg, damage 0.5) and T11 (sp+leg, damage 0.5) — middle-of-pack sign-agreement (0.500, 0.551) but near-zero damage. Predictor said these should damage substantially; they didn't.
- T9 (ari+kk, damage 22.0) — middle-of-pack sign-agreement (0.512) but high damage.
- T15 (kk+sdk, damage 64.5, highest seen) — normal sign-agreement (0.531) but catastrophic collapse (KK drops 97.5 → 50.5; sudoku drops 17.5 pts).

The n = 7 sample happened to contain pairs where mag_agree and damage moved together. The wider sample includes pairs where they don't — and crucially, the variance of damage at fixed sign-agreement is large.

**Implication.** Sign-agreement predictor cannot be rescued by more data. Monotonic decay rules out "real effect of −0.5 partially obscured by small-n noise." Correct interpretation: real effect ≈ −0.3, too small for calibrated prediction.

### 6.4 F37 — L1-ARS Falsified

**Method.** Recompute L1_mean for all n = 12 pairs (L1 atom data for 7 tasks already exists from Phase 2).

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

**F37. L1-ARS → damage flips sign at n = 12.**

| Sample | Pearson r | Bootstrap CI | Spearman r (p) |
|---|---:|---|---:|
| 7 (original F33) | −0.761 | — | — |
| **12** | **+0.058** | **[−0.72, +0.67]** | **+0.179 (p = 0.58)** |

The two-predictor regression (sign_agree + L1_mean → damage) drops from R² = 0.70 (n = 7) to **R² = 0.30 (n = 12)**. Neither predictor survives individually or jointly.

**Why.** T10 (cnt+leg) and T11 (sp+leg) have the *lowest* L1_mean (0.136, 0.110) but also the *lowest* damage (0.5, 0.5) — opposite of the F33 story. T9 (ari+kk) has middle L1_mean but highest damage. The n = 7 sample was monotone on L1; the wider sample is not.

**Implication.** The §3 ↔ §5 unification via grounding bottleneck — the paper's most elegant cross-phase link — does not survive sample expansion. Grounding remains the bottleneck for *single-task* RLVR success (§3 holds) but does not transfer to *multi-task* merge compatibility.

### 6.5 F38 — Routing Probe Null

**Motivation.** §5.4.1 hypothesized a cheap "routing probe" — for each task, hand-write 2 atoms that test whether the base model can already answer cross-position questions about the task's input — could prospectively classify Group membership.

**Method.** Operationalize: 2 routing atoms per task × 8 tasks = 16 atoms, 50 samples per atom, zero-shot base-model accuracy at 3B/4B/8B.

**Example routing atoms:**
- `shortest_path.cell_at`: "What is the character at row R, column C of the grid?"
- `syllogism.premise_with_quantifier`: "Two statements: A: ..., B: .... Which contains the word 'all'?"
- `basic_arithmetic.nextop_after_pos`: "In expression X, what is the next operator after position K?"
- `maze.start_position`: "What are the row/col indices of the start character in the grid?"

**Scores:**

| Task | §5.4 Group (4B) | 3B | 4B | 8B |
|---|:---:|---:|---:|---:|
| basic_arithmetic | 1 | 21% | 35% | 29% |
| syllogism | 1 | 67% | 80% | 75% |
| leg_counting | 1 | 67% | 95% | 6% |
| knights_knaves | 2 | 5% | 50% | 4% |
| shortest_path | 2 | 43% | 71% | 61% |
| maze | 2 | 3% | 6% | 3% |
| mini_sudoku | 2 | 7% | 18% | 12% |
| countdown | 2 | 88% | 100% | 37% |

**F38. Routing score does not predict Group membership.** r(routing_score, expected_group) = −0.35 (3B), −0.31 (4B), −0.25 (8B).

**Inspection.** countdown has 100% routing score at 4B but was the clearest G2 task (Δ_up = −14). basic_arithmetic has 35% routing but was clear G1 (Δ_up = −2.5). The probe measures something — likely *how easy the task's token-level structure is for the base model* — but not "does this task need attention restructuring."

**Re-reading the negative result.** A routing atom like "what character is at row R, col C?" tests whether the base model can handle 2D indexing in general. It does not test whether a *novel* attention pattern is needed to solve the full task. The concept was right but the operationalization cannot capture what it needed to capture.

**Implication.** Given F34 (Group membership is pair-dependent anyway), no single-task probe could have worked. A cheap pair-level probe is what would be needed, and we do not know what shape it should take.

### 6.6 F39 — Per-Feature Residual Search

**Setup.** With linear Pearson and pair-level probes failing, search for any single or combined feature that predicts damage at n = 12. Candidates:

- Weight-space (3B and 4B): mag_agree, frac_agree, conflict_mass, up_conflict, q_conflict
- τ magnitudes (Frobenius): tau_mean, tau_min, tau_max, tau_prod
- Cross-scale transfer: 3B mag_agree vs 4B damage, and 4B mag_agree vs 4B damage

**Single-feature Pearson r against 4B damage at n = 12:**

| Feature | r | Sign correct? |
|---|---:|:---:|
| 3B mag_agree | −0.447 | ✓ |
| 3B q_conflict | +0.452 | ✓ (mirror) |
| 3B up_conflict | +0.394 | ✓ (mirror) |
| tau_prod | −0.449 | ✓ |
| tau_mean | −0.418 | ✓ |
| tau_max | −0.380 | ✓ |
| 4B mag_agree | −0.255 | ✓ |
| 4B q_conflict | +0.251 | ✓ |
| tau_min | −0.309 | ✓ |
| 4B up_conflict | +0.243 | ✓ |

**F39. No single feature exceeds |r| = 0.5 at n = 12.** Two-feature regressions: no combination reaches R² > 0.55 on the full sample. Best: `3B mag_agree + tau_prod`, R² = 0.29.

### 6.7 F40 — Task-Level Incompatibility Score (Overfit)

**Method.** For each task T, compute `incompat(T)` = mean damage across pairs containing T. Predict pair damage as `(incompat(A) + incompat(B)) / 2`.

- Full-sample Pearson: r = **+0.82** (looks promising)
- Leave-one-out Pearson: **r = +0.06** (random)
- Leave-one-out MAE: 7.05 points

**F40. Task-level "incompatibility" signal is entirely overfit.** Full-sample r is an artifact of the pair sharing tasks with the training set. Under proper CV the signal vanishes.

Per-task mean damage (for reference only — do not use for prediction):

| Task | n (pairs containing it) | Mean damage | Min | Max |
|---|---:|---:|---:|---:|
| leg_counting | 5 | 7.8 | 0.5 | 18.0 |
| shortest_path | 3 | 8.5 | 0.5 | 18.0 |
| syllogism | 4 | 9.6 | 4.5 | 19.5 |
| countdown | 2 | 12.2 | 0.5 | 24.0 |
| basic_arithmetic | 4 | 15.8 | 4.5 | 24.0 |
| mini_sudoku | 3 | 19.8 | 18.0 | 22.0 |
| knights_knaves | 3 | 20.7 | 18.0 | 22.0 |

Suggestive (mini_sudoku and knights_knaves appear in pairs that tend to damage each other) but the LOO test kills this as a predictor.

### 6.8 Post-Part-6 Spine Summary

| # | Finding | Status after Part 6 |
|---|---|---|
| F1 / F20 | `mlp.down_proj` always droppable (22/24 cells) | **Strengthened** — 40/48 at 4B across 7 pairs |
| F2 / F6 | 1/N under-shoots; ~3 τ-equivalents sweet spot | Retained |
| F19 | Tasks split into G1/G2 by task identity | **Revised** — split is *pair-dependent*, not task-intrinsic. Only mini_sudoku consistently G2 |
| F23 / F24 | SP, sudoku universal G2 | **Only sudoku survives.** SP classifies G1 with KK |
| F28 | Atomic vs direct τ's compose differently | **Retained** (rescoped to task-class-specific) |
| F29 | 3B sign-agreement predicts 4B damage r = −0.81 | **Falsified as predictor** — decays to −0.45 (n=12) → −0.27 (n=19); CI crosses zero past n=7. Retained as weak directional indicator |
| F30 | TIES/DARE competitive with vanilla | Retained |
| F31 | attn.{k,v,o} rotation nonzero (30–70°) | Retained |
| F33 | L1-ARS predicts 4B damage r = −0.76 | **Falsified** — r = +0.06 at n=12, sign flipped |
| — | Routing-probe hypothesis | **Falsified** — r = −0.31 at 4B |
| — | Per-feature residual search | No |r| > 0.5 at n=12 |
| — | Task-level incompatibility | LOO r = 0.06 (overfit) |

**Surviving positive findings from the full Phase-4 arc (post-Part 6):**

1. `mlp.down_proj` always safe to drop (40/48 cells at 4B; mean-positive at 3B/8B).
2. Simple summation at s = 1.0 produces usable 2-τ multi-skill models; 1/N averages under-shoot.
3. Atomic-curriculum and direct-GRPO τ's compose differently — cross-scale-stable sign per task (M_aa wins arith/leg/KK/countdown; M_dd wins SP at every scale); magnitudes vary.
4. attn.{k,v,o} rotation is nonzero and substantial (30–70° across 7 pairs at 4B/8B).
5. Layer-selective merging works *per pair* (F34 restricted scope). Practitioners can use it, but it requires per-pair testing; no shortcut.

The paper is reframed from "two predictors unify §3 and §5" (Path predictor) to "empirical characterization of RLVR τ compositionality with honest null-predictor catalogue" (Path A). Target venue unchanged: TMLR or NeurIPS workshop.

---

## 7. Cross-Phase Synthesis

### 7.1 Universal Empirical Laws

Five statements hold across every phase and scale we tested:

1. **GRPO is spectrum-preserving.** Every singular value of W shifts by less than 0.1%; effective rank changes by less than 0.01. Across 4 scales × 10 tasks × dozens of layers. This is unusually strong for a supervised or RL update; it implies GRPO's entire expressivity budget lives in the span-orthogonal-to-W direction.

2. **τ is near-orthogonal to W's top singular subspace.** At 4B and 8B, ρ_k ≤ (k/r)² (random-isotropic baseline) at k = 10 and k = 100. The orthogonality grows with scale.

3. **`mlp.down_proj` basis is always frozen.** 0°–0.02° principal angle between U_SFT and U_GRPO across 3B/4B/7B/8B and two model families. This is the single cleanest invariant in the project.

4. **`mlp.down_proj` τ contributions are always droppable from a merge.** Follows from (3) empirically. 40/48 (pair, task) cells at 4B; mean-positive or neutral at 3B and 8B.

5. **Single-task GRPO produces a τ with Frobenius magnitude ~0.7–1.2 at 4B** (after 300 steps from the SFT base). The Frobenius magnitude is insensitive to task identity (within a factor of ~1.6); what varies is the direction of the novel subspace and the depth signature.

### 7.2 Scale-Dependent Facts

- **Rotation amount grows monotonically from 3B to 8B.** `mlp.up_proj` principal angle at k = 100: 1.47° (3B) → 1.99° (4B) → 2.49° (7B) → 2.67° (8B). `attn.q_proj`: 1.65° → 1.81° → 2.03° → 2.07°.
- **Depth signatures reorganize.** 3B has one "early super-cluster" where most tasks concentrate change. 8B separates into three clean bands (early-only, mid-early, mid-late). 4B is intermediate.
- **Group-2 attention-restructuring classification is scale-specific** (before being further revised to pair-dependent in Chapter 6). countdown and knights_knaves are G2 at 4B but G1 at 8B. Shortest_path and sudoku are G2 at 4B and 8B (across Part-3 single-pair evidence).
- **M_aa vs M_dd magnitude is 4B-specific.** Cross-scale Δ(M_aa − M_dd) mean: −3.3 (3B-Qwen2.5), +7.9 (4B-Qwen3), +0.3 (8B-Qwen3). The per-task sign pattern is cross-scale stable; the magnitude is not.

### 7.3 Pair-Dependent Facts (The Biggest Surprise)

- **Group-1/Group-2 layer sufficiency depends on the pair, not the task.** Shortest-path is G2 with Arith but G1 with KK. Syllogism is G1 with SP but G2 with Arith. Only mini_sudoku is stably G2 across pairs.
- **Pair merge damage spans 0.5 to 64.5 at 4B** (n = 19) with no single-feature predictor of |r| > 0.5.
- **Low-L1-grounding pairs can produce low-damage merges** (T10 cnt+leg: L1_mean 0.136, damage 0.5; T11 sp+leg: L1_mean 0.110, damage 0.5). The "grounding bottleneck predicts merge compatibility" story falls at n = 12.
- **Two attention-restructuring tasks can merge catastrophically** (T15 kk+sdk: damage 64.5; KK drops 47 pts). Sign-agreement is normal (0.531) — nothing in the 3B sign-agreement signals this.

### 7.4 What We Originally Claimed and Had to Drop

| Claim | Original evidence | Post-Part-6 status |
|---|---|---|
| Sign-agreement predictor r = −0.81 | n = 7, bootstrap CI [−0.99, −0.38] | **Falsified as predictor.** True effect ≈ −0.25 to −0.35. Retained as weak directional indicator |
| L1-ARS predictor r = −0.76 | n = 7 | **Falsified.** r = +0.06 at n = 12; CI [−0.72, +0.67] |
| Two-predictor regression R² = 0.70 | n = 7 | **Drops to 0.30 at n = 12** |
| Task-intrinsic Group classification | SP+Arith at 4B, partial 3B/8B replication | **Pair-dependent**; only mini_sudoku stably G2 |
| Grounding bottleneck unifies §3 and §5 | F33 at n = 7 | **Falsified at n = 12.** The §3 story stands on its own; the cross-phase bridge does not |
| Task-level routing probe predicts Group | Hypothesis | **Falsified** at r = −0.31 |
| Single-feature residual predictor | Hypothesis | **Falsified.** All |r| < 0.5 at n = 12 |
| Task-level incompatibility score | Full-sample r = 0.82 | **Overfit.** LOO r = 0.06 |

### 7.5 The Five Surviving Positive Findings Across 4 Phases

These are the paper spine (Path A reframe, 2026-05-12):

1. **GRPO's universal structural signature** (Chapter 4): spectrum-preserving; τ near-orthogonal to W's top subspace; `mlp.down_proj` basis frozen; `mlp.up_proj + attn.q_proj` rotate 1.5°–2.7° monotonically with scale; rotation amount is task-invariant while rotation direction is task-specific; depth signatures predict transfer better than global cosine.

2. **The ARS training-regime rule** (Chapter 3): at 4B-Qwen3, ARS correctly predicts the best of {Direct-RLVR, Atomic-Only, Atomic-Warm} on 6/7 tasks. Low-ARS tasks can suffer catastrophic Direct-RLVR failure (maze: 1% vs 21.5% base); atomic curriculum rescues them. The arithmetic exception is a reasoning-compression pathology orthogonal to grounding.

3. **Drop `mlp.down_proj` from merges** (Chapter 5): 40/48 cells at 4B within ±3 of full merge; mean-positive at 3B and 8B. A single, actionable, geometrically-motivated recipe.

4. **Atomic vs direct τ's compose differently** (Chapter 5 F28): cross-scale-stable per-task sign. M_aa wins arith / leg_counting / knights_knaves / countdown at every scale; M_dd wins shortest_path at every scale. Pair this with the training-regime rule to build heterogeneous multi-skill models.

5. **Null-predictor catalogue** (Chapter 6): on n = 19 pairs at 4B, no single cheap pre-probe — sign-agreement, L1-ARS, routing score, global τ cosine, Phase-1 transfer, per-layer flat cosine, novel-subspace angle, τ magnitude — predicts merge damage at a useful effect size. This is a genuine negative-result contribution; the field has implicitly assumed task-intrinsic merge structure and our data falsifies that.

---

## 8. Practical Recipes

### 8.1 For Single-Task RLVR

**Run the ARS probe first (~1 GPU-hour).** Compute L1 and L2 atom accuracies for the target task on the SFT base.

- If ARS < 25% (low): Direct-RLVR may fail catastrophically. Use Atomic-Only (200 steps atom GRPO, no full-task) for feature-routing-style tasks like arithmetic; use Atomic-Warm (atoms → full) for spatial-search tasks like shortest_path or maze.
- If ARS 40–60% (medium): Direct-RLVR. The bottleneck is L3 planning; atoms won't help.
- If ARS > 60% (high): Direct-RLVR usually works. If the task has long reasoning chains (arithmetic-like), watch for reasoning compression; Atomic-Only is safer.

**If atoms have short answers (yes/no, single numbers), expect reasoning compression.** Use L2-long atoms (multi-step simulation) instead. The shortest-path case gained +5.5 points by switching to path-tracing atoms.

**If atoms perform below 20% during training, your verifier is probably too strict.** Atomic rewards need numeric-extraction, set-comparison, yes/no-prefix lenient matching.

### 8.2 For 2-τ Merges

**Always drop `mlp.down_proj` contributions.** Free safety; occasional small gain. No exceptions found across 40+ merge cells at 4B.

**Empirically test aggressive masking per pair.** The `up_proj + gate_proj` mask works for some pairs (saving 82% of keys) and fails for others. No prospective rule tells you which. Run `all_but_down` first, then try `up` only — if `up` matches `all_but_down`, use it; otherwise use `all_but_down`.

**Simple weight summation at s = 1.0 is your default.** TIES and DARE don't consistently beat it on reasoning RLVR τ's (within ±2 pts across 3 tested pairs).

**Expect damage variance pair-to-pair.** Across n = 19 at 4B, damage spans 0.5–64.5. No cheap probe will tell you in advance which pairs will produce which. Build the merge and measure.

**If building a multi-skill model from > 2 τ's, scale s down.** Three τ's at s = 1.0 over-shoots the ~3 τ-equivalents sweet spot. Start at s = 0.7 for three τ's; s = 0.33 for the full 9-τ set.

### 8.3 For Heterogeneous Training Regimes

Pair the ARS regime choice (§8.1) with the merge-regime effect (F28):

- For feature-routing tasks (arithmetic, syllogism, knights_knaves, leg_counting, countdown): train with **atomic curriculum**. Their τ's merge cleaner on feature-routing-dependent pairs.
- For spatial-search tasks (shortest_path, maze): train with **direct GRPO** (or atomic-warm starting). Their τ's need attention restructuring that atomic curriculum alone doesn't produce.

End-to-end: ARS probe → regime choice → train single-task τ → merge with `all_but_down` mask at s = 1.0. This is the heterogeneous recipe that combines all 4 phases.

### 8.4 What Not to Do

- **Don't use 1/N averaging over many τ's.** It under-shoots systematically (8 non-trivial tasks at 41.6% vs 51.3% for s=0.33 or 51.2% for 2-task at 4B).
- **Don't rely on any of the pre-screens we tested** for n > 7 pairs: sign-agreement, L1-ARS, routing probe, global τ cosine. All collapse at larger n.
- **Don't pre-classify a task as "always Group-1" or "always Group-2."** Classification depends on the pair.
- **Don't use partial-credit rewards for GRPO on spatial search.** They kill gradient variance. Binary rewards + hard difficulty transitions work best (shortest-path ideas C, D proved this).
- **Don't heavy-SFT-interleave GRPO** (Idea G collapsed 21% → 5.5%). Light SFT (≤500 samples, lr = 1e-6) is tolerated; heavy SFT overwrites the policy.

---

## 9. Catalogue of Null Results

This is a consolidated list of every predictor or hypothesis that failed in Phases 1–4, with sample size and Pearson r. Useful for the field to avoid re-running these.

| Hypothesis | Predictor | Outcome | Sample | r / R² |
|---|---|---|---|---|
| KL penalty rescues 7B arith self-harm | add `kl_coeff = 0.01–0.1` | No | 1 (single model) | n/a |
| β > 0 in DAPO rescues arith | `beta = 0.01–0.05` | No | — | n/a |
| Doubling rollouts rescues arith | 8 → 16 | No | — | n/a |
| Lower LR fixes harm (preserves gain) | 5e-6 → 1e-6 | Partially — kills gain too | 20 runs (7B/8B v2) | n/a |
| Phase-1 transfer predicts merge damage | |off-diagonal transfer| | No | n = 7 | r ≈ 0 |
| Global τ cosine predicts merge damage | cos(τ_A, τ_B) | No | n = 7 → 19 | r ≈ 0 throughout |
| Per-layer opposing flat cosines predict merge damage | #layers with up flat cos < −0.5 | No | n = 7 | +0.23 |
| Novel-subspace angle predicts merge damage | mean `mlp.up` principal angle between novel | Modest at n = 7 | n = 7 | +0.56 (not re-run n > 7) |
| τ magnitude predicts transfer | ‖τ‖_F | No | n = 10 | 0.459 (p = 0.252) |
| Effective rank of τ predicts transfer | eff_rank(τ) | No | n = 10 | −0.046 |
| τ spectral norm predicts single-task accuracy | σ₁(τ) | No | n = 10 | weak |
| Sign-agreement at 3B predicts 4B merge damage | mag_agree | **Falsified** | n = 7 → 19 | −0.81 → −0.45 → **−0.27** |
| L1-ARS at 3B predicts 4B merge damage | L1_mean | **Falsified** | n = 7 → 12 | −0.76 → **+0.06** |
| Task-level routing probe predicts Group | base accuracy on cross-position atoms | No | 3B/4B/8B | −0.25 to −0.35 |
| Task-level "incompatibility" score predicts | mean damage across pairs containing task | **Overfit** | n = 12 | Full 0.82, LOO 0.06 |
| `tau_prod` (magnitude product) predicts damage | ‖τ_A‖ · ‖τ_B‖ | No | n = 12 | −0.449 |
| 4B mag_agree predicts 4B damage | same-scale sign-agreement | Weaker than 3B | n = 12 | −0.255 |
| 4B q_conflict predicts damage | q-layer sign conflict | No | n = 12 | +0.251 |
| Two-feature regression `3B mag_agree + tau_prod` | — | Weak | n = 12 | R² = 0.29 |
| Progressive difficulty curricula (3×3 → 4×4 → full) beat hard transition | progressive grid-size curriculum | No | shortest-path | 46.5% vs 51.5% |
| Partial-credit rewards help GRPO | graded rewards | **No, actively hurts** | shortest-path | 35.5% vs 44% |
| Heavy SFT interleaving stabilizes GRPO | 5000 samples, lr = 5e-6 every 50 steps | **Collapses training** | — | 21% → 5.5% |
| SFT-L1 (high LR) then full GRPO | SFT grounding at lr = 2e-5 | **Collapses** | — | 2% final |
| Feasible-only training fixes false-infeasible | filter data to feasible grids | No | shortest-path | 40.5% vs 51.5% |
| On-policy RFT on own checkpoint improves it | RFT of B's 51.5% model | Plateaus | — | 51.5% unchanged |
| caesar_cipher improves under any regime | any GRPO | No | every scale | 0% → 0–0.5% |
| cryptarithm improves under any regime | any GRPO | No | every scale | 0% → 0% |
| 4B countdown original GRPO from `sft_qwen3_4b` | wrong base | Training used incompatible SFT | — | ‖τ‖ = 151 outlier |

---

## 10. Limitations and Open Problems

### 10.1 Qwen Family Confound

3B and 7B are Qwen2.5; 4B and 8B are Qwen3. Some "scale effects" are family effects. `basic_arithmetic` depth migrates mid-early in Qwen2.5 but mid-late in Qwen3-8B. The linking experiment (F28) has magnitude differences between 3B-Qwen2.5 (−3.3) and 4B-Qwen3 (+7.9) that may be partly family. Without a matched same-family scaling ladder (e.g., Qwen3 at 3 sizes), we cannot separate these.

**What would resolve this.** Add Qwen3-1.7B and Qwen3-14B to give a same-family 1.7B/4B/8B/14B ladder. Repeat rotation-amount and depth-signature measurements.

### 10.2 n = 19 Is Still Small for Predictor Claims

19 pair merges at 4B was enough to rule out calibrated prediction from the cheap probes we tried. It is not enough to claim a high-dimensional machine-learned predictor couldn't work. A systematic feature-search over per-layer sign agreement, per-depth novel-subspace angle, and ARS atom-level profiles would require n = 50+ with cross-validation discipline to be meaningful. This is the primary remaining open problem for Phase 4.

### 10.3 Single Training Seed per τ

Every GRPO run used seed = 42. Eval-seed variance is small (≤ 3 points typical from the Stream-C seed-1234 re-evals). Training-seed variance is untested. Key question: at n = 19 pairs, how much of the damage variance is "true pair-specific damage" vs "training-seed noise for one-off checkpoints"? A 3-seed × 3-pair mini-study would bound this.

### 10.4 Pair-Level Probe Design Remains Open

The F34 finding that Group membership is pair-dependent — combined with the F38 negative result on single-task routing probes — means any useful pre-probe must measure something about the joint pair. Candidates not yet tested:

- Whether two tasks' atoms produce correlated activations on the SFT base (needs activation probing)
- Whether two tasks' gradient directions at W_SFT are correlated before any training
- Whether training A and then taking a single gradient step on B produces a direction opposing A's τ

None of these are trivially cheap to compute. The practitioner's procedure for now is empirical (build the merge, test).

### 10.5 Atomic Curriculum Matched-Compute Rerun

The 3B and 8B atomic curricula used in Phase 4 Part 5 (F28) are 100 steps L1 + 100 steps L2, identical to 4B Phase 2's original. They did *not* include RFT-Base refinement. The 4B F28 number (+7.9) may benefit from the extra refinement; a matched-compute re-run at 3B and 8B would sharpen the cross-scale magnitude comparison. 1–2 pod-days.

### 10.6 Non-Qwen Replication

Every result is on Qwen2.5 or Qwen3. Replicating Chapter 4 (rotation invariants, `mlp.down_proj` freezing) and Chapter 5 (merge recipes) on Llama-3-8B or Gemma-2-9B would strengthen the "universal GRPO signature" claim enormously. The two families we currently have differ only at the LayerNorm/RoPE scaling level; an independent architecture family would test the invariance more rigorously.

### 10.7 Verifier Quality

All reasoning-gym verifiers are deterministic but not all are perfect. The maze verifier had off-by-one bugs in early versions that contributed to Phase-1 maze volatility. The `mini_sudoku` verifier accepts some invalid grids (partial-correct grids get non-zero reward). These affect downstream numbers but are small compared to the effects we characterize.

### 10.8 Cryptarithm and Caesar Inert at Every Scale

These two tasks produce 0% reward during training at every scale we tested; their τ's are zero or near-zero. They contribute nothing to any Phase 4 merge. This means we effectively have 8 tasks × 19 pairs of merge data, not 10. Scaling this project to a richer task set (or a task where the SFT base has non-zero baseline on the hard ciphers) would be a natural extension.

### 10.9 GRPO-Specific vs RL-General

All findings are GRPO + DAPO loss. Different RL algorithms (PPO with learned value, ReMax, REINFORCE-from-mean) would likely produce different τ's. Whether the universal structural signature (spectrum-preserved, down_proj-frozen) survives non-GRPO RL is open.

---

## 11. Artifacts and Reproducibility

### 11.1 Phase 1 Artifacts

**Code:**
- `train_grpo.py` — main training script (TRL + GRPOTrainer, DDP via torchrun)
- `evaluate_transfer.py` — 10-task evaluator with last-`<answer>` fix
- `environments/{maze,syllogism,gsm8k,battleship,atomic}_env.py`

**Results:** Green cluster `/scratch/sunguha/rlvr_experiments/`

| Path | Contents |
|---|---|
| `outputs/sft_qwen25_3b/` | 3B SFT base |
| `outputs/sft_qwen_4.0b_instruct/` | 4B SFT base |
| `outputs/sft_qwen25_7b/` | 7B SFT base |
| `outputs/sft_qwen3_8b/` | 8B SFT base |
| `outputs/grpo_qwen25_3b_{task}/` | 3B single-task GRPO (10 tasks) |
| `outputs/grpo_qwen4b_{task}/` | 4B single-task GRPO (10 tasks) |
| `outputs/grpo_qwen25_7b_{task}/` | 7B single-task GRPO (10 tasks) v1 + v2 |
| `outputs/grpo_qwen3_8b_{task}/` | 8B single-task GRPO (10 tasks) v1 + v2 |
| `outputs/eval_transfer_{scale}/` | 10×10 eval outputs per scale |

### 11.2 Phase 2 Artifacts

| Path | Contents |
|---|---|
| `phase2_arith/{stage1_l1, stage2_l2, stage3_full}/` | Arithmetic curriculum stages |
| `phase2_arith/direct_rlvr/`, `phase2_arith/atomic_warm_300/` | Additional arith variants |
| `phase2_countdown/stage{1,2,3}_*/` | Countdown stages |
| `phase2_sp/stage{1,2,3}_*/` | Shortest-path stages |
| `phase2_sp/idea_{a,b,c,d,e,f,g,l,rft,rft_base}/` | SP 15-condition deep dive |
| `phase2_sp/idea_rft_base/grpo/` | **The 55.5% RFT-Base SP checkpoint** |
| `phase2_ext/{syllogism,knights_knaves,maze,mini_sudoku}/` | 4 Phase-2-extended tasks |
| `outputs/ars_4b_{task}.json` | 7 ARS probes at 4B |
| `environments/atomic_templates.py` | All 7-task atom definitions |
| `environments/atomic_env.py` | Atomic GRPO environment |
| `probe_atomics.py` | ARS probe script |

### 11.3 Phase 3 Artifacts

| Path | Contents |
|---|---|
| `results/weight_analysis/{3b,4b,7b,8b}/analysis_results.json` | Phase-A/B for each scale |
| `results/weight_analysis/{3b,4b,7b,8b}/subspace_rotation.json` | Projection + angles per scale |
| `results/weight_analysis/8b/eigenvalue_shifts.json` | Full SVD comparison at 8B |
| `results/weight_analysis/8b/per_layer_task_diff.json` | Per-layer cross-task (8B) |
| `results/weight_analysis/8b/rotation_direction.json` | Novel-subspace + flat cos (8B) |
| `analyze_rlvr_weight_changes.py` | τ magnitudes, spectral, transfer correlation |
| `analyze_eigenvalue_shifts.py` | Full-SVD comparison |
| `analyze_subspace_and_rotation.py` | ρ_k + principal angles |
| `analyze_rotation_direction.py` | Novel subspace + flat cos |

### 11.4 Phase 4 Artifacts (Parts 1–5)

| Path | Contents |
|---|---|
| `results/phase2_sp/merged/phase1_all_1overN/` | 10τ 1/N merge (Part 1) |
| `results/phase2_sp/merged/all_tasks_s0.33/` | 9τ s=0.33 merge |
| `results/phase2_sp/merged/basic/` | 2τ SP+Arith s=1.0 merge |
| `results/phase4_pairs/{T1..T7}/basic/` | 7 pair merges (Part 2) |
| `outputs/eval10_phase4_{T1..T7}/` | 7 pair eval dirs |
| `results/phase4_layersel/{up, up_q_clean, all_but_down}/` | 3 layer-selective (Part 3) |
| `results/phase4_triple/basic/` | Triple merge SP+Arith+Syl |
| `results/phase4_replicate_{3b,8b}/{sp_ari/basic, sp_ari_up, sp_ari_all_but_down}/` | 3B/8B replication (Part 4) |
| `results/weight_analysis/{3b,4b,8b}/sign_agreement.json` | Sign agreement at 3 scales |
| `paper_prep/linking/{M_aa, M_dd, M_mix}/` | Part 5 linking merges at 4B |
| `paper_prep/linking_{3b,8b}/{M_aa, M_dd, M_mix}/` | Linking merges at 3B/8B |
| `paper_prep/baselines/{T2,T5,T7}_*/{ties,dare}/` | TIES + DARE baselines |
| `paper_prep/analysis/rotation_attn_kvo_{4b,8b}.json` | attn.{k,v,o} rotation |
| `paper_prep/analysis/bootstrap_r.json` | Bootstrap CI + permutation p |
| `paper_prep/analysis/held_out_predictor.json` | LOO + fixed-split validation |
| `paper_prep/ars/{3b,4b}/ars_<scale>_<task>.json` | ARS probes at 3B/4B |

### 11.5 Phase 4 Part 6 Artifacts

| Path | Contents |
|---|---|
| `paper_prep/gaps/layersel/{T1..T7}_{up, up_q_clean, all_but_down}/` | 18 layer-selective merges (6 pairs × 3) |
| `paper_prep/gaps/eval/layersel/*/` | 144 eval metrics |
| `paper_prep/gaps/newpairs/{T8..T12}/` | Tranche 1: 5 vanilla 4B merges |
| `paper_prep/gaps_pathB/merges/{T13..T19}/` | Tranche 2: 7 vanilla 4B merges |
| `paper_prep/gaps_pathB/evals/{T13..T19}/` | 56 eval metrics |
| `paper_prep/gaps/sign_agreement/new_pairs_{3b,4b}.json` | SA for T8–T12 |
| `paper_prep/gaps_pathB/sa/new_pairs_{3b,4b}.json` | SA for T13–T19 |
| `paper_prep/gaps/routing_probe/{3b,4b,8b}.json` | Routing probe results |
| `paper_prep/gaps_pathB/predictor_n19.json` | n = 19 predictor analysis |
| `paper_prep/gaps/task_incompat.json` | LOO task-incompatibility |
| `paper_prep/gaps/residual_features.json` | 10-feature residual search |

### 11.6 MMLU Evaluation

48 checkpoints evaluated on:
- `mmlu_abstract_algebra` (100 Q)
- `mmlu_elementary_mathematics` (378 Q)
- `mmlu_high_school_mathematics` (270 Q)

Finding: all models within ±3% of their SFT baseline. No catastrophic forgetting.

**Summary by scale (from report_phase2_complete.md §MMLU Evaluation):**

| Model size | SFT baseline | GRPO range | Verdict |
|---|---|---|---|
| **3B** | 47% / 61% / 50% | 44–47% / 59–61% / 50–51% | Slight degradation (1–3%) |
| **4B** | 56% / 71% / 52% | 56–58% / 70–71% / 51–53% | **Maintained or improved** |
| **8B** | 52% / 70% / 52% | 53–55% / 70–71% / 50–52% | Maintained |

(Format: abstract_algebra / elementary_math / high_school_math.)

**SFT baselines:**

| Model | Abstract Algebra | Elementary Math | High School Math |
|---|---:|---:|---:|
| sft_3b | 47.0% | 60.8% | 50.4% |
| sft_4b | 56.0% | 70.6% | 51.5% |
| sft_8b | 52.0% | 70.4% | 51.9% |

**Phase 1 GRPO on single tasks — 4B:**

| Task | Abstract Algebra | Elementary Math | High School Math |
|---|---:|---:|---:|
| grpo_4b_arithmetic | 57.0% | 70.9% | 52.2% |
| grpo_4b_countdown | 56.0% | 70.9% | 52.2% |
| grpo_4b_shortest_path | 58.0% | 69.8% | 51.9% |
| grpo_4b_syllogism | 58.0% | 70.4% | 51.1% |
| grpo_4b_maze | 57.0% | 69.8% | 51.9% |
| grpo_4b_knights_knaves | 57.0% | 70.4% | 52.2% |
| grpo_4b_mini_sudoku | 56.0% | 70.6% | 52.2% |
| grpo_4b_leg_counting | 58.0% | 70.6% | 52.6% |
| grpo_4b_caesar_cipher | 56.0% | 70.4% | 51.5% |
| grpo_4b_cryptarithm | 56.0% | 70.6% | 51.5% |

All 4B GRPO models maintain or slightly improve on the sft_4b baseline.

**Phase 1 GRPO on single tasks — 3B:**

| Task | Abstract Algebra | Elementary Math | High School Math |
|---|---:|---:|---:|
| grpo_3b_arithmetic | 44.0% | 60.8% | 49.6% |
| grpo_3b_countdown | 46.0% | 59.3% | 50.7% |
| grpo_3b_shortest_path | 46.0% | 60.3% | 49.6% |
| grpo_3b_syllogism | 45.0% | 61.1% | 51.5% |
| grpo_3b_maze | 45.0% | 61.4% | 50.4% |
| grpo_3b_knights_knaves | 46.0% | 60.6% | 50.0% |
| grpo_3b_mini_sudoku | 47.0% | 61.1% | 51.1% |
| grpo_3b_leg_counting | 44.0% | 60.6% | 50.0% |

3B models show slight degradation (1–3%) on abstract algebra but maintain elementary/high school math performance.

**Phase 1 GRPO on single tasks — 8B:**

| Task | Abstract Algebra | Elementary Math | High School Math |
|---|---:|---:|---:|
| grpo_8b_arithmetic | 55.0% | 70.9% | 51.5% |
| grpo_8b_countdown | 53.0% | 69.6% | 52.2% |
| grpo_8b_shortest_path | 53.0% | 69.8% | 50.7% |
| grpo_8b_syllogism | 54.0% | 69.8% | 51.9% |
| grpo_8b_knights_knaves | 53.0% | 70.9% | 51.9% |
| grpo_8b_mini_sudoku | 53.0% | 70.9% | 52.2% |
| grpo_8b_leg_counting | 53.0% | 70.1% | 50.4% |

8B models maintain baseline performance (within ±2%).

**Phase 2 atomic curriculum — 4B:**

| Condition | Abstract Algebra | Elementary Math | High School Math |
|---|---:|---:|---:|
| p2_arith_atomic_only | 58.0% | 70.1% | 52.2% |
| p2_arith_atomic_warm | 58.0% | 70.9% | 51.9% |
| p2_sp_atomic_warm | 58.0% | 70.4% | 51.9% |
| p2_sp_rft_base | 58.0% | 70.4% | 51.1% |
| p2_sp_idea_b | 59.0% | 70.4% | 51.5% |

All atomic curriculum variants maintain baseline performance.

**Phase 2 Extended (4 new tasks) — 4B:**

| Condition | Abstract Algebra | Elementary Math | High School Math |
|---|---:|---:|---:|
| p2ext_syllogism_direct | 57.0% | 70.6% | 51.5% |
| p2ext_syllogism_atomic_only | 57.0% | 70.4% | 52.2% |
| p2ext_syllogism_atomic_warm | 55.0% | 70.9% | 52.6% |
| p2ext_knights_knaves_direct | 57.0% | 70.6% | 51.5% |
| p2ext_knights_knaves_atomic_only | 57.0% | 70.6% | 51.5% |
| p2ext_knights_knaves_atomic_warm | 57.0% | 70.4% | 51.5% |
| p2ext_maze_direct | 58.0% | 70.1% | 52.2% |
| p2ext_maze_atomic_only | 58.0% | 70.4% | 51.9% |
| p2ext_maze_atomic_warm | 58.0% | 69.8% | 51.9% |
| p2ext_mini_sudoku_direct | 56.0% | 70.6% | 50.7% |
| p2ext_mini_sudoku_atomic_only | 58.0% | 70.6% | 52.2% |
| p2ext_mini_sudoku_atomic_warm | 57.0% | 70.1% | 52.2% |

All Phase 2 Extended conditions maintain baseline performance.

**Phase 4 layer-selective merges — 4B:**

| Merge type | Abstract Algebra | Elementary Math | High School Math |
|---|---:|---:|---:|
| p4_layersel_all_but_down | 57.0% | 69.8% | 51.9% |
| p4_layersel_up | 57.0% | 69.8% | 52.2% |
| p4_layersel_up_q_clean | 57.0% | 69.8% | 51.9% |

Layer-selective task-vector merging preserves general knowledge. Across all 48 models — 3 scales × 4 phases × 10+ reasoning tasks — no catastrophic forgetting is observed on MMLU math subsets.

### 11.7 Pod Infrastructure

- **Green cluster** P6-B200 nodes (8× B200 @ 178 GB each)
- Pods used: `sunguha-rlvr-phase2-qbgw` (Phase 2–3), `sunguha-rlvr-paper-npdz` (Part 5 + Part 6 tranche A), `sunguha-rlvr-paper2-n25r` (Part 6 tranche B)
- Shared FSx filesystem: `/scratch/sunguha/rlvr_experiments/`
- Framework: TRL 1.0.0 + transformers 4.57.6 + reasoning-gym (from source, pinned commit 2026-03-15)

---

## 12. Chronology (Dated Log)

| Date | Milestone |
|---|---|
| 2026-01 | Phase 1 begins. 3B and 4B transfer matrices computed |
| 2026-02 | Phase 1 completes. 7B v1 + v2 matrices and 8B v1 + v2 matrices |
| 2026-03-15 | Reasoning-gym pinned commit. Verifier bugs identified; some Phase-1 cells re-evaluated |
| 2026-04-17 | Answer-extraction bug found and fixed (`re.findall` with last match). Affects Phase 1 numbers |
| 2026-04-18 | Phase 2 begins. Initial 3-task study (arith, countdown, SP) |
| 2026-04-19 | Phase 3 Phase-A/B analysis runs. Spectrum-preserved finding |
| 2026-04-21 | Phase 3 subspace/rotation runs. `mlp.down_proj` frozen finding |
| 2026-04-22 | Phase 3 novel-subspace analysis. Flat-cosine ±0.9 range observed |
| 2026-04-23 | Phase 4 Part 1: 1/N vs s=0.33 vs 2-task at 4B |
| 2026-04-23 | Phase 2 Idea B (L2-long atoms): 51.5% SP |
| 2026-04-24 | Phase 4 Parts 2, 3, 4: 7 pair merges + layer-selective + 3B/8B replication |
| 2026-04-24 | F29 first appearance: 3B sign-agreement → 4B damage r = −0.81 |
| 2026-04-25 | Phase 2 RFT-Base: 55.5% SP (new best) |
| 2026-04-29 | Phase 2 complete: 7-task analysis published as `report_phase2_complete.md` |
| 2026-05-01 | MMLU forgetting eval across 48 checkpoints |
| 2026-05-05 | Phase 4 Part 5 begins (4 parallel streams on pod-npdz) |
| 2026-05-06 | Part 5 completes: F28 cross-scale, F29 validation, TIES/DARE, F31 attn.{k,v,o} |
| 2026-05-06 | F33 added: L1-ARS at 3B predicts 4B damage r = −0.76 (n = 7) |
| 2026-05-07 | Draft paper written with "two predictors unify §3 and §5" headline |
| 2026-05-09 | Part 6 begins: 3 gap-closing experiments launched |
| 2026-05-10 | Layer-selective on 6 new pairs (F34). SP flips G2 → G1 with KK; syl flips G1 → G2 with arith |
| 2026-05-10 | Tranche 1: n = 7 → 12. Sign-agreement r drops to −0.45; CI crosses zero. L1-ARS flips to +0.06 |
| 2026-05-11 | Routing probe falsified (r = −0.31). Residual search: no |r| > 0.5 |
| 2026-05-11 | Task-level incompatibility: full-sample r = 0.82, LOO r = 0.06 (overfit) |
| 2026-05-12 | Tranche 2 / Path B: n = 12 → 19. Sign-agreement r decays further to −0.27 |
| 2026-05-12 | Paper rewritten (Path A reframe). This master doc consolidates all four phases |

---

## 13. Paper Mapping

Which master-doc sections become which paper sections; what is master-only.

| Paper section | Master doc source | Notes |
|---|---|---|
| §1 Introduction | — (paper-only) | Drawn from §0.1–0.3 + §7.5 for framing |
| §2 Setup | Chapter 1 | Tighten; drop compute breakdowns. Paper lists §1.1, §1.3, §1.4 briefly |
| §3 (Phase 1 — transfer) | Chapter 2 | 3B/4B matrices in main body; 7B/8B in Appendix B. Drop caesar/crypto rows in main body |
| §4 (Phase 2 — ARS) | Chapter 3 | Figures 4.1–4.2, Table 1. Appendix C has per-task atom tables, shortest-path deep dive. MMLU forgetting goes in Appendix |
| §5 (Phase 3 — geometry) | Chapter 4 | Figures 5.1–5.6, Tables 2–5. Appendix F–J has full per-layer data |
| §6 (Implications + bridge) | §4.8, §4.9 | 1-page narrative bridge from Phase 3 to Phase 4 |
| §7 (Phase 4 — merging, Path A) | Chapter 5 + Chapter 6 | Paper §7.10 = master Chapter 6. Paper §7.3 is the null-predictor catalogue distilled from Chapter 9 |
| §8 Cross-phase synthesis | Chapter 7 | Condensed |
| §9 Related work | — (paper-only) | Not in master |
| §10 Limitations | Chapter 10 | Shortened |

**Master-only material:** Chapter 8 (practical recipes), Chapter 9 (full null-result catalogue), Chapter 11 (artifacts), Chapter 12 (chronology), Appendices B (full per-scale transfer matrices), D (full per-task ARS breakdowns), E (full 19-pair merge table), F (TIES/DARE full results), G (hyperparameter table).

**TODOs flagged in this doc:**

- §4.9.2 countdown 4B re-run pending (‖τ‖ = 151 outlier)
- §10.5 matched-compute atomic curricula at 3B/8B (1–2 pod-days)
- §10.6 Non-Qwen replication (Llama-3, Gemma-2)
- §10.3 training-seed variance mini-study (3 seeds × 3 pairs)
- §10.4 pair-level probe design remains open
- §11.4: the 4B T7 checkpoint in F31 attn.{k,v,o} data is missing; 6 of 7 4B cells complete

---

## Appendices

### Appendix A · Glossary of All Metrics

**Frobenius norm ‖M‖_F.** Sqrt-sum-squared of all entries. Equivalent to sqrt of sum of squared singular values.

**Relative magnitude ‖τ‖/‖W‖.** Per-layer ratio of τ magnitude to W magnitude. Values 0.06%–0.12% across scales.

**Cosine similarity cos(τ_A, τ_B).** Dot product of vectorized τ's divided by product of norms. +1 = identical; 0 = orthogonal; −1 = opposite.

**Spectral norm ‖M‖₂ = σ_1(M).** Largest singular value. Maximum amplification M applies to any unit vector.

**Effective rank.** exp(Shannon entropy of normalized singular values). High = information spread across many directions; low = focused on a few directions.

**Singular value decomposition W = UΣV^T.** U (columns = left singular vectors = output directions), V (right = input directions), σ_i = how much direction i is amplified.

**Subspace projection ratio ρ_k(τ).** Two-sided: ‖U_k U_k^T τ V_k V_k^T‖_F² / ‖τ‖_F². Fraction of τ's energy inside W_SFT's top-k singular subspace. ρ_k → 1: τ lives in top-k. ρ_k → (k/r)²: random-isotropic. ρ_k < (k/r)²: anti-aligned, prefers mid/low-rank.

**Principal angles between two k-dim subspaces.** arccos of singular values of U_A^T U_B where U_A, U_B are orthonormal bases. Zero = coincide; 90° = orthogonal.

**Novel subspace.** novel_A = (I − U_SFT U_SFT^T) U_GRPO(A)[:, :k]. The component of task A's top-k basis orthogonal to SFT — "new directions A introduced."

**Pairwise principal angle between novel_A and novel_B.** Same principal-angle formula on the novel subspaces. 0° = tasks rotated same direction; 90° = perpendicular directions.

**Flat rotation cosine.** cos(vec(novel_A), vec(novel_B)). Preserves orientation (principal angles ignore sign). +1 = same direction same orientation; −1 = opposite directions (microscopic interference signature).

**Depth signature.** Per-layer distribution of ‖τ‖_F across layer indices, normalized to sum to 1. Bucketed into early/mid-early/mid-late/late quartiles.

**Depth-profile cosine.** Cosine between two tasks' depth signatures. Measures "do they edit similar layers?"

**Frac sign agreement.** Fraction of shared-nonzero parameters where sign(τ_A) = sign(τ_B). Random baseline is 0.5 for orthogonal τ.

**Magnitude-weighted sign agreement (mag_agree).** Sum of |τ_A| · |τ_B| at sign-agreeing positions, divided by sum of all |τ_A| · |τ_B|. Range [0, 1]. Weighted by how important each parameter is.

**Conflict mass.** 1 − mag_agree. Sum of |τ_A| · |τ_B| at sign-disagreeing positions.

**Atomic Readiness Score (ARS).** Zero-shot mean accuracy of the SFT base on the L1 + L2 atoms of a task. L1 = grounding atoms (parsing input); L2 = local reasoning atoms; L3 is the full task (not part of ARS).

**Pair damage.** max(0, Direct-A − merge) + max(0, Direct-B − merge). Total capability lost by the merge relative to single-task sources.

### Appendix B · Full Per-Scale Transfer Matrices

See §2.4.1–§2.4.6 for all six matrices (3B, 4B, 7B v1, 7B v2, 8B v1, 8B v2).

### Appendix B.1 · Generation Log Analysis (from report_transfer_matrix.md Appendix B)

Sample generations from the Qwen3-8B model (v1) illustrate failure modes per task.

**B.1 Countdown (baseline 32% → GRPO 82%, +50%).** Baseline failure: the model tries many combinations but runs out of tokens without finding a solution, generating verbose "Let me try..." reasoning that explores randomly. GRPO success: shorter, more directed reasoning — tries 2–3 approaches and converges; GRPO reward teaches the model to terminate early with `<answer>` once a solution is found. Countdown has clear binary reward signal, so the model can learn to search more efficiently.

**B.2 Basic Arithmetic (baseline 64% → GRPO 62%, −2%).** Baseline approach: step-by-step computation with explanations — "Subtracting a negative is like adding the positive. So, −66 − (−51) becomes −66 + 51." Average response length ~200 chars. GRPO approach: compressed computation — "−66 − −51 = −15". Average response length ~70 chars. Works for simple problems; fails on multi-step expressions like `−3241 − 2720 − 9730 + 6743 + 1933` where intermediate steps are needed. Why it fails: correctness reward is binary but arithmetic requires process correctness — getting each intermediate step right. GRPO optimizes the final answer, which accidentally teaches step-skipping.

**B.3 Maze (baseline 10% → GRPO 0% for 8B v1).** Baseline approach: the model describes a BFS strategy but can't execute it. It says "Let me visualize the maze" then produces generic text about pathfinding without actually tracing paths. GRPO approach (from corrupted run): collapses to extremely generic responses, no maze-specific reasoning. Why it fails: mazes require spatial reasoning LLMs fundamentally struggle with — tracking row/column positions, checking wall adjacency. Binary reward gives no signal for "almost right" paths.

**B.4 Syllogism (baseline 78% → GRPO 96%, +18%).** Baseline approach: attempts logical analysis but sometimes makes errors in quantifier reasoning. GRPO approach: more systematic — translates premises into formal logic, checks validity step by step. Shorter and more focused. Why it works: syllogisms have binary correctness and learnable reasoning steps; the model can internalize valid syllogism patterns through RL.

**B.5 Caesar Cipher (baseline 0% → GRPO 0%).** Baseline approach: attempts letter-by-letter decryption but makes arithmetic errors on letter shifts (e.g., "V → S" shift 3 but then "P → O" shift 1 in the same word — inconsistent shift application). GRPO v1: same inconsistent behavior; the model knows the algorithm but cannot execute it reliably. GRPO v2 (8B only): with lr=1e-6 and 16 generations, the 8B model gets non-zero reward during training (~0.29) by sometimes getting short words right, but still 0% on eval (eval problems are harder — longer words, higher rotations). Why it fails: Caesar requires perfect character-by-character execution — one wrong letter = whole answer wrong. Reward is extremely sparse.

**B.6 Mini Sudoku (baseline 0% → GRPO 60% for 8B, +60%).** Baseline approach: attempts constraint reasoning but makes errors filling cells, often producing invalid grids. GRPO approach: the 8B model develops a systematic approach — fill each cell by checking row, column, and subgrid constraints. Shorter reasoning with more accurate constraint checking. Why it works for 8B but not 3B: sudoku requires simultaneously tracking multiple constraints; the 8B model has enough capacity, the 3B model doesn't. Sharp capability threshold between 4B (22%) and 8B (60%).

**B.7 Shortest Path (baseline 8% → GRPO 44%, +36%).** Baseline approach: describes BFS algorithm but degenerates into repeating "Left Left Left..." infinitely, cannot control generation length. GRPO approach: produces concise direction sequences. The `stop_strings = ["</answer>"]` combined with GRPO's reward for correct paths teaches the model to output just the directions. Why it works: the task has clear structure (sequence of directions), verifiable correctness, and the model can learn to produce the right format.

### Appendix C · Full Per-Task Atomic Templates

**basic_arithmetic**

| Atom | Layer | Example | Base 4B |
|---|---|---|---|
| number_extraction | L1 | "List all numbers in (14 + 3) × 7" → "14, 3, 7" | 62% |
| operator_identification | L1 | "What operations are used?" → "addition, multiplication" | 70% |
| term_count | L1 | "How many numeric terms?" → "3" | 40% |
| has_parentheses | L1 | "Does this have parentheses?" → "Yes" | 98% |
| single_op_eval | L2 | "What is 14 + 3?" → "17" | 76% |
| order_of_operations | L2 | "What does the innermost part evaluate to?" → "17" | 2% |
| innermost_eval | L2 | "Evaluate (14 + 3)" → "17" | 82% |
| sign_handling | L2 | "What is -5 * -6?" → "30" | 100% |

**countdown**

| Atom | Layer | Example | Base 4B |
|---|---|---|---|
| number_inventory | L1 | "List available numbers" → "12, 70, 87, 95" | 10% |
| target_reading | L1 | "What is the target?" → "50" | 90% |
| how_many_numbers | L1 | "How many numbers?" → "4" | 2% |
| single_op | L2 | "What is 87 - 70?" → "17" | 94% |
| two_number_reachability | L2 | "Can you make 50 from 87 and 70?" → "No" | 50% |
| distance_to_target | L2 | "Target is 50, you have 17. How far?" → "33" | 90% |

**shortest_path**

| Atom | Layer | Example | Base 4B |
|---|---|---|---|
| locate_start | L1 | "Where is * in this grid?" → "row 3, column 0" | 0% |
| locate_goal | L1 | "Where is # in this grid?" → "row 2, column 2" | 0% |
| read_cell | L1 | "What is at row 0, col 0?" → "O" | 0% |
| grid_dimensions | L1 | "How many rows and columns?" → "5 rows, 5 columns" | 0% |
| valid_neighbors | L2 | "From (3,3), which directions non-blocked?" → "up, down, left" | 0% |
| is_move_legal | L2 | "From (0,4), is moving right legal?" → "No" | 54% |
| direction_to_goal | L2 | "Which direction moves closer to #?" → "up, right" | 28% |

**maze**

| Atom | Layer | Example | Base 4B |
|---|---|---|---|
| locate_start | L1 | "Where is start character?" → "row 2, column 3" | 0% |
| locate_goal | L1 | "Where is goal character?" → "row 4, column 1" | 0% |
| read_cell | L1 | "What character at row 0, col 0?" → "#" | 54% |
| grid_size | L1 | "Grid size?" → "5 rows, 5 columns" | 2% |
| valid_moves | L2 | "From (2,3), which moves are valid?" → "up, right" | 8% |
| is_move_valid | L2 | "From (0,0), is moving down valid?" → "Yes" | 62% |
| shortest_path_length | L2 | "What is the Manhattan distance?" → "5" | 12% |

**syllogism**

| Atom | Layer | Example | Base 4B |
|---|---|---|---|
| premise1_subject | L1 | "Subject of premise 1?" → "all humans" | 100% |
| premise2_subject | L1 | "Subject of premise 2?" → "all mortals" | 100% |
| quantifier_type | L1 | "Quantifier in premise 1?" → "all" | 86% |
| conclusion_predicate | L1 | "Predicate of conclusion?" → "are mortal" | 98% |
| single_hop_inference | L2 | "If all A are B, what follows?" → "..." | 46% |
| negation | L2 | "Negate 'some A are B'?" → "no A are B" | 42% |
| validity_check | L2 | "Is this syllogism valid?" → "Yes" | 68% |

**knights_knaves**

| Atom | Layer | Example | Base 4B |
|---|---|---|---|
| count_inhabitants | L1 | "How many characters?" → "3" | 66% |
| list_names | L1 | "List their names" → "A, B, C" | 90% |
| speaker_identification | L1 | "Who said 'X is a knight'?" → "A" | 0% |
| terminology | L1 | "Knights always tell the..." → "truth" | 72% |
| statement_about | L2 | "What is A's claim?" → "..." | 72% |
| if_sage_then | L2 | "If A is a sage, is claim true?" → "Yes" | 96% |
| if_fool_then | L2 | "If A is a fool, is claim true?" → "No" | 100% |
| consistency_check | L2 | "Is assignment consistent?" → "No" | 4% |

**mini_sudoku**

| Atom | Layer | Example | Base 4B |
|---|---|---|---|
| read_cell | L1 | "Value at (1,2)?" → "3" | 24% |
| count_empty | L1 | "How many empty cells?" → "7" | 0% |
| row_values | L1 | "Values in row 1?" → "1, 2, 3, _" | 30% |
| col_values | L1 | "Values in column 2?" → "_, 3, _, 2" | 42% |
| missing_in_row | L2 | "What digit missing in row 1?" → "4" | 16% |
| missing_in_col | L2 | "What digit missing in column 2?" → "1, 4" | 22% |
| can_place | L2 | "Can 3 go in (2,3)?" → "No" | 20% |
| only_option | L2 | "What's the only digit that fits (1,4)?" → "4" | 4% |

### Appendix D · Full Per-Task ARS Breakdowns at 3B and 4B

See §3.4 Table 1 for 4B. For 3B (Qwen2.5) ARS values:

| Task | 3B ARS | 3B L1_mean | 3B L2_mean | 4B ARS | 4B L1_mean | 4B L2_mean |
|---|---:|---:|---:|---:|---:|---:|
| syllogism | 68% | 90% | 46% | 74% | 96% | 52% |
| knights_knaves | 50% | 48% | 52% | 62.5% | 57% | 68% |
| basic_arithmetic | 58% | 55% | 60% | 66% | 67.5% | 65% |
| countdown | 50% | 30% | 70% | 56% | 34% | 78% |
| maze | 15% | 10% | 20% | 20.7% | 14% | 27% |
| shortest_path | 8% | 0% | 16% | 14% | 0% | 27% |
| mini_sudoku | 14% | 18% | 10% | 19.8% | 24% | 15.5% |

3B L1_mean is used as the pair-predictor feature in F33. T4 ari+syl's 3B L1_mean is (55% + 90%) / 2 = 72.5%, which normalizes to 0.500 on the 0–1 scale used in §5.6 and §6.4 tables.

### Appendix E · All 19 Pair Merges — Full 4B Damage + Sign-Agreement Table

| Pair | Source | 3B mag_agree | 4B mag_agree | 3B L1_mean | 4B damage | Notes |
|---|---|---:|---:|---:|---:|---|
| T1 sdk+leg | Part 2 | 0.5300 | 0.5044 | 0.143 | 18.0 | |
| T2 sdk+kk | Part 2 | 0.5306 | 0.5002 | 0.220 | 22.0 | |
| T3 ari+leg | Part 2 | 0.5641 | 0.5695 | 0.290 | 12.5 | |
| T4 ari+syl | Part 2 | 0.6053 | 0.5337 | 0.500 | 4.5 | cleanest in Part 2 |
| T5 sp+syl | Part 2 | 0.5358 | 0.5131 | 0.320 | 7.0 | |
| T6 sp+kk | Part 2 | 0.5131 | 0.5089 | 0.188 | 18.0 | |
| T7 cnt+ari | Part 2 | 0.4931 | 0.5132 | 0.261 | 24.0 | worst in Part 2 |
| T8 leg+syl | Tranche 1 | 0.5376 | — | 0.375 | 7.5 | |
| T9 ari+kk | Tranche 1 | 0.5121 | — | 0.368 | 22.0 | |
| T10 cnt+leg | Tranche 1 | 0.5001 | — | 0.136 | 0.5 | nearly lossless |
| T11 sp+leg | Tranche 1 | 0.5511 | — | 0.110 | 0.5 | nearly lossless |
| T12 sdk+syl | Tranche 1 | 0.5264 | — | 0.352 | 19.5 | |
| T13 ari+sdk | Path B | 0.5469 | — | — | 21.0 | |
| T14 syl+kk | Path B | 0.5080 | — | — | 20.0 | |
| T15 kk+sdk | Path B | 0.5306 | — | — | **64.5** | catastrophic |
| T16 cnt+kk | Path B | 0.5219 | — | — | 10.5 | |
| T17 cnt+syl | Path B | 0.5069 | — | — | 40.0 | |
| T18 cnt+sdk | Path B | 0.5209 | — | — | 45.5 | |
| T19 sp+sdk | Path B | 0.5300 | — | — | 51.0 | |

### Appendix F · TIES / DARE Full Per-Task Results

**TIES** (density = 0.2, magnitude-based trim):

| Pair | arith | cnt | SP | maze | sdk | syl | KK | leg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| T2 sdk+kk | 48.5 | 8.0 | 1.0 | 0.5 | 36.5 | 66.0 | 76.0 | 1.0 |
| T5 sp+syl | 70.0 | 26.0 | 48.0 | 4.5 | 25.5 | 85.0 | 70.0 | 53.0 |
| T7 cnt+ari | 60.5 | — | 23.5 | 1.0 | 16.5 | 85.5 | 73.5 | 40.0 |

(Countdown eval missing for T7 TIES due to checkpoint issue.)

**DARE** (drop rate = 0.5, magnitude rescaling):

| Pair | arith | cnt | SP | maze | sdk | syl | KK | leg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| T2 sdk+kk | 50.0 | 9.0 | 0.5 | 0.5 | 36.0 | 66.0 | 68.5 | 1.0 |
| T5 sp+syl | 69.0 | 27.5 | 51.5 | 3.5 | 24.5 | 84.5 | 67.5 | 55.5 |
| T7 cnt+ari | 62.5 | 43.0 | 27.5 | 1.0 | 14.5 | 86.5 | 74.5 | 43.0 |

**Our geometric recipe (`all_but_down` on T5):** 77.0 arith, 36.5 cnt, 53.5 SP, 6.0 maze, 12.5 sdk, 83.0 syl, 71.5 KK, 59.0 leg. SP = 53.5 beats TIES's 48.0 and DARE's 51.5; no other method beats vanilla by more than method noise.

### Appendix G · Hyperparameters (Consolidated)

**SFT warm-up (all models):**

- Data: `data_gen/data/sft_warmup.jsonl` (1,000 samples)
- Epochs: 3, lr: 2e-5, batch size: 2–4, grad-accum: 2–4, max_seq_length: 1024

**GRPO (standard, v1 for every phase):**

- lr: 5.0e-6 constant
- max_steps: 300 (single-task direct); 100 per atomic stage
- per_device_batch_size: 1, grad_accum: 1
- num_generations: 8
- max_completion_length: 2048 (full) / 256 (atoms)
- max_prompt_length: 512 (most) / 1024 (SP)
- temperature: 1.0, loss: DAPO, beta: 0.0, kl_coeff: 0.0
- epsilon_high: 5.0, mask_truncated_completions: true
- max_grad_norm: 1.0, warmup_ratio: 0.1
- seed: 42

**GRPO v2 (7B/8B Phase-1 low-LR sweep only):**

- lr: 1.0e-6, num_generations: 16 (generation_batch_size: 16). All else identical to v1.

**Evaluation:**

- num_samples: 200 (Phase 2+); 50 (Phase 1)
- seed: 9999
- max_new_tokens: 1024 (Phase 4) / 2048 (Phase 1)
- batch_size: 4–8, DDP 8 GPUs, left padding

**Merging (default):**

- W_merged = W_SFT + s · Σ τ_i
- s = 1.0 (2-task), 0.33 (9-task), 0.1 (10-task 1/N)
- Layer masks: `all_but_down` = exclude keys matching `*.mlp.down_proj.*`; `up` = only `*.mlp.{up,gate}_proj.*`; `up_q_clean` = also include `*.self_attn.q_proj.*`

**TIES:** density = 0.2, magnitude-based trim retaining top |τ|
**DARE:** drop_rate = 0.5, magnitude-rescale by 1 / (1 − drop_rate)

**Bootstrap statistics:** 10,000 resamples (paired on (pair, damage)). Permutation tests: 10,000 label permutations.

---

*End of master document. Last revision: 2026-05-12. Next expected revisions: after matched-compute atomic curriculum re-runs at 3B/8B; after non-Qwen replication; after any predictor work that breaks the n = 19 null-result frontier.*
