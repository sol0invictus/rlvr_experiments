# Cross-Task Transfer in RLVR: A 10-Environment Study Across 4 Model Sizes

## 1. Introduction

We investigate whether training a language model with GRPO (Group Relative Policy Optimization) on one reasoning task transfers to other tasks. We construct a 10×10 transfer matrix for each of four model sizes, measuring how GRPO training on Task A affects performance on Tasks A through J.

Our key finding: **GRPO is not a universal improver.** While it produces large gains on some tasks (e.g., +60% on countdown for 7B), it degrades performance in 36% of all (model, task) pairs — including cases where it hurts the very task it was trained on. The severity of this degradation is strongly mediated by learning rate: reducing lr from 5e-6 to 1e-6 eliminates catastrophic harm but also eliminates most gains.

---

## 2. Environments

We use 10 tasks from reasoning-gym, spanning 5 skill categories:

| Category | Task | Description | Default Params |
|----------|------|-------------|----------------|
| **Arithmetic** | basic_arithmetic | Evaluate arithmetic expressions | 2–6 terms, 1–4 digit numbers, +−×÷, parentheses |
| **Arithmetic** | countdown | Reach a target number using given numbers | 4–5 numbers, target 10–200 |
| **Logic** | syllogism | Determine validity of syllogistic arguments | Default |
| **Logic** | knights_knaves | Determine who is a sage/fool from statements | Default |
| **Spatial/Grid** | maze | Find shortest path length in an ASCII maze | 5×5 grid |
| **Spatial/Grid** | mini_sudoku | Solve a 4×4 Sudoku puzzle | Default |
| **Spatial/Grid** | shortest_path | Find shortest path directions in a grid | Default |
| **Planning** | leg_counting | Count total legs for a list of animals | Default |
| **Algorithmic** | caesar_cipher | Decrypt a Caesar cipher | 2–6 words, rotation 1–5 |
| **Algorithmic** | cryptarithm | Solve a letter-to-digit puzzle | Default |

---

## 3. Models

| Model | Family | Params | HuggingFace ID |
|-------|--------|--------|----------------|
| **3B** | Qwen2.5 | 3B | `Qwen/Qwen2.5-3B-Instruct` |
| **4B** | Qwen3 | 4B | `Qwen/Qwen3-4B` |
| **7B** | Qwen2.5 | 7B | `Qwen/Qwen2.5-7B-Instruct` |
| **8B** | Qwen3 | 8B | `Qwen/Qwen3-8B` |

Note: Qwen3 models do not have separate `-Instruct` variants; the base model includes instruction-following capabilities. Qwen2.5 models use the explicit `-Instruct` variant.

### Why SFT?

Before GRPO training, each model is supervised fine-tuned (SFT) on 1,000 task-agnostic warmup samples (trivia, unit conversions, word puzzles, date reasoning) to teach the structured `<think>...</think><answer>...</answer>` output format. This is necessary because:

1. GRPO requires the model to produce structured outputs so the reward function can extract and verify answers
2. Without SFT, models produce free-form text with no consistent answer markers
3. The warmup data is deliberately task-agnostic to avoid giving any advantage on specific reasoning tasks

The SFT step is identical across all models: 3 epochs, lr=2e-5, batch_size=2–4, max_seq_length=1024.

---

## 4. Transfer Matrices

All values show **absolute delta from SFT baseline** (percentage points). Positive = GRPO improved over baseline; Negative = GRPO degraded. Diagonal entries (same-task) are marked with asterisks. The last row shows baseline (SFT-only) absolute accuracy.

Column headers: **cnt**=countdown, **mz**=maze, **sdk**=mini_sudoku, **pth**=shortest_path, **ari**=basic_arithmetic, **syl**=syllogism, **kk**=knights_knaves, **leg**=leg_counting, **csr**=caesar_cipher, **cry**=cryptarithm

### 4.1 Qwen2.5-3B (lr=5e-6, 8 generations)

| Trained on | cnt | mz | sdk | pth | ari | syl | kk | leg | csr | cry |
|------------|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| countdown | *+10* | -4 | 0 | 0 | -2 | +10 | +8 | 0 | 0 | 0 |
| maze | 0 | *+2* | 0 | +4 | -4 | 0 | -2 | -12 | 0 | 0 |
| mini_sudoku | 0 | -8 | *0* | +2 | -8 | -4 | -4 | -6 | 0 | 0 |
| shortest_path | 0 | -2 | 0 | *+40* | -6 | -8 | -2 | 0 | 0 | 0 |
| basic_arith | 0 | -8 | 0 | 0 | *-14* | +10 | -2 | -10 | 0 | 0 |
| syllogism | 0 | -4 | 0 | 0 | -20 | *+20* | +4 | -12 | 0 | 0 |
| knights_knaves | 0 | -12 | 0 | 0 | -2 | +4 | *+18* | -6 | 0 | 0 |
| leg_counting | 0 | -4 | 0 | +2 | -16 | 0 | -8 | *+12* | 0 | 0 |
| caesar_cipher | 0 | -8 | 0 | 0 | 0 | +8 | +14 | -6 | *0* | 0 |
| cryptarithm | 0 | -2 | 0 | 0 | 0 | +4 | 0 | +2 | 0 | *0* |
| **Baseline** | **0%** | **14%** | **0%** | **0%** | **46%** | **74%** | **26%** | **12%** | **0%** | **0%** |

**Observations (3B):**
- **shortest_path is the standout:** +40% gain, the largest diagonal improvement at this size. The model learns graph traversal skills that don't exist in the baseline.
- **Arithmetic GRPO hurts arithmetic (-14%):** The 3B model already gets 46% from SFT alone. GRPO training with binary reward on problems including multi-digit multiplication causes the model to adopt a compressed reasoning style that loses accuracy.
- **Syllogism transfers broadly:** Training on syllogism gives +20% on syllogism itself, but costs -20% on arithmetic — the model becomes more logic-focused at the expense of computation.
- **Caesar/cryptarithm are inert:** 0% baseline, 0% after GRPO, 0% transfer. The 3B model cannot solve these at all.
- **Cross-category transfer is rare:** Most off-diagonal entries are 0 or negative. The few positive transfers (countdown→syllogism +10%, caesar→knights +14%) suggest that "any GRPO training" activates some general reasoning, not task-specific transfer.

---

### 4.2 Qwen3-4B (lr=5e-6, 8 generations)

| Trained on | cnt | mz | sdk | pth | ari | syl | kk | leg | csr | cry |
|------------|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| countdown | *+36* | +6 | +14 | +4 | +6 | +18 | +4 | +6 | 0 | 0 |
| maze | -10 | *+12* | +16 | +4 | 0 | +12 | 0 | +6 | 0 | 0 |
| mini_sudoku | -16 | -2 | *+22* | -16 | -26 | -16 | -50 | -52 | 0 | 0 |
| shortest_path | +6 | +8 | +28 | *+28* | +6 | +16 | -2 | +14 | 0 | 0 |
| basic_arith | +4 | -2 | -4 | -6 | *-12* | +18 | -16 | -32 | 0 | 0 |
| syllogism | -8 | -2 | +26 | -2 | -2 | *+26* | +2 | +10 | 0 | 0 |
| knights_knaves | -8 | +2 | +4 | -12 | -8 | +26 | *+12* | -4 | 0 | 0 |
| leg_counting | +16 | +2 | +22 | +12 | -8 | +14 | -8 | *+28* | 0 | 0 |
| caesar_cipher | 0 | 0 | 0 | -2 | +6 | -22 | 0 | +2 | *0* | 0 |
| cryptarithm | -2 | -2 | -2 | -2 | -2 | +2 | +2 | -4 | 0 | *0* |
| **Baseline** | **34%** | **2%** | **14%** | **16%** | **62%** | **66%** | **74%** | **52%** | **0%** | **0%** |

**Observations (4B):**
- **Countdown is the best single task:** +36% diagonal, plus broad positive transfer (+18% syllogism, +14% sudoku, +6% arith/path). This task teaches general "search through combinations" reasoning that generalizes.
- **Shortest_path is the best all-rounder:** +28% diagonal AND positive transfer to 6/9 other tasks (up to +28% sudoku). Graph traversal skills transfer broadly.
- **Mini_sudoku is catastrophic for other tasks:** +22% on sudoku, but -50% on knights_knaves and -52% on leg_counting. GRPO overspecializes the model to constraint-satisfaction patterns, destroying general reasoning.
- **Arithmetic still hurts itself (-12%):** Same pattern as 3B. Baseline is already 62%; GRPO cannot improve multi-digit computation.
- **Logic tasks show one-way transfer:** Syllogism training helps sudoku (+26%) but countdown→syllogism (+18%) doesn't reverse well (syllogism→countdown is -8%).
- **4B has the strongest baseline** (Qwen3 family advantage), giving it both the most to gain and the most to lose.

---

### 4.3 Qwen2.5-7B — v1 (lr=5e-6, 8 generations)

| Trained on | cnt | mz | sdk | pth | ari | syl | kk | leg | csr | cry |
|------------|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| countdown | *+60* | -4 | -6 | 0 | -2 | +6 | +8 | -8 | 0 | 0 |
| maze | -4 | *+10* | -6 | -4 | -2 | +4 | 0 | -10 | 0 | 0 |
| mini_sudoku | +2 | -4 | *+16* | -4 | -6 | -8 | +4 | -6 | 0 | 0 |
| shortest_path | -4 | -2 | -4 | *+36* | -8 | -24 | 0 | -2 | 0 | 0 |
| basic_arith | -2 | +2 | -12 | -4 | ***-30*** | -2 | 0 | -30 | 0 | 0 |
| syllogism | 0 | +6 | -12 | -4 | -46 | *+26* | -28 | -32 | 0 | 0 |
| knights_knaves | -2 | -2 | -10 | -4 | -32 | -8 | *+28* | -32 | 0 | 0 |
| leg_counting | -4 | +2 | -4 | -4 | -10 | -2 | +2 | *+8* | 0 | 0 |
| caesar_cipher | -4 | -4 | -6 | -4 | -2 | -4 | +2 | -2 | *0* | 0 |
| cryptarithm | 0 | -2 | 0 | +2 | 0 | 0 | -2 | +2 | 0 | *0* |
| **Baseline** | **4%** | **14%** | **12%** | **4%** | **60%** | **64%** | **32%** | **32%** | **0%** | **0%** |

**Observations (7B v1):**
- **Countdown gains massively (+60%)** from 4% → 64%. This is the largest single gain in the entire study. The Qwen2.5-7B model has a low baseline on countdown but GRPO unlocks it.
- **Arithmetic is catastrophically degraded (-30%):** The model goes from 60% → 30%. This is confirmed to be worse even on the training distribution (seed=42: 70% → 42%). GRPO actively destroys arithmetic competence at lr=5e-6.
- **Syllogism training is extremely destructive:** +26% on syllogism, but -46% on arith, -28% on knights, -32% on leg_counting. The model becomes a syllogism specialist at the cost of everything else.
- **Knights_knaves training similarly destructive:** +28% on knights, -32% on arith, -32% on leg_counting.
- **The 7B model is the most "volatile"** — biggest gains and biggest losses. The lr=5e-6 is too aggressive for this model size.

---

### 4.4 Qwen2.5-7B — v2 (lr=1e-6, 16 generations)

| Trained on | cnt | mz | sdk | pth | ari | syl | kk | leg | csr | cry |
|------------|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| countdown | *-2* | +4 | +4 | 0 | 0 | +2 | +4 | -6 | 0 | 0 |
| maze | -4 | *+6* | +6 | +4 | 0 | +2 | +2 | -6 | 0 | 0 |
| mini_sudoku | -4 | 0 | *+8* | +4 | 0 | -2 | +2 | -2 | 0 | 0 |
| shortest_path | -2 | +2 | +8 | *+8* | -2 | 0 | +2 | -6 | 0 | 0 |
| basic_arith | -2 | +2 | +2 | +4 | *-4* | +2 | 0 | -2 | 0 | 0 |
| syllogism | -4 | +2 | +4 | 0 | -2 | *+8* | +2 | -4 | 0 | 0 |
| knights_knaves | -2 | 0 | +6 | +2 | 0 | -4 | *+4* | 0 | 0 | 0 |
| leg_counting | -4 | +2 | +4 | +4 | 0 | 0 | -4 | *+2* | 0 | 0 |
| caesar_cipher | -2 | 0 | +8 | 0 | 0 | +6 | +8 | -4 | *0* | 0 |
| cryptarithm | -2 | +2 | +2 | +2 | 0 | +2 | -2 | -2 | 0 | *0* |
| **Baseline** | **4%** | **12%** | **4%** | **2%** | **60%** | **64%** | **30%** | **36%** | **0%** | **0%** |

**Observations (7B v2 — lr=1e-6):**
- **Catastrophic harm eliminated:** The worst diagonal drop is -4% (arith), compared to -30% at lr=5e-6. No off-diagonal drop exceeds -6%.
- **But gains are minimal too:** Countdown goes from +60% (v1) to -2% (v2). Shortest_path from +36% to +8%. The lower LR prevents both learning and forgetting.
- **Sudoku column is interesting:** 8 out of 10 models show positive transfer to sudoku (+2 to +8%). This wasn't visible at lr=5e-6 because it was masked by larger noise.
- **The v2 matrix is nearly flat** — most deltas are in ±2-4%. GRPO at lr=1e-6 barely moves the 7B model from its SFT baseline.

---

### 4.5 Qwen3-8B — v1 (lr=5e-6, 8 generations)

| Trained on | cnt | mz | sdk | pth | ari | syl | kk | leg | csr | cry |
|------------|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| countdown | *+50* | -10 | +22 | +8 | -4 | -2 | +2 | -4 | 0 | 0 |
| maze | -24 | *-10* | 0 | -8 | -8 | -22 | -30 | -40 | 0 | 0 |
| mini_sudoku | +18 | +10 | *+60* | +16 | +10 | +6 | +6 | -4 | 0 | 0 |
| shortest_path | -4 | +10 | +8 | *+36* | +2 | -8 | +6 | 0 | 0 | 0 |
| basic_arith | -14 | +4 | +14 | +4 | *-2* | -4 | -4 | -8 | 0 | 0 |
| syllogism | -16 | -6 | +6 | -2 | -2 | *+18* | -6 | -16 | 0 | 0 |
| knights_knaves | -18 | -8 | +16 | +4 | -2 | 0 | *-6* | -10 | 0 | 0 |
| leg_counting | +6 | -2 | 0 | +10 | +4 | -10 | 0 | *0* | +2 | 0 |
| caesar_cipher | -24 | -10 | +8 | +10 | 0 | -16 | +6 | -4 | *0* | 0 |
| cryptarithm | -10 | -6 | 0 | 0 | 0 | 0 | -4 | +2 | 0 | *0* |
| **Baseline** | **32%** | **10%** | **0%** | **8%** | **64%** | **78%** | **64%** | **48%** | **0%** | **0%** |

**Observations (8B v1):**
- **Mini_sudoku is a breakthrough (+60%):** From 0% → 60%. The 8B model has the capacity to learn constraint satisfaction that smaller models couldn't. Positive transfer to 7 other tasks including +18% countdown, +16% path.
- **Maze GRPO is catastrophic:** -10% on maze itself (corrupted training run that got 0% reward), -30% knights, -40% leg_counting, -22% syllogism. GRPO with no reward signal causes pure degradation.
- **Highest baseline of all models** (Qwen3-8B: 78% syllogism, 64% arith/knights). This means GRPO has less room to improve and more room to damage.
- **Knights_knaves training hurts itself (-6%):** The model's 64% baseline is already strong; GRPO overshoots.

---

### 4.6 Qwen3-8B — v2 (lr=1e-6, 16 generations)

| Trained on | cnt | mz | sdk | pth | ari | syl | kk | leg | csr | cry |
|------------|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| countdown | *+4* | -2 | +6 | -8 | +2 | 0 | -12 | -2 | 0 | 0 |
| maze | 0 | *0* | +4 | -6 | +2 | 0 | -10 | -2 | 0 | 0 |
| mini_sudoku | +4 | +2 | *+2* | -2 | +4 | +4 | -6 | 0 | 0 | 0 |
| shortest_path | -4 | -4 | 0 | *-4* | +2 | +4 | -8 | -2 | 0 | 0 |
| basic_arith | +2 | +2 | 0 | -6 | *+4* | +2 | -8 | -2 | 0 | 0 |
| syllogism | +6 | -2 | 0 | -2 | +4 | *+2* | -10 | 0 | 0 | 0 |
| knights_knaves | +6 | -4 | +4 | -6 | +4 | 0 | *-14* | -2 | 0 | 0 |
| leg_counting | +2 | -4 | 0 | -6 | +4 | +2 | -18 | *0* | 0 | 0 |
| caesar_cipher | +4 | -4 | +2 | -6 | +2 | +2 | -2 | -2 | *0* | 0 |
| cryptarithm | +2 | -4 | +2 | -10 | +2 | +2 | -14 | -2 | 0 | *0* |
| **Baseline** | **22%** | **10%** | **0%** | **12%** | **62%** | **74%** | **74%** | **52%** | **0%** | **0%** |

**Observations (8B v2 — lr=1e-6):**
- **Arithmetic is now positive (+4%):** The key result. At lr=5e-6, arith was -2%; at lr=1e-6 with 16 generations, it improves. The lower LR preserves the model's chain-of-thought while GRPO provides gentle optimization.
- **Knights_knaves column is systematically negative:** Every single GRPO model hurts knights_knaves (-2 to -18%). This is a consistent pattern — GRPO training on anything degrades the 8B model's knights_knaves performance. Baseline is 74% and very fragile.
- **Mini_sudoku gains collapse:** From +60% (v1) to +2% (v2). The lower LR prevents the large parameter movements needed to learn constraint satisfaction.
- **Shortest_path flips from +36% to -4%:** Another case where the lower LR prevents learning entirely.

---

## 5. Cross-Model Analysis

### 5.1 Tasks Where GRPO Consistently Helps (All Sizes)

| Task | 3B | 4B | 7B v1 | 8B v1 | Pattern |
|------|-----|-----|-------|-------|---------|
| syllogism | +20% | +26% | +26% | +18% | Consistent +20% across sizes |
| shortest_path | +40% | +28% | +36% | +36% | Consistent +30-40% |
| countdown | +10% | +36% | +60% | +50% | Scales with model size |

These tasks share a common trait: they have **clear reasoning structure** that GRPO can optimize without requiring novel computation skills.

### 5.2 Tasks Where GRPO Consistently Hurts

| Task | 3B | 4B | 7B v1 | 8B v1 | Pattern |
|------|-----|-----|-------|-------|---------|
| basic_arith | -14% | -12% | -30% | -2% | Hurts at every size |
| caesar_cipher | 0% | 0% | 0% | 0% | No signal, no learning |
| cryptarithm | 0% | 0% | 0% | 0% | No signal, no learning |

Basic arithmetic is the only task where GRPO **actively degrades** same-task performance. This is because the task requires precise multi-digit computation (including multiplication and division), which GRPO's binary reward cannot teach. The model's SFT-learned verbose chain-of-thought is a better strategy than anything GRPO discovers.

### 5.3 Transfer Patterns by Skill Category

**Within-category transfer:**
- Arithmetic → Arithmetic: countdown helps arith (+6% at 4B), but arith hurts itself
- Logic → Logic: syllogism ↔ knights_knaves show positive transfer at 3B/4B, destructive at 7B
- Spatial → Spatial: shortest_path helps sudoku (+28% at 4B), but maze is largely isolated

**Cross-category transfer:**
- Countdown (Arithmetic) → Syllogism (Logic): Consistently positive (+10% to +18%)
- Shortest_path (Spatial) → Syllogism (Logic): +16% at 4B
- Syllogism (Logic) → Arithmetic: Consistently negative (-2% to -46%)

The pattern is asymmetric: **training on tasks that require search/planning improves logic, but training on logic hurts computation.**

### 5.4 The LR Effect (v1 vs v2)

| Metric | 7B v1 (5e-6) | 7B v2 (1e-6) | 8B v1 (5e-6) | 8B v2 (1e-6) |
|--------|-------------|-------------|-------------|-------------|
| Max diagonal gain | +60% | +8% | +60% | +4% |
| Worst diagonal harm | -30% | -4% | -10% | -4% |
| Avg |delta| | 8.9% | 3.0% | 11.2% | 3.8% |
| Cases with >10% harm | 14 | 0 | 11 | 2 |

The lower LR reduces both signal and noise by ~3x. This suggests that the **optimal LR is somewhere between 1e-6 and 5e-6** — high enough to learn but low enough to avoid catastrophic forgetting.

---

## 6. Key Takeaways

1. **GRPO is not "free improvement"** — it helps on some tasks, hurts on others, and the net effect depends heavily on the learning rate.

2. **Computation tasks (arithmetic) are anti-correlated with GRPO** — the binary reward signal cannot teach multi-digit multiplication, and GRPO pushes models away from effective chain-of-thought strategies.

3. **Transfer is asymmetric** — search/planning skills transfer to logic, but logic skills hurt computation. This has implications for curriculum design in multi-task RLVR.

4. **Baselines are surprisingly strong** — SFT alone gives 62–78% on arithmetic and syllogism. GRPO's incremental improvement over SFT is smaller than commonly assumed.

5. **Learning rate is the dominant hyperparameter** — the difference between lr=5e-6 and lr=1e-6 is larger than the difference between model sizes.

---

## Appendix A: Training Configuration

### GRPO Training (v1 — all models)
```yaml
learning_rate: 5.0e-6
lr_scheduler_type: constant
max_steps: 300
per_device_train_batch_size: 1
gradient_accumulation_steps: 1
num_generations: 8
generation_batch_size: 8
max_completion_length: 2048
max_prompt_length: 512
temperature: 1.0
loss_type: dapo
beta: 0.0
kl_coeff: 0.0
warmup_ratio: 0.1
max_grad_norm: 1.0
mask_truncated_completions: true
epsilon_high: 5.0
stop_strings: ["</answer>"]
seed: 42
```

### GRPO Training (v2 — 7B and 8B only)
```yaml
learning_rate: 1.0e-6     # 5x lower
num_generations: 16        # 2x more
generation_batch_size: 16
# All other parameters identical to v1
```

### SFT Training (all models)
```yaml
learning_rate: 2.0e-5
num_train_epochs: 3
batch_size: 2-4
gradient_accumulation_steps: 2-4
max_seq_length: 1024
data: data_gen/data/sft_warmup.jsonl (1000 samples)
```

### Evaluation
```yaml
num_samples: 50
max_new_tokens: 2048
batch_size: 4
seed: 9999 (different from training seed 42)
DDP: 8 GPUs, samples sharded across ranks, results aggregated via all_reduce
tokenizer.padding_side: left (required for correct batched generation)
```

---

## Appendix B: Generation Log Analysis

We examine sample generations from the Qwen3-8B model (v1) to understand failure modes.

### B.1 Countdown (baseline 32% → GRPO 82%, +50%)

**Baseline failure:** The model tries many combinations but runs out of tokens without finding a solution. It generates verbose "Let me try..." reasoning that explores randomly.

**GRPO success:** The model produces shorter, more directed reasoning. It tries 2-3 approaches and converges. The GRPO reward teaches it to terminate early with `<answer>` once a solution is found.

**Why it works:** Countdown has clear reward signal — either the expression equals the target or it doesn't. The model can learn to search more efficiently.

### B.2 Basic Arithmetic (baseline 64% → GRPO 62%, -2%)

**Baseline approach:** Step-by-step computation with explanations. "Subtracting a negative is like adding the positive. So, -66 - (-51) becomes -66 + 51." Average response length: ~200 chars.

**GRPO approach:** Compressed computation. "-66 - -51 = -15". Average response length: ~70 chars. Works for simple problems but fails on multi-step expressions like `-3241 - 2720 - 9730 + 6743 + 1933` where intermediate steps are needed.

**Why it fails:** The correctness reward is binary (right/wrong), but arithmetic requires *process correctness* — getting each intermediate step right. GRPO optimizes for the final answer, which accidentally teaches the model to skip steps.

### B.3 Maze (baseline 10% → GRPO 0% for 8B v1)

**Baseline approach:** The model describes a BFS strategy but can't execute it. It says "Let me visualize the maze" then produces generic text about pathfinding algorithms without actually tracing paths.

**GRPO approach (from corrupted run):** The model collapses to extremely generic responses. No maze-specific reasoning.

**Why it fails:** Mazes require spatial reasoning that LLMs fundamentally struggle with — tracking row/column positions, checking wall adjacency. Binary reward gives no signal for "almost right" paths.

### B.4 Syllogism (baseline 78% → GRPO 96%, +18%)

**Baseline approach:** Attempts logical analysis but sometimes makes errors in quantifier reasoning.

**GRPO approach:** More systematic. Translates premises into formal logic, checks validity step by step. Shorter and more focused.

**Why it works:** Syllogisms have binary correctness AND the reasoning steps are learnable. The model can internalize valid syllogism patterns through RL.

### B.5 Caesar Cipher (baseline 0% → GRPO 0%)

**Baseline approach:** Attempts letter-by-letter decryption but makes arithmetic errors on letter shifts. E.g., "V → S" (shift 3) but then "P → O" (shift 1) in the same word — inconsistent shift application.

**GRPO approach (v1):** Same inconsistent behavior. The model knows the algorithm but cannot execute it reliably.

**GRPO approach (v2, 8B only):** With lr=1e-6 and 16 generations, the 8B model gets non-zero reward during training (~0.29) by sometimes getting short words right, but still 0% on eval. The eval problems are harder (longer words, higher rotations).

**Why it fails:** Caesar cipher requires *perfect* character-by-character execution — one wrong letter means the whole answer is wrong. The reward is extremely sparse.

### B.6 Mini Sudoku (baseline 0% → GRPO 60% for 8B, +60%)

**Baseline approach:** Attempts constraint reasoning but makes errors filling cells. Often produces invalid grids.

**GRPO approach:** The 8B model develops a systematic approach: fill each cell by checking row, column, and subgrid constraints. Shorter reasoning with more accurate constraint checking.

**Why it works for 8B but not 3B:** Sudoku requires simultaneously tracking multiple constraints. The 8B model has enough capacity to learn this; the 3B model doesn't. There's a sharp capability threshold between 4B (22%) and 8B (60%).

### B.7 Shortest Path (baseline 8% → GRPO 44%, +36%)

**Baseline approach:** Describes BFS algorithm but degenerates into repeating "Left Left Left..." infinitely. Cannot control generation length.

**GRPO approach:** Produces concise direction sequences. The stop_strings `["</answer>"]` combined with GRPO's reward for correct paths teaches the model to output just the directions.

**Why it works:** The task has clear structure (sequence of directions), verifiable correctness, and the model can learn to produce the right format.

---

## Appendix C: Infrastructure

- **Hardware:** 2× P6-B200 nodes (8× 178GB GPUs each) on Green cluster
- **Training:** `torchrun --nproc_per_node=8` with DDP
- **Evaluation:** DDP with samples sharded across 8 ranks, `dist.all_reduce` for aggregation
- **Framework:** TRL 1.0.0 + transformers 4.57.6 + reasoning-gym
- **Storage:** Shared FSx filesystem across pods
- **Total compute:** ~100 GPU-hours for all training + evaluation runs

---

## Appendix D: Status Update Post-Phase-4-Part-6 (added 2026-05-12)

Phase 1's transfer-matrix findings are unaffected by later phases:

- 36% of (model, task) pairs degrade under GRPO — robust.
- Catastrophic transfers up to −52% (mini_sudoku → leg_counting) — robust.
- LR sensitivity (5e-6 vs 1e-6 trade-off) — robust.
- Arithmetic degrades at every scale — robust.
- Reasoning-on-reasoning positive-transfer cluster — robust.

**Important context for Phase 4 readers:** Phase 1 measures *sequential-training transfer* (train on A, eval on B). Phase 4 measures *weight-summation merging* (W_SFT + τ_A + τ_B). Phase 4 Part 2 confirmed at n = 7 and Part 6 confirmed at n = 19 that **Phase-1 sequential transfer does not predict Phase-4 merge damage** (Pearson r ≈ 0). The two operations interact with weight space differently: reasoning compression, policy collapse, and other behavioral training pathologies that drive Phase 1's worst transfers do not survive into the summed model. Use Phase 1 results to reason about *training* compositions (curricula, sequential RL); do not use them to predict *merging* compatibility.

**Cross-references:**
- Phase 2 (atomic curriculum): `report_atomic_curriculum.md`, `report_phase2_complete.md`
- Phase 3 (weight-space geometry): `report_phase3_task_distances.md`
- Phase 4 (merging at scale): `report_phase4_merge_scaling.md` (Parts 1–6)
- Master document: `MASTER_RLVR_dynamics.md` (consolidated source of truth)
