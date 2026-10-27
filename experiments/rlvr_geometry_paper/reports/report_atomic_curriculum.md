# Phase 2: Atomic Skill Curriculum for RLVR

> **Note:** This report covers the original 3 tasks (arithmetic, countdown, shortest_path) with deep-dive experiments. For the complete 7-task analysis including syllogism, knights_knaves, maze, and mini_sudoku, see **[`report_phase2_complete.md`](report_phase2_complete.md)**.

**Date:** 2026-04-18 (updated 2026-04-23)
**Model:** Qwen3-4B (post-SFT with `<think>/<answer>` format)
**Infrastructure:** Green cluster P6-B200 (8x B200 GPUs), pod `sunguha-rlvr-phase2-qbgw`
**Depends on:** Phase 1 transfer matrix (`report_transfer_matrix.md`)

---

## Summary

Phase 1 showed GRPO hurts 36% of (model, task) pairs. Phase 2 investigates **why** by decomposing tasks into atomic sub-skills (L1 grounding + L2 local reasoning) and training on those atoms before (or instead of) the full task. Phase 2 then systematically explores 15+ training strategies to push shortest_path accuracy from 10% (base) to 55.5% (best).

**Key findings:**

1. **ARS predicts training strategy:** Atomic Readiness Score (zero-shot probe) predicts which approach works best per task.

2. **Long-answer atoms beat short-answer atoms:** Multi-step path-tracing atoms (Idea B, 51.5%) dramatically outperform Yes/No atoms that cause reasoning compression (Atomic-Only, 12%).

3. **On-policy rejection sampling + GRPO is the overall best:** RFT-Base (55.5%) — SFT the base model on verified-correct samples from a trained checkpoint, then GRPO — beats all other approaches.

4. **Partial credit and progressive curricula are harmful to GRPO:** Soft rewards (C: 35.5%, D: 24%) and gradual difficulty (E: 27.5%) reduce the gradient variance GRPO needs.

5. **Cross-task transfer is preserved:** The 55.5% shortest_path model retains arithmetic (70% vs 72% base) and improves countdown (+6%).

| ARS Profile | Best Strategy | Example |
|-------------|---------------|---------|
| High L2, GRPO destructive | **Atomic-Only** (never train on full task) | basic_arithmetic |
| Low L1, GRPO helpful | **Atomic warm-start + full GRPO** | shortest_path |
| High L2, L3 bottleneck | **Direct GRPO** (skip atoms) | countdown |

### Best Results by Task

| Task | Base | Direct-RLVR | Best Approach | Best Accuracy |
|------|------|-------------|---------------|---------------|
| basic_arithmetic | 72% | 32% | Atomic-Only | **78%** |
| countdown | 26% | ~70% | Direct GRPO | **~70%** |
| shortest_path | 10% | 44% | **RFT-Base** | **55.5%** |

### Shortest Path: Complete Exploration (15+ approaches)

| Approach | Accuracy | Key Insight |
|----------|----------|-------------|
| Base (SFT) | 10% | reference |
| Direct-RLVR | 44% | Phase 1 baseline |
| Atomic-Warm-300 (short atoms) | 46% | short L2 atoms + full GRPO |
| A (progressive full-task) | 46.5% | 3x3→4x4→full |
| B (L2-long atoms → full) | 51.5% | path-tracing atoms, hard transition |
| B2 (300+300 steps) | 47% | more steps hurt |
| C (partial credit) | 35.5% | reward too easy for GRPO |
| D (progressive + partial credit) | 24% | worst — PC kills gradient signal |
| E (long atoms → progressive) | 27.5% | progressive after atoms hurts |
| F (feasibility atoms → full) | 42% | fewer false-inf, more wrong paths |
| G (heavy GRPO+SFT interleave) | 5.5% | SFT collapsed the model |
| G2 (light GRPO+SFT interleave) | 40% | improving trend with 500 samples lr=1e-6 |
| G3 (G2 + 150 GRPO) | 45% | matches Direct-RLVR |
| L (feasible-only training) | 40.5% | lost infeasibility detection |
| RFT on B's checkpoint | 51.5% | same ceiling — on-policy SFT doesn't add to trained model |
| **RFT-Base** | **55.5%** | **B's correct traces → base SFT → GRPO 300 steps** |

### RFT-Base Cross-Task Transfer

| Task | Base | RFT-Base (55.5%) | Delta |
|------|------|-----------------|-------|
| shortest_path | 10% | 55% | +45% |
| basic_arithmetic | 72% | 70% | -2% (within noise) |
| countdown | 26% | 32% | +6% (positive transfer) |

---

## Method

### 3-Layer Atomic Taxonomy

- **Layer 1 (Grounding):** Parse the input representation. What is where? What do symbols mean?
- **Layer 2 (Local Reasoning):** Single-step inference over grounded elements. No multi-step planning.
- **Layer 3 (Global Planning):** Multi-step strategy composition — this IS the full task.

### Atomic Readiness Score (ARS)

Before any training, probe the base model on atomic Q&A pairs:

```
ARS(model, task) = mean accuracy across all L1+L2 atoms, zero-shot
```

Cheap diagnostic (<1 hour) that predicts whether atomic curriculum will help.

### Training Conditions

All conditions use GRPO with identical hyperparameters (lr=5e-6, DAPO loss, 8 generations, temperature=1.0).

| Condition | Description |
|-----------|-------------|
| **Base** | SFT model, no GRPO |
| **Direct-RLVR** | 300 steps GRPO on full task |
| **Atomic-Only** | 100 steps L1 atoms → 100 steps L2 atoms, zero-shot eval on full task |
| **Atomic-Warm** | L1(100) → L2(100) → Full(100), 300 total steps |
| **Atomic-Warm-300** | L1(100) → L2(100) → Full(300), 500 total steps (equal full-task compute to Direct-RLVR) |

---

## Tasks and Atomic Decompositions

### basic_arithmetic

**Task:** Calculate multi-step arithmetic expressions (2-6 terms, 1-4 digits, +−×÷, parentheses).
**Phase 1 result:** Direct-RLVR hurt by -12% (4B model, Phase 1 numbers before extraction fix).

| Atom | Layer | Example | Base Accuracy |
|------|-------|---------|--------------|
| number_extraction | L1 | "List all numbers in: (14 + 3) × 7" → "14, 3, 7" | 46% |
| operator_identification | L1 | "What operations are used?" → "addition, multiplication" | 70% |
| term_count | L1 | "How many numeric terms?" → "3" | 40% |
| has_parentheses | L1 | "Does this have parentheses?" → "Yes" | 98% (skip) |
| single_op_eval | L2 | "What is 14 + 3?" → "17" | 76% |
| order_of_operations | L2 | "What does the innermost part evaluate to?" → "17" | 66% |
| innermost_eval | L2 | "Evaluate (14 + 3)" → "17" | 82% (skip) |
| sign_handling | L2 | "What is -5 * -6?" → "30" | 100% (skip) |

**ARS: 72.2%** (L1 avg: 63.5%, L2 avg: 81.0%)

Trained on: number_extraction, operator_identification, term_count (L1); order_of_operations, single_op_eval (L2). Skipped atoms already at ≥82%.

### countdown

**Task:** Reach a target number using 4-5 available numbers with +−×÷.
**Phase 1 result:** Direct-RLVR helped by +36%.

| Atom | Layer | Example | Base Accuracy |
|------|-------|---------|--------------|
| number_inventory | L1 | "List available numbers" → "12, 70, 87, 95" | 10% |
| target_reading | L1 | "What is the target?" → "50" | 90% |
| how_many_numbers | L1 | "How many numbers available?" → "4" | 2% |
| single_op | L2 | "What is 87 - 70?" → "17" | 94% |
| two_number_reachability | L2 | "Can you make 50 from 87 and 70?" → "No" | 50% |
| distance_to_target | L2 | "Target is 50, you have 17. How far?" → "33" | 90% |

**ARS: 56.0%** (L1 avg: 34.0%, L2 avg: 78.0%)

Note: L1 is very weak (number_inventory 10%, how_many_numbers 2%) but L2 is strong (single_op 94%, distance 90%). The bottleneck is L3 (multi-step search), not L1/L2.

### shortest_path

**Task:** Find shortest path from * to # in a grid, moving through O cells, avoiding X cells.
**Phase 1 result:** Direct-RLVR helped by +28%.

| Atom | Layer | Example | Base Accuracy |
|------|-------|---------|--------------|
| locate_start | L1 | "Where is * in this grid?" → "row 3, column 0" | 0% |
| locate_goal | L1 | "Where is # in this grid?" → "row 2, column 2" | 0% |
| read_cell | L1 | "What is at row 0, col 0?" → "O" | 0% |
| grid_dimensions | L1 | "How many rows and columns?" → "5 rows, 5 columns" | 0% |
| valid_neighbors | L2 | "From (3,3), which directions are non-blocked?" → "up, down, left" | 0% |
| is_move_legal | L2 | "From (0,4), is moving right legal?" → "No" | 54% |
| direction_to_goal | L2 | "Which direction moves closer to #?" → "up, right" | 28% |

**ARS: 13.7%** (L1 avg: 0.0%, L2 avg: 27.3%)

Catastrophically low L1 — the model cannot read the grid at all. This is the strongest case for atomic grounding training.

---

## Results

### basic_arithmetic (ARS: 72%)

| Condition | Steps | Accuracy | vs Base | vs Direct |
|-----------|-------|----------|---------|-----------|
| Base (SFT) | 0 | 72% | — | +40% |
| Direct-RLVR | 300 full | 32% | **-40%** | — |
| Stage 1 (L1 atoms) | 100 | 70% | -2% | +38% |
| **Atomic-Only (L1→L2)** | **200** | **78%** | **+6%** | **+46%** |
| Atomic-Warm (100+100+100) | 300 | 70% | -2% | +38% |
| Atomic-Warm-300 (200+300) | 500 | 54% | -18% | +22% |

**Verdict:** Atomic-Only is the best strategy. Full-task GRPO is always destructive — more steps = worse (78% → 70% → 54% → 32%). The model already knows most atoms; order-of-operations training (+6% from L2 stage) is the key contribution. GRPO on the full task compresses reasoning and destroys multi-step computation.

### countdown (ARS: 56%)

| Condition | Steps | Accuracy | vs Base | vs Direct |
|-----------|-------|----------|---------|-----------|
| Base (SFT) | 0 | 26% | — | -44% |
| Direct-RLVR (Phase 1) | 300 full | ~70% | +44% | — |
| Stage 1 (L1 atoms) | 100 | 20% | -6% | -50% |
| Atomic-Only (L1→L2) | 200 | 24% | -2% | -46% |
| Atomic-Warm (100+100+100) | 300 | 26% | 0% | -44% |

**Verdict:** Atomic curriculum does not help. The bottleneck is L3 (multi-step search/planning), not grounding or local reasoning. L2 skills (single-op 94%, distance 90%) are already near-ceiling. Direct-RLVR is the correct strategy — it teaches the search policy that atoms can't capture.

### shortest_path (ARS: 14%)

| Condition | Steps | Accuracy | vs Base | vs Direct |
|-----------|-------|----------|---------|-----------|
| Base (SFT) | 0 | 14% | — | -30% |
| Direct-RLVR (Phase 1) | 300 full | ~44% | +30% | — |
| Stage 1 (L1 atoms) | 100 | 18% | +4% | -26% |
| Atomic-Only (L1→L2) | 200 | 12% | -2% | -32% |
| Atomic-Warm (100+100+100) | 300 | 34% | +20% | -10% |
| **Atomic-Warm-300 (200+300)** | **500** | **46%** | **+32%** | **+2%** |

**Verdict:** Atomic warm-start + full GRPO beats Direct-RLVR when given equal full-task compute (46% vs 44%). The atoms fix catastrophic L1 grounding failure (0% on all grid reading atoms), giving GRPO a better starting point. However, atoms alone (Atomic-Only) fail because the task requires L3 planning that only full-task GRPO teaches.

---

## Key Findings

### 1. ARS Predicts Training Strategy

| ARS Profile | Observed Pattern | Prescription |
|-------------|-----------------|-------------|
| High ARS (>70%), GRPO destructive | Atoms sufficient, full-task GRPO compresses reasoning | **Atomic-Only** |
| Medium ARS (50-70%), GRPO helpful | L3 is the bottleneck, atoms don't address it | **Direct GRPO** |
| Low ARS (<20%), GRPO helpful | Grounding failure blocks GRPO; atoms fix grounding | **Atomic warm-start + GRPO** |

### 2. Full-Task GRPO Compresses Reasoning

Across all 3 tasks, adding full-task GRPO steps after atomic training reduces response length:
- Arithmetic: Atomic-Only 386 chars → Atomic-Warm 234 chars (correct answers)
- This compression skips intermediate computation steps, which helps on easy problems but destroys accuracy on hard multi-step problems

### 3. Reward Function Design is Critical for Atomic Training

Initial shortest_path experiments showed 0% reward across all 100 training steps. Root cause: the reward function required exact string matches (e.g., "row 3, column 0") but the model answered in valid alternative formats ("3, 0", "(3, 0)", "Row 3, Column 0."). With no reward signal, GRPO drifted randomly.

**Fix:** Lenient verification that accepts format variations (numeric extraction, set comparison, yes/no prefix matching). After fixing, shortest_path Stage 1 went from 0% reward → meaningful training signal.

**Lesson for atomic environments:** Unlike full-task eval where answer format is standardized, atomic Q&A can have many correct phrasings. Atomic reward functions must be format-tolerant.

### 4. Reasoning Compression Kills Atomic-Only on Grid Tasks

For shortest_path, Atomic-Only (12%) was worse than base (14%) despite atoms receiving reward. The L2 atoms (is_move_legal → "Yes"/"No", valid_neighbors → "up, left") have short answers, causing GRPO to learn ultra-compressed outputs. When evaluated on the full path-finding task, the model dumps random paths without reasoning.

This doesn't happen for arithmetic because L2 atoms (order_of_operations, single_op_eval) require numeric computation that naturally produces longer reasoning chains.

---

## Experiment Artifacts

**Location:** Green cluster `/scratch/sunguha/rlvr_experiments/results/`

| Artifact | Path |
|----------|------|
| Arithmetic Stage 1 (L1) | `phase2_arith/stage1_l1/` |
| Arithmetic Stage 2 (L1→L2) = Atomic-Only | `phase2_arith/stage2_l2/` |
| Arithmetic Stage 3 (L1→L2→Full) = Atomic-Warm | `phase2_arith/stage3_full/` |
| Arithmetic Direct-RLVR | `phase2_arith/direct_rlvr/` |
| Arithmetic Atomic-Warm-300 | `phase2_arith/atomic_warm_300/` |
| Arithmetic ARS (base) | `outputs/ars_4b_arithmetic_v2.json` |
| Countdown all stages | `phase2_countdown/stage{1,2,3}_*/` |
| Countdown ARS (base) | `phase2_countdown/ars_base_countdown.json` |
| Shortest path all stages | `phase2_sp/stage{1,2,3}_*/` |
| Shortest path Atomic-Warm-300 (best: 46%) | `phase2_sp/atomic_warm_300/` |
| Shortest path SFT-L1 (high LR) | `phase2_sp/sft_l1/` |
| Shortest path SFT-L1 (low LR) | `phase2_sp/sft_l1_lowlr/` |
| Shortest path SFT-L1→Full-300 | `phase2_sp/sft_lowlr_full_300/` |
| Shortest path Progressive grids | `phase2_sp/prog_{l1_3x3,l2_4x4,full_300}/` |
| Shortest path ARS (base) | `phase2_sp/ars_base.json` |

### Code

| File | Purpose |
|------|---------|
| `environments/atomic_templates.py` | Atom definitions for arithmetic, countdown, shortest_path |
| `environments/atomic_env.py` | AtomicEnvironment class for GRPO training on atoms |
| `probe_atomics.py` | ARS probe script (zero-shot atomic accuracy) |
| `compare_ars.py` | Compare ARS between two checkpoints |
| `configs/phase2_arithmetic/` | All arithmetic training configs |
| `configs/phase2_countdown/` | All countdown training configs |
| `configs/phase2_shortest_path/` | All shortest_path training configs |

---

## Methodology Notes

### Evaluation
- 50 samples per task, seed 9999 (different from training seed 42)
- DDP 8 GPUs for eval (torchrun)
- Answer extraction: last `<answer>...</answer>` match (fixed from Phase 1 bug that took the first match, inflating error counts when the model echoed the chat template)

### Hyperparameters (identical across all conditions)
- Learning rate: 5e-6 (constant)
- Loss: DAPO
- Beta: 0.0, KL coeff: 0.0
- Generations per prompt: 8
- Per-device batch size: 1
- Max completion length: 256 (atoms), 2048 (full task)
- Max prompt length: 512 (arithmetic/countdown), 1024 (shortest_path)

### Known Issues Fixed During Study
1. **Answer extraction bug:** `re.search` took the first `<answer>` tag, which matched the echoed chat template. Fixed to use `re.findall` and take the last match. This affected all Phase 1 numbers.
2. **Strict reward function:** Atomic rewards required exact string matches, causing 0% reward on shortest_path. Fixed with lenient verification (numeric extraction, yes/no prefix, set comparison).
3. **Reasoning compression:** GRPO on short-answer atoms causes the model to skip `<think>` tags entirely. System prompt changes ("think deeply") did not fix this — GRPO's policy gradient overwhelms prompt instructions.

---

## Appendix: SFT-for-L1 Experiment (shortest_path)

Hypothesis: Since L1 atoms have fully determined answers (exactly one * position), SFT should be more stable than GRPO — no sparse reward, no compression risk.

### Method
- Generated 2000 SFT samples (500 per L1 atom) with step-by-step reasoning traces
- SFT at lr=2e-5 for 2 epochs on Qwen3-4B base
- Then GRPO-L2 (100 steps) and/or GRPO-Full (300 steps) on top

### Results

| Condition | Accuracy | ARS |
|-----------|----------|-----|
| Base | 14% | 19.4% |
| SFT-L1 | 10% | **65.2%** |
| SFT-L1 → GRPO-L2 | 2% | — |
| SFT-L1 → GRPO-L2 → GRPO-Full-300 | 40% | — |
| SFT-L1 → GRPO-Full-300 (skip L2) | 2% | — |

### ARS Improvement from SFT-L1

| Atom | Before | After SFT | Delta |
|------|--------|-----------|-------|
| locate_start | 0% | 92% | +92% |
| locate_goal | 0% | 88% | +88% |
| grid_dimensions | 0% | 100% | +100% |
| direction_to_goal | 28% | 78% | +50% |
| **L1 avg** | **11.5%** | **83.0%** | **+71.5%** |

### Analysis

SFT massively improved L1 grounding (ARS 19% → 65%) without reasoning compression — avg response length stayed at 235 chars. However:

1. **SFT-L1 → GRPO-L2 collapsed to 2%** — same compression problem as GRPO-only L2 atoms
2. **SFT-L1 → GRPO-Full-300 collapsed to 2%** — the model output garbage tokens (`<think>...シンプ...`) after 300 steps of full-task GRPO. The SFT appears to have made the model *fragile* — it overfit to the atom format and couldn't withstand aggressive GRPO fine-tuning.
3. **SFT-L1 → GRPO-L2 → GRPO-Full-300 recovered to 40%** — but still below GRPO-Atomic-Warm-300 (46%)

**Conclusion:** SFT is excellent for teaching grounding skills, but the resulting model is fragile under subsequent GRPO. The GRPO-only pipeline (with lenient reward) remains the best approach for shortest_path (46%). A lower SFT learning rate or mixing SFT with GRPO data may help, but this is left for future work.

---

## Appendix A: SFT-for-L1 and Progressive Grid Experiments

Attempted to break past 46% with two initial approaches:

### Approach 1: SFT-L1 at Low LR (5e-6, 3 epochs) → GRPO-Full-300
- **Hypothesis:** The high-LR SFT (2e-5) overfit and made the model fragile. Lower LR should preserve robustness.
- **Result:** 40% — better than high-LR SFT (2%), confirming the fragility hypothesis, but still below pure GRPO atoms (46%).

### Approach 2: Progressive Grid Atoms (3x3→4x4→Full)
- **Hypothesis:** L1 atoms at 0% baseline on 5-8 row grids = sparse reward. Train on 3x3 grids first (easy to parse, high reward rate), then graduate to larger grids.
- **Result:** 40% — on par with SFT approach but below GRPO atoms (46%).
- **Why it didn't help:** The atom skills learned on 3x3 grids (trivially small, often just one path) may not transfer to the structural complexity of 5-8 row grids.

---

## Appendix B: Push Past 50% — Ideas A through E

Systematic exploration of 5 ideas to push shortest_path accuracy past 50% within ≤300 GRPO steps.

### Idea A: Progressive Full-Task Curriculum (no atoms)

Train on actual shortest_path (not atoms) with increasing grid size: 3x3 (100 steps) → 4x4 (100 steps) → full (100 steps).

| Stage | Grid | Accuracy |
|-------|------|----------|
| After 3x3 | full eval | 31% |
| After 4x4 | full eval | 37.5% |
| **After full** | **full eval** | **46.5%** |

**Verdict:** Matches the previous best (Atomic-Warm-300 at 46%) but doesn't break through. Progressive difficulty helps during early stages (10→31→37.5%) but converges to the same ceiling. The model gets high reward on easy grids early, giving strong gradient signal, but the full-size grid stage plateaus at the same level as direct training.

### Idea B: Path-Tracing L2 Atoms → Full Task (NEW BEST)

**Key insight:** The original L2 atoms (`is_move_legal` → "Yes", `direction_to_goal` → "down, right") have short answers that cause GRPO reasoning compression. New **L2-long atoms** require multi-step path simulation:

| Atom | Example Q | Example A |
|------|-----------|-----------|
| `sp_execute_path` | "Start at (3,1), execute: up, right, down. Report each step." | "Move up from (3,1) to (2,1) — cell is O; Move right... Final position: row 2, column 1" |
| `sp_trace_and_report` | "Trace path from *: down, right, up. Report cell contents." | "Step 1: down to (4,1) is O (open); Step 2: right to (4,2) is X (blocked)... Final position: row 3, column 1" |
| `sp_count_walls_on_path` | "Scan row 2 left to right. How many walls?" | "Row 2: (2,0)=O, (2,1)=X, (2,2)=O... Wall count: 1" |

These atoms force multi-step reasoning — no compression possible.

**Training:** L2-long atoms (100 steps) → Full task (200 steps) = 300 total steps.

| Stage | Accuracy |
|-------|----------|
| After L2-long atoms | 21.5% |
| **After full task** | **51.5%** |

**Verdict: First approach past 50%.** The path-tracing pretraining teaches the model to navigate grids step-by-step. Unlike short-answer atoms, the long-form trace exercises the exact skill needed for the full task. Going directly to full-difficulty grids (not progressive) forces generalization.

**Training dynamics:** Stage 2 (full task) started with oscillating reward (0-50%) for the first 25 steps, then broke through at step ~50 with consistent 87.5-100% correctness on training batches. Completion length decreased from 1144 → 475 chars as the model learned concise-but-correct paths.

### Idea C: Partial-Credit Reward Shaping

Changed from binary (1.0/0.0) to graduated rewards:
- +0.1 per correct start/goal position mentioned in reasoning
- +0.5 for any valid path (reaches goal, no wall collisions)
- +1.0 for shortest path

**Result: 35.5%** — significantly worse than Direct-RLVR (44%).

**Why it failed:** Mean reward was 0.9 throughout training — too generous. The model got high reward for suboptimal paths, killing the gradient signal GRPO needs (variance between correct and incorrect completions). Partial credit creates a reward landscape where "good enough" is rewarded almost as much as "correct," preventing the model from refining its paths.

### Idea D: Progressive + Partial Credit

Combined progressive grids (3x3→4x4→full, 100 steps each) with partial-credit rewards.

**Result: 24%** — worst of all approaches.

**Why it failed:** Partial credit on easy grids (3x3) gave nearly 100% reward to every completion — zero variance, zero gradient. The model learned nothing in the first two stages and was worse than base when facing full grids. This confirms: **partial credit is consistently harmful for GRPO on this task.**

### Idea E: Long Atoms → Progressive Full-Task

Combined B's winning long atoms with A's progressive grids: L2-long atoms (100 steps, reused B's checkpoint) → 3x3 (70 steps) → 4x4 (70 steps) → full (60 steps).

**Result: 27.5%** — much worse than B (51.5%).

**Why it failed:** After long-atom pretraining, the model already had path-tracing skills. Training on 3x3 grids (where it got 100% reward) caused it to overfit to easy grids — collapsing completion length and losing the ability to handle larger grids. **The difficulty jump from atoms to full grids is essential** — it forces the model to generalize its path-tracing skills rather than memorize solutions to easy instances.

### Key Insight: Difficulty Transitions Matter

The pattern across all 5 ideas reveals a principle: **the optimal training curriculum has exactly one hard transition**, from skill pretraining (atoms) directly to full difficulty. Both progressive easing (A, E) and reward softening (C, D) reduce the difficulty of this transition but also reduce the generalization pressure:

| Transition Type | Example | Result |
|-----------------|---------|--------|
| Hard (atoms → full grids) | Idea B | **51.5%** (best) |
| Gradual (atoms → easy → medium → hard) | Idea E | 27.5% |
| Gradual (no atoms, easy → hard) | Idea A | 46.5% |
| Soft reward (full grids, easy reward) | Idea C | 35.5% |

### Full Shortest Path Results Table

| Condition | Full-task Steps | Atom Steps | Total Steps | Accuracy |
|-----------|----------------|------------|-------------|----------|
| Base (SFT) | 0 | 0 | 0 | 14% |
| Direct-RLVR (Phase 1) | 300 | 0 | 300 | 44% |
| GRPO Atomic-Only (short L2) | 0 | 200 | 200 | 12% |
| GRPO Atomic-Warm | 100 | 200 | 300 | 34% |
| GRPO Atomic-Warm-300 | 300 | 200 | 500 | 46% |
| SFT-L1 (high LR) → Full-300 | 300 | SFT | — | 2% |
| SFT-L1 → GRPO-L2 → Full-300 | 300 | SFT+100 | — | 40% |
| SFT-L1 (low LR) → Full-300 | 300 | SFT | — | 40% |
| Progressive atoms (3x3→4x4→Full) | 300 | 200 | 500 | 40% |
| **Idea A:** Prog full-task (3x3→4x4→full) | 300 | 0 | 300 | 46.5% |
| **Idea B:** L2-long atoms → full | 200 | 100 | 300 | **51.5%** |
| Idea C: Partial credit | 300 | 0 | 300 | 35.5% |
| Idea D: Prog + partial credit | 300 | 0 | 300 | 24% |
| Idea E: L2-long atoms → prog full | 200 | 100 | 300 | 27.5% |

### Experiment Artifacts (Ideas A–E)

| Artifact | Path |
|----------|------|
| Idea A stages | `phase2_sp/idea_a/stage{1,2,3}_*/` |
| Idea B stages (best: 51.5%) | `phase2_sp/idea_b/stage{1,2}_*/` |
| Idea C | `phase2_sp/idea_c/partial_credit/` |
| Idea D stages | `phase2_sp/idea_d/stage{1,2,3}_*/` |
| Idea E stages | `phase2_sp/idea_e/stage{2,3,4}_*/` |
| Configs | `configs/phase2_sp_ideas/` |

### Code Changes for Ideas A–E

| File | Change |
|------|--------|
| `environments/atomic_templates.py` | Added `L2_long` atoms: `sp_execute_path`, `sp_trace_and_report`, `sp_count_walls_on_path` |
| `environments/atomic_env.py` | Extended `_check()` to handle long-form answers (final position extraction, wall count extraction) |
| `environments/partial_credit_sp_env.py` | New environment with graduated rewards (BFS shortest path verification) |
| `train_grpo.py` | Added `partial_credit_sp` environment registration |

---

## Appendix C: GRPO+SFT Interleaving (Ideas G, G2, G3)

### Idea G: Interleaved GRPO + SFT Cycles

**Hypothesis:** Alternate between GRPO (explore) and SFT on BFS-correct solutions (correct mistakes). 3 cycles × (50 GRPO + 50 SFT steps), each cycle with new problem seeds.

| Variant | SFT Samples | SFT LR | Cycle 1 | Cycle 2 | Cycle 3 | Final |
|---------|-------------|--------|---------|---------|---------|-------|
| G (heavy SFT) | 5000 | 5e-6 | 21% | 9.5% | 5.5% | **collapsed** |
| G2 (light SFT) | 500 | 1e-6 | 15.5% | 27% | 40% | improving |

**Key finding:** Heavy SFT (5000 samples, lr=5e-6) destroys GRPO gains — collapses from 21% to 5.5%. Light SFT (500 samples, lr=1e-6) preserves gains and improves monotonically (+11.5%, +13% per cycle). However, extending G2 to 5 cycles showed the trend plateauing and reversing (cycle 4: 38.5%, cycle 5: 34%).

### Idea G3: Continue G2 with Pure GRPO

Added 150 GRPO steps on top of G2's cycle 3 checkpoint (40%) to reach 300 total GRPO steps.

**Result: 45%** — matches Direct-RLVR but doesn't beat B.

---

## Appendix D: Feasibility and Anti-Infeasible Experiments (Ideas F, L)

### Error Analysis of B's 51.5% Model

Detailed analysis on 100 eval samples revealed the dominant failure modes:

| Error Type | Count | Description |
|------------|-------|-------------|
| False infeasible | 23 | Model says "infeasible" when path exists |
| Wrong path | 25 | Model gives incorrect directions |
| Missed infeasible | 6 | Model gives path when none exists |

**If all false-infeasible errors were fixed: 68.3%.** The model has a "give-up reflex" — defaulting to "infeasible" rather than trying to trace a path.

- **Feasible accuracy: 25%** (16/64 feasible problems correct)
- **Infeasible accuracy: 83%** (30/36 infeasible problems correct)

### Idea F: Feasibility + Long Atoms → Full Task

Added 4 anti-false-infeasible atoms (`sp_path_exists`, `sp_find_any_path`, `sp_start_neighbors`, `sp_bfs_first_steps`) combined with L2-long atoms. 100 atom steps → 200 full task steps.

**Result: 42%.** Reduced false-infeasible count (4 vs 8 on rank-0 shard) but introduced more wrong-path errors. The feasibility atoms diluted the path-tracing skills.

### Idea L: Feasible-Only Training Distribution

Filtered training data to only include grids where a valid path exists (3-8 steps). The model never gets rewarded for saying "infeasible."

**Result: 40.5%.** The model learned to always attempt paths but lost the ability to correctly identify infeasible grids, and path quality wasn't better than B's.

---

## Appendix E: Rejection Sampling Fine-Tuning (RFT) — NEW BEST

### Method

On-policy rejection sampling: sample K=8 completions per problem from a trained checkpoint, verify each against BFS ground truth, keep only correct ones. Difficulty-weighted: frontier problems (pass rate ≤25%) get 4x weight, medium (25-50%) get 2x, easy (>50%) get 1x.

### RFT on B's Checkpoint → B's Checkpoint

1. Generated 328 verified-correct samples from B's 51.5% model (210/300 problems had ≥1 correct, avg pass rate 70%)
2. SFT on B's own correct answers (lr=1e-6, 1 epoch): **49.5%** (slight dip)
3. GRPO 100 steps: **51.5%** (recovered to same level)

**Verdict:** On-policy RFT on the already-trained model just recovers to the same equilibrium. No new capability added.

### RFT-Base: B's Data on Base Model → GRPO (NEW BEST: 55.5%)

**Key insight:** Use B's verified-correct completions as SFT data for the *base* model, then GRPO.

1. Same 328 RFT samples from B's checkpoint (no new data generation needed)
2. SFT the base SFT model on these samples (lr=1e-6, 1 epoch): **15.5%** (up from 10% base)
3. GRPO 300 steps: **55.5%**

**Result: 55.5% — new best, +4% over B (51.5%), +11.5% over Direct-RLVR (44%).**

### Why RFT-Base Works Better Than B

B's training path: base → atom GRPO (100 steps) → full GRPO (200 steps) = 51.5%
RFT-Base path: base → SFT on B's correct traces (328 samples) → full GRPO (300 steps) = 55.5%

The RFT-Base model gets:
1. **More full-task GRPO steps** (300 vs B's 200) — same total budget but all on the actual task
2. **Better initialization** — SFT on verified-correct traces gives a clean starting point without the atom-format artifacts that B's atom stage introduces
3. **On-policy data quality** — B's correct answers are high-quality demonstrations generated by a model that already learned path-tracing, providing richer signal than atoms

The GRPO reward trajectory confirms this: RFT-Base started at mean reward 0.45 (vs B's stage 2 starting near 0) and maintained 0.48+ throughout, meaning more gradient signal per step.

### Complete Shortest Path Results Table

| Approach | Accuracy | GRPO Steps | Notes |
|----------|----------|-----------|-------|
| Base (SFT) | 10% | 0 | reference |
| Direct-RLVR | 44% | 300 | Phase 1 baseline |
| Atomic-Warm-300 | 46% | 300+200 atom | short L2 atoms |
| A (prog 3x3→4x4→full) | 46.5% | 300 | progressive full-task |
| B (L2-long atoms → full) | 51.5% | 300 | long atoms, hard transition |
| B2 (300+300) | 47% | 600 | more steps hurt |
| C (partial credit) | 35.5% | 300 | reward too easy |
| D (prog + partial credit) | 24% | 300 | PC consistently harmful |
| E (long atoms → prog full) | 27.5% | 300 | progressive after atoms hurts |
| F (feasibility atoms → full) | 42% | 300 | fewer false-inf, more wrong paths |
| G (heavy GRPO+SFT ×3) | 5.5% | 150 | SFT collapsed it |
| G2 (light GRPO+SFT ×3) | 40% | 150 | improving trend |
| G3 (G2 + 150 GRPO) | 45% | 300 | matches Direct-RLVR |
| L (feasible-only) | 40.5% | 300 | lost infeasibility detection |
| RFT on B | 51.5% | 400 | same ceiling as B |
| **RFT-Base** | **55.5%** | **300** | **B's data → base model → GRPO** |

### Experiment Artifacts (Ideas F–RFT)

| Artifact | Path |
|----------|------|
| Idea F stages | `phase2_sp/idea_f/` |
| G/G2/G3 cycles | `phase2_sp/idea_g/`, `phase2_sp/idea_g2/`, `phase2_sp/idea_g3/` |
| Idea L (feasible-only) | `phase2_sp/idea_l/` |
| RFT data (328 samples) | `phase2_sp/idea_rft/rft_data.jsonl` |
| RFT on B | `phase2_sp/idea_rft/sft/`, `phase2_sp/idea_rft/grpo/` |
| **RFT-Base (best: 55.5%)** | `phase2_sp/idea_rft_base/sft/`, `phase2_sp/idea_rft_base/grpo/` |

### Code Added for Ideas F–RFT

| File | Purpose |
|------|---------|
| `environments/atomic_templates.py` | Added `L2_feasibility` atoms (path_exists, find_any_path, start_neighbors, bfs_first_steps) |
| `environments/feasible_only_sp_env.py` | Feasible-only environment with BFS filtering |
| `generate_rft_data.py` | On-policy rejection sampling data generator (file-based DDP gather) |
| `generate_sp_sft_bfs.py` | BFS solution generator for SFT data |
| `eval_best_of_n_sp.py` | Best-of-N evaluation script |
| `analyze_idea_b_errors.py` | Error analysis (false-infeasible breakdown) |

---

## Status update post Phase 4 Part 6 (2026-05-12)

This early Phase 2 report (3-task focus + shortest-path 15-condition deep dive) is unaffected by Phase 4 Part 6. ARS as a single-task training-regime predictor remains robust on 6/7 tasks. The proposal that L1 ARS would also predict multi-task merge compatibility (Phase 4 F33, r = −0.76 at n = 7) was tested at n = 12 and falsified (r = +0.06, sign flipped). The Phase 2 → Phase 4 grounding-bottleneck unification was retracted; ARS remains valid within Phase 2 alone.

**Cross-references:**
- Phase 2 complete (7-task analysis): `report_phase2_complete.md`
- Phase 4 (merging at scale, including Part 6 falsification): `report_phase4_merge_scaling.md`
- Master document: `MASTER_RLVR_dynamics.md`
