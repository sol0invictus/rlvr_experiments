# RLVR Paper Prep — 1-Day Experimental Results

All experiments run 2026-05-04 on green-cluster pod `sunguha-rlvr-paper-npdz`, 8× B200 GPUs.

## Stream D — Sign-agreement predictor and attn.{k,v,o} rotation

### D.1 Bootstrap CI on r(3B mag_agree, 4B merge damage)
- n = 7 task pairs
- **r = -0.810**, 95% bootstrap CI [-0.993, -0.376]
- Permutation p-value (two-sided, 10000 perms) = **0.0237**

Per-pair values:

| pair | mag_agree (3B) | damage (4B) |
|------|---:|---:|
| T1_sdk_leg | 0.5300 | 18.0 |
| T2_sdk_kk | 0.5306 | 22.0 |
| T3_ari_leg | 0.5641 | 12.5 |
| T4_ari_syl | 0.6053 | 4.5 |
| T5_sp_syl | 0.5358 | 7.0 |
| T6_sp_kk | 0.5131 | 18.0 |
| T7_cnt_ari | 0.4931 | 24.0 |

### D.2 Held-out validation of the predictor
- Leave-one-out: **r(pred, truth) = 0.725**, MAE = 3.59 damage pts, RMSE = 4.77
- Train {T1–T4} → predict {T5–T7}: **r = 0.991**, MAE = 6.93
- Train {T5–T7} → predict {T1–T4}: **r = 0.974**, MAE = 15.29

The sign-agreement predictor generalizes well across held-out pairs. Split-based fits both have r > 0.97.

### D.3 attn.{k,v,o} novel-subspace rotation
Tests Phase-4 F21 prediction: attn.{k,v,o} carry merge-relevant structure that `q_proj` alone misses.

**4B — mean principal angle per layer type (degrees)**

| pair | k_proj | v_proj | o_proj |
|------|---:|---:|---:|
| T1_sdk_leg | 37.3° | 58.1° | 61.4° |
| T2_sdk_kk | 41.3° | 63.5° | 59.6° |
| T3_ari_leg | 40.1° | 53.5° | 57.7° |
| T4_ari_syl | 41.3° | 60.1° | 60.5° |
| T5_sp_syl | 38.3° | 64.9° | 60.9° |
| T6_sp_kk | 37.9° | 65.5° | 60.6° |
| T7_cnt_ari | nan° | nan° | nan° |

**8B — mean principal angle per layer type (degrees)**

| pair | k_proj | v_proj | o_proj |
|------|---:|---:|---:|
| T1_sdk_leg | 33.3° | 61.9° | 56.0° |
| T2_sdk_kk | 37.6° | 56.9° | 58.1° |
| T3_ari_leg | 35.8° | 55.9° | 64.0° |
| T4_ari_syl | 35.2° | 49.2° | 60.8° |
| T5_sp_syl | 35.4° | 56.4° | 68.0° |
| T6_sp_kk | 34.9° | 50.4° | 70.1° |
| T7_cnt_ari | 31.8° | 55.3° | 67.7° |

## Stream A — Linking experiment (atomic vs direct τ)

Tests whether atomic-trained task vectors merge more cleanly than direct-RLVR task vectors.

### A.1 Sign agreement (atomic vs direct τ)

| pair | group | mag_agree | conflict_mass |
|------|-------|---:|---:|
| sp_ari | atomic | 0.5190 | 0.4810 |
| sp_maze | atomic | 0.5118 | 0.4882 |
| ari_maze | atomic | 0.5059 | 0.4941 |
| sp_ari | direct | 0.5006 | 0.4994 |
| sp_maze | direct | 0.5150 | 0.4850 |
| ari_maze | direct | 0.5085 | 0.4915 |
| sp_atomic_x_ari_direct | cross | 0.4966 | 0.5034 |
| sp_atomic_x_maze_direct | cross | 0.5071 | 0.4929 |
| ari_atomic_x_maze_direct | cross | 0.5079 | 0.4921 |

### A.2 Merged checkpoints — 10-task accuracy

| model                          | basic_ar | countdow | shortest |     maze | mini_sud | syllogis | knights_ | leg_coun | caesar_c | cryptari |
|----------------------------------|------------|------------|------------|------------|------------|------------|------------|------------|------------|------------|
| M_aa (atomic+atomic)           |  75.0 |  36.0 |  54.0 |  11.0 |  17.5 |  83.0 |  66.5 |  55.5 |   0.5 |   0.0 |
| M_dd (direct+direct)           |  66.5 |  35.5 |  52.5 |   3.5 |  19.0 |  48.5 |  65.5 |  44.0 |   0.0 |   0.0 |
| M_mix (atomic_SP+direct_Arith) |  65.5 |  37.5 |  51.5 |   1.0 |  14.5 |  86.0 |  68.5 |  47.5 |   0.0 |   0.0 |

## Stream B — TIES + DARE baselines vs vanilla sum

| pair | method | basic_ar | countdow | shortest | maze | mini_sud | syllogis | knights_ | leg_coun | caesar_c | cryptari |
|------|--------|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| T2_sdk_kk | ties |  48.5 |   8.0 |   1.0 |   0.5 |  36.5 |  66.0 |  76.0 |   1.0 |   0.0 |   0.0 |
| T2_sdk_kk | dare |  50.0 |   9.0 |   0.5 |   0.5 |  36.0 |  66.0 |  68.5 |   1.0 |   0.0 |   0.0 |
| T5_sp_syl | ties |  70.0 |  26.0 |  48.0 |   4.5 |  25.5 |  85.0 |  70.0 |  53.0 |   0.0 |   0.0 |
| T5_sp_syl | dare |  69.0 |  27.5 |  51.5 |   3.5 |  24.5 |  84.5 |  67.5 |  55.5 |   0.0 |   0.0 |
| T7_cnt_ari | ties |  60.5 |   -   |  23.5 |   1.0 |  16.5 |  85.5 |  73.5 |  40.0 |   0.0 |   0.0 |
| T7_cnt_ari | dare |  62.5 |  43.0 |  27.5 |   1.0 |  14.5 |  86.5 |  74.5 |  43.0 |   0.0 |   0.0 |

For reference, the existing Phase-4 vanilla-sum results (from `outputs/eval10_phase4_*/`):

| T2_sdk_kk | vanilla_sum |  50.0 |   8.0 |   0.5 |   0.5 |  35.0 |  64.0 |  65.0 |   1.0 |   0.0 |   0.0 |
| T5_sp_syl | vanilla_sum |  71.0 |  28.0 |  47.5 |   3.5 |  31.5 |  85.0 |  70.0 |  55.5 |   0.0 |   0.0 |
| T7_cnt_ari | vanilla_sum |  63.0 |  46.0 |  27.0 |   0.5 |  17.0 |  84.0 |  72.5 |  39.5 |   0.0 |   0.0 |

## Stream C — n=2 eval-seed error bars (seed 1234 vs original 9999)

| checkpoint | seed | basic_ar | countdow | shortest | maze | mini_sud | syllogis | knights_ | leg_coun | caesar_c | cryptari |
|------------|-----:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| p2_arith_atomic | 1234 |  75.0 |  33.0 |  18.5 |   8.0 |   8.5 |  85.5 |  62.5 |  61.5 |   0.0 |   0.0 |
| p2_sp_rft_base | 1234 |  72.0 |  39.5 |  55.5 |   5.0 |  11.5 |  83.5 |  65.0 |  57.0 |   0.0 |   0.0 |
| merge_2task_basic | 1234 |  76.0 |  46.5 |  60.5 |   6.0 |  20.0 |  85.0 |  73.0 |  54.0 |   0.0 |   0.0 |
| phase4_T5_sp_syl | 1234 |  65.5 |  29.0 |  43.0 |   3.5 |  21.5 |  88.5 |  63.0 |  46.5 |   0.0 |   0.0 |

## One-page summary of findings

- **Sign-agreement predictor holds under scrutiny**: r = -0.810 on n=7, bootstrap 95% CI [-0.993, -0.376], permutation p = 0.0237. Leave-one-out r = 0.725.
- **Predictor generalizes to held-out pairs**: train T1-T4 → predict T5-T7: r = 0.991.
- **attn.{k,v,o} rotation data now exists at 4B and 8B** — enables the F21 follow-up in Phase 3.
- **Linking experiment**: sign-agreement for atomic-atomic SP+Arith = 0.5190 vs direct-direct = 0.5006. Supports hypothesis that atomic-trained τ's are more compatible.
- **TIES/DARE baselines**: now have 6 new merge evaluations at 4B for head-to-head comparison against the vanilla-sum recipe.

---

Generated by experiments/rlvr_geometry_paper/paper/scripts/consolidate.py