# Config index

Generated from `MANIFEST.csv`. `Original` is the path under `rlvr_experiments/` that scripts actually read.


## `not_in_paper/early_smoke_tests/` (6)

| File | Paper | Role | Task | LR | Steps | Gens | Original |
|---|---|---|---|---|---|---|---|
| `chain_maze_test.yaml` | none | 0.5B maze smoke test | maze | 2e-05 | 20 | 4 | `configs/chain_maze_test.yaml` |
| `config_battleship.yaml` | none | 0.5B battleship | battleship | 1e-5 | - | 4 | `configs/config_battleship.yaml` |
| `config_countdown_grpo.yaml` | none | DeepSeek-R1-Distill-Qwen-1.5B countdown | countdown | 5e-06 | -1 | 8 | `configs/config_countdown_grpo.yaml` |
| `config_gsm8k.yaml` | none | TinyLlama GSM8K |  | 1e-05 | 500 | 4 | `configs/config_gsm8k.yaml` |
| `config_maze.yaml` | none | 0.5B maze GRPO | maze | 0.0001 | 1000 | 4 | `configs/config_maze.yaml` |
| `data_gen_maze.yaml` | none | Maze SFT data generation | maze |  | - | - | `configs/data_gen_maze.yaml` |

## `not_in_paper/hp_sweep_prelim/` (11)

| File | Paper | Role | Task | LR | Steps | Gens | Original |
|---|---|---|---|---|---|---|---|
| `A1_grpo_baseline.yaml` | none (no mention in paper or reports) | Preliminary loss/lr/num_gen/beta sweep, countdown on Qwen3-4B-Instruct-2507 (run_sweep.sh) | countdown | 5e-06 | 350 | 8 | `configs/hp_sweep/A1_grpo_baseline.yaml` |
| `A2_dr_grpo.yaml` | none (no mention in paper or reports) | Preliminary loss/lr/num_gen/beta sweep, countdown on Qwen3-4B-Instruct-2507 (run_sweep.sh) | countdown | 5e-06 | 350 | 8 | `configs/hp_sweep/A2_dr_grpo.yaml` |
| `A3_dapo.yaml` | none (no mention in paper or reports) | Preliminary loss/lr/num_gen/beta sweep, countdown on Qwen3-4B-Instruct-2507 (run_sweep.sh) | countdown | 5e-06 | 350 | 8 | `configs/hp_sweep/A3_dapo.yaml` |
| `A4_cispo.yaml` | none (no mention in paper or reports) | Preliminary loss/lr/num_gen/beta sweep, countdown on Qwen3-4B-Instruct-2507 (run_sweep.sh) | countdown | 5e-06 | 350 | 8 | `configs/hp_sweep/A4_cispo.yaml` |
| `B1_lr_5e5.yaml` | none (no mention in paper or reports) | Preliminary loss/lr/num_gen/beta sweep, countdown on Qwen3-4B-Instruct-2507 (run_sweep.sh) | countdown | 5e-05 | 350 | 8 | `configs/hp_sweep/B1_lr_5e5.yaml` |
| `B3_lr_5e7.yaml` | none (no mention in paper or reports) | Preliminary loss/lr/num_gen/beta sweep, countdown on Qwen3-4B-Instruct-2507 (run_sweep.sh) | countdown | 5e-07 | 350 | 8 | `configs/hp_sweep/B3_lr_5e7.yaml` |
| `C1_numgen_4.yaml` | none (no mention in paper or reports) | Preliminary loss/lr/num_gen/beta sweep, countdown on Qwen3-4B-Instruct-2507 (run_sweep.sh) | countdown | 5e-06 | 350 | 4 | `configs/hp_sweep/C1_numgen_4.yaml` |
| `C3_numgen_16.yaml` | none (no mention in paper or reports) | Preliminary loss/lr/num_gen/beta sweep, countdown on Qwen3-4B-Instruct-2507 (run_sweep.sh) | countdown | 5e-06 | 350 | 16 | `configs/hp_sweep/C3_numgen_16.yaml` |
| `D1_beta_0.yaml` | none (no mention in paper or reports) | Preliminary loss/lr/num_gen/beta sweep, countdown on Qwen3-4B-Instruct-2507 (run_sweep.sh) | countdown | 5e-06 | 350 | 8 | `configs/hp_sweep/D1_beta_0.yaml` |
| `D2_beta_001.yaml` | none (no mention in paper or reports) | Preliminary loss/lr/num_gen/beta sweep, countdown on Qwen3-4B-Instruct-2507 (run_sweep.sh) | countdown | 5e-06 | 350 | 8 | `configs/hp_sweep/D2_beta_001.yaml` |
| `D3_beta_01.yaml` | none (no mention in paper or reports) | Preliminary loss/lr/num_gen/beta sweep, countdown on Qwen3-4B-Instruct-2507 (run_sweep.sh) | countdown | 5e-06 | 350 | 8 | `configs/hp_sweep/D3_beta_01.yaml` |

## `not_in_paper/pilot_1p5b_sft_base/` (5)

| File | Paper | Role | Task | LR | Steps | Gens | Original |
|---|---|---|---|---|---|---|---|
| `config_qwen3_basic_arithmetic.yaml` | none | Early pilot GRPO, basic_arithmetic, init sft_qwen_1.5b | basic_arithmetic | 5e-06 | 300 | 8 | `configs/config_qwen3_basic_arithmetic.yaml` |
| `config_qwen3_countdown.yaml` | none | Early pilot GRPO, countdown, init sft_qwen_1.5b | countdown | 5e-06 | 300 | 8 | `configs/config_qwen3_countdown.yaml` |
| `config_sft_qwen0.5b.yaml` | none | SFT warm-up producing sft_qwen_1.5b (file name says 0.5b but loads Qwen2.5-1.5B-Instruct) |  | 2e-05 | - | - | `configs/config_sft_qwen0.5b.yaml` |
| `config_qwen3_caesar_cipher.yaml` | none | Early pilot GRPO, caesar_cipher, init sft_qwen_1.5b | caesar_cipher | 5e-06 | 300 | 8 | `configs/qwen3_new_tasks/config_qwen3_caesar_cipher.yaml` |
| `config_qwen3_shortest_path.yaml` | none | Early pilot GRPO, shortest_path, init sft_qwen_1.5b | shortest_path | 5e-06 | 300 | 8 | `configs/qwen3_new_tasks/config_qwen3_shortest_path.yaml` |

## `not_in_paper/pilot_eval_configs/` (5)

| File | Paper | Role | Task | LR | Steps | Gens | Original |
|---|---|---|---|---|---|---|---|
| `eval_caesar_base.yaml` | none | Early pilot evaluation config (pre evaluate_transfer.py protocol) | caesar_cipher |  | - | - | `experiments/rlvr_geometry_paper/configs/eval/eval_caesar_base.yaml` |
| `eval_caesar_qwen25_7b_base.yaml` | none | Early pilot evaluation config (pre evaluate_transfer.py protocol) | caesar_cipher |  | - | - | `experiments/rlvr_geometry_paper/configs/eval/eval_caesar_qwen25_7b_base.yaml` |
| `eval_countdown.yaml` | none | Early pilot evaluation config (pre evaluate_transfer.py protocol) | countdown |  | - | - | `configs/eval/eval_countdown.yaml` |
| `eval_countdown_grpo.yaml` | none | Early pilot evaluation config (pre evaluate_transfer.py protocol) | countdown |  | - | - | `experiments/rlvr_geometry_paper/configs/eval/eval_countdown_grpo.yaml` |
| `eval_countdown_sft_base.yaml` | none | Early pilot evaluation config (pre evaluate_transfer.py protocol) | countdown |  | - | - | `experiments/rlvr_geometry_paper/configs/eval/eval_countdown_sft_base.yaml` |

## `not_in_paper/pilot_other/` (2)

| File | Paper | Role | Task | LR | Steps | Gens | Original |
|---|---|---|---|---|---|---|---|
| `config_qwen25_7b_caesar_grpo.yaml` | none | Early 7B caesar_cipher GRPO. Superseded by transfer_7b/train_caesar_cipher.yaml | caesar_cipher | 5e-06 | 500 | 8 | `experiments/rlvr_geometry_paper/configs/config_qwen25_7b_caesar_grpo.yaml` |
| `config_qwen_4b_sft_caesar.yaml` | none | SFT on caesar_cipher, 4B |  | 2e-05 | - | - | `experiments/rlvr_geometry_paper/configs/config_qwen_4b_sft_caesar.yaml` |

## `sec3_setup_sft_warmup/` (4)

| File | Paper | Role | Task | LR | Steps | Gens | Original |
|---|---|---|---|---|---|---|---|
| `config_sft_qwen3_4b.yaml` | Sec. 3 (Tab. 1, 4B) | SFT warm-up, 4B. UNVERIFIED: loads Qwen3-4B-Base and writes outputs/sft_qwen3_4b, but downstream runs load /scratch/.../sft_qwen_4.0b_instruct |  | 2e-05 | - | - | `experiments/rlvr_geometry_paper/configs/config_sft_qwen3_4b.yaml` |
| `config_sft_qwen25_3b.yaml` | Sec. 3 (Tab. 1, 3B) | SFT warm-up, Qwen2.5-3B-Instruct -> W_sft (3B) |  | 2e-05 | - | - | `experiments/rlvr_geometry_paper/configs/transfer_3b/config_sft_qwen25_3b.yaml` |
| `config_sft_qwen25_7b.yaml` | Sec. 3 (Tab. 1, 7B) | SFT warm-up, Qwen2.5-7B-Instruct -> W_sft (7B) |  | 2e-05 | - | - | `experiments/rlvr_geometry_paper/configs/transfer_7b/config_sft_qwen25_7b.yaml` |
| `config_sft_qwen3_8b.yaml` | Sec. 3 (Tab. 1, 8B) | SFT warm-up, Qwen3-8B -> W_sft (8B) |  | 2e-05 | - | - | `experiments/rlvr_geometry_paper/configs/transfer_8b/config_sft_qwen3_8b.yaml` |

## `sec4_transfer_matrix/3b_qwen25/` (11)

| File | Paper | Role | Task | LR | Steps | Gens | Original |
|---|---|---|---|---|---|---|---|
| `train_basic_arithmetic.yaml` | Sec. 4 | Single-task GRPO on basic_arithmetic, 3B (row of 3B transfer matrix) | basic_arithmetic | 5e-06 | 300 | 8 | `experiments/rlvr_geometry_paper/configs/transfer_3b/train_basic_arithmetic.yaml` |
| `train_caesar_cipher.yaml` | Sec. 4 | Single-task GRPO on caesar_cipher, 3B (row of 3B transfer matrix) | caesar_cipher | 5e-06 | 300 | 8 | `experiments/rlvr_geometry_paper/configs/transfer_3b/train_caesar_cipher.yaml` |
| `train_countdown.yaml` | Sec. 4 | Single-task GRPO on countdown, 3B (row of 3B transfer matrix) | countdown | 5e-06 | 300 | 8 | `experiments/rlvr_geometry_paper/configs/transfer_3b/train_countdown.yaml` |
| `train_cryptarithm.yaml` | Sec. 4 | Single-task GRPO on cryptarithm, 3B (row of 3B transfer matrix) | cryptarithm | 5e-06 | 300 | 8 | `experiments/rlvr_geometry_paper/configs/transfer_3b/train_cryptarithm.yaml` |
| `train_knights_knaves.yaml` | Sec. 4 | Single-task GRPO on knights_knaves, 3B (row of 3B transfer matrix) | knights_knaves | 5e-06 | 300 | 8 | `experiments/rlvr_geometry_paper/configs/transfer_3b/train_knights_knaves.yaml` |
| `train_leg_counting.yaml` | Sec. 4 | Single-task GRPO on leg_counting, 3B (row of 3B transfer matrix) | leg_counting | 5e-06 | 300 | 8 | `experiments/rlvr_geometry_paper/configs/transfer_3b/train_leg_counting.yaml` |
| `train_maze.yaml` | Sec. 4 | Single-task GRPO on maze, 3B (row of 3B transfer matrix) | maze | 5e-06 | 300 | 8 | `experiments/rlvr_geometry_paper/configs/transfer_3b/train_maze.yaml` |
| `train_mini_sudoku.yaml` | Sec. 4 | Single-task GRPO on mini_sudoku, 3B (row of 3B transfer matrix) | mini_sudoku | 5e-06 | 300 | 8 | `experiments/rlvr_geometry_paper/configs/transfer_3b/train_mini_sudoku.yaml` |
| `train_shortest_path.yaml` | Sec. 4 | Single-task GRPO on shortest_path, 3B (row of 3B transfer matrix) | shortest_path | 5e-06 | 300 | 8 | `experiments/rlvr_geometry_paper/configs/transfer_3b/train_shortest_path.yaml` |
| `train_syllogism.yaml` | Sec. 4 | Single-task GRPO on syllogism, 3B (row of 3B transfer matrix) | syllogism | 5e-06 | 300 | 8 | `experiments/rlvr_geometry_paper/configs/transfer_3b/train_syllogism.yaml` |
| `train_tower_of_hanoi.yaml` | Sec. 4 | Single-task GRPO on tower_of_hanoi, 3B (row of 3B transfer matrix) (NOT one of the paper's ten tasks) | tower_of_hanoi | 5e-06 | 300 | 8 | `experiments/rlvr_geometry_paper/configs/transfer_3b/train_tower_of_hanoi.yaml` |

## `sec4_transfer_matrix/4b_qwen3/` (4)

| File | Paper | Role | Task | LR | Steps | Gens | Original |
|---|---|---|---|---|---|---|---|
| `train_caesar_cipher.yaml` | Sec. 4 | Single-task GRPO on caesar_cipher, 4B (row of 4B transfer matrix) | caesar_cipher | 5e-06 | 300 | 8 | `experiments/rlvr_geometry_paper/configs/transfer_4b/train_caesar_cipher.yaml` |
| `train_cryptarithm.yaml` | Sec. 4 | Single-task GRPO on cryptarithm, 4B (row of 4B transfer matrix) | cryptarithm | 5e-06 | 300 | 8 | `experiments/rlvr_geometry_paper/configs/transfer_4b/train_cryptarithm.yaml` |
| `train_knights_knaves.yaml` | Sec. 4 | Single-task GRPO on knights_knaves, 4B (row of 4B transfer matrix) | knights_knaves | 5e-06 | 300 | 8 | `experiments/rlvr_geometry_paper/configs/transfer_4b/train_knights_knaves.yaml` |
| `train_leg_counting.yaml` | Sec. 4 | Single-task GRPO on leg_counting, 4B (row of 4B transfer matrix) | leg_counting | 5e-06 | 300 | 8 | `experiments/rlvr_geometry_paper/configs/transfer_4b/train_leg_counting.yaml` |

## `sec4_transfer_matrix/4b_qwen3/unverified_provenance/` (2)

| File | Paper | Role | Task | LR | Steps | Gens | Original |
|---|---|---|---|---|---|---|---|
| `config_qwen3_4b_countdown_grpo.yaml` | Sec. 4 (?) | 4B countdown GRPO, 300 steps. Predates transfer_4b/; may be the 4B countdown row. UNVERIFIED | countdown | 5e-06 | 300 | 8 | `experiments/rlvr_geometry_paper/configs/config_qwen3_4b_countdown_grpo.yaml` |
| `config_qwen_4b_instruct_caesar_grpo.yaml` | Sec. 4 (?) | 4B caesar_cipher GRPO. Superseded by transfer_4b/train_caesar_cipher.yaml. UNVERIFIED | caesar_cipher | 5e-06 | 300 | 8 | `experiments/rlvr_geometry_paper/configs/config_qwen_4b_instruct_caesar_grpo.yaml` |

## `sec4_transfer_matrix/appA_7b_qwen25_v1/` (11)

| File | Paper | Role | Task | LR | Steps | Gens | Original |
|---|---|---|---|---|---|---|---|
| `train_basic_arithmetic.yaml` | App. A | Single-task GRPO on basic_arithmetic, 7B, lr 5e-6 / 8 gens (v1) | basic_arithmetic | 5e-06 | 300 | 8 | `experiments/rlvr_geometry_paper/configs/transfer_7b/train_basic_arithmetic.yaml` |
| `train_caesar_cipher.yaml` | App. A | Single-task GRPO on caesar_cipher, 7B, lr 5e-6 / 8 gens (v1) | caesar_cipher | 5e-06 | 300 | 8 | `experiments/rlvr_geometry_paper/configs/transfer_7b/train_caesar_cipher.yaml` |
| `train_countdown.yaml` | App. A | Single-task GRPO on countdown, 7B, lr 5e-6 / 8 gens (v1) | countdown | 5e-06 | 300 | 8 | `experiments/rlvr_geometry_paper/configs/transfer_7b/train_countdown.yaml` |
| `train_cryptarithm.yaml` | App. A | Single-task GRPO on cryptarithm, 7B, lr 5e-6 / 8 gens (v1) | cryptarithm | 5e-06 | 300 | 8 | `experiments/rlvr_geometry_paper/configs/transfer_7b/train_cryptarithm.yaml` |
| `train_knights_knaves.yaml` | App. A | Single-task GRPO on knights_knaves, 7B, lr 5e-6 / 8 gens (v1) | knights_knaves | 5e-06 | 300 | 8 | `experiments/rlvr_geometry_paper/configs/transfer_7b/train_knights_knaves.yaml` |
| `train_leg_counting.yaml` | App. A | Single-task GRPO on leg_counting, 7B, lr 5e-6 / 8 gens (v1) | leg_counting | 5e-06 | 300 | 8 | `experiments/rlvr_geometry_paper/configs/transfer_7b/train_leg_counting.yaml` |
| `train_maze.yaml` | App. A | Single-task GRPO on maze, 7B, lr 5e-6 / 8 gens (v1) | maze | 5e-06 | 300 | 8 | `experiments/rlvr_geometry_paper/configs/transfer_7b/train_maze.yaml` |
| `train_mini_sudoku.yaml` | App. A | Single-task GRPO on mini_sudoku, 7B, lr 5e-6 / 8 gens (v1) | mini_sudoku | 5e-06 | 300 | 8 | `experiments/rlvr_geometry_paper/configs/transfer_7b/train_mini_sudoku.yaml` |
| `train_shortest_path.yaml` | App. A | Single-task GRPO on shortest_path, 7B, lr 5e-6 / 8 gens (v1) | shortest_path | 5e-06 | 300 | 8 | `experiments/rlvr_geometry_paper/configs/transfer_7b/train_shortest_path.yaml` |
| `train_syllogism.yaml` | App. A | Single-task GRPO on syllogism, 7B, lr 5e-6 / 8 gens (v1) | syllogism | 5e-06 | 300 | 8 | `experiments/rlvr_geometry_paper/configs/transfer_7b/train_syllogism.yaml` |
| `train_tower_of_hanoi.yaml` | App. A | Single-task GRPO on tower_of_hanoi, 7B, lr 5e-6 / 8 gens (v1) (NOT one of the paper's ten tasks) | tower_of_hanoi | 5e-06 | 300 | 8 | `experiments/rlvr_geometry_paper/configs/transfer_7b/train_tower_of_hanoi.yaml` |

## `sec4_transfer_matrix/appA_7b_qwen25_v2_lr1e-6_g16/` (10)

| File | Paper | Role | Task | LR | Steps | Gens | Original |
|---|---|---|---|---|---|---|---|
| `train_basic_arithmetic.yaml` | App. A (LR sweep) | Single-task GRPO on basic_arithmetic, 7B, lr 1e-6 / 16 gens (v2) | basic_arithmetic | 1e-06 | 300 | 16 | `experiments/rlvr_geometry_paper/configs/transfer_7b_v2/train_basic_arithmetic.yaml` |
| `train_caesar_cipher.yaml` | App. A (LR sweep) | Single-task GRPO on caesar_cipher, 7B, lr 1e-6 / 16 gens (v2) | caesar_cipher | 1e-06 | 300 | 16 | `experiments/rlvr_geometry_paper/configs/transfer_7b_v2/train_caesar_cipher.yaml` |
| `train_countdown.yaml` | App. A (LR sweep) | Single-task GRPO on countdown, 7B, lr 1e-6 / 16 gens (v2) | countdown | 1e-06 | 300 | 16 | `experiments/rlvr_geometry_paper/configs/transfer_7b_v2/train_countdown.yaml` |
| `train_cryptarithm.yaml` | App. A (LR sweep) | Single-task GRPO on cryptarithm, 7B, lr 1e-6 / 16 gens (v2) | cryptarithm | 1e-06 | 300 | 16 | `experiments/rlvr_geometry_paper/configs/transfer_7b_v2/train_cryptarithm.yaml` |
| `train_knights_knaves.yaml` | App. A (LR sweep) | Single-task GRPO on knights_knaves, 7B, lr 1e-6 / 16 gens (v2) | knights_knaves | 1e-06 | 300 | 16 | `experiments/rlvr_geometry_paper/configs/transfer_7b_v2/train_knights_knaves.yaml` |
| `train_leg_counting.yaml` | App. A (LR sweep) | Single-task GRPO on leg_counting, 7B, lr 1e-6 / 16 gens (v2) | leg_counting | 1e-06 | 300 | 16 | `experiments/rlvr_geometry_paper/configs/transfer_7b_v2/train_leg_counting.yaml` |
| `train_maze.yaml` | App. A (LR sweep) | Single-task GRPO on maze, 7B, lr 1e-6 / 16 gens (v2) | maze | 1e-06 | 300 | 16 | `experiments/rlvr_geometry_paper/configs/transfer_7b_v2/train_maze.yaml` |
| `train_mini_sudoku.yaml` | App. A (LR sweep) | Single-task GRPO on mini_sudoku, 7B, lr 1e-6 / 16 gens (v2) | mini_sudoku | 1e-06 | 300 | 16 | `experiments/rlvr_geometry_paper/configs/transfer_7b_v2/train_mini_sudoku.yaml` |
| `train_shortest_path.yaml` | App. A (LR sweep) | Single-task GRPO on shortest_path, 7B, lr 1e-6 / 16 gens (v2) | shortest_path | 1e-06 | 300 | 16 | `experiments/rlvr_geometry_paper/configs/transfer_7b_v2/train_shortest_path.yaml` |
| `train_syllogism.yaml` | App. A (LR sweep) | Single-task GRPO on syllogism, 7B, lr 1e-6 / 16 gens (v2) | syllogism | 1e-06 | 300 | 16 | `experiments/rlvr_geometry_paper/configs/transfer_7b_v2/train_syllogism.yaml` |

## `sec4_transfer_matrix/appA_8b_qwen3_v1/` (10)

| File | Paper | Role | Task | LR | Steps | Gens | Original |
|---|---|---|---|---|---|---|---|
| `train_basic_arithmetic.yaml` | App. A | Single-task GRPO on basic_arithmetic, 8B, lr 5e-6 / 8 gens (v1) | basic_arithmetic | 5e-06 | 300 | 8 | `experiments/rlvr_geometry_paper/configs/transfer_8b/train_basic_arithmetic.yaml` |
| `train_caesar_cipher.yaml` | App. A | Single-task GRPO on caesar_cipher, 8B, lr 5e-6 / 8 gens (v1) | caesar_cipher | 5e-06 | 300 | 8 | `experiments/rlvr_geometry_paper/configs/transfer_8b/train_caesar_cipher.yaml` |
| `train_countdown.yaml` | App. A | Single-task GRPO on countdown, 8B, lr 5e-6 / 8 gens (v1) | countdown | 5e-06 | 300 | 8 | `experiments/rlvr_geometry_paper/configs/transfer_8b/train_countdown.yaml` |
| `train_cryptarithm.yaml` | App. A | Single-task GRPO on cryptarithm, 8B, lr 5e-6 / 8 gens (v1) | cryptarithm | 5e-06 | 300 | 8 | `experiments/rlvr_geometry_paper/configs/transfer_8b/train_cryptarithm.yaml` |
| `train_knights_knaves.yaml` | App. A | Single-task GRPO on knights_knaves, 8B, lr 5e-6 / 8 gens (v1) | knights_knaves | 5e-06 | 300 | 8 | `experiments/rlvr_geometry_paper/configs/transfer_8b/train_knights_knaves.yaml` |
| `train_leg_counting.yaml` | App. A | Single-task GRPO on leg_counting, 8B, lr 5e-6 / 8 gens (v1) | leg_counting | 5e-06 | 300 | 8 | `experiments/rlvr_geometry_paper/configs/transfer_8b/train_leg_counting.yaml` |
| `train_maze.yaml` | App. A | Single-task GRPO on maze, 8B, lr 5e-6 / 8 gens (v1) | maze | 5e-06 | 300 | 8 | `experiments/rlvr_geometry_paper/configs/transfer_8b/train_maze.yaml` |
| `train_mini_sudoku.yaml` | App. A | Single-task GRPO on mini_sudoku, 8B, lr 5e-6 / 8 gens (v1) | mini_sudoku | 5e-06 | 300 | 8 | `experiments/rlvr_geometry_paper/configs/transfer_8b/train_mini_sudoku.yaml` |
| `train_shortest_path.yaml` | App. A | Single-task GRPO on shortest_path, 8B, lr 5e-6 / 8 gens (v1) | shortest_path | 5e-06 | 300 | 8 | `experiments/rlvr_geometry_paper/configs/transfer_8b/train_shortest_path.yaml` |
| `train_syllogism.yaml` | App. A | Single-task GRPO on syllogism, 8B, lr 5e-6 / 8 gens (v1) | syllogism | 5e-06 | 300 | 8 | `experiments/rlvr_geometry_paper/configs/transfer_8b/train_syllogism.yaml` |

## `sec4_transfer_matrix/appA_8b_qwen3_v2_lr1e-6_g16/` (10)

| File | Paper | Role | Task | LR | Steps | Gens | Original |
|---|---|---|---|---|---|---|---|
| `train_basic_arithmetic.yaml` | App. A (LR sweep) | Single-task GRPO on basic_arithmetic, 8B, lr 1e-6 / 16 gens (v2) | basic_arithmetic | 1e-06 | 300 | 16 | `experiments/rlvr_geometry_paper/configs/transfer_8b_v2/train_basic_arithmetic.yaml` |
| `train_caesar_cipher.yaml` | App. A (LR sweep) | Single-task GRPO on caesar_cipher, 8B, lr 1e-6 / 16 gens (v2) | caesar_cipher | 1e-06 | 300 | 16 | `experiments/rlvr_geometry_paper/configs/transfer_8b_v2/train_caesar_cipher.yaml` |
| `train_countdown.yaml` | App. A (LR sweep) | Single-task GRPO on countdown, 8B, lr 1e-6 / 16 gens (v2) | countdown | 1e-06 | 300 | 16 | `experiments/rlvr_geometry_paper/configs/transfer_8b_v2/train_countdown.yaml` |
| `train_cryptarithm.yaml` | App. A (LR sweep) | Single-task GRPO on cryptarithm, 8B, lr 1e-6 / 16 gens (v2) | cryptarithm | 1e-06 | 300 | 16 | `experiments/rlvr_geometry_paper/configs/transfer_8b_v2/train_cryptarithm.yaml` |
| `train_knights_knaves.yaml` | App. A (LR sweep) | Single-task GRPO on knights_knaves, 8B, lr 1e-6 / 16 gens (v2) | knights_knaves | 1e-06 | 300 | 16 | `experiments/rlvr_geometry_paper/configs/transfer_8b_v2/train_knights_knaves.yaml` |
| `train_leg_counting.yaml` | App. A (LR sweep) | Single-task GRPO on leg_counting, 8B, lr 1e-6 / 16 gens (v2) | leg_counting | 1e-06 | 300 | 16 | `experiments/rlvr_geometry_paper/configs/transfer_8b_v2/train_leg_counting.yaml` |
| `train_maze.yaml` | App. A (LR sweep) | Single-task GRPO on maze, 8B, lr 1e-6 / 16 gens (v2) | maze | 1e-06 | 300 | 16 | `experiments/rlvr_geometry_paper/configs/transfer_8b_v2/train_maze.yaml` |
| `train_mini_sudoku.yaml` | App. A (LR sweep) | Single-task GRPO on mini_sudoku, 8B, lr 1e-6 / 16 gens (v2) | mini_sudoku | 1e-06 | 300 | 16 | `experiments/rlvr_geometry_paper/configs/transfer_8b_v2/train_mini_sudoku.yaml` |
| `train_shortest_path.yaml` | App. A (LR sweep) | Single-task GRPO on shortest_path, 8B, lr 1e-6 / 16 gens (v2) | shortest_path | 1e-06 | 300 | 16 | `experiments/rlvr_geometry_paper/configs/transfer_8b_v2/train_shortest_path.yaml` |
| `train_syllogism.yaml` | App. A (LR sweep) | Single-task GRPO on syllogism, 8B, lr 1e-6 / 16 gens (v2) | syllogism | 1e-06 | 300 | 16 | `experiments/rlvr_geometry_paper/configs/transfer_8b_v2/train_syllogism.yaml` |

## `sec5_atomic_curriculum/appD_shortest_path_deep_dive/A_progressive_full_task/` (3)

| File | Paper | Role | Task | LR | Steps | Gens | Original |
|---|---|---|---|---|---|---|---|
| `idea_a_stage1_3x3.yaml` | App. D | [A] progressive full task, stage 1: 3x3 grids | shortest_path | 5e-06 | 100 | 8 | `experiments/rlvr_geometry_paper/configs/phase2_sp_ideas/idea_a_stage1_3x3.yaml` |
| `idea_a_stage2_4x4.yaml` | App. D | [A] progressive full task, stage 2: 4x4 grids | shortest_path | 5e-06 | 100 | 8 | `experiments/rlvr_geometry_paper/configs/phase2_sp_ideas/idea_a_stage2_4x4.yaml` |
| `idea_a_stage3_full.yaml` | App. D | [A] progressive full task, stage 3: full grids | shortest_path | 5e-06 | 100 | 8 | `experiments/rlvr_geometry_paper/configs/phase2_sp_ideas/idea_a_stage3_full.yaml` |

## `sec5_atomic_curriculum/appD_shortest_path_deep_dive/B2_300plus300_steps/` (2)

| File | Paper | Role | Task | LR | Steps | Gens | Original |
|---|---|---|---|---|---|---|---|
| `idea_b2_stage1_long_atoms_300.yaml` | App. D | [B2] stage 1: L2-long atoms, 300 steps | shortest_path | 5e-06 | 300 | 8 | `experiments/rlvr_geometry_paper/configs/phase2_sp_ideas/idea_b2_stage1_long_atoms_300.yaml` |
| `idea_b2_stage2_full_300.yaml` | App. D | [B2] stage 2: full task, 300 steps | shortest_path | 5e-06 | 300 | 8 | `experiments/rlvr_geometry_paper/configs/phase2_sp_ideas/idea_b2_stage2_full_300.yaml` |

## `sec5_atomic_curriculum/appD_shortest_path_deep_dive/B_L2long_atoms_then_full/` (2)

| File | Paper | Role | Task | LR | Steps | Gens | Original |
|---|---|---|---|---|---|---|---|
| `idea_b_stage1_long_atoms.yaml` | App. D | [B] stage 1: L2-long (path-tracing) atoms, 100 steps | shortest_path | 5e-06 | 100 | 8 | `experiments/rlvr_geometry_paper/configs/phase2_sp_ideas/idea_b_stage1_long_atoms.yaml` |
| `idea_b_stage2_full.yaml` | App. D | [B] stage 2: full task, 200 steps (best atomic-curriculum result, 51.5%) | shortest_path | 5e-06 | 200 | 8 | `experiments/rlvr_geometry_paper/configs/phase2_sp_ideas/idea_b_stage2_full.yaml` |

## `sec5_atomic_curriculum/appD_shortest_path_deep_dive/C_partial_credit_reward/` (1)

| File | Paper | Role | Task | LR | Steps | Gens | Original |
|---|---|---|---|---|---|---|---|
| `idea_c_partial_credit.yaml` | App. D | [C] partial-credit reward on full task, 300 steps | partial_credit_sp | 5e-06 | 300 | 8 | `experiments/rlvr_geometry_paper/configs/phase2_sp_ideas/idea_c_partial_credit.yaml` |

## `sec5_atomic_curriculum/appD_shortest_path_deep_dive/D_progressive_plus_partial_credit/` (3)

| File | Paper | Role | Task | LR | Steps | Gens | Original |
|---|---|---|---|---|---|---|---|
| `idea_d_stage1_3x3_pc.yaml` | App. D | [D] stage 1: 3x3, partial credit | partial_credit_sp | 5e-06 | 100 | 8 | `experiments/rlvr_geometry_paper/configs/phase2_sp_ideas/idea_d_stage1_3x3_pc.yaml` |
| `idea_d_stage2_4x4_pc.yaml` | App. D | [D] stage 2: 4x4, partial credit | partial_credit_sp | 5e-06 | 100 | 8 | `experiments/rlvr_geometry_paper/configs/phase2_sp_ideas/idea_d_stage2_4x4_pc.yaml` |
| `idea_d_stage3_full_pc.yaml` | App. D | [D] stage 3: full, partial credit | partial_credit_sp | 5e-06 | 100 | 8 | `experiments/rlvr_geometry_paper/configs/phase2_sp_ideas/idea_d_stage3_full_pc.yaml` |

## `sec5_atomic_curriculum/appD_shortest_path_deep_dive/E_L2long_atoms_then_progressive/` (3)

| File | Paper | Role | Task | LR | Steps | Gens | Original |
|---|---|---|---|---|---|---|---|
| `idea_e_stage2_3x3.yaml` | App. D | [E] stage 2: 3x3 (stage 1 = idea B stage 1 atoms) | shortest_path | 5e-06 | 70 | 8 | `experiments/rlvr_geometry_paper/configs/phase2_sp_ideas/idea_e_stage2_3x3.yaml` |
| `idea_e_stage3_4x4.yaml` | App. D | [E] stage 3: 4x4 | shortest_path | 5e-06 | 70 | 8 | `experiments/rlvr_geometry_paper/configs/phase2_sp_ideas/idea_e_stage3_4x4.yaml` |
| `idea_e_stage4_full.yaml` | App. D | [E] stage 4: full | shortest_path | 5e-06 | 60 | 8 | `experiments/rlvr_geometry_paper/configs/phase2_sp_ideas/idea_e_stage4_full.yaml` |

## `sec5_atomic_curriculum/appD_shortest_path_deep_dive/F_feasibility_atoms/` (2)

| File | Paper | Role | Task | LR | Steps | Gens | Original |
|---|---|---|---|---|---|---|---|
| `idea_f_stage1_feasibility_atoms.yaml` | App. D | [F] stage 1: feasibility atoms | shortest_path | 5e-06 | 100 | 8 | `experiments/rlvr_geometry_paper/configs/phase2_sp_ideas/idea_f_stage1_feasibility_atoms.yaml` |
| `idea_f_stage2_full.yaml` | App. D | [F] stage 2: full task, 200 steps | shortest_path | 5e-06 | 200 | 8 | `experiments/rlvr_geometry_paper/configs/phase2_sp_ideas/idea_f_stage2_full.yaml` |

## `sec5_atomic_curriculum/appD_shortest_path_deep_dive/G_G2_grpo_sft_interleave/` (11)

| File | Paper | Role | Task | LR | Steps | Gens | Original |
|---|---|---|---|---|---|---|---|
| `grpo_cycle1.yaml` | App. D | [G/G2] cycle 1: GRPO 50 steps. File settings are the G2 (light) variant; shared by experiments/rlvr_geometry_paper/scripts/run/run_idea_g.sh and experiments/rlvr_geometry_paper/scripts/run/run_idea_g2.sh | shortest_path | 5e-06 | 50 | 8 | `experiments/rlvr_geometry_paper/configs/phase2_sp_ideas/idea_g/grpo_cycle1.yaml` |
| `grpo_cycle2.yaml` | App. D | [G/G2] cycle 2: GRPO 50 steps. File settings are the G2 (light) variant; shared by experiments/rlvr_geometry_paper/scripts/run/run_idea_g.sh and experiments/rlvr_geometry_paper/scripts/run/run_idea_g2.sh | shortest_path | 5e-06 | 50 | 8 | `experiments/rlvr_geometry_paper/configs/phase2_sp_ideas/idea_g/grpo_cycle2.yaml` |
| `grpo_cycle3.yaml` | App. D | [G/G2] cycle 3: GRPO 50 steps. File settings are the G2 (light) variant; shared by experiments/rlvr_geometry_paper/scripts/run/run_idea_g.sh and experiments/rlvr_geometry_paper/scripts/run/run_idea_g2.sh | shortest_path | 5e-06 | 50 | 8 | `experiments/rlvr_geometry_paper/configs/phase2_sp_ideas/idea_g/grpo_cycle3.yaml` |
| `grpo_cycle4.yaml` | App. D | [G2 ext.] cycle 4: GRPO 50 steps (experiments/rlvr_geometry_paper/scripts/run/run_idea_g2_continue.sh). Not a numbered table condition | shortest_path | 5e-06 | 50 | 8 | `experiments/rlvr_geometry_paper/configs/phase2_sp_ideas/idea_g/grpo_cycle4.yaml` |
| `grpo_cycle5.yaml` | App. D | [G2 ext.] cycle 5: GRPO 50 steps (experiments/rlvr_geometry_paper/scripts/run/run_idea_g2_continue.sh). Not a numbered table condition | shortest_path | 5e-06 | 50 | 8 | `experiments/rlvr_geometry_paper/configs/phase2_sp_ideas/idea_g/grpo_cycle5.yaml` |
| `sft_cycle1.yaml` | App. D | [G/G2] cycle 1: SFT, lr 1e-6 (light, G2 setting) |  | 1e-06 | - | - | `experiments/rlvr_geometry_paper/configs/phase2_sp_ideas/idea_g/sft_cycle1.yaml` |
| `sft_cycle2.yaml` | App. D | [G/G2] cycle 2: SFT, lr 1e-6 (light, G2 setting) |  | 1e-06 | - | - | `experiments/rlvr_geometry_paper/configs/phase2_sp_ideas/idea_g/sft_cycle2.yaml` |
| `sft_cycle3.yaml` | App. D | [G/G2] cycle 3: SFT, lr 1e-6 (light, G2 setting) |  | 1e-06 | - | - | `experiments/rlvr_geometry_paper/configs/phase2_sp_ideas/idea_g/sft_cycle3.yaml` |
| `sft_cycle4.yaml` | App. D | [G2 ext.] cycle 4: SFT, lr 1e-6 (experiments/rlvr_geometry_paper/scripts/run/run_idea_g2_continue.sh). Not a numbered table condition |  | 1e-06 | - | - | `experiments/rlvr_geometry_paper/configs/phase2_sp_ideas/idea_g/sft_cycle4.yaml` |
| `sft_cycle5.yaml` | App. D | [G2 ext.] cycle 5: SFT, lr 1e-6 (experiments/rlvr_geometry_paper/scripts/run/run_idea_g2_continue.sh). Not a numbered table condition |  | 1e-06 | - | - | `experiments/rlvr_geometry_paper/configs/phase2_sp_ideas/idea_g/sft_cycle5.yaml` |
| `idea_g3_continue.yaml` | App. D | [G3] G2 + 150 more GRPO steps | shortest_path | 5e-06 | 150 | 8 | `experiments/rlvr_geometry_paper/configs/phase2_sp_ideas/idea_g3_continue.yaml` |

## `sec5_atomic_curriculum/appD_shortest_path_deep_dive/L_feasible_only/` (1)

| File | Paper | Role | Task | LR | Steps | Gens | Original |
|---|---|---|---|---|---|---|---|
| `idea_l_stage2_feasible.yaml` | App. D | [L] feasible-only training, init from idea B stage 1 atoms | feasible_only_sp | 5e-06 | 200 | 8 | `experiments/rlvr_geometry_paper/configs/phase2_sp_ideas/idea_l_stage2_feasible.yaml` |

## `sec5_atomic_curriculum/appD_shortest_path_deep_dive/RFT_Base_best_55.5pct/` (2)

| File | Paper | Role | Task | LR | Steps | Gens | Original |
|---|---|---|---|---|---|---|---|
| `idea_rft_base_grpo.yaml` | App. D | [RFT-Base] GRPO 300 steps after RFT-Base SFT (best shortest_path result, 55.5%) | shortest_path | 5e-06 | 300 | 8 | `experiments/rlvr_geometry_paper/configs/phase2_sp_ideas/idea_rft_base_grpo.yaml` |
| `idea_rft_base_sft.yaml` | App. D | [RFT-Base] SFT the SFT-base model on B's verified traces |  | 1e-06 | - | - | `experiments/rlvr_geometry_paper/configs/phase2_sp_ideas/idea_rft_base_sft.yaml` |

## `sec5_atomic_curriculum/appD_shortest_path_deep_dive/RFT_on_B_checkpoint/` (2)

| File | Paper | Role | Task | LR | Steps | Gens | Original |
|---|---|---|---|---|---|---|---|
| `idea_rft_grpo.yaml` | App. D | [RFT] GRPO 100 steps after RFT SFT | shortest_path | 5e-06 | 100 | 8 | `experiments/rlvr_geometry_paper/configs/phase2_sp_ideas/idea_rft_grpo.yaml` |
| `idea_rft_sft.yaml` | App. D | [RFT] light SFT on B's own verified traces (init: idea B stage 2) |  | 1e-06 | - | - | `experiments/rlvr_geometry_paper/configs/phase2_sp_ideas/idea_rft_sft.yaml` |

## `sec5_atomic_curriculum/appD_shortest_path_deep_dive/atomic_warm/` (5)

| File | Paper | Role | Task | LR | Steps | Gens | Original |
|---|---|---|---|---|---|---|---|
| `continue_best_300.yaml` | App. D | [Atomic-Warm] continues Atomic-Warm-300 for 300 more steps. Not a numbered table condition | shortest_path | 5e-06 | 300 | 8 | `experiments/rlvr_geometry_paper/configs/phase2_shortest_path/continue_best_300.yaml` |
| `stage1_l1_atoms.yaml` | App. D | [Atomic-Warm] stage 1: L1 atoms, 100 steps | shortest_path | 5e-06 | 100 | 8 | `experiments/rlvr_geometry_paper/configs/phase2_shortest_path/stage1_l1_atoms.yaml` |
| `stage2_l2_atoms.yaml` | App. D | [Atomic-Warm] stage 2: L2 atoms, 100 steps | shortest_path | 5e-06 | 100 | 8 | `experiments/rlvr_geometry_paper/configs/phase2_shortest_path/stage2_l2_atoms.yaml` |
| `stage3_full_300steps.yaml` | App. D | [Atomic-Warm-300] stage 3: full task, 300 steps (init: stage 2) | shortest_path | 5e-06 | 300 | 8 | `experiments/rlvr_geometry_paper/configs/phase2_shortest_path/stage3_full_300steps.yaml` |
| `stage3_full_task.yaml` | App. D | [Atomic-Warm] stage 3: full task, 100 steps | shortest_path | 5e-06 | 100 | 8 | `experiments/rlvr_geometry_paper/configs/phase2_shortest_path/stage3_full_task.yaml` |

## `sec5_atomic_curriculum/appD_shortest_path_deep_dive/progressive_atoms/` (3)

| File | Paper | Role | Task | LR | Steps | Gens | Original |
|---|---|---|---|---|---|---|---|
| `prog_full_300.yaml` | App. D | progressive atoms, full task 300 steps (init: prog_l2_4x4). Not a numbered table condition | shortest_path | 5e-06 | 300 | 8 | `experiments/rlvr_geometry_paper/configs/phase2_shortest_path/prog_full_300.yaml` |
| `prog_l1_3x3.yaml` | App. D | progressive atoms, L1 3x3, 100 steps (experiments/rlvr_geometry_paper/scripts/run/run_sp_push_higher.sh). Not a numbered table condition | shortest_path | 5e-06 | 100 | 8 | `experiments/rlvr_geometry_paper/configs/phase2_shortest_path/prog_l1_3x3.yaml` |
| `prog_l2_4x4.yaml` | App. D | progressive atoms, L2 4x4, 100 steps (experiments/rlvr_geometry_paper/scripts/run/run_sp_push_higher.sh). Not a numbered table condition | shortest_path | 5e-06 | 100 | 8 | `experiments/rlvr_geometry_paper/configs/phase2_shortest_path/prog_l2_4x4.yaml` |

## `sec5_atomic_curriculum/appD_shortest_path_deep_dive/sft_for_l1_ablation/` (6)

| File | Paper | Role | Task | LR | Steps | Gens | Original |
|---|---|---|---|---|---|---|---|
| `sft_direct_full_300.yaml` | App. D | full-task GRPO 300 steps directly after L1 SFT | shortest_path | 5e-06 | 300 | 8 | `experiments/rlvr_geometry_paper/configs/phase2_shortest_path/sft_direct_full_300.yaml` |
| `sft_l1_atoms.yaml` | App. D | SFT on L1 atoms, lr 2e-5 (experiments/rlvr_geometry_paper/scripts/run/run_sp_sft_atoms.sh) |  | 2e-05 | - | - | `experiments/rlvr_geometry_paper/configs/phase2_shortest_path/sft_l1_atoms.yaml` |
| `sft_l1_low_lr.yaml` | App. D | SFT on L1 atoms, low lr 5e-6 |  | 5e-06 | - | - | `experiments/rlvr_geometry_paper/configs/phase2_shortest_path/sft_l1_low_lr.yaml` |
| `sft_lowlr_full_300.yaml` | App. D | full-task GRPO 300 steps after low-lr L1 SFT | shortest_path | 5e-06 | 300 | 8 | `experiments/rlvr_geometry_paper/configs/phase2_shortest_path/sft_lowlr_full_300.yaml` |
| `sft_then_full_300.yaml` | App. D | full-task GRPO 300 steps after L1-SFT + L2 | shortest_path | 5e-06 | 300 | 8 | `experiments/rlvr_geometry_paper/configs/phase2_shortest_path/sft_then_full_300.yaml` |
| `sft_then_l2_grpo.yaml` | App. D | GRPO on L2 atoms after L1 SFT | shortest_path | 5e-06 | 100 | 8 | `experiments/rlvr_geometry_paper/configs/phase2_shortest_path/sft_then_l2_grpo.yaml` |

## `sec5_atomic_curriculum/per_task/basic_arithmetic/` (5)

| File | Paper | Role | Task | LR | Steps | Gens | Original |
|---|---|---|---|---|---|---|---|
| `direct_rlvr_4b.yaml` | Sec. 5 / App. C | Direct-RLVR baseline on full task, 4B, 300 steps | basic_arithmetic | 5e-06 | 300 | 8 | `experiments/rlvr_geometry_paper/configs/phase2_arithmetic/direct_rlvr_4b.yaml` |
| `stage1_l1_atoms.yaml` | Sec. 5 / App. C | Atomic-Warm stage 1: L1 atoms, 100 steps (init: SFT base) | basic_arithmetic | 5e-06 | 100 | 8 | `experiments/rlvr_geometry_paper/configs/phase2_arithmetic/stage1_l1_atoms.yaml` |
| `stage2_l2_atoms.yaml` | Sec. 5 / App. C | Atomic-Warm stage 2: L2 atoms, 100 steps (init: stage 1) | basic_arithmetic | 5e-06 | 100 | 8 | `experiments/rlvr_geometry_paper/configs/phase2_arithmetic/stage2_l2_atoms.yaml` |
| `stage3_full_300steps.yaml` | Sec. 5 / App. C | Atomic-Warm stage 3: full task, 300-step variant (init: stage 2) | basic_arithmetic | 5e-06 | 300 | 8 | `experiments/rlvr_geometry_paper/configs/phase2_arithmetic/stage3_full_300steps.yaml` |
| `stage3_full_task.yaml` | Sec. 5 / App. C | Atomic-Warm stage 3: full task, 100 steps (init: stage 2) | basic_arithmetic | 5e-06 | 100 | 8 | `experiments/rlvr_geometry_paper/configs/phase2_arithmetic/stage3_full_task.yaml` |

## `sec5_atomic_curriculum/per_task/countdown/` (4)

| File | Paper | Role | Task | LR | Steps | Gens | Original |
|---|---|---|---|---|---|---|---|
| `direct_rlvr_4b.yaml` | Sec. 5 / App. C | Direct-RLVR baseline on full task, 4B, 300 steps | countdown | 5e-06 | 300 | 8 | `experiments/rlvr_geometry_paper/configs/phase2_countdown/direct_rlvr_4b.yaml` |
| `stage1_l1_atoms.yaml` | Sec. 5 / App. C | Atomic-Warm stage 1: L1 atoms, 100 steps (init: SFT base) | countdown | 5e-06 | 100 | 8 | `experiments/rlvr_geometry_paper/configs/phase2_countdown/stage1_l1_atoms.yaml` |
| `stage2_l2_atoms.yaml` | Sec. 5 / App. C | Atomic-Warm stage 2: L2 atoms, 100 steps (init: stage 1) | countdown | 5e-06 | 100 | 8 | `experiments/rlvr_geometry_paper/configs/phase2_countdown/stage2_l2_atoms.yaml` |
| `stage3_full_task.yaml` | Sec. 5 / App. C | Atomic-Warm stage 3: full task, 100 steps (init: stage 2) | countdown | 5e-06 | 100 | 8 | `experiments/rlvr_geometry_paper/configs/phase2_countdown/stage3_full_task.yaml` |

## `sec5_atomic_curriculum/per_task/knights_knaves/` (4)

| File | Paper | Role | Task | LR | Steps | Gens | Original |
|---|---|---|---|---|---|---|---|
| `direct_rlvr.yaml` | Sec. 5 / App. C | Direct-RLVR baseline on full task, 4B, 300 steps | knights_knaves | 5e-06 | 300 | 8 | `experiments/rlvr_geometry_paper/configs/phase2_extended/knights_knaves/direct_rlvr.yaml` |
| `stage1_l1.yaml` | Sec. 5 / App. C | Atomic-Warm stage 1: L1 atoms, 100 steps (init: SFT base) | knights_knaves | 5e-06 | 100 | 8 | `experiments/rlvr_geometry_paper/configs/phase2_extended/knights_knaves/stage1_l1.yaml` |
| `stage2_l2.yaml` | Sec. 5 / App. C | Atomic-Warm stage 2: L2 atoms, 100 steps (init: stage 1) | knights_knaves | 5e-06 | 100 | 8 | `experiments/rlvr_geometry_paper/configs/phase2_extended/knights_knaves/stage2_l2.yaml` |
| `stage3_full.yaml` | Sec. 5 / App. C | Atomic-Warm stage 3: full task, 100 steps (init: stage 2) | knights_knaves | 5e-06 | 100 | 8 | `experiments/rlvr_geometry_paper/configs/phase2_extended/knights_knaves/stage3_full.yaml` |

## `sec5_atomic_curriculum/per_task/maze/` (4)

| File | Paper | Role | Task | LR | Steps | Gens | Original |
|---|---|---|---|---|---|---|---|
| `direct_rlvr.yaml` | Sec. 5 / App. C | Direct-RLVR baseline on full task, 4B, 300 steps | maze | 5e-06 | 300 | 8 | `experiments/rlvr_geometry_paper/configs/phase2_extended/maze/direct_rlvr.yaml` |
| `stage1_l1.yaml` | Sec. 5 / App. C | Atomic-Warm stage 1: L1 atoms, 100 steps (init: SFT base) | maze | 5e-06 | 100 | 8 | `experiments/rlvr_geometry_paper/configs/phase2_extended/maze/stage1_l1.yaml` |
| `stage2_l2.yaml` | Sec. 5 / App. C | Atomic-Warm stage 2: L2 atoms, 100 steps (init: stage 1) | maze | 5e-06 | 100 | 8 | `experiments/rlvr_geometry_paper/configs/phase2_extended/maze/stage2_l2.yaml` |
| `stage3_full.yaml` | Sec. 5 / App. C | Atomic-Warm stage 3: full task, 100 steps (init: stage 2) | maze | 5e-06 | 100 | 8 | `experiments/rlvr_geometry_paper/configs/phase2_extended/maze/stage3_full.yaml` |

## `sec5_atomic_curriculum/per_task/mini_sudoku/` (4)

| File | Paper | Role | Task | LR | Steps | Gens | Original |
|---|---|---|---|---|---|---|---|
| `direct_rlvr.yaml` | Sec. 5 / App. C | Direct-RLVR baseline on full task, 4B, 300 steps | mini_sudoku | 5e-06 | 300 | 8 | `experiments/rlvr_geometry_paper/configs/phase2_extended/mini_sudoku/direct_rlvr.yaml` |
| `stage1_l1.yaml` | Sec. 5 / App. C | Atomic-Warm stage 1: L1 atoms, 100 steps (init: SFT base) | mini_sudoku | 5e-06 | 100 | 8 | `experiments/rlvr_geometry_paper/configs/phase2_extended/mini_sudoku/stage1_l1.yaml` |
| `stage2_l2.yaml` | Sec. 5 / App. C | Atomic-Warm stage 2: L2 atoms, 100 steps (init: stage 1) | mini_sudoku | 5e-06 | 100 | 8 | `experiments/rlvr_geometry_paper/configs/phase2_extended/mini_sudoku/stage2_l2.yaml` |
| `stage3_full.yaml` | Sec. 5 / App. C | Atomic-Warm stage 3: full task, 100 steps (init: stage 2) | mini_sudoku | 5e-06 | 100 | 8 | `experiments/rlvr_geometry_paper/configs/phase2_extended/mini_sudoku/stage3_full.yaml` |

## `sec5_atomic_curriculum/per_task/syllogism/` (4)

| File | Paper | Role | Task | LR | Steps | Gens | Original |
|---|---|---|---|---|---|---|---|
| `direct_rlvr.yaml` | Sec. 5 / App. C | Direct-RLVR baseline on full task, 4B, 300 steps | syllogism | 5e-06 | 300 | 8 | `experiments/rlvr_geometry_paper/configs/phase2_extended/syllogism/direct_rlvr.yaml` |
| `stage1_l1.yaml` | Sec. 5 / App. C | Atomic-Warm stage 1: L1 atoms, 100 steps (init: SFT base) | syllogism | 5e-06 | 100 | 8 | `experiments/rlvr_geometry_paper/configs/phase2_extended/syllogism/stage1_l1.yaml` |
| `stage2_l2.yaml` | Sec. 5 / App. C | Atomic-Warm stage 2: L2 atoms, 100 steps (init: stage 1) | syllogism | 5e-06 | 100 | 8 | `experiments/rlvr_geometry_paper/configs/phase2_extended/syllogism/stage2_l2.yaml` |
| `stage3_full.yaml` | Sec. 5 / App. C | Atomic-Warm stage 3: full task, 100 steps (init: stage 2) | syllogism | 5e-06 | 100 | 8 | `experiments/rlvr_geometry_paper/configs/phase2_extended/syllogism/stage3_full.yaml` |
