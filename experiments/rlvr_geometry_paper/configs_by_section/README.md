# Configs by paper section

Paper: **The Geometry and Transfer of RLVR in Small Reasoning Models** (section numbers follow `experiments/rlvr_geometry_paper/paper/main.tex`).

This tree is a **read-only, shareable view** of `../configs/`. Every file here is a copy of an original
with a comment header added (paper section, role, init model, task, LR/steps/generations, original path).
The YAML body is unmodified (checked: each copy parses identically to its original). The training and
eval scripts (`run_*.sh`, `train_grpo.py`, ...) still read from `../configs/`, so **edit and launch from
the originals**, not from here. Full file-by-file listing: [`INDEX.md`](INDEX.md) and [`MANIFEST.csv`](MANIFEST.csv).

## Layout

| Folder | Paper location | Files |
|---|---|---|
| `sec3_setup_sft_warmup/` | Sec. 3 Setup: SFT warm-up that gives W_sft for each scale | 4 |
| `sec4_transfer_matrix/3b_qwen25/` | Sec. 4: single-task GRPO runs behind the 3B transfer matrix | 11 |
| `sec4_transfer_matrix/4b_qwen3/` | Sec. 4: 4B runs (only 4 of 10 tasks live here, see gaps) | 4 (+2 unverified) |
| `sec4_transfer_matrix/appA_7b_*`, `appA_8b_*` | App. A: 7B/8B matrices; `v1` = lr 5e-6 / 8 gens, `v2` = lr 1e-6 / 16 gens (the LR sweep) | 11 + 10 + 10 + 10 |
| `sec5_atomic_curriculum/per_task/<task>/` | Sec. 5 + App. C: Direct-RLVR vs Atomic-Warm (L1 -> L2 -> full) for 6 tasks | 5 + 4 + 4 x 4 |
| `sec5_atomic_curriculum/appD_shortest_path_deep_dive/` | App. D: the 16-condition shortest_path study; folders named by the condition letter in the App. D table | 46 |
| `sec6_weight_geometry/` | Sec. 6: **no YAML configs** (analysis scripts only), README points to them | 0 |
| `sec7_task_vector_merging/` | Sec. 7 + App. F: **no YAML configs** (merge scripts only), README points to them | 0 |
| `not_in_paper/` | Preliminary sweeps, 1.5B pilots, early smoke tests, pilot eval configs | 29 |

Total: 162 copied configs. `../cer_project/` (38 configs, separate CER/maze side study) is deliberately excluded.

## Known gaps and caveats (read before sharing)

1. **4B matrix is incomplete.** `transfer_4b/` only holds caesar_cipher, cryptarithm, knights_knaves and
   leg_counting. The 4B configs for basic_arithmetic, countdown, maze, mini_sudoku, shortest_path and
   syllogism are not in `configs/` under a transfer name. The Sec. 5 `direct_rlvr*.yaml` files are 4B,
   300-step single-task GRPO and may be the same runs, but I did not verify that. Two earlier 4B configs
   sit in `4b_qwen3/unverified_provenance/`.
2. **4B SFT warm-up provenance.** `config_sft_qwen3_4b.yaml` loads `Qwen3-4B-Base` and writes
   `outputs/sft_qwen3_4b`, but every downstream run loads `/scratch/sunguha/sft_qwen_4.0b_instruct`.
   No config in the repo is a clear producer of that checkpoint. The Sec. 3 table in the paper says `Qwen/Qwen3-4B`.
3. **Shortest_path condition G vs G2** share the same files in `G_G2_grpo_sft_interleave/`. The checked-in
   SFT files carry the G2 (light, lr 1e-6) setting; `experiments/rlvr_geometry_paper/scripts/run/run_idea_g2.sh` also sets 500 SFT samples at data-generation time.
4. **Not every SP config is a table condition.** `progressive_atoms/`, `sft_for_l1_ablation/` and G cycles 4-5
   are extra runs (labelled as such in each header).
5. `tower_of_hanoi` configs exist for 3B and 7B v1 but it is not one of the paper's ten tasks.
6. Init-model paths (`/scratch/sunguha/...`, `results/phase2_*/...`) are from the original cluster and will
   not exist on a colleague's machine; they must be re-pointed.
7. `not_in_paper/hp_sweep_prelim/` is not cited in the paper or any report. It may have motivated the
   DAPO / beta = 0 recipe, but that is a guess.
8. Section numbers in `CLAUDE.md` (Phase 1 = Sec. 5, Phase 2 = Sec. 4) and in `experiments/rlvr_geometry_paper/paper/drafts/paper_section_*.md`
   are out of date relative to `main.tex`, where Transfer = Sec. 4, Atomic = Sec. 5, Geometry = Sec. 6, Merging = Sec. 7.
   This tree follows `main.tex`.
