# Sec. 6: The Weight-Space Geometry of GRPO

There are **no YAML configs** for this section. It is analysis-only (CPU-bound SVDs) on the single-task
GRPO checkpoints trained with the Sec. 4 configs (`../sec4_transfer_matrix/`) against their SFT bases
(`../sec3_setup_sft_warmup/`). Metric definitions are in App. H of the paper.

Scripts (in `rlvr_experiments/`, driven by CLI arguments):

- `analyze_rlvr_weight_changes.py`, `analyze_weight_distance.py`, `run_weight_analysis.sh`: task-vector magnitude, relative change
- `analyze_eigenvalue_shifts.py`: spectrum preservation
- `analyze_subspace_and_rotation.py`, `analyze_rotation_direction.py`, `analyze_rotation_direction_4b.py`: principal angles, novel subspace
- `analyze_per_layer_task_diff.py`: depth signature
- `analyze_sign_agreement.py`, `analyze_sign_agreement_multi.py`: sign agreement (a falsified predictor, reported as a null)

Source report: `report_phase3_task_distances.md`.
