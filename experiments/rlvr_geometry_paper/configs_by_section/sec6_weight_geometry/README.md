# Sec. 6: The Weight-Space Geometry of GRPO

There are **no YAML configs** for this section. It is analysis-only (CPU-bound SVDs) on the single-task
GRPO checkpoints trained with the Sec. 4 configs (`../sec4_transfer_matrix/`) against their SFT bases
(`../sec3_setup_sft_warmup/`). Metric definitions are in App. H of the paper.

Scripts (in `rlvr_experiments/`, driven by CLI arguments):

- `experiments/rlvr_geometry_paper/scripts/analysis/analyze_rlvr_weight_changes.py`, `experiments/rlvr_geometry_paper/scripts/analysis/analyze_weight_distance.py`, `experiments/rlvr_geometry_paper/scripts/run/run_weight_analysis.sh`: task-vector magnitude, relative change
- `experiments/rlvr_geometry_paper/scripts/analysis/analyze_eigenvalue_shifts.py`: spectrum preservation
- `experiments/rlvr_geometry_paper/scripts/analysis/analyze_subspace_and_rotation.py`, `experiments/rlvr_geometry_paper/scripts/analysis/analyze_rotation_direction.py`, `experiments/rlvr_geometry_paper/scripts/analysis/analyze_rotation_direction_4b.py`: principal angles, novel subspace
- `experiments/rlvr_geometry_paper/scripts/analysis/analyze_per_layer_task_diff.py`: depth signature
- `experiments/rlvr_geometry_paper/scripts/analysis/analyze_sign_agreement.py`, `experiments/rlvr_geometry_paper/scripts/analysis/analyze_sign_agreement_multi.py`: sign agreement (a falsified predictor, reported as a null)

Source report: `experiments/rlvr_geometry_paper/reports/report_phase3_task_distances.md`.
