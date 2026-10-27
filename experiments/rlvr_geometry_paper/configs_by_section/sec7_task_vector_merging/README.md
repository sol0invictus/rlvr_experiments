# Sec. 7: Characterizing Task-Vector Merging

There are **no YAML configs** for this section. Merging is driven by scripts with CLI arguments over the
Sec. 4 single-task checkpoints and the Sec. 5 curriculum checkpoints (`../sec4_transfer_matrix/`,
`../sec5_atomic_curriculum/`). Full sweep tables are in App. F.

Scripts (in `rlvr_experiments/`):

- `experiments/rlvr_geometry_paper/scripts/merge/merge_task_vectors.py`: addition, scaled addition, depth-selective, layer-type-selective, TIES, DARE
- `experiments/rlvr_geometry_paper/scripts/merge/merge_layer_selective.py`, `experiments/rlvr_geometry_paper/scripts/merge/merge_all_tasks.py`, `experiments/rlvr_geometry_paper/scripts/merge/merge_triple.py`, `experiments/rlvr_geometry_paper/scripts/merge/merge_phase1_all.py`
- `experiments/rlvr_geometry_paper/scripts/run/run_merge_eval.sh`: merge + evaluate
- `experiments/rlvr_geometry_paper/paper/scripts/build_replicate_merges.py`, `experiments/rlvr_geometry_paper/paper/scripts/f28_replicate_pipeline.sh`: 3B/8B replication. The pipeline
  writes its own per-scale configs to `configs/phase5_replicate_<scale>/` at run time; those are not checked in.
- `experiments/rlvr_geometry_paper/paper/scripts/run_ars_batch.sh`, `ars_vs_merge.py`, `ars_vs_sign_agreement.py`: predictor analyses (null results)

Source report: `experiments/rlvr_geometry_paper/reports/report_phase4_merge_scaling.md`.
