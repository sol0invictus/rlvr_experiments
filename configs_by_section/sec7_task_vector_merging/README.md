# Sec. 7: Characterizing Task-Vector Merging

There are **no YAML configs** for this section. Merging is driven by scripts with CLI arguments over the
Sec. 4 single-task checkpoints and the Sec. 5 curriculum checkpoints (`../sec4_transfer_matrix/`,
`../sec5_atomic_curriculum/`). Full sweep tables are in App. F.

Scripts (in `rlvr_experiments/`):

- `merge_task_vectors.py`: addition, scaled addition, depth-selective, layer-type-selective, TIES, DARE
- `merge_layer_selective.py`, `merge_all_tasks.py`, `merge_triple.py`, `merge_phase1_all.py`
- `run_merge_eval.sh`: merge + evaluate
- `paper_prep/build_replicate_merges.py`, `paper_prep/f28_replicate_pipeline.sh`: 3B/8B replication. The pipeline
  writes its own per-scale configs to `configs/phase5_replicate_<scale>/` at run time; those are not checked in.
- `paper_prep/run_ars_batch.sh`, `ars_vs_merge.py`, `ars_vs_sign_agreement.py`: predictor analyses (null results)

Source report: `report_phase4_merge_scaling.md`.
