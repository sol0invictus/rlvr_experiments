#!/usr/bin/env bash
# Baseline diagnostics on a 1x A100 80GB VM with the data volume at /data (scripts/vm/setup.sh):
# three 60-step runs of the arithmetic baseline, each with one change (see the headers of
# experiments/atom_decomposition/configs/diag/*.yaml). No weights are saved, only trainer_state.json
# and generation logs under /data/rlvr/curriculum/diag/<name>. Ends by deleting the VM.
#
#   tmux new -d -s diag 'bash scripts/vm/curriculum_diag.sh'
set -uo pipefail
source /data/env.sh
R=/data/rlvr/curriculum/diag
D=experiments/atom_decomposition/configs/diag
mkdir -p "$R"
failed=()

finish() {
    printf 'failed=%s\nend=%s\n' "${failed[*]:-none}" "$(date -u +%FT%TZ)" > "$R/diag.done"
    nvidia-smi > /data/logs/nvidia_smi_end_diag.txt 2>&1 || true
    sync
    cd "$HOME"   # leave /data so it can be unmounted
    bash "$HOME/self_delete.sh"
}
trap finish EXIT
rm -f "$R/diag.done"   # a marker left by an earlier run would tell the local watcher we are done

# fastest and most informative first
for name in diag_is diag_nosleep diag_hf; do
    [ -f "$R/$name/trainer_state.json" ] && continue
    bash scripts/vm/run_job.sh $name 50m python train_grpo.py $D/$name.yaml --output_dir $R/$name \
        || failed+=($name)
done
du -sh "$R"/* || true
echo "DIAG DONE (failed: ${failed[*]:-none})"
