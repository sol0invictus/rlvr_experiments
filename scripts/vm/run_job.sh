#!/usr/bin/env bash
# Run one named job step on the VM with a hard time limit, logging to /data/logs
# and leaving a status marker the local watcher can poll:
#   /data/rlvr/status/<name>.running   while it runs
#   /data/rlvr/status/<name>.done      exit code + timestamps when finished
#
#   bash scripts/vm/run_job.sh <name> <timeout> <command...>
#   e.g. bash scripts/vm/run_job.sh smoke_sft 30m python train_sft.py configs/replication/smoke_sft_qwen25_0.5b.yaml
set -uo pipefail
source /data/env.sh

name=$1; limit=$2; shift 2
status=/data/rlvr/status
log=/data/logs/$name.log
mkdir -p "$status"
rm -f "$status/$name.done"
date -u +%FT%TZ > "$status/$name.running"

echo "[$name] $(date -u +%FT%TZ) start (timeout $limit): $*" | tee -a "$log"
timeout --signal=TERM --kill-after=2m "$limit" "$@" >> "$log" 2>&1
code=$?
echo "[$name] $(date -u +%FT%TZ) exit $code" | tee -a "$log"

printf 'exit=%s\nstart=%s\nend=%s\n' "$code" "$(cat "$status/$name.running")" "$(date -u +%FT%TZ)" > "$status/$name.done"
rm -f "$status/$name.running"
exit $code
