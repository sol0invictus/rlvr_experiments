#!/usr/bin/env bash
# Delete this VM through the Hyperstack API so it stops billing, after cleanly
# unmounting and detaching the /data volume (a volume still attached when its VM is
# deleted ends up RESERVED instead of AVAILABLE).
#
# Needs these files on the VM's ROOT disk (never on the /data volume, which outlives the VM):
#   ~/.hs/key        Hyperstack API key (mode 600)
#   ~/.hs/vm_id      this VM's id
#   ~/.hs/volume_id  the data volume's id (optional; no volume = skip detach)
# Run from outside /data. Also used as a watchdog:  (sleep 8h; bash ~/self_delete.sh) &
set -uo pipefail
cd "$HOME"
API=https://infrahub-api.nexgencloud.com/v1
key_file=$HOME/.hs/key
log=$HOME/self_delete.log
if [ ! -s "$key_file" ] || [ ! -s "$HOME/.hs/vm_id" ]; then
    echo "[self_delete] missing ~/.hs/key or ~/.hs/vm_id; not deleting" | tee -a "$log"
    exit 1
fi
vm_id=$(cat "$HOME/.hs/vm_id")
vol_id=$(cat "$HOME/.hs/volume_id" 2>/dev/null || true)
api() {  # api METHOD PATH [JSON]
    curl -s -X "$1" -H "api_key: $(cat "$key_file")" -H "accept: application/json" \
        -H "content-type: application/json" -H "user-agent: rlvr-experiments-hs/1.0" \
        ${3:+-d "$3"} "$API$2"
}
say() { echo "[self_delete] $(date -u +%FT%TZ) $*" | tee -a "$log"; }

say "starting for VM $vm_id (volume ${vol_id:-none})"
if [ -n "$vol_id" ] && mountpoint -q /data; then
    cp "$log" /data/logs/self_delete.log 2>/dev/null || true
    sync
    sudo umount /data 2>>"$log" || { say "umount busy; lazy unmount"; sudo umount -l /data; }
    sync
fi
if [ -n "$vol_id" ]; then
    say "detach: $(api POST /core/virtual-machines/$vm_id/detach-volumes "{\"volume_ids\": [$vol_id]}" | head -c 300)"
    for i in $(seq 1 20); do
        st=$(api GET /core/volumes | python3 -c "import json,sys; print(next((v.get('status') for v in json.load(sys.stdin).get('volumes', []) if v.get('id') == $vol_id), 'unknown'))")
        say "volume $vol_id status: $st"
        [ "$st" = "available" ] || [ "$st" = "AVAILABLE" ] && break
        sleep 15
    done
fi
for attempt in 1 2 3 4 5; do
    out=$(api DELETE /core/virtual-machines/$vm_id)
    say "delete attempt $attempt: $(echo "$out" | head -c 200)"
    echo "$out" | grep -q '"status": *true' && exit 0
    sleep 30
done
exit 1
