#!/usr/bin/env bash
# Local safety net for an unattended VM run. Every 5 minutes: if the run reported done,
# or the VM is unreachable for 30 minutes, or the hard cap is reached, delete the VM.
# (The VM also deletes itself; this covers the case where that fails.)
#
#   bash scripts/hyperstack/watch.sh VM_ID IP HOURS_CAP
set -uo pipefail
vm_id=$1; ip=$2; cap_h=${3:-8.5}
here=$(cd "$(dirname "$0")" && pwd)
start=$(date +%s); cap_s=$(python3 -c "print(int(float('$cap_h') * 3600))")
unreachable_since=""
ssh_cmd=(ssh -i "$HOME/.ssh/hyperstack_ed25519" -o BatchMode=yes -o ConnectTimeout=15 "ubuntu@$ip")

delete_vm() {
    echo "[watch] $(date -u +%FT%TZ) $1 -> deleting VM $vm_id"
    python3 -I "$here/hs.py" delete "$vm_id"
}

while true; do
    now=$(date +%s)
    state=$(python3 -I "$here/hs.py" show "$vm_id" 2>/dev/null | python3 -c "import json,sys; print(json.load(sys.stdin).get('status'))" 2>/dev/null)
    case "$state" in
        ""|None|DELETED|DELETING) echo "[watch] $(date -u +%FT%TZ) VM $vm_id is gone ($state); done"; exit 0;;
    esac
    if (( now - start > cap_s )); then delete_vm "hard cap ${cap_h}h reached"; exit 0; fi
    if "${ssh_cmd[@]}" 'test -f /data/rlvr/status/overnight.done' 2>/dev/null; then
        sleep 120   # give the VM's own self-delete a chance first
        delete_vm "run reported done"; exit 0
    fi
    if "${ssh_cmd[@]}" true 2>/dev/null; then
        unreachable_since=""
        echo "[watch] $(date -u +%FT%TZ) $state; $("${ssh_cmd[@]}" 'ls /data/rlvr/status | tr "\n" " "' 2>/dev/null | tail -c 300)"
    else
        unreachable_since=${unreachable_since:-$now}
        if (( now - unreachable_since > 1800 )); then delete_vm "unreachable for 30 min"; exit 0; fi
    fi
    sleep 300
done
