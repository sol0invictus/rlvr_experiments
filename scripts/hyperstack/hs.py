"""
Minimal Hyperstack (Infrahub API) helper, run from the local machine.

Reads HYPERSTACK_API_KEY from the environment; never prints it. Every command that
creates, changes or deletes a paid resource logs a line to the cost log so spend can
be reconciled against Hyperstack billing.

    python scripts/hyperstack/hs.py credit
    python scripts/hyperstack/hs.py vms
    python scripts/hyperstack/hs.py volumes
    python scripts/hyperstack/hs.py create-vm NAME --flavor n3-RTX-A6000x1 [--volume-id ID]
    python scripts/hyperstack/hs.py show VM_ID
    python scripts/hyperstack/hs.py hibernate VM_ID | restore VM_ID | delete VM_ID
    python scripts/hyperstack/hs.py create-volume NAME --size 250
    python scripts/hyperstack/hs.py attach VM_ID VOLUME_ID
    python scripts/hyperstack/hs.py delete-volume VOLUME_ID
"""

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone

API = "https://infrahub-api.nexgencloud.com/v1"
ENVIRONMENT = "default-CANADA-1"
IMAGE = "Ubuntu Server 24.04 LTS R570 CUDA 12.8"
KEY_NAME = "rlvr-key"
COST_LOG = os.environ.get("HS_COST_LOG", os.path.expanduser("~/.config/hyperstack/cost_log.jsonl"))


def call(method: str, path: str, body=None):
    key = os.environ.get("HYPERSTACK_API_KEY")
    if not key:
        sys.exit("HYPERSTACK_API_KEY is not set")
    req = urllib.request.Request(
        API + path,
        method=method,
        data=json.dumps(body).encode() if body is not None else None,
        # Cloudflare in front of the API rejects urllib's default User-Agent.
        headers={"api_key": key, "accept": "application/json", "content-type": "application/json",
                 "user-agent": "rlvr-experiments-hs/1.0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        sys.exit(f"HTTP {e.code} on {method} {path}: {e.read().decode()[:500]}")


def log_cost_event(event: str, **fields):
    os.makedirs(os.path.dirname(COST_LOG), exist_ok=True)
    rec = {"utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "event": event, **fields}
    with open(COST_LOG, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"[cost-log] {rec}")


def vm_summary(vm: dict) -> dict:
    keep = ("id", "name", "status", "power_state", "vm_state", "floating_ip", "fixed_ip", "created_at")
    out = {k: vm.get(k) for k in keep}
    out["flavor"] = (vm.get("flavor") or {}).get("name")
    out["volumes"] = [v.get("volume", v).get("id") for v in vm.get("volume_attachments") or []]
    return out


def my_ip() -> str:
    with urllib.request.urlopen("https://api.ipify.org", timeout=20) as r:
        return r.read().decode().strip()


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("credit")
    sub.add_parser("vms")
    sub.add_parser("volumes")
    c = sub.add_parser("create-vm")
    c.add_argument("name")
    c.add_argument("--flavor", required=True)
    c.add_argument("--volume-id", type=int, default=None)
    for name in ("show", "hibernate", "restore", "delete"):
        sub.add_parser(name).add_argument("vm_id", type=int)
    v = sub.add_parser("create-volume")
    v.add_argument("name")
    v.add_argument("--size", type=int, required=True, help="GB")
    a = sub.add_parser("attach")
    a.add_argument("vm_id", type=int)
    a.add_argument("volume_id", type=int)
    sub.add_parser("delete-volume").add_argument("volume_id", type=int)
    args = p.parse_args()

    if args.cmd == "credit":
        print(json.dumps(call("GET", "/billing/user-credit/credit").get("data"), indent=2))
    elif args.cmd == "vms":
        for vm in call("GET", "/core/virtual-machines").get("instances", []):
            print(json.dumps(vm_summary(vm)))
    elif args.cmd == "volumes":
        for vol in call("GET", "/core/volumes").get("volumes", []):
            print(json.dumps({k: vol.get(k) for k in ("id", "name", "size", "status", "volume_type", "created_at")}))
    elif args.cmd == "create-vm":
        body = {
            "name": args.name,
            "environment_name": ENVIRONMENT,
            "image_name": IMAGE,
            "flavor_name": args.flavor,
            "key_name": KEY_NAME,
            "count": 1,
            "assign_floating_ip": True,
            # SSH only, and only from this machine's current public IP.
            "security_rules": [{
                "direction": "ingress", "protocol": "tcp", "ethertype": "IPv4",
                "remote_ip_prefix": f"{my_ip()}/32", "port_range_min": 22, "port_range_max": 22,
            }],
        }
        res = call("POST", "/core/virtual-machines", body)
        vms = [vm_summary(vm) for vm in res.get("instances", [])]
        print(json.dumps(vms, indent=2))
        for vm in vms:
            log_cost_event("vm_created", vm_id=vm["id"], name=args.name, flavor=args.flavor)
            if args.volume_id is not None:
                print("Attach the volume once the VM is ACTIVE: hs.py attach", vm["id"], args.volume_id)
    elif args.cmd == "show":
        print(json.dumps(vm_summary(call("GET", f"/core/virtual-machines/{args.vm_id}").get("instance", {})), indent=2))
    elif args.cmd == "hibernate":
        print(call("GET", f"/core/virtual-machines/{args.vm_id}/hibernate"))
        log_cost_event("vm_hibernated", vm_id=args.vm_id)
    elif args.cmd == "restore":
        print(call("GET", f"/core/virtual-machines/{args.vm_id}/hibernate-restore"))
        log_cost_event("vm_restored", vm_id=args.vm_id)
    elif args.cmd == "delete":
        print(call("DELETE", f"/core/virtual-machines/{args.vm_id}"))
        log_cost_event("vm_deleted", vm_id=args.vm_id)
    elif args.cmd == "create-volume":
        res = call("POST", "/core/volumes", {
            "name": args.name, "environment_name": ENVIRONMENT, "volume_type": "Cloud-SSD", "size": args.size,
        })
        vol = res.get("volume", res)
        print(json.dumps(vol, indent=2))
        log_cost_event("volume_created", volume_id=vol.get("id"), name=args.name, size_gb=args.size)
    elif args.cmd == "attach":
        print(call("POST", f"/core/virtual-machines/{args.vm_id}/attach-volumes", {"volume_ids": [args.volume_id]}))
        log_cost_event("volume_attached", vm_id=args.vm_id, volume_id=args.volume_id)
    elif args.cmd == "delete-volume":
        print(call("DELETE", f"/core/volumes/{args.volume_id}"))
        log_cost_event("volume_deleted", volume_id=args.volume_id)


if __name__ == "__main__":
    main()
