#!/usr/bin/env bash
# One-time setup on a fresh GPU VM (Ubuntu 24.04 + CUDA 12.8 image). Idempotent.
#
#   VOLUME_DEV=/dev/vdb bash setup.sh      # format (if blank) + mount a data volume at /data
#   bash setup.sh                          # no volume: /data lives on the root disk
#
# Everything persistent (repo, venv, HF cache, outputs) goes under /data so it
# survives deleting the VM when a volume is attached.
set -euo pipefail

DATA=/data
BRANCH=${BRANCH:-claude/rlvr-study-restart-568397}
REPO=${REPO:-https://github.com/sol0invictus/rlvr_experiments.git}

# --- Data volume -------------------------------------------------------------
if [ -n "${VOLUME_DEV:-}" ]; then
    if ! sudo blkid "$VOLUME_DEV" >/dev/null 2>&1; then
        echo "[setup] formatting blank volume $VOLUME_DEV"
        sudo mkfs.ext4 -q "$VOLUME_DEV"
    fi
    sudo mkdir -p "$DATA"
    if ! mountpoint -q "$DATA"; then
        sudo mount "$VOLUME_DEV" "$DATA"
    fi
    grep -q " $DATA " /etc/fstab || echo "$VOLUME_DEV $DATA ext4 defaults,nofail 0 2" | sudo tee -a /etc/fstab >/dev/null
fi
sudo mkdir -p "$DATA"
sudo chown "$USER":"$USER" "$DATA"
mkdir -p "$DATA"/rlvr/status "$DATA"/hf "$DATA"/logs

# --- System packages ---------------------------------------------------------
sudo apt-get update -qq
sudo apt-get install -y -qq git tmux python3-venv python3-dev build-essential >/dev/null

# --- Code --------------------------------------------------------------------
if [ ! -d "$DATA/rlvr_experiments/.git" ]; then
    git clone -q --branch "$BRANCH" "$REPO" "$DATA/rlvr_experiments"
else
    git -C "$DATA/rlvr_experiments" fetch -q origin "$BRANCH"
    git -C "$DATA/rlvr_experiments" checkout -q "$BRANCH"
    git -C "$DATA/rlvr_experiments" reset -q --hard "origin/$BRANCH"
fi

# --- Python env --------------------------------------------------------------
if [ ! -x "$DATA/venv/bin/python" ]; then
    python3 -m venv "$DATA/venv"
fi
"$DATA/venv/bin/pip" install -q --upgrade pip
"$DATA/venv/bin/pip" install -q -r "$DATA/rlvr_experiments/requirements-replication.txt"

cat > "$DATA/env.sh" <<EOF
export HF_HOME=$DATA/hf
export TOKENIZERS_PARALLELISM=false
export PATH=$DATA/venv/bin:\$PATH
cd $DATA/rlvr_experiments
EOF

source "$DATA/env.sh"
python - <<'PY'
import torch, transformers, trl, vllm, reasoning_gym
print("torch", torch.__version__, "cuda", torch.version.cuda, "gpu", torch.cuda.get_device_name(0))
print("transformers", transformers.__version__, "trl", trl.__version__, "vllm", vllm.__version__)
PY
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
df -h "$DATA" | tail -1
echo "[setup] done; source $DATA/env.sh"
