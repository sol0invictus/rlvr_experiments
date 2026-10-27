"""
Stream A — Linking experiment.

Test: do atomic-trained τ's merge BETTER than direct-RLVR τ's?

Computes sign-agreement + conflict_mass between:
  - atomic-atomic pairs: τ_atomic(SP) × τ_atomic(Arith), × τ_atomic(maze)
  - direct-direct pairs: same pairings with τ_direct

Then builds 3 merged checkpoints (s=1.0, basic sum):
  - M_aa  = W_sft + τ_atomic(SP) + τ_atomic(Arith)
  - M_dd  = W_sft + τ_direct(SP) + τ_direct(Arith)   [pure Phase-1]
  - M_mix = W_sft + τ_atomic(SP) + τ_direct(Arith)

Saves sign_agreement_linking.json and the 3 merged checkpoints.
Eval is run separately by pipeline_eval.sh.
"""
import json, os, sys, shutil
from pathlib import Path
import numpy as np
from safetensors.torch import load_file, save_file
import torch

BASE = "/code-fsx/sunguha/rlvr_experiments/experiments/rlvr_geometry_paper"
SFT = "/code-fsx/sunguha/sft_qwen_4.0b_instruct"

ATOMIC = {
    "shortest_path":    f"{BASE}/results/phase2_sp/idea_rft_base/grpo",
    "basic_arithmetic": f"{BASE}/results/phase2_arith/stage2_l2",
    "maze":             f"{BASE}/results/phase2_ext/maze/stage2_l2",
}
DIRECT = {
    "shortest_path":    f"{BASE}/results/grpo_qwen4b_shortest_path",
    "basic_arithmetic": f"{BASE}/results/grpo_qwen4b_basic_arithmetic",
    "maze":             f"{BASE}/results/grpo_qwen4b_maze",
}
PAIRS = [
    ("sp", "ari"),
    ("sp", "maze"),
    ("ari", "maze"),
]
TASK_NAME = {"sp": "shortest_path", "ari": "basic_arithmetic", "maze": "maze"}

OUT_DIR = Path(f"{BASE}/paper/data/linking")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_sd(p):
    files = sorted(Path(p).glob("model*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no safetensors in {p}")
    sd = {}
    for f in files:
        sd.update(load_file(str(f)))
    return sd


def compute_tau(base, ckpt):
    print(f"  loading {ckpt}")
    sd = load_sd(ckpt)
    tau = {}
    for k, v in base.items():
        if k in sd and sd[k].shape == v.shape:
            tau[k] = (sd[k].float() - v.float()).to(torch.float32)
    return tau


def sign_agreement_stats(tau_a, tau_b):
    """Same metrics as analyze_sign_agreement_multi.py."""
    tot_nonzero = 0
    tot_agree_count = 0
    tot_mag = 0.0
    tot_mag_agree = 0.0
    tot_conflict_mass = 0.0
    per_type = {}
    for k in tau_a:
        if k not in tau_b: continue
        a = tau_a[k].cpu().numpy().ravel()
        b = tau_b[k].cpu().numpy().ravel()
        mask = (a != 0) & (b != 0)
        if not mask.any(): continue
        a = a[mask]; b = b[mask]
        agree = (np.sign(a) == np.sign(b))
        mag = np.abs(a) * np.abs(b)
        n = len(a)
        tot_nonzero += n
        tot_agree_count += int(agree.sum())
        tot_mag += float(mag.sum())
        tot_mag_agree += float(mag[agree].sum())
        tot_conflict_mass += float(mag[~agree].sum())
        if "up_proj" in k or "gate_proj" in k: lt = "up"
        elif "q_proj" in k: lt = "q"
        elif "k_proj" in k: lt = "k"
        elif "v_proj" in k: lt = "v"
        elif "o_proj" in k: lt = "o"
        elif "down_proj" in k: lt = "down"
        else: lt = "other"
        d = per_type.setdefault(lt, {"mag":0.0, "conflict":0.0})
        d["mag"] += float(mag.sum())
        d["conflict"] += float(mag[~agree].sum())
    return {
        "frac_agree": tot_agree_count / max(tot_nonzero, 1),
        "mag_agree": tot_mag_agree / max(tot_mag, 1e-12),
        "conflict_mass": tot_conflict_mass / max(tot_mag, 1e-12),
        "per_type": {lt: {"conflict_frac": d["conflict"]/max(d["mag"],1e-12)} for lt, d in per_type.items()},
        "tot_nonzero": tot_nonzero,
    }


def tau_norm(tau):
    sq = 0.0
    for v in tau.values():
        sq += float((v.float() ** 2).sum())
    return sq ** 0.5


def merge_basic(base_sd, taus, out_dir, ref_ckpt):
    """W_sft + sum(taus). Copies tokenizer/config from ref_ckpt."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    merged = {}
    for k, v in base_sd.items():
        w = v.float().clone()
        for tau in taus:
            if k in tau:
                w = w + tau[k]
        merged[k] = w.to(v.dtype)
    save_file(merged, os.path.join(out_dir, "model.safetensors"))
    for fname in ["config.json", "tokenizer.json", "tokenizer_config.json",
                  "special_tokens_map.json", "generation_config.json",
                  "chat_template.jinja", "added_tokens.json"]:
        src = Path(ref_ckpt) / fname
        if src.exists():
            shutil.copy(src, Path(out_dir) / fname)


def main():
    print("=== Stream A: Linking experiment ===", flush=True)
    print(f"Loading SFT base from {SFT}", flush=True)
    base = load_sd(SFT)

    # compute taus
    taus_atomic = {}
    taus_direct = {}
    for name, path in ATOMIC.items():
        if not Path(path).exists():
            print(f"  WARN: atomic {name} path {path} missing — skipping")
            continue
        taus_atomic[name] = compute_tau(base, path)
        print(f"  tau_atomic[{name}] ‖τ‖={tau_norm(taus_atomic[name]):.3f}")
    for name, path in DIRECT.items():
        if not Path(path).exists():
            print(f"  WARN: direct {name} path {path} missing")
            continue
        taus_direct[name] = compute_tau(base, path)
        print(f"  tau_direct[{name}] ‖τ‖={tau_norm(taus_direct[name]):.3f}")

    # sign agreement stats
    result = {"atomic": {}, "direct": {}, "cross": {}}
    for a, b in PAIRS:
        na, nb = TASK_NAME[a], TASK_NAME[b]
        if na in taus_atomic and nb in taus_atomic:
            print(f"  sign-agreement atomic {a}×{b}", flush=True)
            result["atomic"][f"{a}_{b}"] = sign_agreement_stats(taus_atomic[na], taus_atomic[nb])
        if na in taus_direct and nb in taus_direct:
            print(f"  sign-agreement direct {a}×{b}", flush=True)
            result["direct"][f"{a}_{b}"] = sign_agreement_stats(taus_direct[na], taus_direct[nb])
        if na in taus_atomic and nb in taus_direct:
            print(f"  sign-agreement cross {a}_atomic × {b}_direct", flush=True)
            result["cross"][f"{a}_atomic_x_{b}_direct"] = sign_agreement_stats(taus_atomic[na], taus_direct[nb])

    with open(OUT_DIR / "sign_agreement_linking.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"  wrote sign_agreement_linking.json", flush=True)

    # build 3 merged checkpoints
    sp_a, ari_a = taus_atomic.get("shortest_path"), taus_atomic.get("basic_arithmetic")
    sp_d, ari_d = taus_direct.get("shortest_path"), taus_direct.get("basic_arithmetic")

    if sp_a and ari_a:
        out = OUT_DIR / "M_aa"
        print(f"  building M_aa -> {out}", flush=True)
        merge_basic(base, [sp_a, ari_a], str(out), ATOMIC["basic_arithmetic"])
    if sp_d and ari_d:
        out = OUT_DIR / "M_dd"
        print(f"  building M_dd -> {out}", flush=True)
        merge_basic(base, [sp_d, ari_d], str(out), DIRECT["basic_arithmetic"])
    if sp_a and ari_d:
        out = OUT_DIR / "M_mix"
        print(f"  building M_mix -> {out}", flush=True)
        merge_basic(base, [sp_a, ari_d], str(out), DIRECT["basic_arithmetic"])

    print("DONE stream A build", flush=True)


if __name__ == "__main__":
    main()
