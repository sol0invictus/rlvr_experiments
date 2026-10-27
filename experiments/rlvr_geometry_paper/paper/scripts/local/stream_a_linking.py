"""
Stream A — Linking experiment.

Test whether atomic-trained task vectors merge better than direct-RLVR task
vectors.

Steps:
  1. Compute 6 task vectors (atomic/direct × {arith, SP, maze}) vs SFT base.
  2. Sign-agreement for atomic-atomic pairs and direct-direct pairs. Save:
       experiments/rlvr_geometry_paper/results/weight_analysis/4b/sign_agreement_linking.json
  3. Build 3 merged models at s=1.0:
       M_atomic_atomic = W_sft + tau_atomic(SP)   + tau_atomic(Arith)
       M_direct_direct = W_sft + tau_direct(SP)   + tau_direct(Arith)
       M_mixed         = W_sft + tau_atomic(SP)   + tau_direct(Arith)

Artifacts saved under experiments/rlvr_geometry_paper/results/ and the merged models under
experiments/rlvr_geometry_paper/results/linking/{aa,dd,mix}.
"""

import json
import os
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

BASE = "/scratch/sunguha/sft_qwen_4.0b_instruct"

TAUS_CFG = {
    "atomic_arith":  "experiments/rlvr_geometry_paper/results/phase2_arith/stage2_l2",
    "atomic_sp":     "experiments/rlvr_geometry_paper/results/phase2_sp/idea_rft_base/grpo",
    "atomic_maze":   "experiments/rlvr_geometry_paper/results/phase2_ext/maze/stage2_l2",
    "direct_arith":  "experiments/rlvr_geometry_paper/results/grpo_qwen4b_basic_arithmetic",
    "direct_sp":     "experiments/rlvr_geometry_paper/results/grpo_qwen4b_shortest_path",
    "direct_maze":   "experiments/rlvr_geometry_paper/results/grpo_qwen4b_maze",
}


def load_sd(p):
    p = Path(p)
    out = {}
    for f in sorted(p.glob("model*.safetensors")):
        out.update(load_file(str(f)))
    return out


def compute_tau(base, sd):
    t = {}
    for k in base:
        if k in sd and sd[k].shape == base[k].shape:
            t[k] = sd[k].float() - base[k].float()
    return t


def sign_agreement(tau_a, tau_b):
    total_params = 0
    n_agree = 0
    mag_num = 0.0
    mag_den = 0.0
    conf_num = 0.0
    conf_den = 0.0
    up_num = 0.0
    up_den = 0.0
    q_num = 0.0
    q_den = 0.0
    for k in tau_a:
        if k not in tau_b:
            continue
        a = tau_a[k]
        b = tau_b[k]
        if a.shape != b.shape:
            continue
        sa = a.sign()
        sb = b.sign()
        agree = (sa == sb) & (sa != 0)
        n_agree += agree.sum().item()
        total_params += a.numel()
        w = a.abs() * b.abs()
        mag_num += (w * agree.float()).sum().item()
        mag_den += w.sum().item()
        disagree = (sa != sb) & (sa != 0) & (sb != 0)
        conf_num += (w * disagree.float()).sum().item()
        conf_den += w.sum().item()
        if "up_proj" in k or "gate_proj" in k:
            up_num += (w * disagree.float()).sum().item()
            up_den += w.sum().item()
        elif "q_proj" in k:
            q_num += (w * disagree.float()).sum().item()
            q_den += w.sum().item()
    return {
        "frac_agree": n_agree / total_params if total_params else 0,
        "mag_agree": mag_num / mag_den if mag_den else 0,
        "conflict_mass": conf_num / conf_den if conf_den else 0,
        "up_conflict": up_num / up_den if up_den else 0,
        "q_conflict": q_num / q_den if q_den else 0,
    }


def save_merged(base_path, merged, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for f in Path(base_path).iterdir():
        if f.suffix in (".safetensors", ".bin"):
            continue
        if f.name == "trainer_state.json":
            continue
        if f.is_file():
            shutil.copy2(str(f), out_dir)
    save_file(merged, os.path.join(out_dir, "model.safetensors"))
    print(f"  Saved -> {out_dir}")


def main():
    print(f"Loading base: {BASE}")
    base = load_sd(BASE)

    taus = {}
    for name, path in TAUS_CFG.items():
        assert os.path.exists(path), f"Missing {path}"
        print(f"Computing tau: {name} <- {path}")
        sd = load_sd(path)
        taus[name] = compute_tau(base, sd)
        del sd

    # -- Sign agreement: 3 atomic-atomic pairs + 3 direct-direct pairs --
    pairs = [
        ("atomic_sp_arith",   "atomic_sp",   "atomic_arith"),
        ("atomic_sp_maze",    "atomic_sp",   "atomic_maze"),
        ("atomic_arith_maze", "atomic_arith","atomic_maze"),
        ("direct_sp_arith",   "direct_sp",   "direct_arith"),
        ("direct_sp_maze",    "direct_sp",   "direct_maze"),
        ("direct_arith_maze", "direct_arith","direct_maze"),
    ]
    rows = []
    print("\nSign agreement:")
    print(f'{"pair":22} {"frac_agree":>10} {"mag_agree":>10} {"conflict":>10} {"up_conf":>10} {"q_conf":>10}')
    for pname, a, b in pairs:
        stats = sign_agreement(taus[a], taus[b])
        stats["pair"] = pname
        stats["kind"] = pname.split("_")[0]
        rows.append(stats)
        print(f'{pname:22} {stats["frac_agree"]:10.4f} {stats["mag_agree"]:10.4f} '
              f'{stats["conflict_mass"]:10.4f} {stats["up_conflict"]:10.4f} {stats["q_conflict"]:10.4f}')

    out_path = "experiments/rlvr_geometry_paper/results/weight_analysis/4b/sign_agreement_linking.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"Saved: {out_path}")

    # -- Build 3 merged models at s=1.0 --
    def merge(tau_sp, tau_ar, out_dir):
        print(f"\nMerging -> {out_dir}")
        merged = {}
        for k in base:
            w = base[k].float()
            if k in tau_sp:
                w = w + tau_sp[k]
            if k in tau_ar:
                w = w + tau_ar[k]
            merged[k] = w.to(base[k].dtype)
        save_merged(BASE, merged, out_dir)

    merge(taus["atomic_sp"], taus["atomic_arith"], "experiments/rlvr_geometry_paper/results/linking/aa")
    merge(taus["direct_sp"], taus["direct_arith"], "experiments/rlvr_geometry_paper/results/linking/dd")
    merge(taus["atomic_sp"], taus["direct_arith"], "experiments/rlvr_geometry_paper/results/linking/mix")

    print("\nStream A merge phase complete.")


if __name__ == "__main__":
    main()
