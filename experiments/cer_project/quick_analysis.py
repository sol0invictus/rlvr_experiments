"""Quick Phase 1 analysis - run on pod."""
import json, os, numpy as np
from pathlib import Path

diag_dir = "/scratch/sunguha/cer_project/phase1_instrumented_grpo/token_diagnostics"
files = sorted(Path(diag_dir).glob("binned_step_*.json"))

decoder = json.JSONDecoder()
all_data = []
for f in files:
    with open(f) as fh:
        raw = fh.read()
    data, _ = decoder.raw_decode(raw)
    all_data.append(data)

print(f"Loaded {len(all_data)} steps: [{all_data[0]['step']}..{all_data[-1]['step']}]")
print(f"Mean entropy over training: {np.mean([d['mean_entropy'] for d in all_data]):.4f}")

# Analyze last 50% of training (stabilized)
late_data = [d for d in all_data if d["step"] >= 250]
print(f"\nAnalyzing last {len(late_data)} steps (steps 250-500)")

# Collect weighted (entropy, grad_proxy) pairs
all_ent = []
all_gp = []
for d in late_data:
    for b in d["bins"]:
        for _ in range(b["count"]):
            all_ent.append(b["mean_entropy"])
            all_gp.append(b["mean_grad_proxy"])

ent_arr = np.array(all_ent)
gp_arr = np.array(all_gp)

# Re-bin with fixed edges
n_bins = 15
edges = np.linspace(0, np.percentile(ent_arr, 95), n_bins + 1)
bin_idx = np.digitize(ent_arr, edges) - 1
bin_idx = np.clip(bin_idx, 0, n_bins - 1)

print(f"\n{'Bin':>4} {'Entropy':>10} {'Grad Proxy':>12} {'Count':>8} {'Ratio vs bin0':>14}")
print("-" * 52)

bin0_gp = None
for b in range(n_bins):
    mask = bin_idx == b
    if mask.sum() == 0:
        continue
    mean_ent = ent_arr[mask].mean()
    mean_gp = gp_arr[mask].mean()
    count = int(mask.sum())
    if bin0_gp is None:
        bin0_gp = mean_gp
    ratio = mean_gp / max(bin0_gp, 1e-8)
    print(f"{b:>4} {mean_ent:>10.4f} {mean_gp:>12.6f} {count:>8d} {ratio:>14.1f}x")

# Key finding
print("\n" + "=" * 60)
print("KEY FINDING: Gradient Signal Concentration")
print("=" * 60)
low_ent_mask = ent_arr < np.percentile(ent_arr, 25)
mid_ent_mask = (ent_arr >= np.percentile(ent_arr, 25)) & (ent_arr <= np.percentile(ent_arr, 75))
high_ent_mask = ent_arr > np.percentile(ent_arr, 75)
print(f"  Low entropy (P<25):   mean_gp = {gp_arr[low_ent_mask].mean():.6f}  ({int(low_ent_mask.sum())} tokens)")
print(f"  Mid entropy (P25-75): mean_gp = {gp_arr[mid_ent_mask].mean():.6f}  ({int(mid_ent_mask.sum())} tokens)")
print(f"  High entropy (P>75):  mean_gp = {gp_arr[high_ent_mask].mean():.6f}  ({int(high_ent_mask.sum())} tokens)")
low_mean = gp_arr[low_ent_mask].mean()
mid_mean = gp_arr[mid_ent_mask].mean()
high_mean = gp_arr[high_ent_mask].mean()
print(f"  Concentration ratio (high/low): {high_mean / max(low_mean, 1e-8):.0f}x")
print(f"  Mid/Low ratio: {mid_mean / max(low_mean, 1e-8):.0f}x")

# Time evolution
print("\n" + "=" * 60)
print("TIME EVOLUTION")
print("=" * 60)
for d in all_data[::10]:
    bins = d["bins"]
    if len(bins) < 5:
        continue
    low_gp = bins[0]["mean_grad_proxy"]
    high_gp = bins[-1]["mean_grad_proxy"]
    mid_gp = bins[len(bins)//2]["mean_grad_proxy"]
    ratio = high_gp / max(low_gp, 1e-8)
    print(f"  Step {d['step']:3d}: ent={d['mean_entropy']:.3f} | low={low_gp:.4f} mid={mid_gp:.4f} high={high_gp:.4f} | hi/lo={ratio:.0f}x")

# Monotonic vs inverted-U assessment
print("\n" + "=" * 60)
print("SHAPE ASSESSMENT")
print("=" * 60)
bin_means = []
for b in range(n_bins):
    mask = bin_idx == b
    if mask.sum() > 0:
        bin_means.append(gp_arr[mask].mean())
bin_means = np.array(bin_means)
peak_idx = np.argmax(bin_means)
n = len(bin_means)
print(f"  Peak bin: {peak_idx}/{n} (0=leftmost, {n-1}=rightmost)")
print(f"  Peak value: {bin_means[peak_idx]:.6f}")
print(f"  Shape: {'MONOTONIC INCREASING' if peak_idx >= n-2 else 'INVERTED-U' if 2 <= peak_idx <= n-3 else 'LEFT-PEAKED'}")
print(f"  Implication for CER: {'STRONG' if bin_means[-1]/max(bin_means[0], 1e-8) > 50 else 'MODERATE'} gradient waste on low-entropy tokens")
