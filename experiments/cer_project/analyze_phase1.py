"""
Analyze Phase 1 token diagnostics to validate the inverted-U hypothesis.

Reads the binned diagnostic files and produces:
1. Plot of mean gradient proxy vs entropy bin
2. Summary statistics
3. Go/no-go assessment for Phase 2
"""

import json
import os
import argparse
import numpy as np
from pathlib import Path

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def load_binned_data(diagnostics_dir):
    """Load all binned_step_*.json files and aggregate."""
    all_bins = {}
    files = sorted(Path(diagnostics_dir).glob("binned_step_*.json"))
    if not files:
        raise FileNotFoundError(f"No binned diagnostic files found in {diagnostics_dir}")

    steps = []
    decoder = json.JSONDecoder()
    for f in files:
        with open(f) as fh:
            raw = fh.read()
        data, _ = decoder.raw_decode(raw)
        steps.append(data['step'])
        for b in data['bins']:
            key = (b['entropy_lo'], b['entropy_hi'])
            if key not in all_bins:
                all_bins[key] = {'counts': [], 'grad_proxies': [], 'entropies': []}
            all_bins[key]['counts'].append(b['count'])
            all_bins[key]['grad_proxies'].append(b['mean_grad_proxy'])
            all_bins[key]['entropies'].append(b['mean_entropy'])

    return all_bins, steps


def compute_aggregate(all_bins):
    """Compute weighted average across all steps for each bin."""
    results = []
    for (lo, hi), data in sorted(all_bins.items()):
        counts = np.array(data['counts'])
        gps = np.array(data['grad_proxies'])
        ents = np.array(data['entropies'])
        total = counts.sum()
        if total == 0:
            continue
        weighted_gp = (gps * counts).sum() / total
        weighted_ent = (ents * counts).sum() / total
        results.append({
            'entropy_lo': lo,
            'entropy_hi': hi,
            'mean_entropy': float(weighted_ent),
            'mean_grad_proxy': float(weighted_gp),
            'total_tokens': int(total),
        })
    return results


def check_inverted_u(results):
    """Check if the gradient proxy follows an inverted-U shape."""
    if len(results) < 5:
        return False, "Too few bins to assess shape"

    entropies = np.array([r['mean_entropy'] for r in results])
    grad_proxies = np.array([r['mean_grad_proxy'] for r in results])

    peak_idx = np.argmax(grad_proxies)
    n = len(grad_proxies)

    if peak_idx == 0 or peak_idx == n - 1:
        return False, f"Peak at edge (idx={peak_idx}/{n}), not inverted-U"

    left_increasing = np.mean(np.diff(grad_proxies[:peak_idx+1]) > 0)
    right_decreasing = np.mean(np.diff(grad_proxies[peak_idx:]) < 0)

    is_inverted_u = left_increasing > 0.5 and right_decreasing > 0.5
    peak_entropy = entropies[peak_idx]
    peak_gp = grad_proxies[peak_idx]
    edge_gp = (grad_proxies[0] + grad_proxies[-1]) / 2
    prominence = peak_gp / (edge_gp + 1e-8)

    msg = (f"Peak at entropy={peak_entropy:.2f} (bin {peak_idx}/{n}), "
           f"prominence={prominence:.2f}x over edges, "
           f"left_increasing={left_increasing:.0%}, right_decreasing={right_decreasing:.0%}")

    return is_inverted_u, msg


def plot_results(results, output_path, steps):
    if not HAS_MPL:
        print("matplotlib not available, skipping plot")
        return

    entropies = [r['mean_entropy'] for r in results]
    grad_proxies = [r['mean_grad_proxy'] for r in results]
    counts = [r['total_tokens'] for r in results]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    ax1.bar(range(len(entropies)), grad_proxies, color='steelblue', alpha=0.8)
    ax1.set_xlabel('Entropy Bin (low → high)')
    ax1.set_ylabel('Mean |log_prob × advantage|')
    ax1.set_title('Gradient Proxy vs Token Entropy\n(Inverted-U = premise validated)')
    ax1.set_xticks(range(len(entropies)))
    ax1.set_xticklabels([f'{e:.1f}' for e in entropies], rotation=45, fontsize=7)

    ax2.bar(range(len(counts)), counts, color='coral', alpha=0.8)
    ax2.set_xlabel('Entropy Bin (low → high)')
    ax2.set_ylabel('Token Count')
    ax2.set_title('Distribution of Tokens Across Entropy Bins')
    ax2.set_xticks(range(len(entropies)))
    ax2.set_xticklabels([f'{e:.1f}' for e in entropies], rotation=45, fontsize=7)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnostics_dir", type=str,
                        default="/scratch/sunguha/cer_project/phase1_instrumented_grpo/token_diagnostics")
    parser.add_argument("--output_plot", type=str, default=None)
    args = parser.parse_args()

    print(f"Loading diagnostics from {args.diagnostics_dir}")
    all_bins, steps = load_binned_data(args.diagnostics_dir)
    print(f"  Found data from {len(steps)} steps: [{min(steps)}..{max(steps)}]")

    results = compute_aggregate(all_bins)
    print(f"\n{'='*60}")
    print("AGGREGATED RESULTS (all steps)")
    print(f"{'='*60}")
    print(f"{'Entropy':>10} {'Grad Proxy':>12} {'Count':>10}")
    print(f"{'-'*10} {'-'*12} {'-'*10}")
    for r in results:
        print(f"{r['mean_entropy']:>10.3f} {r['mean_grad_proxy']:>12.6f} {r['total_tokens']:>10d}")

    is_inverted_u, msg = check_inverted_u(results)
    print(f"\n{'='*60}")
    print("INVERTED-U ASSESSMENT")
    print(f"{'='*60}")
    print(f"  Result: {'CONFIRMED' if is_inverted_u else 'NOT CONFIRMED'}")
    print(f"  Details: {msg}")

    if is_inverted_u:
        print("\n  → GO for Phase 2: The premise holds. Tokens at intermediate")
        print("    competence (entropy) contribute most to GRPO learning.")
    else:
        print("\n  → INVESTIGATE: The inverted-U is not clean. Check:")
        print("    - Are there enough training steps? (need 100+)")
        print("    - Is the entropy range appropriate?")
        print("    - Try margin-based competence instead of entropy.")

    output_plot = args.output_plot or os.path.join(
        os.path.dirname(args.diagnostics_dir), "phase1_inverted_u.png"
    )
    plot_results(results, output_plot, steps)


if __name__ == "__main__":
    main()
