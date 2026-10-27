"""
RLVR Weight Change Analysis — Phases A, B, C

Phase A: Weight-only analysis (task vectors, cosine similarity, layer-wise magnitude)
Phase B: Spectral deep-dive (SVD, effective rank, singular value spectra)
Phase C: Gradient alignment (optional, more expensive)

Usage:
    python experiments/rlvr_geometry_paper/scripts/analysis/analyze_rlvr_weight_changes.py --model_size 4b --output_dir experiments/rlvr_geometry_paper/results/weight_analysis
"""

import argparse
import json
import os
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import gc

import torch
import numpy as np
from safetensors import safe_open
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr


# ============================================================================
# Configuration
# ============================================================================

MODEL_CONFIGS = {
    "3b": {
        "sft_path": "/code-fsx/sunguha/rlvr_experiments/outputs/sft_qwen25_3b",
        "results_prefix": "grpo_qwen25_3b_",
        "model_name": "Qwen2.5-3B",
    },
    "4b": {
        "sft_path": "/code-fsx/sunguha/sft_qwen_4.0b_instruct",
        "results_prefix": "grpo_qwen4b_",
        "alt_prefixes": ["grpo_qwen_4b_instruct_countdown_4to5"],  # countdown uses this full name
        "model_name": "Qwen3-4B",
        "task_overrides": {
            # Use phase2 direct_rlvr which was trained from sft_qwen_4.0b_instruct (matches other tasks)
            "countdown": "/code-fsx/sunguha/rlvr_experiments/results/phase2_countdown/direct_rlvr",
        },
    },
    "7b": {
        "sft_path": "/code-fsx/sunguha/rlvr_experiments/outputs/sft_qwen25_7b",
        "results_prefix": "grpo_qwen25_7b_",
        "model_name": "Qwen2.5-7B",
    },
    "8b": {
        "sft_path": "/code-fsx/sunguha/rlvr_experiments/outputs/sft_qwen3_8b",
        "results_prefix": "grpo_qwen3_8b_",
        "model_name": "Qwen3-8B",
    },
}

TASKS = [
    "countdown",
    "maze",
    "mini_sudoku",
    "shortest_path",
    "basic_arithmetic",
    "syllogism",
    "knights_knaves",
    "leg_counting",
    "caesar_cipher",
    "cryptarithm",
]

# Transfer matrix from Phase 1 (4B model, v1)
TRANSFER_MATRIX_4B = {
    "countdown": {"countdown": 36, "maze": 6, "mini_sudoku": 14, "shortest_path": 4, "basic_arithmetic": 6, "syllogism": 18, "knights_knaves": 4, "leg_counting": 6},
    "maze": {"countdown": -10, "maze": 12, "mini_sudoku": 16, "shortest_path": 4, "basic_arithmetic": 0, "syllogism": 12, "knights_knaves": 0, "leg_counting": 6},
    "mini_sudoku": {"countdown": -16, "maze": -2, "mini_sudoku": 22, "shortest_path": -16, "basic_arithmetic": -26, "syllogism": -16, "knights_knaves": -50, "leg_counting": -52},
    "shortest_path": {"countdown": 6, "maze": 8, "mini_sudoku": 28, "shortest_path": 28, "basic_arithmetic": 6, "syllogism": 16, "knights_knaves": -2, "leg_counting": 14},
    "basic_arithmetic": {"countdown": 4, "maze": -2, "mini_sudoku": -4, "shortest_path": -6, "basic_arithmetic": -12, "syllogism": 18, "knights_knaves": -16, "leg_counting": -32},
    "syllogism": {"countdown": -8, "maze": -2, "mini_sudoku": 26, "shortest_path": -2, "basic_arithmetic": -2, "syllogism": 26, "knights_knaves": 2, "leg_counting": 10},
    "knights_knaves": {"countdown": -8, "maze": 2, "mini_sudoku": 4, "shortest_path": -12, "basic_arithmetic": -8, "syllogism": 26, "knights_knaves": 12, "leg_counting": -4},
    "leg_counting": {"countdown": 16, "maze": 2, "mini_sudoku": 22, "shortest_path": 12, "basic_arithmetic": -8, "syllogism": 14, "knights_knaves": -8, "leg_counting": 28},
    "caesar_cipher": {"countdown": 0, "maze": 0, "mini_sudoku": 0, "shortest_path": -2, "basic_arithmetic": 6, "syllogism": -22, "knights_knaves": 0, "leg_counting": 2},
    "cryptarithm": {"countdown": -2, "maze": -2, "mini_sudoku": -2, "shortest_path": -2, "basic_arithmetic": -2, "syllogism": 2, "knights_knaves": 2, "leg_counting": -4},
}


# ============================================================================
# Utilities
# ============================================================================

def load_state_dict_from_safetensors(model_path: str) -> dict:
    """Load state dict from safetensors files."""
    model_path = Path(model_path)
    state_dict = {}

    safetensor_files = sorted(model_path.glob("*.safetensors"))
    if not safetensor_files:
        raise ValueError(f"No safetensors files found in {model_path}")

    for sf_path in safetensor_files:
        with safe_open(sf_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)

    return state_dict


def get_layer_type(key: str) -> str:
    """Categorize parameter by layer type."""
    if "self_attn" in key or "attention" in key:
        if "q_proj" in key:
            return "attn_q"
        elif "k_proj" in key:
            return "attn_k"
        elif "v_proj" in key:
            return "attn_v"
        elif "o_proj" in key:
            return "attn_o"
        return "attention"
    elif "mlp" in key:
        if "gate" in key:
            return "mlp_gate"
        elif "up" in key:
            return "mlp_up"
        elif "down" in key:
            return "mlp_down"
        return "mlp"
    elif "embed" in key:
        return "embedding"
    elif "norm" in key or "ln" in key:
        return "norm"
    elif "lm_head" in key:
        return "lm_head"
    return "other"


def get_layer_index(key: str) -> Optional[int]:
    """Extract layer index from parameter name."""
    import re
    match = re.search(r'layers\.(\d+)\.', key)
    return int(match.group(1)) if match else None


# ============================================================================
# Phase A: Weight-only Analysis
# ============================================================================

def compute_task_vector(sft_sd: dict, task_sd: dict) -> dict:
    """Compute task vector (delta) between SFT and task checkpoint."""
    task_vector = {}
    for key in sft_sd:
        if key in task_sd:
            sft_t = sft_sd[key].float()
            task_t = task_sd[key].float()
            if sft_t.shape == task_t.shape:
                task_vector[key] = task_t - sft_t
    return task_vector


def compute_task_vector_similarity(tv1: dict, tv2: dict) -> Tuple[float, dict]:
    """
    Compute cosine similarity between two task vectors.
    Returns global similarity and per-layer similarities.
    """
    common_keys = set(tv1.keys()) & set(tv2.keys())

    # Global similarity
    all_v1, all_v2 = [], []
    per_layer = {}

    for key in common_keys:
        v1 = tv1[key].flatten()
        v2 = tv2[key].flatten()

        if v1.numel() > 0 and v2.numel() > 0:
            all_v1.append(v1)
            all_v2.append(v2)

            # Per-layer cosine
            norm1, norm2 = v1.norm(), v2.norm()
            if norm1 > 1e-8 and norm2 > 1e-8:
                per_layer[key] = (v1 @ v2 / (norm1 * norm2)).item()
            else:
                per_layer[key] = 0.0

    v1_cat = torch.cat(all_v1)
    v2_cat = torch.cat(all_v2)

    norm1, norm2 = v1_cat.norm(), v2_cat.norm()
    if norm1 > 1e-8 and norm2 > 1e-8:
        global_sim = (v1_cat @ v2_cat / (norm1 * norm2)).item()
    else:
        global_sim = 0.0

    return global_sim, per_layer


def compute_task_vector_magnitude(tv: dict) -> Tuple[float, dict]:
    """Compute magnitude (Frobenius norm) of task vector."""
    all_params = []
    per_layer = {}

    for key, delta in tv.items():
        flat = delta.flatten()
        all_params.append(flat)
        per_layer[key] = flat.norm().item()

    global_mag = torch.cat(all_params).norm().item()
    return global_mag, per_layer


def phase_a_analysis(sft_sd: dict, task_checkpoints: Dict[str, dict]) -> dict:
    """
    Phase A: Weight-only analysis.

    Computes:
    1. Task vector magnitudes
    2. Task vector cosine similarity matrix
    3. Layer-wise breakdown
    """
    print("\n" + "="*80)
    print("PHASE A: Weight-only Analysis")
    print("="*80)

    results = {
        "task_magnitudes": {},
        "similarity_matrix": {},
        "per_layer_similarity": {},
        "layer_type_magnitudes": {},
    }

    # Compute task vectors
    task_vectors = {}
    for task, ckpt_sd in tqdm(task_checkpoints.items(), desc="Computing task vectors"):
        tv = compute_task_vector(sft_sd, ckpt_sd)
        task_vectors[task] = tv

        # Magnitude
        global_mag, per_layer_mag = compute_task_vector_magnitude(tv)
        results["task_magnitudes"][task] = {
            "global": global_mag,
            "per_layer": {k: v for k, v in sorted(per_layer_mag.items(), key=lambda x: -x[1])[:20]}
        }

        # Aggregate by layer type
        layer_type_mags = defaultdict(list)
        for key, mag in per_layer_mag.items():
            lt = get_layer_type(key)
            layer_type_mags[lt].append(mag)

        results["layer_type_magnitudes"][task] = {
            lt: {"mean": np.mean(mags), "sum": np.sum(mags), "count": len(mags)}
            for lt, mags in layer_type_mags.items()
        }

    # Compute similarity matrix
    tasks = list(task_vectors.keys())
    n = len(tasks)
    sim_matrix = np.zeros((n, n))

    print("\nComputing similarity matrix...")
    for i, t1 in enumerate(tqdm(tasks)):
        for j, t2 in enumerate(tasks):
            if i <= j:
                sim, per_layer = compute_task_vector_similarity(task_vectors[t1], task_vectors[t2])
                sim_matrix[i, j] = sim
                sim_matrix[j, i] = sim

                if i != j:
                    results["per_layer_similarity"][f"{t1}_vs_{t2}"] = {
                        k: v for k, v in sorted(per_layer.items(), key=lambda x: -abs(x[1]))[:10]
                    }

    results["similarity_matrix"] = {
        "tasks": tasks,
        "matrix": sim_matrix.tolist()
    }

    # Print summary
    print("\nTask Vector Magnitudes:")
    for task, mag in sorted(results["task_magnitudes"].items(), key=lambda x: -x[1]["global"]):
        print(f"  {task:20s}: {mag['global']:.4f}")

    print("\nTask Vector Similarity Matrix:")
    print(f"{'':15s}", end="")
    for t in tasks:
        print(f"{t[:8]:>10s}", end="")
    print()
    for i, t1 in enumerate(tasks):
        print(f"{t1[:15]:15s}", end="")
        for j, t2 in enumerate(tasks):
            print(f"{sim_matrix[i,j]:10.3f}", end="")
        print()

    # Clean up task vectors to free memory
    del task_vectors
    gc.collect()

    return results


# ============================================================================
# Phase B: Spectral Analysis
# ============================================================================

def compute_spectral_metrics(delta: torch.Tensor, max_rank: int = 100) -> dict:
    """
    Compute spectral metrics for a weight delta matrix.

    For 2D matrices: full SVD analysis
    For 1D/other: basic norm metrics only
    """
    if delta.dim() != 2:
        return {
            "frob_norm": delta.norm().item(),
            "spectral_norm": None,
            "effective_rank": None,
            "top_singular_values": None,
        }

    # Ensure contiguous for SVD
    delta = delta.float().contiguous()

    try:
        # For large matrices, use truncated SVD
        if min(delta.shape) > max_rank:
            # Use randomized SVD for large matrices
            U, S, V = torch.svd_lowrank(delta, q=max_rank)
            S = S[:max_rank]
        else:
            U, S, Vh = torch.linalg.svd(delta, full_matrices=False)

        spectral_norm = S[0].item()
        frob_norm = S.norm().item()

        # Effective rank: exp(entropy of normalized singular values)
        S_norm = S / S.sum()
        S_norm = S_norm[S_norm > 1e-10]  # Filter near-zero
        entropy = -(S_norm * S_norm.log()).sum()
        eff_rank = entropy.exp().item()

        # Normalized effective rank (0-1 scale)
        max_eff_rank = len(S)
        norm_eff_rank = eff_rank / max_eff_rank if max_eff_rank > 0 else 0

        return {
            "frob_norm": frob_norm,
            "spectral_norm": spectral_norm,
            "effective_rank": eff_rank,
            "normalized_eff_rank": norm_eff_rank,
            "top_singular_values": S[:min(20, len(S))].tolist(),
            "condition_number": (S[0] / S[-1]).item() if S[-1] > 1e-10 else float('inf'),
        }
    except Exception as e:
        print(f"  SVD failed: {e}")
        return {
            "frob_norm": delta.norm().item(),
            "spectral_norm": None,
            "effective_rank": None,
            "error": str(e),
        }


def phase_b_analysis(sft_sd: dict, task_checkpoints: Dict[str, dict], top_k_layers: int = 20) -> dict:
    """
    Phase B: Spectral deep-dive.

    For each task, compute SVD of weight deltas AND compare SFT vs GRPO spectral properties.
    """
    print("\n" + "="*80)
    print("PHASE B: Spectral Analysis")
    print("="*80)

    results = {
        "per_task_spectral": {},
        "layer_spectral_summary": {},
        "sft_vs_grpo_comparison": {},
    }

    for task, ckpt_sd in tqdm(task_checkpoints.items(), desc="Spectral analysis"):
        print(f"\n--- {task} ---")

        # Compute deltas and magnitudes
        deltas_with_mag = []
        for key in sft_sd:
            if key in ckpt_sd:
                sft_t = sft_sd[key].float()
                task_t = ckpt_sd[key].float()
                if sft_t.shape == task_t.shape and sft_t.dim() == 2:
                    delta = task_t - sft_t
                    mag = delta.norm().item()
                    deltas_with_mag.append((key, sft_t, task_t, delta, mag))

        # Sort by magnitude and take top-k
        deltas_with_mag.sort(key=lambda x: -x[4])
        top_deltas = deltas_with_mag[:top_k_layers]

        task_results = {}
        comparison_results = {}

        for key, sft_t, task_t, delta, mag in top_deltas:
            # Spectral metrics for delta
            spectral_delta = compute_spectral_metrics(delta)
            spectral_delta["magnitude"] = mag
            spectral_delta["layer_type"] = get_layer_type(key)
            spectral_delta["layer_idx"] = get_layer_index(key)

            # Spectral metrics for SFT and GRPO weights themselves
            spectral_sft = compute_spectral_metrics(sft_t)
            spectral_grpo = compute_spectral_metrics(task_t)

            # Compute comparison metrics
            if spectral_sft.get("spectral_norm") and spectral_grpo.get("spectral_norm"):
                comparison = {
                    "sft_spectral_norm": spectral_sft["spectral_norm"],
                    "grpo_spectral_norm": spectral_grpo["spectral_norm"],
                    "spectral_norm_ratio": spectral_grpo["spectral_norm"] / spectral_sft["spectral_norm"],
                    "sft_eff_rank": spectral_sft.get("effective_rank"),
                    "grpo_eff_rank": spectral_grpo.get("effective_rank"),
                    "eff_rank_change": (spectral_grpo.get("effective_rank", 0) - spectral_sft.get("effective_rank", 0)),
                    "delta_spectral_norm": spectral_delta.get("spectral_norm"),
                    "delta_eff_rank": spectral_delta.get("effective_rank"),
                }
                comparison_results[key] = comparison

            task_results[key] = spectral_delta

            eff_rank = spectral_delta.get("effective_rank", "N/A")
            if isinstance(eff_rank, float):
                ratio = comparison.get("spectral_norm_ratio", 0) if comparison_results.get(key) else 0
                print(f"  {key[:50]:50s} mag={mag:.4f} delta_eff_rank={eff_rank:.2f} spec_ratio={ratio:.4f}")

        results["per_task_spectral"][task] = task_results
        results["sft_vs_grpo_comparison"][task] = comparison_results

        # Aggregate by layer type
        layer_type_spectral = defaultdict(list)
        for key, metrics in task_results.items():
            lt = metrics["layer_type"]
            if metrics.get("effective_rank") is not None:
                comp = comparison_results.get(key, {})
                layer_type_spectral[lt].append({
                    "eff_rank": metrics["effective_rank"],
                    "norm_eff_rank": metrics.get("normalized_eff_rank", 0),
                    "spectral_norm": metrics["spectral_norm"],
                    "frob_norm": metrics["frob_norm"],
                    "spectral_norm_ratio": comp.get("spectral_norm_ratio", 1.0),
                    "eff_rank_change": comp.get("eff_rank_change", 0),
                })

        results["layer_spectral_summary"][task] = {
            lt: {
                "mean_eff_rank": np.mean([x["eff_rank"] for x in vals]),
                "mean_norm_eff_rank": np.mean([x["norm_eff_rank"] for x in vals]),
                "mean_spectral_norm": np.mean([x["spectral_norm"] for x in vals]),
                "mean_spectral_norm_ratio": np.mean([x["spectral_norm_ratio"] for x in vals]),
                "mean_eff_rank_change": np.mean([x["eff_rank_change"] for x in vals]),
                "count": len(vals),
            }
            for lt, vals in layer_type_spectral.items()
        }

        # Print summary for this task
        print(f"\n  Layer type summary for {task}:")
        for lt, summary in results["layer_spectral_summary"][task].items():
            print(f"    {lt}: spec_ratio={summary['mean_spectral_norm_ratio']:.4f}, eff_rank_change={summary['mean_eff_rank_change']:.2f}")

    return results


# ============================================================================
# Phase C: Correlation with Transfer Matrix
# ============================================================================

def phase_c_analysis(phase_a_results: dict, phase_b_results: dict, transfer_matrix: dict) -> dict:
    """
    Phase C: Correlate weight metrics with transfer performance.

    Test hypotheses:
    1. Similar task vectors → positive transfer
    2. Low effective rank → more transferable
    3. High magnitude changes → more volatile
    """
    print("\n" + "="*80)
    print("PHASE C: Transfer Correlation Analysis")
    print("="*80)

    results = {
        "similarity_vs_transfer": {},
        "magnitude_vs_transfer": {},
        "eff_rank_vs_transfer": {},
        "correlations": {},
    }

    sim_matrix = np.array(phase_a_results["similarity_matrix"]["matrix"])
    tasks = phase_a_results["similarity_matrix"]["tasks"]

    # Build transfer performance vectors
    transfer_data = []
    similarity_data = []

    for i, train_task in enumerate(tasks):
        if train_task not in transfer_matrix:
            continue
        for j, eval_task in enumerate(tasks):
            if eval_task not in transfer_matrix.get(train_task, {}):
                continue
            transfer_delta = transfer_matrix[train_task][eval_task]
            task_sim = sim_matrix[i, j]

            transfer_data.append(transfer_delta)
            similarity_data.append(task_sim)

            results["similarity_vs_transfer"][f"{train_task}_on_{eval_task}"] = {
                "transfer_delta": transfer_delta,
                "task_vector_sim": task_sim,
            }

    # Compute correlation
    if len(transfer_data) > 5:
        pearson_r, pearson_p = pearsonr(similarity_data, transfer_data)
        spearman_r, spearman_p = spearmanr(similarity_data, transfer_data)

        results["correlations"]["similarity_vs_transfer"] = {
            "pearson_r": pearson_r,
            "pearson_p": pearson_p,
            "spearman_r": spearman_r,
            "spearman_p": spearman_p,
            "interpretation": "Positive r means similar task vectors tend to transfer better"
        }

        print(f"\nTask Vector Similarity vs Transfer Performance:")
        print(f"  Pearson r = {pearson_r:.3f} (p = {pearson_p:.4f})")
        print(f"  Spearman r = {spearman_r:.3f} (p = {spearman_p:.4f})")

    # Magnitude vs diagonal (self-improvement)
    mag_data = []
    diag_transfer_data = []

    for task in tasks:
        if task in phase_a_results["task_magnitudes"] and task in transfer_matrix:
            if task in transfer_matrix[task]:
                mag = phase_a_results["task_magnitudes"][task]["global"]
                diag = transfer_matrix[task][task]
                mag_data.append(mag)
                diag_transfer_data.append(diag)
                results["magnitude_vs_transfer"][task] = {
                    "magnitude": mag,
                    "diagonal_transfer": diag,
                }

    if len(mag_data) > 3:
        pearson_r, pearson_p = pearsonr(mag_data, diag_transfer_data)
        results["correlations"]["magnitude_vs_diagonal"] = {
            "pearson_r": pearson_r,
            "pearson_p": pearson_p,
            "interpretation": "Does larger weight change correlate with better same-task performance?"
        }
        print(f"\nMagnitude vs Same-Task Performance:")
        print(f"  Pearson r = {pearson_r:.3f} (p = {pearson_p:.4f})")

    # Effective rank vs transfer (from Phase B)
    eff_rank_data = []
    off_diag_transfer = []

    for task in tasks:
        if task in phase_b_results.get("layer_spectral_summary", {}):
            # Average effective rank across layer types
            layer_summary = phase_b_results["layer_spectral_summary"][task]
            eff_ranks = [v["mean_eff_rank"] for v in layer_summary.values() if "mean_eff_rank" in v]
            if eff_ranks and task in transfer_matrix:
                avg_eff_rank = np.mean(eff_ranks)
                # Average off-diagonal transfer
                off_diag = [v for k, v in transfer_matrix[task].items() if k != task]
                if off_diag:
                    avg_off_diag = np.mean(off_diag)
                    eff_rank_data.append(avg_eff_rank)
                    off_diag_transfer.append(avg_off_diag)
                    results["eff_rank_vs_transfer"][task] = {
                        "avg_eff_rank": avg_eff_rank,
                        "avg_off_diag_transfer": avg_off_diag,
                    }

    if len(eff_rank_data) > 3:
        pearson_r, pearson_p = pearsonr(eff_rank_data, off_diag_transfer)
        results["correlations"]["eff_rank_vs_transfer"] = {
            "pearson_r": pearson_r,
            "pearson_p": pearson_p,
            "interpretation": "Negative r would mean lower rank changes transfer better (more 'skill-like')"
        }
        print(f"\nEffective Rank vs Off-Diagonal Transfer:")
        print(f"  Pearson r = {pearson_r:.3f} (p = {pearson_p:.4f})")

    return results


# ============================================================================
# Visualization
# ============================================================================

def create_visualizations(results: dict, output_dir: Path, model_name: str):
    """Generate plots for the analysis."""

    # 1. Task vector similarity heatmap
    if "phase_a" in results and "similarity_matrix" in results["phase_a"]:
        sim_data = results["phase_a"]["similarity_matrix"]
        tasks = sim_data["tasks"]
        matrix = np.array(sim_data["matrix"])

        fig, ax = plt.subplots(figsize=(12, 10))
        im = ax.imshow(matrix, cmap='RdBu_r', vmin=-1, vmax=1)

        ax.set_xticks(range(len(tasks)))
        ax.set_yticks(range(len(tasks)))
        ax.set_xticklabels([t[:12] for t in tasks], rotation=45, ha='right')
        ax.set_yticklabels([t[:12] for t in tasks])

        for i in range(len(tasks)):
            for j in range(len(tasks)):
                ax.text(j, i, f'{matrix[i,j]:.2f}', ha='center', va='center', fontsize=8)

        plt.colorbar(im, label='Cosine Similarity')
        ax.set_title(f'Task Vector Similarity Matrix ({model_name})')
        plt.tight_layout()
        plt.savefig(output_dir / 'task_vector_similarity.png', dpi=150)
        plt.close()

    # 2. Task magnitude bar chart
    if "phase_a" in results and "task_magnitudes" in results["phase_a"]:
        mags = results["phase_a"]["task_magnitudes"]
        tasks = list(mags.keys())
        values = [mags[t]["global"] for t in tasks]

        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(range(len(tasks)), values)
        ax.set_xticks(range(len(tasks)))
        ax.set_xticklabels([t[:12] for t in tasks], rotation=45, ha='right')
        ax.set_ylabel('Task Vector Magnitude (Frobenius Norm)')
        ax.set_title(f'GRPO Weight Change Magnitude ({model_name})')
        plt.tight_layout()
        plt.savefig(output_dir / 'task_magnitudes.png', dpi=150)
        plt.close()

    # 3. Effective rank by task and layer type
    if "phase_b" in results and "layer_spectral_summary" in results["phase_b"]:
        summary = results["phase_b"]["layer_spectral_summary"]

        layer_types = set()
        for task_data in summary.values():
            layer_types.update(task_data.keys())
        layer_types = sorted(layer_types)

        tasks = list(summary.keys())
        n_tasks = len(tasks)
        n_types = len(layer_types)

        fig, ax = plt.subplots(figsize=(14, 8))
        x = np.arange(n_tasks)
        width = 0.8 / n_types

        for i, lt in enumerate(layer_types):
            values = []
            for task in tasks:
                if lt in summary[task]:
                    values.append(summary[task][lt].get("mean_norm_eff_rank", 0))
                else:
                    values.append(0)
            ax.bar(x + i * width, values, width, label=lt)

        ax.set_xticks(x + width * n_types / 2)
        ax.set_xticklabels([t[:12] for t in tasks], rotation=45, ha='right')
        ax.set_ylabel('Normalized Effective Rank')
        ax.set_title(f'Effective Rank of Weight Changes by Layer Type ({model_name})')
        ax.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig(output_dir / 'effective_rank_by_layer.png', dpi=150)
        plt.close()

    # 4. Similarity vs Transfer scatter
    if "phase_c" in results and "similarity_vs_transfer" in results["phase_c"]:
        data = results["phase_c"]["similarity_vs_transfer"]

        sims = [v["task_vector_sim"] for v in data.values()]
        transfers = [v["transfer_delta"] for v in data.values()]
        labels = list(data.keys())

        fig, ax = plt.subplots(figsize=(10, 8))
        scatter = ax.scatter(sims, transfers, alpha=0.6, s=50)

        # Add labels for notable points
        for i, label in enumerate(labels):
            if abs(transfers[i]) > 20 or abs(sims[i]) > 0.5:
                ax.annotate(label.replace('_on_', '→')[:20], (sims[i], transfers[i]), fontsize=7)

        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Task Vector Cosine Similarity')
        ax.set_ylabel('Transfer Delta (%)')
        ax.set_title(f'Task Similarity vs Transfer Performance ({model_name})')

        # Add correlation info
        if "correlations" in results["phase_c"]:
            corr = results["phase_c"]["correlations"].get("similarity_vs_transfer", {})
            if "pearson_r" in corr:
                ax.text(0.02, 0.98, f'r = {corr["pearson_r"]:.3f}, p = {corr["pearson_p"]:.4f}',
                       transform=ax.transAxes, va='top', fontsize=10)

        plt.tight_layout()
        plt.savefig(output_dir / 'similarity_vs_transfer.png', dpi=150)
        plt.close()

    print(f"\nPlots saved to {output_dir}/")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="RLVR Weight Change Analysis")
    parser.add_argument("--model_size", type=str, default="4b", choices=["3b", "4b", "7b", "8b"])
    parser.add_argument("--output_dir", type=str, default="experiments/rlvr_geometry_paper/results/weight_analysis")
    parser.add_argument("--skip_phase_b", action="store_true", help="Skip spectral analysis (faster)")
    parser.add_argument("--tasks", type=str, default=None, help="Comma-separated task list (default: all)")
    args = parser.parse_args()

    config = MODEL_CONFIGS[args.model_size]
    output_dir = Path(args.output_dir) / args.model_size
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine tasks to analyze
    tasks_to_analyze = args.tasks.split(",") if args.tasks else TASKS

    print(f"Analyzing {config['model_name']} ({args.model_size})")
    print(f"SFT checkpoint: {config['sft_path']}")
    print(f"Tasks: {tasks_to_analyze}")
    print(f"Output: {output_dir}")

    # Load SFT checkpoint
    print("\nLoading SFT checkpoint...")
    sft_sd = load_state_dict_from_safetensors(config['sft_path'])
    print(f"  Loaded {len(sft_sd)} parameters")

    # Load task checkpoints
    results_dir = Path("/code-fsx/sunguha/rlvr_experiments/results")
    task_checkpoints = {}
    task_overrides = config.get('task_overrides', {})

    for task in tasks_to_analyze:
        # Check for explicit task override first
        if task in task_overrides:
            task_dir = Path(task_overrides[task])
        else:
            # Try main prefix first, then alternative prefixes
            prefixes = [config['results_prefix']] + config.get('alt_prefixes', [])
            task_dir = None

            for prefix in prefixes:
                candidate = results_dir / f"{prefix}{task}"
                if candidate.exists():
                    task_dir = candidate
                    break

        if task_dir is None or not task_dir.exists():
            print(f"  Warning: No checkpoint found for {task}, skipping")
            continue

        # Find model files
        if list(task_dir.glob("*.safetensors")):
            ckpt_path = task_dir
        else:
            ckpt_dirs = sorted(task_dir.glob("checkpoint-*"))
            if ckpt_dirs:
                ckpt_path = ckpt_dirs[-1]
            else:
                print(f"  Warning: No checkpoint found in {task_dir}, skipping")
                continue

        print(f"  Loading {task} from {ckpt_path}")
        task_checkpoints[task] = load_state_dict_from_safetensors(ckpt_path)

    print(f"\nLoaded {len(task_checkpoints)} task checkpoints")

    # Run analyses
    all_results = {"model": config['model_name'], "model_size": args.model_size}

    # Phase A
    all_results["phase_a"] = phase_a_analysis(sft_sd, task_checkpoints)

    # Phase B (optional)
    if not args.skip_phase_b:
        all_results["phase_b"] = phase_b_analysis(sft_sd, task_checkpoints)

    # Phase C (correlation with transfer matrix)
    if args.model_size == "4b":
        all_results["phase_c"] = phase_c_analysis(
            all_results["phase_a"],
            all_results.get("phase_b", {}),
            TRANSFER_MATRIX_4B
        )

    # Save results
    output_file = output_dir / "analysis_results.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {output_file}")

    # Generate visualizations
    create_visualizations(all_results, output_dir, config['model_name'])

    # Generate summary report
    generate_report(all_results, output_dir)


def generate_report(results: dict, output_dir: Path):
    """Generate a markdown report summarizing the findings."""

    report = f"""# RLVR Weight Change Analysis Report

**Model:** {results['model']} ({results['model_size']})
**Generated:** {__import__('datetime').datetime.now().isoformat()}

## Phase A: Task Vector Analysis

### Task Vector Magnitudes

How much did GRPO training change the weights for each task?

| Task | Magnitude | Interpretation |
|------|-----------|----------------|
"""

    if "phase_a" in results:
        mags = results["phase_a"]["task_magnitudes"]
        sorted_mags = sorted(mags.items(), key=lambda x: -x[1]["global"])
        for task, data in sorted_mags:
            mag = data["global"]
            if mag > 10:
                interp = "Large change"
            elif mag > 5:
                interp = "Moderate change"
            else:
                interp = "Small change"
            report += f"| {task} | {mag:.4f} | {interp} |\n"

        report += """
### Task Vector Similarity Matrix

Do different tasks learn similar weight changes? High similarity suggests related skills.

"""
        sim_data = results["phase_a"]["similarity_matrix"]
        tasks = sim_data["tasks"]
        matrix = np.array(sim_data["matrix"])

        # Find most similar and most different pairs
        pairs = []
        for i in range(len(tasks)):
            for j in range(i+1, len(tasks)):
                pairs.append((tasks[i], tasks[j], matrix[i,j]))

        pairs_sorted = sorted(pairs, key=lambda x: -x[2])

        report += "**Most Similar Task Pairs:**\n"
        for t1, t2, sim in pairs_sorted[:5]:
            report += f"- {t1} ↔ {t2}: {sim:.3f}\n"

        report += "\n**Most Different Task Pairs:**\n"
        for t1, t2, sim in pairs_sorted[-5:]:
            report += f"- {t1} ↔ {t2}: {sim:.3f}\n"

    if "phase_b" in results:
        report += """
## Phase B: Spectral Analysis

### Effective Rank by Task

Lower effective rank suggests more "focused" changes (potentially more skill-like).
Higher effective rank suggests diffuse changes (potentially more disruptive).

| Task | Avg Normalized Eff Rank | Interpretation |
|------|------------------------|----------------|
"""
        summary = results["phase_b"]["layer_spectral_summary"]
        for task, layer_data in summary.items():
            if layer_data:
                avg_rank = np.mean([v.get("mean_norm_eff_rank", 0) for v in layer_data.values()])
                if avg_rank < 0.1:
                    interp = "Very low rank (focused)"
                elif avg_rank < 0.3:
                    interp = "Low rank"
                elif avg_rank < 0.5:
                    interp = "Medium rank"
                else:
                    interp = "High rank (diffuse)"
                report += f"| {task} | {avg_rank:.3f} | {interp} |\n"

    if "phase_c" in results:
        report += """
## Phase C: Correlation with Transfer Performance

### Key Findings

"""
        corrs = results["phase_c"].get("correlations", {})

        if "similarity_vs_transfer" in corrs:
            c = corrs["similarity_vs_transfer"]
            report += f"""**Task Vector Similarity vs Transfer:**
- Pearson r = {c['pearson_r']:.3f} (p = {c['pearson_p']:.4f})
- {c['interpretation']}
- {"Significant" if c['pearson_p'] < 0.05 else "Not significant"} at α=0.05

"""

        if "magnitude_vs_diagonal" in corrs:
            c = corrs["magnitude_vs_diagonal"]
            report += f"""**Weight Change Magnitude vs Same-Task Performance:**
- Pearson r = {c['pearson_r']:.3f} (p = {c['pearson_p']:.4f})
- {c['interpretation']}

"""

        if "eff_rank_vs_transfer" in corrs:
            c = corrs["eff_rank_vs_transfer"]
            report += f"""**Effective Rank vs Off-Diagonal Transfer:**
- Pearson r = {c['pearson_r']:.3f} (p = {c['pearson_p']:.4f})
- {c['interpretation']}

"""

    report += """
## Conclusions

*(To be filled in after reviewing the quantitative results)*

## Visualizations

- `task_vector_similarity.png` - Heatmap of task vector cosine similarities
- `task_magnitudes.png` - Bar chart of weight change magnitudes
- `effective_rank_by_layer.png` - Effective rank breakdown by layer type
- `similarity_vs_transfer.png` - Scatter plot of similarity vs transfer performance
"""

    report_path = output_dir / "analysis_report.md"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Report saved to {report_path}")


if __name__ == "__main__":
    main()
