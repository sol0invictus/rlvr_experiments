# RLVR Weight Change Analysis Report

**Model:** Qwen2.5-3B (3b)
**Generated:** 2026-04-20T20:38:33.734289

## Phase A: Task Vector Analysis

### Task Vector Magnitudes

How much did GRPO training change the weights for each task?

| Task | Magnitude | Interpretation |
|------|-----------|----------------|
| shortest_path | 0.8725 | Small change |
| syllogism | 0.8600 | Small change |
| countdown | 0.8372 | Small change |
| leg_counting | 0.7824 | Small change |
| caesar_cipher | 0.7686 | Small change |
| mini_sudoku | 0.7542 | Small change |
| basic_arithmetic | 0.7485 | Small change |
| maze | 0.7376 | Small change |
| knights_knaves | 0.7308 | Small change |
| cryptarithm | 0.0000 | Small change |

### Task Vector Similarity Matrix

Do different tasks learn similar weight changes? High similarity suggests related skills.

**Most Similar Task Pairs:**
- basic_arithmetic ↔ syllogism: 0.131
- basic_arithmetic ↔ leg_counting: 0.081
- shortest_path ↔ basic_arithmetic: 0.078
- shortest_path ↔ leg_counting: 0.065
- maze ↔ leg_counting: 0.060

**Most Different Task Pairs:**
- syllogism ↔ cryptarithm: 0.000
- knights_knaves ↔ cryptarithm: 0.000
- leg_counting ↔ cryptarithm: 0.000
- caesar_cipher ↔ cryptarithm: 0.000
- countdown ↔ basic_arithmetic: -0.009

## Phase B: Spectral Analysis

### Effective Rank by Task

Lower effective rank suggests more "focused" changes (potentially more skill-like).
Higher effective rank suggests diffuse changes (potentially more disruptive).

| Task | Avg Normalized Eff Rank | Interpretation |
|------|------------------------|----------------|
| countdown | 0.870 | High rank (diffuse) |
| maze | 0.821 | High rank (diffuse) |
| mini_sudoku | 0.963 | High rank (diffuse) |
| shortest_path | 0.892 | High rank (diffuse) |
| basic_arithmetic | 0.778 | High rank (diffuse) |
| syllogism | 0.767 | High rank (diffuse) |
| knights_knaves | 0.827 | High rank (diffuse) |
| leg_counting | 0.840 | High rank (diffuse) |
| caesar_cipher | 0.772 | High rank (diffuse) |
| cryptarithm | 0.010 | Very low rank (focused) |

## Conclusions

*(To be filled in after reviewing the quantitative results)*

## Visualizations

- `task_vector_similarity.png` - Heatmap of task vector cosine similarities
- `task_magnitudes.png` - Bar chart of weight change magnitudes
- `effective_rank_by_layer.png` - Effective rank breakdown by layer type
- `similarity_vs_transfer.png` - Scatter plot of similarity vs transfer performance
