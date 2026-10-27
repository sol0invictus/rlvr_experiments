# RLVR Weight Change Analysis Report

**Model:** Qwen2.5-7B (7b)
**Generated:** 2026-04-21T19:14:22.126524

## Phase A: Task Vector Analysis

### Task Vector Magnitudes

How much did GRPO training change the weights for each task?

| Task | Magnitude | Interpretation |
|------|-----------|----------------|
| countdown | 1.5770 | Small change |
| shortest_path | 1.5282 | Small change |
| knights_knaves | 1.4397 | Small change |
| syllogism | 1.3059 | Small change |
| mini_sudoku | 1.3009 | Small change |
| caesar_cipher | 1.2918 | Small change |
| leg_counting | 1.2693 | Small change |
| maze | 1.2155 | Small change |
| basic_arithmetic | 1.2090 | Small change |
| cryptarithm | 0.0000 | Small change |

### Task Vector Similarity Matrix

Do different tasks learn similar weight changes? High similarity suggests related skills.

**Most Similar Task Pairs:**
- basic_arithmetic ↔ syllogism: 0.106
- syllogism ↔ knights_knaves: 0.092
- basic_arithmetic ↔ leg_counting: 0.080
- syllogism ↔ leg_counting: 0.067
- basic_arithmetic ↔ knights_knaves: 0.057

**Most Different Task Pairs:**
- knights_knaves ↔ cryptarithm: 0.000
- leg_counting ↔ cryptarithm: 0.000
- caesar_cipher ↔ cryptarithm: 0.000
- countdown ↔ syllogism: -0.001
- countdown ↔ knights_knaves: -0.003

## Phase B: Spectral Analysis

### Effective Rank by Task

Lower effective rank suggests more "focused" changes (potentially more skill-like).
Higher effective rank suggests diffuse changes (potentially more disruptive).

| Task | Avg Normalized Eff Rank | Interpretation |
|------|------------------------|----------------|
| countdown | 0.939 | High rank (diffuse) |
| maze | 0.941 | High rank (diffuse) |
| mini_sudoku | 0.932 | High rank (diffuse) |
| shortest_path | 0.915 | High rank (diffuse) |
| basic_arithmetic | 0.933 | High rank (diffuse) |
| syllogism | 0.918 | High rank (diffuse) |
| knights_knaves | 0.935 | High rank (diffuse) |
| leg_counting | 0.916 | High rank (diffuse) |
| caesar_cipher | 0.924 | High rank (diffuse) |
| cryptarithm | 0.010 | Very low rank (focused) |

## Conclusions

*(To be filled in after reviewing the quantitative results)*

## Visualizations

- `task_vector_similarity.png` - Heatmap of task vector cosine similarities
- `task_magnitudes.png` - Bar chart of weight change magnitudes
- `effective_rank_by_layer.png` - Effective rank breakdown by layer type
- `similarity_vs_transfer.png` - Scatter plot of similarity vs transfer performance
