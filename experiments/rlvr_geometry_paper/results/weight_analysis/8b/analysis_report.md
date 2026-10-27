# RLVR Weight Change Analysis Report

**Model:** Qwen3-8B (8b)
**Generated:** 2026-04-20T17:09:53.107968

## Phase A: Task Vector Analysis

### Task Vector Magnitudes

How much did GRPO training change the weights for each task?

| Task | Magnitude | Interpretation |
|------|-----------|----------------|
| mini_sudoku | 1.6960 | Small change |
| knights_knaves | 1.2363 | Small change |
| countdown | 1.2072 | Small change |
| shortest_path | 1.1913 | Small change |
| leg_counting | 1.1557 | Small change |
| caesar_cipher | 1.1528 | Small change |
| basic_arithmetic | 1.1373 | Small change |
| syllogism | 0.9785 | Small change |
| cryptarithm | 0.0000 | Small change |

### Task Vector Similarity Matrix

Do different tasks learn similar weight changes? High similarity suggests related skills.

**Most Similar Task Pairs:**
- syllogism ↔ knights_knaves: 0.083
- basic_arithmetic ↔ syllogism: 0.065
- basic_arithmetic ↔ leg_counting: 0.065
- knights_knaves ↔ leg_counting: 0.056
- countdown ↔ syllogism: 0.056

**Most Different Task Pairs:**
- syllogism ↔ cryptarithm: 0.000
- knights_knaves ↔ cryptarithm: 0.000
- leg_counting ↔ cryptarithm: 0.000
- caesar_cipher ↔ cryptarithm: 0.000
- mini_sudoku ↔ syllogism: -0.008

## Phase B: Spectral Analysis

### Effective Rank by Task

Lower effective rank suggests more "focused" changes (potentially more skill-like).
Higher effective rank suggests diffuse changes (potentially more disruptive).

| Task | Avg Normalized Eff Rank | Interpretation |
|------|------------------------|----------------|
| countdown | 0.921 | High rank (diffuse) |
| mini_sudoku | 0.921 | High rank (diffuse) |
| shortest_path | 0.920 | High rank (diffuse) |
| basic_arithmetic | 0.889 | High rank (diffuse) |
| syllogism | 0.873 | High rank (diffuse) |
| knights_knaves | 0.894 | High rank (diffuse) |
| leg_counting | 0.887 | High rank (diffuse) |
| caesar_cipher | 0.799 | High rank (diffuse) |
| cryptarithm | 0.010 | Very low rank (focused) |

## Conclusions

*(To be filled in after reviewing the quantitative results)*

## Visualizations

- `task_vector_similarity.png` - Heatmap of task vector cosine similarities
- `task_magnitudes.png` - Bar chart of weight change magnitudes
- `effective_rank_by_layer.png` - Effective rank breakdown by layer type
- `similarity_vs_transfer.png` - Scatter plot of similarity vs transfer performance
