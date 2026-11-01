"""Instrumented solvers, one module per reasoning-gym task. Each exposes GRAPHS, trace(sample, graph)
and difficulty_features(sample)."""

from . import arithmetic, gcd, maze, mini_sudoku

TASKS = {
    "basic_arithmetic": arithmetic,
    "mini_sudoku": mini_sudoku,
    "maze": maze,
    "gcd": gcd,
}
