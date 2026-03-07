"""
Atomic Skill Extractors Package

Each module implements an extract_atoms(instance, rng) function
that takes a reasoning-gym instance and returns a list of atomic Q&A dicts.
"""

from data_gen.atomic_extractors.countdown import extract_atoms as extract_countdown
from data_gen.atomic_extractors.maze import extract_atoms as extract_maze
from data_gen.atomic_extractors.mini_sudoku import extract_atoms as extract_mini_sudoku
from data_gen.atomic_extractors.tower_of_hanoi import extract_atoms as extract_tower_of_hanoi
from data_gen.atomic_extractors.knights_knaves import extract_atoms as extract_knights_knaves
from data_gen.atomic_extractors.syllogism import extract_atoms as extract_syllogism
from data_gen.atomic_extractors.graph_color import extract_atoms as extract_graph_color
from data_gen.atomic_extractors.caesar_cipher import extract_atoms as extract_caesar_cipher
from data_gen.atomic_extractors.shortest_path import extract_atoms as extract_shortest_path
from data_gen.atomic_extractors.basic_arithmetic import extract_atoms as extract_basic_arithmetic

EXTRACTORS = {
    "countdown": extract_countdown,
    "maze": extract_maze,
    "mini_sudoku": extract_mini_sudoku,
    "tower_of_hanoi": extract_tower_of_hanoi,
    "knights_knaves": extract_knights_knaves,
    "syllogism": extract_syllogism,
    "graph_color": extract_graph_color,
    "caesar_cipher": extract_caesar_cipher,
    "shortest_path": extract_shortest_path,
    "basic_arithmetic": extract_basic_arithmetic,
}
