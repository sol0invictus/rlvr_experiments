"""
Atomic skill templates for RLVR tasks.

Each template takes a reasoning_gym sample dict and returns (question, answer)
for a specific atomic sub-skill. All answers are programmatically verifiable.

Supported tasks: basic_arithmetic, countdown

Layers:
  L1 (Grounding): Parse the input — extract numbers, operators, structure
  L2 (Local Reasoning): Single-step computation — one operation, comparisons, reachability
"""

import re
import random
from typing import Tuple, Optional


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_expression(expr: str):
    """Extract numbers and operators from an expression string."""
    numbers = [int(n) for n in re.findall(r'-?\d+', expr)]
    operators = re.findall(r'(?<!\d)[+\-*/](?!\d)', expr.replace('--', '- -'))
    ops_clean = []
    for op in re.findall(r'[\+\-\*/]', expr):
        ops_clean.append(op)
    return numbers, ops_clean


def _find_innermost_parens(expr: str) -> Optional[str]:
    """Find the innermost parenthesized sub-expression."""
    match = re.search(r'\(([^()]+)\)', expr)
    if match:
        return match.group(1).strip()
    return None


# ---------------------------------------------------------------------------
# L1 — Grounding atoms
# ---------------------------------------------------------------------------

def atom_number_extraction(sample: dict, rng: random.Random) -> Tuple[str, str]:
    """L1: Extract all numbers (with their signs) from the expression."""
    expr = sample['metadata']['expression']
    numbers = re.findall(r'-?\d+', expr)
    q = (f"List all the numbers (including negative signs) in this expression, "
         f"in order, separated by commas: {expr}")
    a = ", ".join(numbers)
    return q, a


def atom_operator_identification(sample: dict, rng: random.Random) -> Tuple[str, str]:
    """L1: Identify the operators used in the expression."""
    expr = sample['metadata']['expression']
    op_map = {'+': 'addition', '-': 'subtraction', '*': 'multiplication', '/': 'division'}
    ops_found = []
    seen = set()
    for ch in expr:
        if ch in op_map and ch not in seen:
            ops_found.append(op_map[ch])
            seen.add(ch)
    q = f"What arithmetic operations are used in this expression (list them): {expr}"
    a = ", ".join(ops_found) if ops_found else "none"
    return q, a


def atom_term_count(sample: dict, rng: random.Random) -> Tuple[str, str]:
    """L1: Count how many numeric terms are in the expression."""
    expr = sample['metadata']['expression']
    num_terms = sample['metadata']['num_terms']
    q = f"How many numeric terms are in this expression: {expr}"
    a = str(num_terms)
    return q, a


def atom_has_parentheses(sample: dict, rng: random.Random) -> Tuple[str, str]:
    """L1: Does the expression contain parentheses?"""
    expr = sample['metadata']['expression']
    has_parens = '(' in expr
    q = f"Does this expression contain parentheses? Answer Yes or No: {expr}"
    a = "Yes" if has_parens else "No"
    return q, a


# ---------------------------------------------------------------------------
# L2 — Local reasoning atoms
# ---------------------------------------------------------------------------

def atom_single_op_eval(sample: dict, rng: random.Random) -> Tuple[str, str]:
    """L2: Evaluate a single two-operand arithmetic operation."""
    ops = [
        ('+', lambda a, b: a + b),
        ('-', lambda a, b: a - b),
        ('*', lambda a, b: a * b),
    ]
    op_sym, op_fn = rng.choice(ops)
    num_digits = sample['metadata'].get('num_digits', 2)
    max_val = 10 ** min(num_digits, 3)
    a_val = rng.randint(-max_val, max_val)
    b_val = rng.randint(-max_val, max_val)
    if op_sym == '/' and b_val == 0:
        b_val = rng.randint(1, max_val)
    result = op_fn(a_val, b_val)
    q = f"What is {a_val} {op_sym} {b_val}?"
    a = str(result)
    return q, a


def atom_order_of_operations(sample: dict, rng: random.Random) -> Tuple[str, str]:
    """L2: Evaluate the first step of a multi-step expression.

    Instead of asking "what to compute first" (ambiguous to verify),
    we ask: "What is the result after performing the first required
    operation?" This has a single correct numeric answer.
    """
    expr = sample['metadata']['expression']
    inner = _find_innermost_parens(expr)
    if inner:
        try:
            result = eval(inner)
            if isinstance(result, float) and result == int(result):
                result = int(result)
            q = (f"In the expression {expr}, the innermost parentheses contain: {inner}\n"
                 f"What does {inner} evaluate to?")
            a = str(result)
            return q, a
        except Exception:
            pass
    # No parens — find first */ operation (precedence)
    mult_match = re.search(r'(-?\d+)\s*([*/])\s*(-?\d+)', expr)
    if mult_match:
        left, op, right = mult_match.group(1), mult_match.group(2), mult_match.group(3)
        try:
            result = eval(f"{left} {op} {right}")
            if isinstance(result, float) and result == int(result):
                result = int(result)
            q = (f"In the expression {expr}, multiplication/division comes before addition/subtraction.\n"
                 f"What is {left} {op} {right}?")
            a = str(result)
            return q, a
        except Exception:
            pass
    # Fallback: just first pair
    a_val = rng.randint(-50, 50)
    b_val = rng.randint(-50, 50)
    op = rng.choice(['+', '-', '*'])
    result = eval(f"{a_val} {op} {b_val}")
    if isinstance(result, float) and result == int(result):
        result = int(result)
    q = f"What is {a_val} {op} {b_val}?"
    a = str(result)
    return q, a


def atom_innermost_eval(sample: dict, rng: random.Random) -> Tuple[str, str]:
    """L2: Evaluate the innermost parenthesized sub-expression."""
    expr = sample['metadata']['expression']
    inner = _find_innermost_parens(expr)
    if inner:
        try:
            result = eval(inner)
            if isinstance(result, float) and result == int(result):
                result = int(result)
            q = f"Evaluate the innermost parenthesized part of: {expr}\nThe innermost part is: ({inner}). What does it equal?"
            a = str(result)
            return q, a
        except Exception:
            pass
    a_val = rng.randint(-100, 100)
    b_val = rng.randint(-100, 100)
    op = rng.choice(['+', '-', '*'])
    result = eval(f"{a_val} {op} {b_val}")
    if isinstance(result, float) and result == int(result):
        result = int(result)
    q = f"What is ({a_val} {op} {b_val})?"
    a = str(result)
    return q, a


def atom_sign_handling(sample: dict, rng: random.Random) -> Tuple[str, str]:
    """L2: Determine the sign of a product/subtraction involving negatives."""
    scenario = rng.choice(['neg_times_neg', 'neg_times_pos', 'sub_neg'])
    if scenario == 'neg_times_neg':
        a_val = -rng.randint(1, 20)
        b_val = -rng.randint(1, 20)
        result = a_val * b_val
        q = f"What is {a_val} * {b_val}? (pay attention to signs)"
    elif scenario == 'neg_times_pos':
        a_val = -rng.randint(1, 20)
        b_val = rng.randint(1, 20)
        result = a_val * b_val
        q = f"What is {a_val} * {b_val}? (pay attention to signs)"
    else:
        a_val = rng.randint(1, 100)
        b_val = -rng.randint(1, 100)
        result = a_val - b_val
        q = f"What is {a_val} - {b_val}? (remember: subtracting a negative)"
    a = str(result)
    return q, a


# ===========================================================================
# COUNTDOWN atoms
# ===========================================================================

def countdown_number_inventory(sample: dict, rng: random.Random) -> Tuple[str, str]:
    """L1: List the available numbers."""
    numbers = sample['metadata']['numbers']
    q = f"List all available numbers (separated by commas): {sample['question'][:120]}"
    a = ", ".join(str(n) for n in sorted(numbers))
    return q, a


def countdown_target_reading(sample: dict, rng: random.Random) -> Tuple[str, str]:
    """L1: What is the target number?"""
    target = sample['metadata']['target']
    q = f"What is the target number in this problem? {sample['question'][:120]}"
    a = str(target)
    return q, a


def countdown_how_many_numbers(sample: dict, rng: random.Random) -> Tuple[str, str]:
    """L1: How many numbers are available?"""
    numbers = sample['metadata']['numbers']
    q = f"How many numbers are available to use? {sample['question'][:120]}"
    a = str(len(numbers))
    return q, a


def countdown_single_op(sample: dict, rng: random.Random) -> Tuple[str, str]:
    """L2: Evaluate a single operation on two of the available numbers."""
    numbers = sample['metadata']['numbers']
    if len(numbers) < 2:
        a_val, b_val = 10, 5
    else:
        pair = rng.sample(numbers, 2)
        a_val, b_val = pair[0], pair[1]
    op = rng.choice(['+', '-', '*'])
    result = eval(f"{a_val} {op} {b_val}")
    if isinstance(result, float) and result == int(result):
        result = int(result)
    q = f"Using numbers from the problem, what is {a_val} {op} {b_val}?"
    a = str(result)
    return q, a


def countdown_two_number_reachability(sample: dict, rng: random.Random) -> Tuple[str, str]:
    """L2: Can you reach the target using just two numbers and one operation?"""
    target = sample['metadata']['target']
    numbers = sample['metadata']['numbers']
    if len(numbers) < 2:
        q = f"Can you make {target} from 10 and 5 using one operation (+, -, *, /)? Answer Yes or No, and if Yes show the expression."
        a = "Yes, 10 - 5 = 5" if target == 5 else "No"
        return q, a

    for i in range(len(numbers)):
        for j in range(len(numbers)):
            if i == j:
                continue
            x, y = numbers[i], numbers[j]
            if x + y == target:
                q = f"Can you make {target} from {x} and {y} using one operation (+, -, *, /)? Answer Yes or No, and if Yes show the expression."
                a = f"Yes, {x} + {y} = {target}"
                return q, a
            if x - y == target:
                q = f"Can you make {target} from {x} and {y} using one operation (+, -, *, /)? Answer Yes or No, and if Yes show the expression."
                a = f"Yes, {x} - {y} = {target}"
                return q, a
            if x * y == target:
                q = f"Can you make {target} from {x} and {y} using one operation (+, -, *, /)? Answer Yes or No, and if Yes show the expression."
                a = f"Yes, {x} * {y} = {target}"
                return q, a
            if y != 0 and x / y == target and x % y == 0:
                q = f"Can you make {target} from {x} and {y} using one operation (+, -, *, /)? Answer Yes or No, and if Yes show the expression."
                a = f"Yes, {x} / {y} = {target}"
                return q, a

    pair = rng.sample(numbers, 2)
    q = f"Can you make {target} from {pair[0]} and {pair[1]} using one operation (+, -, *, /)? Answer Yes or No, and if Yes show the expression."
    a = "No"
    return q, a


def countdown_distance_to_target(sample: dict, rng: random.Random) -> Tuple[str, str]:
    """L2: How far is a partial result from the target?"""
    target = sample['metadata']['target']
    numbers = sample['metadata']['numbers']
    if len(numbers) < 2:
        partial = 10
    else:
        pair = rng.sample(numbers, 2)
        op = rng.choice(['+', '-'])
        partial = eval(f"{pair[0]} {op} {pair[1]}")
        if isinstance(partial, float) and partial == int(partial):
            partial = int(partial)
    distance = abs(target - partial)
    q = f"The target is {target}. You currently have {partial}. How far away are you from the target (absolute difference)?"
    a = str(distance)
    return q, a


# ===========================================================================
# SHORTEST_PATH atoms
# ===========================================================================

def _find_in_grid(matrix, char):
    """Find (row, col) of a character in the grid."""
    for r, row in enumerate(matrix):
        for c, cell in enumerate(row):
            if cell == char:
                return r, c
    return None, None


def _format_grid(matrix):
    """Format grid matrix as readable string."""
    return "\n".join(" ".join(row) for row in matrix)


def sp_locate_start(sample: dict, rng: random.Random) -> Tuple[str, str]:
    """L1: Find the starting position (*) in the grid."""
    matrix = sample['metadata']['matrix']
    r, c = _find_in_grid(matrix, '*')
    grid_str = _format_grid(matrix)
    q = f"In this grid, what are the row and column of the start position (*)? Rows and columns are 0-indexed.\n\n{grid_str}"
    a = f"row {r}, column {c}"
    return q, a


def sp_locate_goal(sample: dict, rng: random.Random) -> Tuple[str, str]:
    """L1: Find the goal position (#) in the grid."""
    matrix = sample['metadata']['matrix']
    r, c = _find_in_grid(matrix, '#')
    grid_str = _format_grid(matrix)
    q = f"In this grid, what are the row and column of the goal position (#)? Rows and columns are 0-indexed.\n\n{grid_str}"
    a = f"row {r}, column {c}"
    return q, a


def sp_read_cell(sample: dict, rng: random.Random) -> Tuple[str, str]:
    """L1: What is at a specific position in the grid?"""
    matrix = sample['metadata']['matrix']
    rows = len(matrix)
    cols = len(matrix[0])
    r = rng.randint(0, rows - 1)
    c = rng.randint(0, cols - 1)
    grid_str = _format_grid(matrix)
    cell = matrix[r][c]
    cell_names = {'O': 'O', 'X': 'X', '*': '*', '#': '#'}
    q = f"What is at row {r}, column {c} in this grid? Answer with the cell character (O, X, *, or #).\n\n{grid_str}"
    a = cell_names.get(cell, cell)
    return q, a


def sp_grid_dimensions(sample: dict, rng: random.Random) -> Tuple[str, str]:
    """L1: How many rows and columns does the grid have?"""
    matrix = sample['metadata']['matrix']
    rows = len(matrix)
    cols = len(matrix[0])
    grid_str = _format_grid(matrix)
    q = f"How many rows and columns does this grid have?\n\n{grid_str}"
    a = f"{rows} rows, {cols} columns"
    return q, a


def sp_valid_neighbors(sample: dict, rng: random.Random) -> Tuple[str, str]:
    """L2: List all non-blocked neighbors of a given cell."""
    matrix = sample['metadata']['matrix']
    rows = len(matrix)
    cols = len(matrix[0])
    # Pick a non-blocked cell
    open_cells = [(r, c) for r in range(rows) for c in range(cols) if matrix[r][c] != 'X']
    if not open_cells:
        return sp_read_cell(sample, rng)
    r, c = rng.choice(open_cells)
    grid_str = _format_grid(matrix)
    directions = [('up', -1, 0), ('down', 1, 0), ('left', 0, -1), ('right', 0, 1)]
    valid = []
    for name, dr, dc in directions:
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols and matrix[nr][nc] != 'X':
            valid.append(name)
    q = f"From row {r}, column {c} in this grid, which directions (up/down/left/right) lead to a non-blocked cell?\n\n{grid_str}"
    a = ", ".join(valid) if valid else "none"
    return q, a


def sp_is_move_legal(sample: dict, rng: random.Random) -> Tuple[str, str]:
    """L2: Is a specific move from a cell legal?"""
    matrix = sample['metadata']['matrix']
    rows = len(matrix)
    cols = len(matrix[0])
    open_cells = [(r, c) for r in range(rows) for c in range(cols) if matrix[r][c] != 'X']
    if not open_cells:
        return sp_read_cell(sample, rng)
    r, c = rng.choice(open_cells)
    directions = [('up', -1, 0), ('down', 1, 0), ('left', 0, -1), ('right', 0, 1)]
    dir_name, dr, dc = rng.choice(directions)
    nr, nc = r + dr, c + dc
    legal = 0 <= nr < rows and 0 <= nc < cols and matrix[nr][nc] != 'X'
    grid_str = _format_grid(matrix)
    q = f"From row {r}, column {c}, is moving {dir_name} legal (not blocked and within bounds)? Answer Yes or No.\n\n{grid_str}"
    a = "Yes" if legal else "No"
    return q, a


def sp_direction_to_goal(sample: dict, rng: random.Random) -> Tuple[str, str]:
    """L2: Which direction (up/down/left/right) moves closer to the goal?"""
    matrix = sample['metadata']['matrix']
    sr, sc = _find_in_grid(matrix, '*')
    gr, gc = _find_in_grid(matrix, '#')
    if sr is None or gr is None:
        return sp_read_cell(sample, rng)
    options = []
    if gr < sr:
        options.append("up")
    if gr > sr:
        options.append("down")
    if gc < sc:
        options.append("left")
    if gc > sc:
        options.append("right")
    grid_str = _format_grid(matrix)
    q = f"The start (*) is at row {sr}, col {sc}. The goal (#) is at row {gr}, col {gc}. Which direction(s) move closer to the goal (ignoring walls)?"
    a = ", ".join(options) if options else "already at goal"
    return q, a


def sp_execute_path(sample: dict, rng: random.Random) -> Tuple[str, str]:
    """L2-long: Execute a sequence of moves and report the final position.

    Produces multi-step reasoning answers to avoid GRPO compression.
    """
    matrix = sample['metadata']['matrix']
    rows = len(matrix)
    cols = len(matrix[0])
    open_cells = [(r, c) for r in range(rows) for c in range(cols) if matrix[r][c] != 'X']
    if not open_cells:
        return sp_read_cell(sample, rng)

    start_r, start_c = rng.choice(open_cells)
    dir_map = {'up': (-1, 0), 'down': (1, 0), 'left': (0, -1), 'right': (0, 1)}
    dir_names = list(dir_map.keys())

    num_moves = rng.randint(2, 4)
    moves = []
    cr, cc = start_r, start_c
    trace_lines = []
    for _ in range(num_moves):
        d = rng.choice(dir_names)
        dr, dc = dir_map[d]
        nr, nc = cr + dr, cc + dc
        if 0 <= nr < rows and 0 <= nc < cols and matrix[nr][nc] != 'X':
            trace_lines.append(f"Move {d} from ({cr},{cc}) to ({nr},{nc}) — cell is {matrix[nr][nc]}")
            cr, cc = nr, nc
        else:
            trace_lines.append(f"Move {d} from ({cr},{cc}) — blocked or out of bounds, stay at ({cr},{cc})")
        moves.append(d)

    grid_str = _format_grid(matrix)
    move_str = ", ".join(moves)
    q = (f"Starting at row {start_r}, column {start_c}, execute these moves in order: {move_str}. "
         f"If a move is blocked (wall or out of bounds), stay in place. "
         f"Report each step and your final position as 'row R, column C'.\n\n{grid_str}")
    trace = "; ".join(trace_lines)
    a = f"{trace}. Final position: row {cr}, column {cc}"
    return q, a


def sp_trace_and_report(sample: dict, rng: random.Random) -> Tuple[str, str]:
    """L2-long: Trace a path from start and report what each cell contains.

    Multi-step trace: the model must read the grid at each position.
    """
    matrix = sample['metadata']['matrix']
    rows = len(matrix)
    cols = len(matrix[0])
    sr, sc = _find_in_grid(matrix, '*')
    if sr is None:
        return sp_read_cell(sample, rng)

    dir_map = {'up': (-1, 0), 'down': (1, 0), 'left': (0, -1), 'right': (0, 1)}
    dir_names = list(dir_map.keys())

    num_moves = rng.randint(2, 4)
    moves = []
    cr, cc = sr, sc
    report_lines = []
    for step in range(1, num_moves + 1):
        d = rng.choice(dir_names)
        moves.append(d)
        dr, dc = dir_map[d]
        nr, nc = cr + dr, cc + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            cell = matrix[nr][nc]
            if cell == 'X':
                report_lines.append(f"Step {step}: {d} to ({nr},{nc}) is X (blocked)")
            else:
                report_lines.append(f"Step {step}: {d} to ({nr},{nc}) is {cell} (open)")
                cr, cc = nr, nc
        else:
            report_lines.append(f"Step {step}: {d} from ({cr},{cc}) is out of bounds")

    grid_str = _format_grid(matrix)
    move_str = ", ".join(moves)
    q = (f"Starting from * at row {sr}, column {sc}, trace this path: {move_str}. "
         f"For each step, report the target cell contents and whether you moved or were blocked. "
         f"End with your final position.\n\n{grid_str}")
    report = "; ".join(report_lines)
    a = f"{report}. Final position: row {cr}, column {cc}"
    return q, a


def sp_count_walls_on_path(sample: dict, rng: random.Random) -> Tuple[str, str]:
    """L2-long: Count how many walls block a straight-line path between two cells."""
    matrix = sample['metadata']['matrix']
    rows = len(matrix)
    cols = len(matrix[0])

    if rng.random() < 0.5 and rows > 1:
        # Horizontal scan across a row
        r = rng.randint(0, rows - 1)
        wall_count = sum(1 for c in range(cols) if matrix[r][c] == 'X')
        cells = [f"({r},{c})={matrix[r][c]}" for c in range(cols)]
        grid_str = _format_grid(matrix)
        q = (f"Scan row {r} from left to right. For each cell, report its contents. "
             f"How many walls (X) are in row {r}?\n\n{grid_str}")
        cell_report = ", ".join(cells)
        a = f"Row {r}: {cell_report}. Wall count: {wall_count}"
    else:
        # Vertical scan down a column
        c = rng.randint(0, cols - 1)
        wall_count = sum(1 for r in range(rows) if matrix[r][c] == 'X')
        cells = [f"({r},{c})={matrix[r][c]}" for r in range(rows)]
        grid_str = _format_grid(matrix)
        q = (f"Scan column {c} from top to bottom. For each cell, report its contents. "
             f"How many walls (X) are in column {c}?\n\n{grid_str}")
        cell_report = ", ".join(cells)
        a = f"Column {c}: {cell_report}. Wall count: {wall_count}"
    return q, a


# ===========================================================================
# SHORTEST_PATH anti-false-infeasible atoms (L2_feasibility)
# ===========================================================================

def _bfs_path(matrix, start, goal):
    """BFS returning list of direction strings, or None if infeasible."""
    from collections import deque
    rows, cols = len(matrix), len(matrix[0])
    sr, sc = start
    gr, gc = goal
    queue = deque([(sr, sc, [])])
    visited = {(sr, sc)}
    dir_map = [('up', -1, 0), ('down', 1, 0), ('left', 0, -1), ('right', 0, 1)]
    while queue:
        r, c, path = queue.popleft()
        if r == gr and c == gc:
            return path
        for name, dr, dc in dir_map:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited and matrix[nr][nc] != 'X':
                visited.add((nr, nc))
                queue.append((nr, nc, path + [name]))
    return None


def sp_path_exists(sample: dict, rng: random.Random) -> Tuple[str, str]:
    """L2_feasibility: Is there a path from * to #? Requires checking connectivity."""
    matrix = sample['metadata']['matrix']
    sr, sc = _find_in_grid(matrix, '*')
    gr, gc = _find_in_grid(matrix, '#')
    if sr is None or gr is None:
        return sp_read_cell(sample, rng)
    path = _bfs_path(matrix, (sr, sc), (gr, gc))
    grid_str = _format_grid(matrix)
    feasible = path is not None
    q = (f"Is there a path from * to # in this grid, moving only through O cells "
         f"(up/down/left/right, no diagonals)? Answer Yes or No, then briefly explain why.\n\n{grid_str}")
    if feasible:
        a = f"Yes, a path exists. One route: {', '.join(path[:6])}{'...' if len(path) > 6 else ''}"
    else:
        a = "No, there is no path. The goal is completely blocked by walls."
    return q, a


def sp_find_any_path(sample: dict, rng: random.Random) -> Tuple[str, str]:
    """L2_feasibility: Find ANY valid path (not necessarily shortest). Multi-step answer."""
    matrix = sample['metadata']['matrix']
    sr, sc = _find_in_grid(matrix, '*')
    gr, gc = _find_in_grid(matrix, '#')
    if sr is None or gr is None:
        return sp_read_cell(sample, rng)
    path = _bfs_path(matrix, (sr, sc), (gr, gc))
    if path is None:
        return sp_path_exists(sample, rng)
    grid_str = _format_grid(matrix)
    # Trace the path step by step
    cr, cc = sr, sc
    trace = []
    for move in path:
        dr, dc = {'up': (-1,0), 'down': (1,0), 'left': (0,-1), 'right': (0,1)}[move]
        nr, nc = cr + dr, cc + dc
        trace.append(f"{move} to ({nr},{nc})")
        cr, cc = nr, nc
    q = (f"Find a valid path from * at ({sr},{sc}) to # at ({gr},{gc}). "
         f"List each move and the cell you reach. The path does not need to be shortest.\n\n{grid_str}")
    trace_str = "; ".join(trace)
    a = f"Path: {trace_str}. Directions: {' '.join(path)}"
    return q, a


def sp_start_neighbors(sample: dict, rng: random.Random) -> Tuple[str, str]:
    """L2_feasibility: List all non-blocked neighbors of * with their contents."""
    matrix = sample['metadata']['matrix']
    rows, cols = len(matrix), len(matrix[0])
    sr, sc = _find_in_grid(matrix, '*')
    if sr is None:
        return sp_read_cell(sample, rng)
    grid_str = _format_grid(matrix)
    directions = [('up', -1, 0), ('down', 1, 0), ('left', 0, -1), ('right', 0, 1)]
    report = []
    open_count = 0
    for name, dr, dc in directions:
        nr, nc = sr + dr, sc + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            cell = matrix[nr][nc]
            status = "open" if cell != 'X' else "blocked (wall)"
            report.append(f"{name}: ({nr},{nc}) is {cell} — {status}")
            if cell != 'X':
                open_count += 1
        else:
            report.append(f"{name}: out of bounds")
    q = (f"The start * is at ({sr},{sc}). List each neighbor (up/down/left/right), "
         f"what it contains, and whether it's open or blocked.\n\n{grid_str}")
    report_str = "; ".join(report)
    a = f"{report_str}. Open neighbors: {open_count}"
    return q, a


def sp_bfs_first_steps(sample: dict, rng: random.Random) -> Tuple[str, str]:
    """L2_feasibility: What are the first 2-3 moves of a valid path from * to #?"""
    matrix = sample['metadata']['matrix']
    sr, sc = _find_in_grid(matrix, '*')
    gr, gc = _find_in_grid(matrix, '#')
    if sr is None or gr is None:
        return sp_read_cell(sample, rng)
    path = _bfs_path(matrix, (sr, sc), (gr, gc))
    if path is None:
        return sp_path_exists(sample, rng)
    grid_str = _format_grid(matrix)
    n_steps = min(3, len(path))
    first_moves = path[:n_steps]
    cr, cc = sr, sc
    trace = []
    for move in first_moves:
        dr, dc = {'up': (-1,0), 'down': (1,0), 'left': (0,-1), 'right': (0,1)}[move]
        nr, nc = cr + dr, cc + dc
        trace.append(f"{move} to ({nr},{nc}) which is {matrix[nr][nc]}")
        cr, cc = nr, nc
    q = (f"Start at * ({sr},{sc}), goal is # ({gr},{gc}). "
         f"What are the first {n_steps} moves of a valid path toward the goal? "
         f"Report each move and what cell you reach.\n\n{grid_str}")
    trace_str = "; ".join(trace)
    a = f"First {n_steps} moves: {trace_str}. Directions so far: {' '.join(first_moves)}"
    return q, a


# ===========================================================================
# SYLLOGISM atoms
# ===========================================================================

def syllogism_premise1_subject(sample: dict, rng: random.Random) -> Tuple[str, str]:
    """L1: Extract the subject of premise 1."""
    metadata = sample['metadata']
    premise1 = metadata.get('premise1', '')
    # Parse "No/All/Some X are Y" format
    parts = premise1.split()
    if len(parts) >= 4:
        # Find the word before "are"
        try:
            are_idx = parts.index('are')
            subject = ' '.join(parts[1:are_idx])
        except ValueError:
            subject = parts[1] if len(parts) > 1 else "unknown"
    else:
        subject = "unknown"
    q = f"In this syllogism, what is the subject (first term) of premise 1?\n\nPremise 1: {premise1}"
    a = subject
    return q, a


def syllogism_premise2_subject(sample: dict, rng: random.Random) -> Tuple[str, str]:
    """L1: Extract the subject of premise 2."""
    metadata = sample['metadata']
    premise2 = metadata.get('premise2', '')
    parts = premise2.split()
    if len(parts) >= 4:
        try:
            are_idx = parts.index('are')
            subject = ' '.join(parts[1:are_idx])
        except ValueError:
            subject = parts[1] if len(parts) > 1 else "unknown"
    else:
        subject = "unknown"
    q = f"In this syllogism, what is the subject (first term) of premise 2?\n\nPremise 2: {premise2}"
    a = subject
    return q, a


def syllogism_quantifier_type(sample: dict, rng: random.Random) -> Tuple[str, str]:
    """L1: Identify the quantifier type of a randomly selected premise."""
    metadata = sample['metadata']
    premise_num = rng.choice([1, 2])
    premise = metadata.get(f'premise{premise_num}', '')
    premise_lower = premise.lower()
    if premise_lower.startswith('no '):
        quantifier = 'universal negative'
    elif premise_lower.startswith('all '):
        quantifier = 'universal affirmative'
    elif premise_lower.startswith('some ') and ' not ' in premise_lower:
        quantifier = 'particular negative'
    elif premise_lower.startswith('some '):
        quantifier = 'particular affirmative'
    else:
        quantifier = 'unknown'
    q = f"What type of quantifier is used in this statement? Answer with one of: universal affirmative, universal negative, particular affirmative, particular negative.\n\nStatement: {premise}"
    a = quantifier
    return q, a


def syllogism_conclusion_predicate(sample: dict, rng: random.Random) -> Tuple[str, str]:
    """L1: Extract the predicate of the conclusion."""
    metadata = sample['metadata']
    conclusion = metadata.get('conclusion', '')
    parts = conclusion.split()
    if 'are' in parts:
        are_idx = parts.index('are')
        predicate = ' '.join(parts[are_idx+1:])
    else:
        predicate = parts[-1] if parts else "unknown"
    q = f"What is the predicate (second term) of this conclusion?\n\nConclusion: {conclusion}"
    a = predicate
    return q, a


def syllogism_single_hop_inference(sample: dict, rng: random.Random) -> Tuple[str, str]:
    """L2: Given one premise is true, what directly follows?"""
    metadata = sample['metadata']
    premise1 = metadata.get('premise1', '')
    premise2 = metadata.get('premise2', '')
    # Parse premise 2: "All X are Y" means if something is X, it is Y
    p2_lower = premise2.lower()
    if p2_lower.startswith('all '):
        parts = premise2.split()
        if 'are' in parts:
            are_idx = parts.index('are')
            subject = ' '.join(parts[1:are_idx])
            predicate = ' '.join(parts[are_idx+1:])
            q = f"Given that '{premise2}' is true, if something is a {subject}, what can we conclude about it?"
            a = f"It is a {predicate}"
            return q, a
    # Fallback: simpler question
    q = f"Given the statement '{premise2}' is true, does it tell us anything about all members of the subject category? Answer Yes or No."
    a = "Yes" if p2_lower.startswith('all ') else "No"
    return q, a


def syllogism_negation(sample: dict, rng: random.Random) -> Tuple[str, str]:
    """L2: What is the negation of a given statement?"""
    metadata = sample['metadata']
    premise = metadata.get(f'premise{rng.choice([1,2])}', '')
    premise_lower = premise.lower()
    parts = premise.split()
    if 'are' in parts:
        are_idx = parts.index('are')
        subject = ' '.join(parts[1:are_idx])
        predicate = ' '.join(parts[are_idx+1:])
        if premise_lower.startswith('all '):
            negation = f"Some {subject} are not {predicate}"
        elif premise_lower.startswith('no '):
            negation = f"Some {subject} are {predicate}"
        elif premise_lower.startswith('some ') and ' not ' not in premise_lower:
            negation = f"No {subject} are {predicate}"
        else:
            negation = f"All {subject} are {predicate}"
    else:
        negation = "Cannot determine"
    q = f"What is the logical negation of this statement?\n\nStatement: {premise}"
    a = negation
    return q, a


def syllogism_validity_check(sample: dict, rng: random.Random) -> Tuple[str, str]:
    """L2: Is a specific conclusion valid given the premises?"""
    metadata = sample['metadata']
    is_valid = metadata.get('is_valid', False)
    premise1 = metadata.get('premise1', '')
    premise2 = metadata.get('premise2', '')
    conclusion = metadata.get('conclusion', '')
    q = f"Given these premises, does the conclusion logically follow?\n\nPremise 1: {premise1}\nPremise 2: {premise2}\nConclusion: {conclusion}\n\nAnswer Yes or No."
    a = "Yes" if is_valid else "No"
    return q, a


# ===========================================================================
# KNIGHTS_KNAVES atoms
# ===========================================================================

def kk_count_inhabitants(sample: dict, rng: random.Random) -> Tuple[str, str]:
    """L1: How many inhabitants are there?"""
    metadata = sample['metadata']
    names = metadata.get('names', [])
    q = f"How many inhabitants are mentioned in this puzzle?\n\n{sample['question'][:300]}"
    a = str(len(names))
    return q, a


def kk_list_names(sample: dict, rng: random.Random) -> Tuple[str, str]:
    """L1: List the names of all inhabitants."""
    metadata = sample['metadata']
    names = metadata.get('names', [])
    q = f"List all the names of inhabitants mentioned in this puzzle, separated by commas.\n\n{sample['question'][:300]}"
    a = ", ".join(names)
    return q, a


def kk_speaker_identification(sample: dict, rng: random.Random) -> Tuple[str, str]:
    """L1: Who made a specific statement?"""
    metadata = sample['metadata']
    names = metadata.get('names', [])
    if not names:
        return kk_count_inhabitants(sample, rng)
    speaker_idx = rng.randint(0, len(names) - 1)
    speaker = names[speaker_idx]
    q = f"In this puzzle, does {speaker} make a statement? Answer Yes or No based on the problem.\n\n{sample['question'][:400]}"
    # All named inhabitants make statements in knights_knaves
    a = "Yes"
    return q, a


def kk_terminology(sample: dict, rng: random.Random) -> Tuple[str, str]:
    """L1: What terms are used for truth-tellers and liars?"""
    metadata = sample['metadata']
    terms = metadata.get('knight_knave_terms', {})
    knight_term = terms.get('knight', 'knight')
    knave_term = terms.get('knave', 'knave')
    which = rng.choice(['truth', 'lie'])
    if which == 'truth':
        q = f"In this puzzle, what is the term used for inhabitants who always tell the truth?\n\n{sample['question'][:200]}"
        a = knight_term
    else:
        q = f"In this puzzle, what is the term used for inhabitants who always lie?\n\n{sample['question'][:200]}"
        a = knave_term
    return q, a


def kk_statement_about(sample: dict, rng: random.Random) -> Tuple[str, str]:
    """L2: What does person X's statement claim about person Y?"""
    metadata = sample['metadata']
    names = metadata.get('names', [])
    statements = metadata.get('statements', [])
    terms = metadata.get('knight_knave_terms', {})
    knight_term = terms.get('knight', 'sage')
    knave_term = terms.get('knave', 'fool')

    if len(names) < 2 or not statements:
        return kk_count_inhabitants(sample, rng)

    # Find a statement that refers to another person
    for i, stmt in enumerate(statements):
        if isinstance(stmt, list) and len(stmt) >= 2:
            if stmt[0] == 'lying' and isinstance(stmt[1], int):
                speaker = names[i]
                target = names[stmt[1]]
                q = f"In this puzzle, {speaker} makes a statement about {target}. Does {speaker} claim that {target} is a {knight_term} or a {knave_term}?"
                a = knave_term
                return q, a
            elif stmt[0] == 'telling-truth' and isinstance(stmt[1], int):
                speaker = names[i]
                target = names[stmt[1]]
                q = f"In this puzzle, {speaker} makes a statement about {target}. Does {speaker} claim that {target} is a {knight_term} or a {knave_term}?"
                a = knight_term
                return q, a

    # Fallback
    return kk_terminology(sample, rng)


def kk_if_sage_then(sample: dict, rng: random.Random) -> Tuple[str, str]:
    """L2: If person X is a sage, what must be true about their statement?"""
    metadata = sample['metadata']
    names = metadata.get('names', [])
    terms = metadata.get('knight_knave_terms', {})
    knight_term = terms.get('knight', 'sage')

    if not names:
        return kk_count_inhabitants(sample, rng)

    person = rng.choice(names)
    q = f"If {person} is a {knight_term} (always tells the truth), then their statement must be:"
    a = "true"
    return q, a


def kk_if_fool_then(sample: dict, rng: random.Random) -> Tuple[str, str]:
    """L2: If person X is a fool, what must be true about their statement?"""
    metadata = sample['metadata']
    names = metadata.get('names', [])
    terms = metadata.get('knight_knave_terms', {})
    knave_term = terms.get('knave', 'fool')

    if not names:
        return kk_count_inhabitants(sample, rng)

    person = rng.choice(names)
    q = f"If {person} is a {knave_term} (always lies), then their statement must be:"
    a = "false"
    return q, a


def kk_consistency_check(sample: dict, rng: random.Random) -> Tuple[str, str]:
    """L2: Is a hypothetical assignment consistent with a statement?"""
    metadata = sample['metadata']
    names = metadata.get('names', [])
    solution = metadata.get('solution', [])
    terms = metadata.get('knight_knave_terms', {})
    knight_term = terms.get('knight', 'sage')
    knave_term = terms.get('knave', 'fool')

    if not names or not solution:
        return kk_count_inhabitants(sample, rng)

    # Use the actual solution
    person_idx = rng.randint(0, len(names) - 1)
    person = names[person_idx]
    is_knight = solution[person_idx]
    role = knight_term if is_knight else knave_term

    q = f"Based on the solution to this puzzle, is {person} a {knight_term} or a {knave_term}?\n\n{sample['question'][:300]}"
    a = role
    return q, a


# ===========================================================================
# MAZE atoms (uses '>' for wall, 'e' for path, '3' for start, 'z' for goal)
# ===========================================================================

def _maze_find_char(grid, char):
    """Find (row, col) of a character in the maze grid."""
    for r, row in enumerate(grid):
        for c, cell in enumerate(row):
            if cell == char:
                return r, c
    return None, None


def _maze_format_grid(grid):
    """Format maze grid as string."""
    return "\n".join(grid)


def maze_locate_start(sample: dict, rng: random.Random) -> Tuple[str, str]:
    """L1: Find the starting position ('3') in the maze."""
    metadata = sample['metadata']
    grid = metadata.get('grid', [])
    start_char = metadata.get('start', '3')
    r, c = _maze_find_char(grid, start_char)
    grid_str = _maze_format_grid(grid)
    q = f"In this maze, what are the row and column of the start position ('{start_char}')? Rows and columns are 0-indexed.\n\n{grid_str}"
    a = f"row {r}, column {c}"
    return q, a


def maze_locate_goal(sample: dict, rng: random.Random) -> Tuple[str, str]:
    """L1: Find the goal position ('z') in the maze."""
    metadata = sample['metadata']
    grid = metadata.get('grid', [])
    goal_char = metadata.get('goal', 'z')
    r, c = _maze_find_char(grid, goal_char)
    grid_str = _maze_format_grid(grid)
    q = f"In this maze, what are the row and column of the goal position ('{goal_char}')? Rows and columns are 0-indexed.\n\n{grid_str}"
    a = f"row {r}, column {c}"
    return q, a


def maze_read_cell(sample: dict, rng: random.Random) -> Tuple[str, str]:
    """L1: What is at a specific position in the maze?"""
    metadata = sample['metadata']
    grid = metadata.get('grid', [])
    wall_char = metadata.get('wall', '>')
    path_char = metadata.get('path', 'e')
    rows = len(grid)
    cols = len(grid[0]) if grid else 0
    r = rng.randint(0, rows - 1)
    c = rng.randint(0, cols - 1)
    cell = grid[r][c]
    grid_str = _maze_format_grid(grid)
    q = f"What is at row {r}, column {c} in this maze? Answer with the character you see.\n\n{grid_str}"
    a = cell
    return q, a


def maze_grid_size(sample: dict, rng: random.Random) -> Tuple[str, str]:
    """L1: How large is the maze?"""
    metadata = sample['metadata']
    grid_size = metadata.get('grid_size', 0)
    grid = metadata.get('grid', [])
    rows = len(grid)
    cols = len(grid[0]) if grid else 0
    q = f"How many rows and columns does this maze have?\n\n{_maze_format_grid(grid)}"
    a = f"{rows} rows, {cols} columns"
    return q, a


def maze_valid_moves(sample: dict, rng: random.Random) -> Tuple[str, str]:
    """L2: From a given position, which directions are passable?"""
    metadata = sample['metadata']
    grid = metadata.get('grid', [])
    wall_char = metadata.get('wall', '>')
    rows = len(grid)
    cols = len(grid[0]) if grid else 0

    # Find a passable cell
    passable_cells = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != wall_char:
                passable_cells.append((r, c))

    if not passable_cells:
        return maze_read_cell(sample, rng)

    r, c = rng.choice(passable_cells)
    directions = [('up', -1, 0), ('down', 1, 0), ('left', 0, -1), ('right', 0, 1)]
    valid = []
    for name, dr, dc in directions:
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] != wall_char:
            valid.append(name)

    grid_str = _maze_format_grid(grid)
    q = f"From row {r}, column {c} in this maze, which directions (up/down/left/right) lead to a passable cell (not a wall '{wall_char}')?\n\n{grid_str}"
    a = ", ".join(valid) if valid else "none"
    return q, a


def maze_is_move_valid(sample: dict, rng: random.Random) -> Tuple[str, str]:
    """L2: Is a specific move from a given position valid?"""
    metadata = sample['metadata']
    grid = metadata.get('grid', [])
    wall_char = metadata.get('wall', '>')
    rows = len(grid)
    cols = len(grid[0]) if grid else 0

    passable_cells = [(r, c) for r in range(rows) for c in range(cols) if grid[r][c] != wall_char]
    if not passable_cells:
        return maze_read_cell(sample, rng)

    r, c = rng.choice(passable_cells)
    directions = [('up', -1, 0), ('down', 1, 0), ('left', 0, -1), ('right', 0, 1)]
    dir_name, dr, dc = rng.choice(directions)
    nr, nc = r + dr, c + dc
    valid = 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] != wall_char

    grid_str = _maze_format_grid(grid)
    q = f"From row {r}, column {c}, is moving {dir_name} valid (not into a wall and within bounds)? Answer Yes or No.\n\n{grid_str}"
    a = "Yes" if valid else "No"
    return q, a


def maze_shortest_path_length(sample: dict, rng: random.Random) -> Tuple[str, str]:
    """L2: What is the shortest path length (given in metadata)?"""
    metadata = sample['metadata']
    path_length = metadata.get('shortest_path_length', 0)
    grid_str = _maze_format_grid(metadata.get('grid', []))
    q = f"This maze has a shortest path from start to goal. What is the length of that shortest path?\n\n{grid_str}"
    a = str(path_length)
    return q, a


# ===========================================================================
# MINI_SUDOKU atoms
# ===========================================================================

def sudoku_read_cell(sample: dict, rng: random.Random) -> Tuple[str, str]:
    """L1: What number is in a specific cell?"""
    metadata = sample['metadata']
    puzzle = metadata.get('puzzle', [])
    if not puzzle:
        return ("What is 2+2?", "4")  # fallback
    rows = len(puzzle)
    cols = len(puzzle[0])
    r = rng.randint(0, rows - 1)
    c = rng.randint(0, cols - 1)
    val = puzzle[r][c]
    puzzle_str = "\n".join(" ".join(str(x) if x != 0 else "_" for x in row) for row in puzzle)
    q = f"What number is in row {r+1}, column {c+1} of this puzzle? (Answer with the number, or '_' if empty)\n\n{puzzle_str}"
    a = str(val) if val != 0 else "_"
    return q, a


def sudoku_count_empty(sample: dict, rng: random.Random) -> Tuple[str, str]:
    """L1: How many empty cells are in the puzzle?"""
    metadata = sample['metadata']
    num_empty = metadata.get('num_empty', 0)
    puzzle = metadata.get('puzzle', [])
    puzzle_str = "\n".join(" ".join(str(x) if x != 0 else "_" for x in row) for row in puzzle)
    q = f"How many empty cells (marked with '_') are in this puzzle?\n\n{puzzle_str}"
    a = str(num_empty)
    return q, a


def sudoku_row_values(sample: dict, rng: random.Random) -> Tuple[str, str]:
    """L1: What values are already placed in a specific row?"""
    metadata = sample['metadata']
    puzzle = metadata.get('puzzle', [])
    if not puzzle:
        return ("What is 2+2?", "4")
    row_idx = rng.randint(0, len(puzzle) - 1)
    row = puzzle[row_idx]
    values = [str(x) for x in row if x != 0]
    puzzle_str = "\n".join(" ".join(str(x) if x != 0 else "_" for x in r) for r in puzzle)
    q = f"What numbers are already placed in row {row_idx+1}? List them separated by commas.\n\n{puzzle_str}"
    a = ", ".join(values) if values else "none"
    return q, a


def sudoku_col_values(sample: dict, rng: random.Random) -> Tuple[str, str]:
    """L1: What values are already placed in a specific column?"""
    metadata = sample['metadata']
    puzzle = metadata.get('puzzle', [])
    if not puzzle:
        return ("What is 2+2?", "4")
    col_idx = rng.randint(0, len(puzzle[0]) - 1)
    values = [str(puzzle[r][col_idx]) for r in range(len(puzzle)) if puzzle[r][col_idx] != 0]
    puzzle_str = "\n".join(" ".join(str(x) if x != 0 else "_" for x in row) for row in puzzle)
    q = f"What numbers are already placed in column {col_idx+1}? List them separated by commas.\n\n{puzzle_str}"
    a = ", ".join(values) if values else "none"
    return q, a


def sudoku_missing_in_row(sample: dict, rng: random.Random) -> Tuple[str, str]:
    """L2: What numbers are missing from a specific row?"""
    metadata = sample['metadata']
    puzzle = metadata.get('puzzle', [])
    if not puzzle:
        return ("What is 2+2?", "4")
    row_idx = rng.randint(0, len(puzzle) - 1)
    row = puzzle[row_idx]
    present = set(x for x in row if x != 0)
    all_nums = set(range(1, len(puzzle) + 1))  # 1-4 for 4x4
    missing = sorted(all_nums - present)
    puzzle_str = "\n".join(" ".join(str(x) if x != 0 else "_" for x in r) for r in puzzle)
    q = f"What numbers (1-{len(puzzle)}) are missing from row {row_idx+1}? List them separated by commas.\n\n{puzzle_str}"
    a = ", ".join(str(x) for x in missing) if missing else "none"
    return q, a


def sudoku_missing_in_col(sample: dict, rng: random.Random) -> Tuple[str, str]:
    """L2: What numbers are missing from a specific column?"""
    metadata = sample['metadata']
    puzzle = metadata.get('puzzle', [])
    if not puzzle:
        return ("What is 2+2?", "4")
    col_idx = rng.randint(0, len(puzzle[0]) - 1)
    present = set(puzzle[r][col_idx] for r in range(len(puzzle)) if puzzle[r][col_idx] != 0)
    all_nums = set(range(1, len(puzzle) + 1))
    missing = sorted(all_nums - present)
    puzzle_str = "\n".join(" ".join(str(x) if x != 0 else "_" for x in row) for row in puzzle)
    q = f"What numbers (1-{len(puzzle)}) are missing from column {col_idx+1}? List them separated by commas.\n\n{puzzle_str}"
    a = ", ".join(str(x) for x in missing) if missing else "none"
    return q, a


def sudoku_can_place(sample: dict, rng: random.Random) -> Tuple[str, str]:
    """L2: Can a specific number be placed in a specific empty cell?"""
    metadata = sample['metadata']
    puzzle = metadata.get('puzzle', [])
    solution = metadata.get('solution', [])
    if not puzzle or not solution:
        return ("What is 2+2?", "4")

    # Find empty cells
    empty_cells = [(r, c) for r in range(len(puzzle)) for c in range(len(puzzle[0])) if puzzle[r][c] == 0]
    if not empty_cells:
        return sudoku_count_empty(sample, rng)

    r, c = rng.choice(empty_cells)
    correct_val = solution[r][c]
    test_val = rng.choice([correct_val, rng.randint(1, len(puzzle))])

    # Check if test_val can be placed (not in row, col, or 2x2 box)
    row_vals = set(puzzle[r])
    col_vals = set(puzzle[i][c] for i in range(len(puzzle)))
    # 2x2 box for 4x4 sudoku
    box_r, box_c = (r // 2) * 2, (c // 2) * 2
    box_vals = set(puzzle[box_r + i][box_c + j] for i in range(2) for j in range(2))

    can_place = test_val not in row_vals and test_val not in col_vals and test_val not in box_vals

    puzzle_str = "\n".join(" ".join(str(x) if x != 0 else "_" for x in row) for row in puzzle)
    q = f"Can the number {test_val} be placed in row {r+1}, column {c+1}? Consider row, column, and 2x2 box constraints. Answer Yes or No.\n\n{puzzle_str}"
    a = "Yes" if can_place else "No"
    return q, a


def sudoku_only_option(sample: dict, rng: random.Random) -> Tuple[str, str]:
    """L2: Find a cell where only one number can go."""
    metadata = sample['metadata']
    puzzle = metadata.get('puzzle', [])
    solution = metadata.get('solution', [])
    if not puzzle or not solution:
        return ("What is 2+2?", "4")

    n = len(puzzle)
    # Find cells with only one possible value
    for r in range(n):
        for c in range(n):
            if puzzle[r][c] == 0:
                row_vals = set(x for x in puzzle[r] if x != 0)
                col_vals = set(puzzle[i][c] for i in range(n) if puzzle[i][c] != 0)
                box_r, box_c = (r // 2) * 2, (c // 2) * 2
                box_vals = set(puzzle[box_r + i][box_c + j] for i in range(2) for j in range(2) if puzzle[box_r + i][box_c + j] != 0)
                taken = row_vals | col_vals | box_vals
                possible = set(range(1, n + 1)) - taken
                if len(possible) == 1:
                    puzzle_str = "\n".join(" ".join(str(x) if x != 0 else "_" for x in row) for row in puzzle)
                    q = f"In row {r+1}, column {c+1}, only one number can be placed based on Sudoku rules. What is that number?\n\n{puzzle_str}"
                    a = str(list(possible)[0])
                    return q, a

    # Fallback if no single-option cell found
    return sudoku_missing_in_row(sample, rng)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

SYLLOGISM_ATOMS = {
    'L1': {
        'premise1_subject': syllogism_premise1_subject,
        'premise2_subject': syllogism_premise2_subject,
        'quantifier_type': syllogism_quantifier_type,
        'conclusion_predicate': syllogism_conclusion_predicate,
    },
    'L2': {
        'single_hop_inference': syllogism_single_hop_inference,
        'negation': syllogism_negation,
        'validity_check': syllogism_validity_check,
    },
}

KNIGHTS_KNAVES_ATOMS = {
    'L1': {
        'count_inhabitants': kk_count_inhabitants,
        'list_names': kk_list_names,
        'speaker_identification': kk_speaker_identification,
        'terminology': kk_terminology,
    },
    'L2': {
        'statement_about': kk_statement_about,
        'if_sage_then': kk_if_sage_then,
        'if_fool_then': kk_if_fool_then,
        'consistency_check': kk_consistency_check,
    },
}

MAZE_ATOMS = {
    'L1': {
        'locate_start': maze_locate_start,
        'locate_goal': maze_locate_goal,
        'read_cell': maze_read_cell,
        'grid_size': maze_grid_size,
    },
    'L2': {
        'valid_moves': maze_valid_moves,
        'is_move_valid': maze_is_move_valid,
        'shortest_path_length': maze_shortest_path_length,
    },
}

MINI_SUDOKU_ATOMS = {
    'L1': {
        'read_cell': sudoku_read_cell,
        'count_empty': sudoku_count_empty,
        'row_values': sudoku_row_values,
        'col_values': sudoku_col_values,
    },
    'L2': {
        'missing_in_row': sudoku_missing_in_row,
        'missing_in_col': sudoku_missing_in_col,
        'can_place': sudoku_can_place,
        'only_option': sudoku_only_option,
    },
}

ARITHMETIC_ATOMS = {
    'L1': {
        'number_extraction': atom_number_extraction,
        'operator_identification': atom_operator_identification,
        'term_count': atom_term_count,
        'has_parentheses': atom_has_parentheses,
    },
    'L2': {
        'single_op_eval': atom_single_op_eval,
        'order_of_operations': atom_order_of_operations,
        'innermost_eval': atom_innermost_eval,
        'sign_handling': atom_sign_handling,
    },
}

COUNTDOWN_ATOMS = {
    'L1': {
        'number_inventory': countdown_number_inventory,
        'target_reading': countdown_target_reading,
        'how_many_numbers': countdown_how_many_numbers,
    },
    'L2': {
        'single_op': countdown_single_op,
        'two_number_reachability': countdown_two_number_reachability,
        'distance_to_target': countdown_distance_to_target,
    },
}

SHORTEST_PATH_ATOMS = {
    'L1': {
        'locate_start': sp_locate_start,
        'locate_goal': sp_locate_goal,
        'read_cell': sp_read_cell,
        'grid_dimensions': sp_grid_dimensions,
    },
    'L2': {
        'valid_neighbors': sp_valid_neighbors,
        'is_move_legal': sp_is_move_legal,
        'direction_to_goal': sp_direction_to_goal,
    },
    'L2_long': {
        'execute_path': sp_execute_path,
        'trace_and_report': sp_trace_and_report,
        'count_walls_on_path': sp_count_walls_on_path,
    },
    'L2_feasibility': {
        'path_exists': sp_path_exists,
        'find_any_path': sp_find_any_path,
        'start_neighbors': sp_start_neighbors,
        'bfs_first_steps': sp_bfs_first_steps,
    },
}

TASK_ATOMS = {
    'basic_arithmetic': ARITHMETIC_ATOMS,
    'countdown': COUNTDOWN_ATOMS,
    'shortest_path': SHORTEST_PATH_ATOMS,
    'syllogism': SYLLOGISM_ATOMS,
    'knights_knaves': KNIGHTS_KNAVES_ATOMS,
    'maze': MAZE_ATOMS,
    'mini_sudoku': MINI_SUDOKU_ATOMS,
}

ALL_ATOMS = {}
for task_atoms in TASK_ATOMS.values():
    for layer, atoms in task_atoms.items():
        for name, fn in atoms.items():
            ALL_ATOMS[f"{layer}_{name}"] = fn
