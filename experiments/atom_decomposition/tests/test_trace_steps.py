"""trace_steps: each kind of first mistake is found on a real (abridged) model trace.

Run from the repo root:  python -m pytest experiments/atom_decomposition/tests/test_trace_steps.py -q
"""

from experiments.atom_decomposition.trace_steps import _gold, analyse, final_int, left_to_right


def first_error(expr, text):
    return analyse(expr, _gold(expr), text)["first_error"]


def test_left_to_right_and_answers():
    assert left_to_right("2 + 3 * 4") == 20 and left_to_right("-(2 + 3) * 4") == -20
    assert final_int("<think>x</think>\n<answer>4137.0</answer>") == 4137
    assert final_int("<think>x</think>\nThe result is -53,482,124,988.") == -53482124988


def test_l2_arithmetic_in_a_chain_of_steps():
    t = ("<think>\nFirst, calculate -30 * 66 = -1980\nThen, -1980 * -37 = 73260\n"
         "Next, 73260 * 84 = 6163440\nFinally, 53 + 6163440 = 6163493\n</think>\n\nThe result is 6163493.")
    fe = first_error("53 + -30 * 66 * -37 * 84", t)
    assert fe["layer"] == "L2" and fe["kind"] == "arithmetic" and fe["op"] == "*" and fe["digits"] == [5, 2]


def test_multi_op_step_and_precedence():
    t = "<think>\n414 + 338 + -(459)×468 = 414 + 338 + -214,  414 + 338 = 752\n</think><answer>538</answer>"
    assert first_error("414 + 338 + -459 * 468", t)["sub"] == "arithmetic"
    assert first_error("2 + 3 * 4", "<think>2 + 3 * 4 = 20</think><answer>20</answer>")["sub"] == "precedence"
    fe = first_error("2 + 3 * 4", "<think>2 + 3 = 5\n5 * 4 = 20</think><answer>20</answer>")
    assert fe == {"layer": "L1", "kind": "plan", "sub": "precedence"}


def test_l1_sign_carry_and_grounding():
    t = ("<think>\n-11 - -23 = -11 + 23 = 12\nThen: -( -12 ) = 12\n"
         "Now, -(12) + (-73) = -12 -73 = -85\n</think>\n\n<answer>-85</answer>")
    assert first_error("-( -( -11 - -23 ) ) + -73", t)["kind"] == "sign carry"
    assert first_error("10 - 17", "<think>10 - 71 = -61</think><answer>-61</answer>")["kind"] == "grounding"


def test_correct_trace_has_no_error_and_loops_count_once():
    t = "<think>\n93 + 53 = 146\n146 - (-31) = 177\n177 - (-71) = 248\n</think>\n\nThe result is 248."
    r = analyse("93 + 53 - -31 - -71", 248, t)
    assert r["correct"] == 1 and r["first_error"] is None and len(r["single"]) == 3
    loop = "<think>" + "743 + 743 = 460,  " * 20 + "</think><answer>460</answer>"
    assert len(analyse("743 + 743", 1486, loop)["single"]) == 1
