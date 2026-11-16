"""Tests for the grading and loop metrics that every reported number depends on."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from grading import last_boxed, repetition, same_answer  # noqa: E402
from report import sign_test  # noqa: E402


def test_last_boxed_takes_the_last_box_and_handles_nested_braces():
    assert last_boxed(r"first \boxed{3}, then \boxed{\frac{1}{2}}") == r"\frac{1}{2}"
    assert last_boxed("no answer") is None
    assert last_boxed(r"\boxed{12") is None  # unclosed box


def test_same_answer_is_numeric_and_ignores_formatting():
    assert same_answer("1,000", "1000")
    assert same_answer(r"\$18", "18")
    assert same_answer("18.0", "18")
    assert not same_answer("17", "18")
    assert not same_answer(None, "18")


def test_repetition_separates_a_verbatim_loop_from_progressing_text():
    progressing = list(range(2000))
    loop = list(range(100)) * 20
    assert repetition(progressing) == 0.0
    assert repetition(loop) > 0.8
    assert repetition([1, 2, 3]) == 0.0  # shorter than one n-gram


def test_sign_test_matches_the_binomial_reference():
    assert sign_test(11, 0) == 0.001  # 2 / 2**11
    assert sign_test(9, 4) == 0.267
    assert sign_test(0, 0) == 1.0
