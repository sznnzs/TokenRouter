"""Smoke tests for answer extraction utilities."""
from tokenrouter.utils import extract_answer


def test_extract_answer_from_boxed_expression():
    text = "Let's solve it carefully. The final answer is \\boxed{42}."
    assert extract_answer(text) == "42"


def test_extract_answer_from_final_answer_section():
    text = "Reasoning...\n**Final Answer:** 128"
    assert extract_answer(text) == "128"
