"""Tests for src.utils.metrics."""

from src.utils.metrics import precision_at_k


def test_precision_at_k_perfect():
    recommended = ["a", "b", "c"]
    relevant = {"a", "b", "c"}
    assert precision_at_k(recommended, relevant, k=3) == 1.0


def test_precision_at_k_partial():
    recommended = ["a", "b", "c", "d", "e"]
    relevant = {"a", "c", "e"}
    assert precision_at_k(recommended, relevant, k=5) == 0.6


def test_precision_at_k_empty():
    assert precision_at_k([], set(), k=10) == 0.0


def test_precision_at_k_no_overlap():
    recommended = ["x", "y"]
    relevant = {"a", "b"}
    assert precision_at_k(recommended, relevant, k=2) == 0.0
