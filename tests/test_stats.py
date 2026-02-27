import numpy as np

from babappa.engine import benjamini_hochberg
from babappa.stats import rank_p_value


def test_rank_p_value_right_and_left() -> None:
    null = np.array([0.1, 0.2, 0.3, 0.4], dtype=float)
    p_right = rank_p_value(0.35, null, tail="right")
    p_left = rank_p_value(0.35, null, tail="left")
    assert 0.0 < p_right <= 1.0
    assert 0.0 < p_left <= 1.0
    assert p_right < p_left


def test_rank_p_value_two_sided() -> None:
    null = np.array([1.0, 1.1, 1.2, 1.3], dtype=float)
    p = rank_p_value(2.0, null, tail="two-sided")
    assert 0.0 < p <= 1.0


def test_bh_known_toy_example() -> None:
    p = [0.01, 0.04, 0.03, 0.002]
    q = benjamini_hochberg(p)
    expected = [0.02, 0.04, 0.04, 0.008]
    assert np.allclose(np.asarray(q, dtype=float), np.asarray(expected, dtype=float), atol=1e-12)


def test_bh_q_not_below_p_and_in_unit_interval() -> None:
    p = [0.2, 0.01, 0.5, 0.001, 0.07]
    q = np.asarray(benjamini_hochberg(p), dtype=float)
    p_arr = np.asarray(p, dtype=float)
    assert np.all(q >= p_arr)
    assert np.all((q >= 0.0) & (q <= 1.0))


def test_bh_monotone_in_sorted_p_order() -> None:
    p = np.asarray([0.12, 0.8, 0.05, 0.01, 0.2, 0.6], dtype=float)
    q = np.asarray(benjamini_hochberg(p.tolist()), dtype=float)
    order = np.argsort(p)
    q_sorted = q[order]
    assert np.all(np.diff(q_sorted) >= -1e-12)
