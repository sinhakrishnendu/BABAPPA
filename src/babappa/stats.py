from __future__ import annotations

import numpy as np


def rank_p_value(
    observed: float,
    null_values: np.ndarray,
    *,
    tail: str = "right",
) -> float:
    """Rank-based Monte Carlo p-value with add-one correction.

    tail='right':   P(T >= T_obs)
    tail='left':    P(T <= T_obs)
    tail='two-sided': P(|T-c| >= |T_obs-c|), where c is pooled median.
    """
    if null_values.ndim != 1:
        raise ValueError("null_values must be a 1D array.")
    n = int(null_values.size)
    if n <= 0:
        raise ValueError("null_values must be non-empty.")

    mode = tail.lower()
    if mode == "right":
        b = int(np.count_nonzero(null_values >= observed))
        return (b + 1.0) / (n + 1.0)

    if mode == "left":
        b = int(np.count_nonzero(null_values <= observed))
        return (b + 1.0) / (n + 1.0)

    if mode == "two-sided":
        pooled = np.concatenate([null_values, np.array([observed], dtype=float)])
        center = float(np.median(pooled))
        observed_dev = abs(observed - center)
        null_dev = np.abs(null_values - center)
        b = int(np.count_nonzero(null_dev >= observed_dev))
        return (b + 1.0) / (n + 1.0)

    raise ValueError(f"Unsupported tail: {tail}")
