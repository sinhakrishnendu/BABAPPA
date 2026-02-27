from __future__ import annotations

import numpy as np

from babappa.model import EnergyModel
from babappa.representation import FEATURE_NAMES


def _make_quadratic_model(seed: int = 7) -> EnergyModel:
    rng = np.random.default_rng(seed)
    dim = len(FEATURE_NAMES)
    a = rng.normal(size=(dim, dim))
    precision = a.T @ a + 0.5 * np.eye(dim)
    covariance = np.linalg.pinv(precision)
    mean = rng.normal(size=dim)
    return EnergyModel.create(
        feature_names=FEATURE_NAMES,
        mean=mean,
        covariance=covariance,
        precision=precision,
        training_samples=500,
        training_alignments=5,
        max_training_length=300,
        ridge=1e-5,
        seed=seed,
        objective=1.0,
        neutral_spec=None,
    )


def _finite_diff_gradient(model: EnergyModel, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    grad = np.zeros_like(x, dtype=float)
    for j in range(x.size):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[j] += eps
        x_minus[j] -= eps
        f_plus = float(model.energy(x_plus[None, :])[0])
        f_minus = float(model.energy(x_minus[None, :])[0])
        grad[j] = (f_plus - f_minus) / (2.0 * eps)
    return grad


def _finite_diff_laplacian(model: EnergyModel, x: np.ndarray, eps: float = 1e-4) -> float:
    center = float(model.energy(x[None, :])[0])
    acc = 0.0
    for j in range(x.size):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[j] += eps
        x_minus[j] -= eps
        f_plus = float(model.energy(x_plus[None, :])[0])
        f_minus = float(model.energy(x_minus[None, :])[0])
        acc += (f_plus - 2.0 * center + f_minus) / (eps**2)
    return float(acc)


def test_quadratic_energy_gradient_matches_finite_difference() -> None:
    model = _make_quadratic_model(seed=11)
    rng = np.random.default_rng(23)
    points = rng.normal(size=(6, len(FEATURE_NAMES)))

    analytic = model.gradient(points)
    numeric = np.vstack([_finite_diff_gradient(model, p, eps=1e-6) for p in points])
    max_abs_error = float(np.max(np.abs(analytic - numeric)))
    assert max_abs_error <= 1e-6


def test_quadratic_energy_laplacian_matches_trace_and_numeric() -> None:
    model = _make_quadratic_model(seed=13)
    rng = np.random.default_rng(29)
    points = rng.normal(size=(4, len(FEATURE_NAMES)))

    analytic = model.laplacian(points)
    expected = np.full(points.shape[0], float(np.trace(model.precision)), dtype=float)
    assert float(np.max(np.abs(analytic - expected))) <= 1e-6

    numeric = np.array([_finite_diff_laplacian(model, p, eps=1e-4) for p in points], dtype=float)
    max_abs_error = float(np.max(np.abs(analytic - numeric)))
    assert max_abs_error <= 1e-4

