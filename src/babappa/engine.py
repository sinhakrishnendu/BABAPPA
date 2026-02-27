from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .hash_utils import sha256_json
from .io import Alignment
from .model import EnergyModel
from .neutral import GY94NeutralSimulator, NeutralSpec
from .representation import FEATURE_NAMES, alignment_to_matrix
from .stats import rank_p_value


@dataclass
class AnalysisResult:
    alignment_name: str
    gene_length: int
    dispersion: float
    mu0: float
    sigma0: float
    identifiability_index: float
    p_value: float
    calibration_size: int
    calibration_mode: str
    tail: str
    n_used: int
    model_hash: str
    phi_hash: str
    m0_hash: str
    seed_gene: int | None
    seed_calib_base: int | None
    q_value: float | None = None
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "alignment_name": self.alignment_name,
            "gene_length": self.gene_length,
            "dispersion": self.dispersion,
            "mu0": self.mu0,
            "sigma0": self.sigma0,
            "identifiability_index": self.identifiability_index,
            "p_value": self.p_value,
            "q_value": self.q_value,
            "calibration_size": self.calibration_size,
            "calibration_mode": self.calibration_mode,
            "tail": self.tail,
            "n_used": self.n_used,
            "model_hash": self.model_hash,
            "phi_hash": self.phi_hash,
            "M0_hash": self.m0_hash,
            "seed_gene": self.seed_gene,
            "seed_calib_base": self.seed_calib_base,
            "warnings": list(self.warnings),
        }


def _score_matching_objective(
    features: np.ndarray, mean: np.ndarray, precision: np.ndarray
) -> float:
    centered = features - mean
    gradients = centered @ precision.T
    quadratic = 0.5 * np.mean(np.sum(gradients * gradients, axis=1))
    return float(quadratic + np.trace(precision))


def _as_feature_matrix(alignments: list[Alignment]) -> np.ndarray:
    if not alignments:
        raise ValueError("At least one alignment is required.")
    matrices = [alignment_to_matrix(alignment) for alignment in alignments]
    return np.vstack(matrices)


def train_energy_model(
    alignments: list[Alignment],
    *,
    max_columns: int | None = None,
    ridge: float = 1e-5,
    seed: int | None = None,
    neutral_spec: NeutralSpec | None = None,
    training_mode: str = "alignment",
) -> EnergyModel:
    """Train a frozen BABAPPA neutral energy model."""
    if ridge <= 0:
        raise ValueError("ridge must be > 0")

    features = _as_feature_matrix(alignments)
    n_rows, n_features = features.shape

    rng = np.random.default_rng(seed)
    if max_columns is not None:
        if max_columns <= 0:
            raise ValueError("max_columns must be > 0")
        if max_columns < n_rows:
            chosen = rng.choice(n_rows, size=max_columns, replace=False)
            features = features[chosen]
            n_rows = features.shape[0]

    mean = features.mean(axis=0)
    centered = features - mean
    denom = max(n_rows - 1, 1)
    covariance = (centered.T @ centered) / denom
    covariance = covariance + ridge * np.eye(n_features, dtype=float)
    precision = np.linalg.pinv(covariance)
    objective = _score_matching_objective(features, mean, precision)

    return EnergyModel.create(
        feature_names=FEATURE_NAMES,
        mean=mean,
        covariance=covariance,
        precision=precision,
        training_samples=n_rows,
        training_alignments=len(alignments),
        max_training_length=max(alignment.length for alignment in alignments),
        ridge=ridge,
        seed=seed,
        objective=objective,
        neutral_spec=neutral_spec,
        training_mode=training_mode,
    )


def train_energy_model_from_neutral_spec(
    *,
    neutral_spec: NeutralSpec,
    sim_replicates: int,
    sim_length: int,
    max_columns: int | None = None,
    ridge: float = 1e-5,
    seed: int | None = None,
) -> EnergyModel:
    if sim_replicates <= 0:
        raise ValueError("sim_replicates must be > 0")
    if sim_length <= 0:
        raise ValueError("sim_length must be > 0")

    simulator = GY94NeutralSimulator(neutral_spec)
    rng = np.random.default_rng(seed)
    alignments: list[Alignment] = []
    for _ in range(sim_replicates):
        sub_seed = int(rng.integers(0, 2**32 - 1))
        alignments.append(simulator.simulate_alignment(length_nt=sim_length, seed=sub_seed))

    return train_energy_model(
        alignments,
        max_columns=max_columns,
        ridge=ridge,
        seed=seed,
        neutral_spec=neutral_spec,
        training_mode="simulation",
    )


def compute_dispersion(features: np.ndarray, model: EnergyModel) -> tuple[float, np.ndarray]:
    energies = model.energy(features)
    mean_energy = float(np.mean(energies))
    dispersion = float(np.mean((energies - mean_energy) ** 2))
    return dispersion, energies


def _calibrate_gaussian(
    *,
    observed_dispersion: float,
    gene_length: int,
    model: EnergyModel,
    calibration_size: int,
    seed: int | None,
) -> tuple[float, float, np.ndarray]:
    rng = np.random.default_rng(seed)
    null_dispersions = np.empty(calibration_size, dtype=float)
    for idx in range(calibration_size):
        simulated_features = rng.multivariate_normal(
            mean=model.mean,
            cov=model.covariance,
            size=gene_length,
            check_valid="ignore",
        )
        null_dispersions[idx], _ = compute_dispersion(simulated_features, model)

    mu0 = float(np.mean(null_dispersions))
    sigma0 = float(np.std(null_dispersions, ddof=0))
    return mu0, sigma0, null_dispersions


def _calibrate_phylo(
    *,
    observed_dispersion: float,
    gene_length: int,
    model: EnergyModel,
    calibration_size: int,
    seed: int | None,
) -> tuple[float, float, np.ndarray]:
    if model.neutral_spec is None:
        raise ValueError(
            "Model lacks neutral tree/codon specification. "
            "Re-train with --tree or use calibration_mode='gaussian'."
        )

    simulator = GY94NeutralSimulator(model.neutral_spec)
    rng = np.random.default_rng(seed)
    null_dispersions = np.empty(calibration_size, dtype=float)
    for idx in range(calibration_size):
        sub_seed = int(rng.integers(0, 2**32 - 1))
        null_alignment = simulator.simulate_alignment(length_nt=gene_length, seed=sub_seed)
        features = alignment_to_matrix(null_alignment)
        null_dispersions[idx], _ = compute_dispersion(features, model)

    mu0 = float(np.mean(null_dispersions))
    sigma0 = float(np.std(null_dispersions, ddof=0))
    return mu0, sigma0, null_dispersions


def calibrate_dispersion(
    *,
    observed_dispersion: float,
    gene_length: int,
    model: EnergyModel,
    calibration_size: int = 1000,
    seed: int | None = None,
    calibration_mode: str = "auto",
    tail: str = "right",
) -> tuple[float, float, float, np.ndarray, str, str]:
    if calibration_size <= 0:
        raise ValueError("calibration_size must be > 0")
    if gene_length <= 0:
        raise ValueError("gene_length must be > 0")

    mode = calibration_mode.lower()
    if mode == "auto":
        mode = "phylo" if model.neutral_spec is not None else "gaussian"

    if mode == "phylo":
        mu0, sigma0, null_dispersions = _calibrate_phylo(
            observed_dispersion=observed_dispersion,
            gene_length=gene_length,
            model=model,
            calibration_size=calibration_size,
            seed=seed,
        )
        p_value = rank_p_value(observed_dispersion, null_dispersions, tail=tail)
        return p_value, mu0, sigma0, null_dispersions, mode, tail.lower()

    if mode == "gaussian":
        mu0, sigma0, null_dispersions = _calibrate_gaussian(
            observed_dispersion=observed_dispersion,
            gene_length=gene_length,
            model=model,
            calibration_size=calibration_size,
            seed=seed,
        )
        p_value = rank_p_value(observed_dispersion, null_dispersions, tail=tail)
        return p_value, mu0, sigma0, null_dispersions, mode, tail.lower()

    raise ValueError(f"Unsupported calibration_mode: {calibration_mode}")


def analyze_alignment(
    alignment: Alignment,
    *,
    model: EnergyModel,
    calibration_size: int = 1000,
    seed: int | None = None,
    name: str = "alignment",
    strict_scaling: bool = False,
    calibration_mode: str = "auto",
    tail: str = "right",
    override_scaling: bool = False,
    seed_gene: int | None = None,
    scaling_warn_ratio: float = 0.1,
    scaling_fail_ratio: float = 0.25,
) -> AnalysisResult:
    features = alignment_to_matrix(alignment)
    observed_dispersion, _ = compute_dispersion(features, model)
    p_value, mu0, sigma0, _, used_mode, used_tail = calibrate_dispersion(
        observed_dispersion=observed_dispersion,
        gene_length=alignment.length,
        model=model,
        calibration_size=calibration_size,
        seed=seed,
        calibration_mode=calibration_mode,
        tail=tail,
    )
    identifiability = 0.0 if sigma0 == 0 else (observed_dispersion - mu0) / sigma0

    warnings: list[str] = []
    ratio = alignment.length / max(model.training_samples, 1)
    if ratio > scaling_fail_ratio and not override_scaling:
        raise ValueError(
            f"Scaling guardrail failure: L/n={ratio:.3f} > {scaling_fail_ratio}. "
            "Use --override-scaling to bypass."
        )
    if ratio > scaling_warn_ratio:
        msg = (
            f"Scaling warning: L/n={ratio:.3f} exceeds {scaling_warn_ratio}. "
            "Inference may be unstable."
        )
        if strict_scaling:
            raise ValueError(msg)
        warnings.append(msg)

    if calibration_size < 1000:
        warnings.append(
            "Calibration warning: N < 1000 may produce unstable Monte Carlo estimates."
        )

    if sigma0 == 0:
        warnings.append("Null dispersion variance is zero; identifiability index set to 0.")

    if used_mode == "gaussian":
        warnings.append(
            "Calibration used Gaussian feature null. "
            "Use a tree-trained model to enable phylogenetic null calibration."
        )

    model_hash = sha256_json(model.to_dict())
    phi_hash = sha256_json(
        {
            "schema_version": 1,
            "name": "phi_default_v1",
            "feature_names": list(FEATURE_NAMES),
        }
    )
    m0_hash = sha256_json(model.neutral_spec.to_dict()) if model.neutral_spec else "NA"

    return AnalysisResult(
        alignment_name=name,
        gene_length=alignment.length,
        dispersion=observed_dispersion,
        mu0=mu0,
        sigma0=sigma0,
        identifiability_index=float(identifiability),
        p_value=float(p_value),
        calibration_size=calibration_size,
        calibration_mode=used_mode,
        tail=used_tail,
        n_used=model.training_samples,
        model_hash=model_hash,
        phi_hash=phi_hash,
        m0_hash=m0_hash,
        seed_gene=seed_gene,
        seed_calib_base=seed,
        warnings=tuple(warnings),
    )


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    n = len(p_values)
    if n == 0:
        return []
    order = np.argsort(p_values)
    ranked = np.array([p_values[i] for i in order], dtype=float)
    adjusted = np.empty(n, dtype=float)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        candidate = min(prev, ranked[i] * n / rank)
        adjusted[i] = candidate
        prev = candidate

    result = np.empty(n, dtype=float)
    for sorted_idx, original_idx in enumerate(order):
        result[original_idx] = adjusted[sorted_idx]
    return result.tolist()


def analyze_batch(
    alignments: list[tuple[str, Alignment]],
    *,
    model: EnergyModel,
    calibration_size: int = 1000,
    seed: int | None = None,
    strict_scaling: bool = False,
    calibration_mode: str = "auto",
    tail: str = "right",
    override_scaling: bool = False,
    scaling_warn_ratio: float = 0.1,
    scaling_fail_ratio: float = 0.25,
) -> list[AnalysisResult]:
    rng = np.random.default_rng(seed)
    if alignments:
        lmax = max(alignment.length for _, alignment in alignments)
        ratio = lmax / max(model.training_samples, 1)
        if ratio > scaling_fail_ratio and not override_scaling:
            raise ValueError(
                f"Scaling guardrail failure: Lmax/n={ratio:.3f} > {scaling_fail_ratio}. "
                "Use --override-scaling to bypass."
            )
    results: list[AnalysisResult] = []
    for name, alignment in alignments:
        sub_seed = int(rng.integers(0, 2**32 - 1))
        calib_seed = int(rng.integers(0, 2**32 - 1))
        result = analyze_alignment(
            alignment,
            model=model,
            calibration_size=calibration_size,
            seed=calib_seed,
            name=name,
            strict_scaling=strict_scaling,
            calibration_mode=calibration_mode,
            tail=tail,
            override_scaling=override_scaling,
            seed_gene=sub_seed,
            scaling_warn_ratio=scaling_warn_ratio,
            scaling_fail_ratio=scaling_fail_ratio,
        )
        results.append(result)

    q_values = benjamini_hochberg([result.p_value for result in results])
    for idx, q_value in enumerate(q_values):
        results[idx].q_value = float(q_value)

    return results
