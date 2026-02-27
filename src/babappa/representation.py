from __future__ import annotations

from collections import Counter

import numpy as np

from .io import Alignment

GAP_CHARS = {"-", "."}
AMBIGUOUS_CHARS = {"?", "X", "N"}

AA_HYDROPHOBICITY = {
    "A": 1.8,
    "C": 2.5,
    "D": -3.5,
    "E": -3.5,
    "F": 2.8,
    "G": -0.4,
    "H": -3.2,
    "I": 4.5,
    "K": -3.9,
    "L": 3.8,
    "M": 1.9,
    "N": -3.5,
    "P": -1.6,
    "Q": -3.5,
    "R": -4.5,
    "S": -0.8,
    "T": -0.7,
    "V": 4.2,
    "W": -0.9,
    "Y": -1.3,
}

FEATURE_NAMES = (
    "gap_fraction",
    "ambiguous_fraction",
    "major_fraction",
    "minor_fraction",
    "normalized_entropy",
    "gc_fraction",
    "hydrophobicity_mean",
    "hydrophobicity_std",
)


def _safe_entropy(freqs: np.ndarray) -> float:
    if freqs.size <= 1:
        return 0.0
    entropy = float(-np.sum(freqs * np.log(freqs + 1e-12)))
    denom = float(np.log(freqs.size))
    if denom <= 0:
        return 0.0
    return entropy / denom


def column_to_vector(column: str) -> np.ndarray:
    """Deterministic representation map phi(column) -> R^d."""
    symbols = [c.upper() for c in column if not c.isspace()]
    total = len(symbols)
    if total == 0:
        return np.zeros(len(FEATURE_NAMES), dtype=float)

    gap_fraction = sum(c in GAP_CHARS for c in symbols) / total
    ambiguous_fraction = sum(c in AMBIGUOUS_CHARS for c in symbols) / total

    valid = [c for c in symbols if c not in GAP_CHARS and c not in AMBIGUOUS_CHARS]
    if not valid:
        return np.array(
            [
                gap_fraction,
                ambiguous_fraction,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            dtype=float,
        )

    counts = Counter(valid)
    frequencies = np.array(list(counts.values()), dtype=float)
    frequencies /= frequencies.sum()

    major_fraction = float(frequencies.max())
    minor_fraction = 1.0 - major_fraction
    normalized_entropy = _safe_entropy(frequencies)
    gc_fraction = sum(c in {"G", "C"} for c in valid) / len(valid)

    hydrophobic_values = [AA_HYDROPHOBICITY[s] for s in valid if s in AA_HYDROPHOBICITY]
    if hydrophobic_values:
        hydro_mean = float(np.mean(hydrophobic_values))
        hydro_std = float(np.std(hydrophobic_values, ddof=0))
    else:
        hydro_mean = 0.0
        hydro_std = 0.0

    return np.array(
        [
            gap_fraction,
            ambiguous_fraction,
            major_fraction,
            minor_fraction,
            normalized_entropy,
            gc_fraction,
            hydro_mean,
            hydro_std,
        ],
        dtype=float,
    )


def alignment_to_matrix(alignment: Alignment) -> np.ndarray:
    matrix = np.vstack([column_to_vector(column) for column in alignment.iter_columns()])
    return matrix.astype(float, copy=False)
