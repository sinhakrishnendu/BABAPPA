from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np


NUCLEOTIDES = ("T", "C", "A", "G")

GENETIC_CODE = {
    "TTT": "F",
    "TTC": "F",
    "TTA": "L",
    "TTG": "L",
    "TCT": "S",
    "TCC": "S",
    "TCA": "S",
    "TCG": "S",
    "TAT": "Y",
    "TAC": "Y",
    "TAA": "*",
    "TAG": "*",
    "TGT": "C",
    "TGC": "C",
    "TGA": "*",
    "TGG": "W",
    "CTT": "L",
    "CTC": "L",
    "CTA": "L",
    "CTG": "L",
    "CCT": "P",
    "CCC": "P",
    "CCA": "P",
    "CCG": "P",
    "CAT": "H",
    "CAC": "H",
    "CAA": "Q",
    "CAG": "Q",
    "CGT": "R",
    "CGC": "R",
    "CGA": "R",
    "CGG": "R",
    "ATT": "I",
    "ATC": "I",
    "ATA": "I",
    "ATG": "M",
    "ACT": "T",
    "ACC": "T",
    "ACA": "T",
    "ACG": "T",
    "AAT": "N",
    "AAC": "N",
    "AAA": "K",
    "AAG": "K",
    "AGT": "S",
    "AGC": "S",
    "AGA": "R",
    "AGG": "R",
    "GTT": "V",
    "GTC": "V",
    "GTA": "V",
    "GTG": "V",
    "GCT": "A",
    "GCC": "A",
    "GCA": "A",
    "GCG": "A",
    "GAT": "D",
    "GAC": "D",
    "GAA": "E",
    "GAG": "E",
    "GGT": "G",
    "GGC": "G",
    "GGA": "G",
    "GGG": "G",
}

SENSE_CODONS = tuple(c for c in sorted("".join(p) for p in itertools.product(NUCLEOTIDES, repeat=3)) if GENETIC_CODE[c] != "*")
CODON_INDEX = {codon: idx for idx, codon in enumerate(SENSE_CODONS)}
TRANSITIONS = {
    ("A", "G"),
    ("G", "A"),
    ("C", "T"),
    ("T", "C"),
}


@dataclass(frozen=True)
class GY94Model:
    codons: tuple[str, ...]
    frequencies: np.ndarray
    rate_matrix: np.ndarray


def _is_transition(a: str, b: str) -> bool:
    return (a, b) in TRANSITIONS


def default_codon_frequencies() -> dict[str, float]:
    p = 1.0 / len(SENSE_CODONS)
    return {codon: p for codon in SENSE_CODONS}


def normalize_codon_frequencies(raw: dict[str, float] | None) -> dict[str, float]:
    if raw is None:
        return default_codon_frequencies()

    cleaned: dict[str, float] = {}
    for codon, value in raw.items():
        key = codon.strip().upper()
        if key not in CODON_INDEX:
            raise ValueError(f"Invalid or stop codon in frequencies: {codon}")
        val = float(value)
        if val < 0:
            raise ValueError(f"Codon frequency cannot be negative: {codon}={value}")
        cleaned[key] = val

    if not cleaned:
        raise ValueError("Codon frequency table is empty.")

    for codon in SENSE_CODONS:
        cleaned.setdefault(codon, 0.0)

    total = float(sum(cleaned.values()))
    if total <= 0:
        raise ValueError("Codon frequencies must have positive total mass.")
    return {codon: val / total for codon, val in cleaned.items()}


def build_gy94_rate_matrix(
    *,
    kappa: float,
    omega: float,
    codon_frequencies: dict[str, float] | None = None,
) -> GY94Model:
    if kappa <= 0:
        raise ValueError("kappa must be > 0")
    if omega <= 0:
        raise ValueError("omega must be > 0")

    freq_map = normalize_codon_frequencies(codon_frequencies)
    codons = SENSE_CODONS
    n = len(codons)
    q = np.zeros((n, n), dtype=float)
    pi = np.array([freq_map[codon] for codon in codons], dtype=float)

    for i, source in enumerate(codons):
        aa_source = GENETIC_CODE[source]
        for j, target in enumerate(codons):
            if i == j:
                continue
            diff_positions = [k for k in range(3) if source[k] != target[k]]
            if len(diff_positions) != 1:
                continue
            pos = diff_positions[0]
            rate = 1.0
            if _is_transition(source[pos], target[pos]):
                rate *= kappa
            if GENETIC_CODE[target] != aa_source:
                rate *= omega
            q[i, j] = rate * pi[j]

    q[np.diag_indices_from(q)] = -np.sum(q, axis=1)
    expected_rate = float(-np.sum(pi * np.diag(q)))
    if expected_rate <= 0:
        raise ValueError("Rate matrix has non-positive expected substitution rate.")
    q /= expected_rate

    return GY94Model(codons=codons, frequencies=pi, rate_matrix=q)


def transition_matrix(rate_matrix: np.ndarray, branch_length: float) -> np.ndarray:
    if branch_length < 0:
        raise ValueError("Branch lengths must be non-negative.")
    if branch_length == 0:
        return np.eye(rate_matrix.shape[0], dtype=float)

    eigenvalues, eigenvectors = np.linalg.eig(rate_matrix)
    inv_eigenvectors = np.linalg.inv(eigenvectors)
    exp_diag = np.diag(np.exp(eigenvalues * branch_length))
    p_complex = eigenvectors @ exp_diag @ inv_eigenvectors
    p = np.real(p_complex)
    p[p < 0] = 0.0
    row_sums = p.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    p /= row_sums
    return p
