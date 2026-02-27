from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .engine import analyze_alignment, train_energy_model_from_neutral_spec
from .io import Alignment
from .neutral import GY94NeutralSimulator, NeutralSpec
from .stats import rank_p_value


@dataclass
class BenchmarkResult:
    summary: dict[str, float]
    records: list[dict[str, object]]
    metadata: dict[str, object]


def _column_entropy(column: str) -> float:
    symbols = [c for c in column.upper() if c not in {"-", "N", "X", "?"}]
    if not symbols:
        return 0.0
    counts: dict[str, int] = {}
    for c in symbols:
        counts[c] = counts.get(c, 0) + 1
    freqs = np.array(list(counts.values()), dtype=float)
    freqs /= freqs.sum()
    return float(-np.sum(freqs * np.log(freqs + 1e-12)))


def mean_site_entropy(alignment: Alignment) -> float:
    values = [_column_entropy(alignment.column(i)) for i in range(alignment.length)]
    return float(np.mean(values))


def _replace_codon_block(seq: str, start: int, replacement: str) -> str:
    return seq[:start] + replacement + seq[start + 3 :]


def simulate_mixture_omega_alignment(
    *,
    neutral_spec: NeutralSpec,
    length_nt: int,
    selected_fraction: float,
    omega_alt: float,
    seed: int | None = None,
) -> Alignment:
    if not (0.0 <= selected_fraction <= 1.0):
        raise ValueError("selected_fraction must be in [0, 1].")
    if omega_alt <= 0:
        raise ValueError("omega_alt must be > 0.")
    if length_nt <= 0:
        raise ValueError("length_nt must be > 0.")

    rng = np.random.default_rng(seed)
    seed_neutral = int(rng.integers(0, 2**32 - 1))
    seed_alt = int(rng.integers(0, 2**32 - 1))

    neutral_sim = GY94NeutralSimulator(neutral_spec)
    alt_spec = NeutralSpec(
        tree_newick=neutral_spec.tree_newick,
        kappa=neutral_spec.kappa,
        omega=omega_alt,
        codon_frequencies=neutral_spec.codon_frequencies,
        simulator=neutral_spec.simulator,
    )
    alt_sim = GY94NeutralSimulator(alt_spec)

    neutral_alignment = neutral_sim.simulate_alignment(length_nt=length_nt, seed=seed_neutral)
    alt_alignment = alt_sim.simulate_alignment(length_nt=length_nt, seed=seed_alt)

    n_codons = (length_nt + 2) // 3
    n_selected = int(round(selected_fraction * n_codons))
    selected_codon_indices = set(
        int(x) for x in rng.choice(n_codons, size=n_selected, replace=False)
    )

    out_sequences = list(neutral_alignment.sequences)
    for codon_idx in selected_codon_indices:
        nt_start = codon_idx * 3
        nt_end = min(nt_start + 3, length_nt)
        if nt_end - nt_start < 3:
            continue
        for i in range(len(out_sequences)):
            repl = alt_alignment.sequences[i][nt_start:nt_end]
            out_sequences[i] = _replace_codon_block(out_sequences[i], nt_start, repl)

    return Alignment(names=neutral_alignment.names, sequences=tuple(out_sequences))


def run_benchmark(
    *,
    neutral_spec: NeutralSpec,
    training_replicates: int,
    training_length_nt: int,
    gene_length_nt: int,
    calibration_size: int,
    n_null_genes: int,
    n_alt_genes: int,
    selected_fraction: float,
    omega_alt: float,
    alpha: float = 0.05,
    tail: str = "right",
    seed: int | None = None,
) -> BenchmarkResult:
    if n_null_genes <= 0:
        raise ValueError("n_null_genes must be > 0.")
    if n_alt_genes <= 0:
        raise ValueError("n_alt_genes must be > 0.")
    if calibration_size <= 0:
        raise ValueError("calibration_size must be > 0.")
    if not (0 < alpha < 1):
        raise ValueError("alpha must be in (0, 1).")

    rng = np.random.default_rng(seed)
    model = train_energy_model_from_neutral_spec(
        neutral_spec=neutral_spec,
        sim_replicates=training_replicates,
        sim_length=training_length_nt,
        seed=int(rng.integers(0, 2**32 - 1)),
    )
    neutral_sim = GY94NeutralSimulator(neutral_spec)

    # Shared baseline null for efficiency and reproducibility.
    baseline_null = np.empty(calibration_size, dtype=float)
    for i in range(calibration_size):
        s = int(rng.integers(0, 2**32 - 1))
        a = neutral_sim.simulate_alignment(length_nt=gene_length_nt, seed=s)
        baseline_null[i] = mean_site_entropy(a)

    records: list[dict[str, object]] = []

    def _append_record(label: str, alignment: Alignment) -> None:
        local_seed = int(rng.integers(0, 2**32 - 1))
        babappa = analyze_alignment(
            alignment,
            model=model,
            calibration_size=calibration_size,
            calibration_mode="phylo",
            tail=tail,
            seed=local_seed,
            name=label,
        )
        entropy_obs = mean_site_entropy(alignment)
        entropy_p = rank_p_value(entropy_obs, baseline_null, tail=tail)
        records.append(
            {
                "label": label,
                "truth": "alt" if label.startswith("alt") else "null",
                "babappa_p_value": babappa.p_value,
                "babappa_sig": babappa.p_value <= alpha,
                "babappa_dispersion": babappa.dispersion,
                "entropy_p_value": float(entropy_p),
                "entropy_sig": bool(entropy_p <= alpha),
                "entropy_mean": entropy_obs,
            }
        )

    for i in range(n_null_genes):
        s = int(rng.integers(0, 2**32 - 1))
        aln = neutral_sim.simulate_alignment(length_nt=gene_length_nt, seed=s)
        _append_record(f"null_{i+1}", aln)

    for i in range(n_alt_genes):
        s = int(rng.integers(0, 2**32 - 1))
        aln = simulate_mixture_omega_alignment(
            neutral_spec=neutral_spec,
            length_nt=gene_length_nt,
            selected_fraction=selected_fraction,
            omega_alt=omega_alt,
            seed=s,
        )
        _append_record(f"alt_{i+1}", aln)

    null_rows = [r for r in records if r["truth"] == "null"]
    alt_rows = [r for r in records if r["truth"] == "alt"]

    def _mean_bool(rows: list[dict[str, object]], key: str) -> float:
        if not rows:
            return float("nan")
        return float(np.mean([1.0 if bool(r[key]) else 0.0 for r in rows]))

    summary = {
        "alpha": float(alpha),
        "n_null_genes": float(n_null_genes),
        "n_alt_genes": float(n_alt_genes),
        "type1_babappa": _mean_bool(null_rows, "babappa_sig"),
        "type1_entropy": _mean_bool(null_rows, "entropy_sig"),
        "power_babappa": _mean_bool(alt_rows, "babappa_sig"),
        "power_entropy": _mean_bool(alt_rows, "entropy_sig"),
    }

    metadata = {
        "training_replicates": training_replicates,
        "training_length_nt": training_length_nt,
        "gene_length_nt": gene_length_nt,
        "calibration_size": calibration_size,
        "selected_fraction": selected_fraction,
        "omega_alt": omega_alt,
        "tail": tail,
        "neutral_spec": neutral_spec.to_dict(),
    }

    return BenchmarkResult(summary=summary, records=records, metadata=metadata)
