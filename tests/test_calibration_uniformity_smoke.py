from __future__ import annotations

import numpy as np

from babappa.engine import analyze_alignment, train_energy_model_from_neutral_spec
from babappa.neutral import GY94NeutralSimulator, NeutralSpec


def test_null_calibration_smoke_not_degenerate() -> None:
    spec = NeutralSpec(tree_newick="(A:0.1,B:0.1,C:0.1);", kappa=2.0, omega=1.0)
    model = train_energy_model_from_neutral_spec(
        neutral_spec=spec,
        sim_replicates=12,
        sim_length=180,
        seed=101,
    )
    simulator = GY94NeutralSimulator(spec)
    rng = np.random.default_rng(202)

    p_values: list[float] = []
    for gene_idx in range(200):
        seed = int(rng.integers(0, 2**32 - 1))
        aln = simulator.simulate_alignment(length_nt=150, seed=seed)
        res = analyze_alignment(
            aln,
            model=model,
            calibration_size=199,
            calibration_mode="phylo",
            tail="right",
            seed=seed,
            name=f"null_{gene_idx+1}",
        )
        p_values.append(float(res.p_value))

    p = np.asarray(p_values, dtype=float)
    frac_sig = float(np.mean(p < 0.05))
    assert 0.4 <= float(np.mean(p)) <= 0.6
    assert 0.01 <= frac_sig <= 0.10

