from babappa.evaluation import run_benchmark
from babappa.neutral import NeutralSpec


def test_run_benchmark_small() -> None:
    spec = NeutralSpec(tree_newick="(A:0.1,B:0.1,C:0.1);", kappa=2.0, omega=1.0)
    result = run_benchmark(
        neutral_spec=spec,
        training_replicates=4,
        training_length_nt=60,
        gene_length_nt=60,
        calibration_size=8,
        n_null_genes=3,
        n_alt_genes=3,
        selected_fraction=0.4,
        omega_alt=2.5,
        alpha=0.1,
        tail="right",
        seed=7,
    )
    assert "type1_babappa" in result.summary
    assert "power_babappa" in result.summary
    assert len(result.records) == 6
