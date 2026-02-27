from pathlib import Path

from babappa.cli import main
from babappa.engine import analyze_alignment, train_energy_model_from_neutral_spec
from babappa.model import load_model
from babappa.neutral import GY94NeutralSimulator, NeutralSpec
from babappa.phylo import parse_newick


def test_parse_newick_tree() -> None:
    tree = parse_newick("((A:0.1,B:0.2):0.3,C:0.4);")
    assert sorted(tree.leaf_names()) == ["A", "B", "C"]
    assert all(length >= 0 for length in tree.branch_lengths())


def test_phylo_simulation_and_calibration() -> None:
    spec = NeutralSpec(tree_newick="(A:0.1,B:0.1,C:0.1);", kappa=2.0, omega=1.0)
    model = train_energy_model_from_neutral_spec(
        neutral_spec=spec,
        sim_replicates=5,
        sim_length=90,
        seed=11,
    )
    simulator = GY94NeutralSimulator(spec)
    observed = simulator.simulate_alignment(length_nt=90, seed=22)
    result = analyze_alignment(
        observed,
        model=model,
        calibration_size=30,
        calibration_mode="phylo",
        seed=33,
        name="obs",
    )
    assert result.calibration_mode == "phylo"
    assert 0.0 <= result.p_value <= 1.0


def test_cli_train_from_tree_embeds_neutral_spec(tmp_path: Path) -> None:
    model_path = tmp_path / "model_phylo.json"
    obs_path = tmp_path / "obs.fasta"
    out_path = tmp_path / "out.json"
    obs_path.write_text(">A\nACGTACGTACGT\n>B\nACGTACGTACGT\n", encoding="utf-8")

    assert (
        main(
            [
                "train",
                "--tree",
                "(A:0.1,B:0.1);",
                "--sim-replicates",
                "3",
                "--sim-length",
                "60",
                "--model",
                str(model_path),
                "--seed",
                "7",
            ]
        )
        == 0
    )
    model = load_model(model_path)
    assert model.neutral_spec is not None

    assert (
        main(
            [
                "analyze",
                "--model",
                str(model_path),
                "--alignment",
                str(obs_path),
                "--calibration-size",
                "10",
                "--calibration-mode",
                "auto",
                "--output",
                str(out_path),
                "--json",
            ]
        )
        == 0
    )
