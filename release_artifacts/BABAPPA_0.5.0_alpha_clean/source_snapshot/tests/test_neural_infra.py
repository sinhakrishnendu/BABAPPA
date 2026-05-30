from typer.testing import CliRunner

from babappa.align import AlignmentConfig, align_simulation_directory
from babappa.cli import app
from babappa.datasets import DatasetIndexConfig, build_dataset_index, read_tsv
from babappa.simulate import SimulationConfig, simulate_families
from babappa.tensors import TensorBuildConfig, build_tensor_dataset
from babappa.training import (
    NeuralDatasetConfig,
    get_torch_environment,
    inspect_neural_dataset,
    resolve_tensor_file,
    safe_import_torch,
)


runner = CliRunner()


def test_check_neural_env_function_does_not_crash() -> None:
    env = get_torch_environment()

    assert "torch_available" in env
    assert "recommended_device" in env


def test_inspect_neural_dataset_without_torch(tmp_path) -> None:
    dataset_dir = _prepare_dataset(tmp_path)

    summary = inspect_neural_dataset(
        NeuralDatasetConfig(dataset_dir=str(dataset_dir), split="train")
    )

    assert summary["n_rows"] > 0
    assert summary["example_shape"] is not None
    assert "class_counts" in summary


def test_resolve_tensor_file(tmp_path) -> None:
    dataset_dir = _prepare_dataset(tmp_path)
    rows = read_tsv(dataset_dir / "splits.tsv")

    tensor_file = resolve_tensor_file(rows[0]["tensor_file"], dataset_dir)

    assert tensor_file.exists()
    assert tensor_file.name.endswith(".tensor.npz")


def test_cli_check_neural_env_exits_0() -> None:
    result = runner.invoke(app, ["check-neural-env"])

    assert result.exit_code == 0
    assert "Torch available" in result.output


def test_cli_inspect_neural_data_exits_0(tmp_path) -> None:
    dataset_dir = _prepare_dataset(tmp_path)

    result = runner.invoke(
        app,
        [
            "inspect-neural-data",
            "--dataset-dir",
            str(dataset_dir),
            "--split",
            "train",
            "--methods",
            "identity,codon_dropout",
            "--max-items",
            "8",
        ],
    )

    assert result.exit_code == 0
    assert "Example tensor shape" in result.output


def test_smoke_neural_batch_behavior(tmp_path) -> None:
    dataset_dir = _prepare_dataset(tmp_path)
    torch, _error = safe_import_torch()

    result = runner.invoke(
        app,
        [
            "smoke-neural-batch",
            "--dataset-dir",
            str(dataset_dir),
            "--split",
            "train",
            "--methods",
            "identity,codon_dropout",
            "--batch-size",
            "4",
        ],
    )

    if torch is None:
        assert result.exit_code != 0
        assert "PyTorch is not available" in result.output
    else:
        assert result.exit_code == 0
        assert "X shape" in result.output


def test_invalid_split_fails(tmp_path) -> None:
    dataset_dir = _prepare_dataset(tmp_path)

    result = runner.invoke(
        app,
        [
            "inspect-neural-data",
            "--dataset-dir",
            str(dataset_dir),
            "--split",
            "bad_split",
        ],
    )

    assert result.exit_code != 0
    assert "split must be one of" in result.output


def _prepare_dataset(tmp_path):
    sim_dir = tmp_path / "sim"
    align_dir = tmp_path / "align"
    tensor_dir = tmp_path / "tensors"
    dataset_dir = tmp_path / "dataset"
    simulate_families(
        SimulationConfig(
            outdir=str(sim_dir),
            n_families=10,
            n_taxa=6,
            n_codons=60,
            seed=42,
            positive_rate=0.5,
            saturation_tier="moderate",
        )
    )
    align_simulation_directory(
        AlignmentConfig(
            sim_dir=str(sim_dir),
            outdir=str(align_dir),
            methods=["identity", "codon_dropout"],
            seed=42,
            dropout_rate=0.02,
        )
    )
    build_tensor_dataset(
        TensorBuildConfig(
            sim_dir=str(sim_dir),
            align_dir=str(align_dir),
            outdir=str(tensor_dir),
            methods=["identity", "codon_dropout"],
        )
    )
    build_dataset_index(
        DatasetIndexConfig(
            tensor_dir=str(tensor_dir),
            outdir=str(dataset_dir),
            methods=["identity", "codon_dropout"],
            seed=42,
        )
    )
    return dataset_dir
