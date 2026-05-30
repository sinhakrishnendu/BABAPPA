from typer.testing import CliRunner

from babappa.align import AlignmentConfig, align_simulation_directory
from babappa.cli import app
from babappa.datasets import DatasetIndexConfig, build_dataset_index, read_tsv
from babappa.simulate import SimulationConfig, simulate_families
from babappa.tensors import TensorBuildConfig, build_tensor_dataset


runner = CliRunner()


def test_build_dataset_index_writes_expected_files_and_rows(tmp_path) -> None:
    tensor_dir = _prepare_tensor_dataset(tmp_path)
    index_dir = tmp_path / "dataset_index"

    summary = build_dataset_index(
        DatasetIndexConfig(
            tensor_dir=str(tensor_dir),
            outdir=str(index_dir),
            methods=["identity", "codon_dropout"],
            seed=42,
        )
    )

    assert (index_dir / "dataset_index.json").exists()
    assert (index_dir / "features.tsv").exists()
    assert (index_dir / "splits.tsv").exists()
    assert summary["n_rows"] == 20
    assert summary["n_families"] == 10


def test_validate_dataset_index_succeeds(tmp_path) -> None:
    tensor_dir = _prepare_tensor_dataset(tmp_path)
    index_dir = tmp_path / "dataset_validate"
    build_dataset_index(
        DatasetIndexConfig(tensor_dir=str(tensor_dir), outdir=str(index_dir), seed=42)
    )

    result = runner.invoke(app, ["validate-index", "--index-dir", str(index_dir)])

    assert result.exit_code == 0
    assert "Status" in result.output
    assert "ok" in result.output


def test_family_disjoint_splitting(tmp_path) -> None:
    tensor_dir = _prepare_tensor_dataset(tmp_path)
    index_dir = tmp_path / "dataset_splits"
    build_dataset_index(
        DatasetIndexConfig(tensor_dir=str(tensor_dir), outdir=str(index_dir), seed=42)
    )
    split_rows = read_tsv(index_dir / "splits.tsv")
    family_to_splits = {}
    for row in split_rows:
        family_to_splits.setdefault(row["family_id"], set()).add(row["split"])

    assert all(len(splits) == 1 for splits in family_to_splits.values())


def test_required_feature_columns_exist(tmp_path) -> None:
    tensor_dir = _prepare_tensor_dataset(tmp_path)
    index_dir = tmp_path / "dataset_features"
    build_dataset_index(
        DatasetIndexConfig(tensor_dir=str(tensor_dir), outdir=str(index_dir), seed=42)
    )
    rows = read_tsv(index_dir / "features.tsv")
    required_columns = {
        "codon_id_mean",
        "codon_id_std",
        "unique_codon_id_fraction",
        "mean_taxon_codon_id_std",
        "mean_site_codon_id_std",
        "gap_codon_fraction",
    }

    assert rows
    assert required_columns.issubset(rows[0].keys())


def test_gene_label_present_and_binary(tmp_path) -> None:
    tensor_dir = _prepare_tensor_dataset(tmp_path)
    index_dir = tmp_path / "dataset_labels"
    build_dataset_index(
        DatasetIndexConfig(tensor_dir=str(tensor_dir), outdir=str(index_dir), seed=42)
    )
    rows = read_tsv(index_dir / "features.tsv")

    assert rows
    assert {row["gene_label"] for row in rows}.issubset({"0", "1"})


def test_cli_index_dataset_exits_successfully(tmp_path) -> None:
    tensor_dir = _prepare_tensor_dataset(tmp_path)
    index_dir = tmp_path / "dataset_cli"

    result = runner.invoke(
        app,
        [
            "index-dataset",
            "--tensor-dir",
            str(tensor_dir),
            "--outdir",
            str(index_dir),
            "--methods",
            "identity,codon_dropout",
            "--seed",
            "42",
        ],
    )

    assert result.exit_code == 0
    assert "Dataset index path:" in result.output


def test_cli_validate_index_exits_successfully(tmp_path) -> None:
    tensor_dir = _prepare_tensor_dataset(tmp_path)
    index_dir = tmp_path / "dataset_cli_validate"
    build_dataset_index(
        DatasetIndexConfig(tensor_dir=str(tensor_dir), outdir=str(index_dir), seed=42)
    )

    result = runner.invoke(app, ["validate-index", "--index-dir", str(index_dir)])

    assert result.exit_code == 0
    assert "ok" in result.output


def test_invalid_split_fractions_fail_gracefully(tmp_path) -> None:
    tensor_dir = _prepare_tensor_dataset(tmp_path)

    result = runner.invoke(
        app,
        [
            "index-dataset",
            "--tensor-dir",
            str(tensor_dir),
            "--outdir",
            str(tmp_path / "dataset_bad_fractions"),
            "--train-fraction",
            "0.9",
            "--val-fraction",
            "0.1",
            "--calib-fraction",
            "0.05",
            "--test-fraction",
            "0.05",
        ],
    )

    assert result.exit_code != 0
    assert "split fractions must sum to 1.0" in result.output


def test_validate_index_fails_when_referenced_tensor_missing(tmp_path) -> None:
    tensor_dir = _prepare_tensor_dataset(tmp_path)
    index_dir = tmp_path / "dataset_corrupt"
    build_dataset_index(
        DatasetIndexConfig(
            tensor_dir=str(tensor_dir),
            outdir=str(index_dir),
            methods=["identity", "codon_dropout"],
            seed=42,
        )
    )
    first_split = read_tsv(index_dir / "splits.tsv")[0]
    (tensor_dir / first_split["tensor_file"]).unlink()

    result = runner.invoke(app, ["validate-index", "--index-dir", str(index_dir)])

    assert result.exit_code != 0
    assert "missing tensor_file referenced by splits.tsv" in result.output


def _prepare_tensor_dataset(tmp_path):
    sim_dir = tmp_path / "sim"
    align_dir = tmp_path / "align"
    tensor_dir = tmp_path / "tensors"
    simulate_families(
        SimulationConfig(
            outdir=str(sim_dir),
            n_families=10,
            n_taxa=5,
            n_codons=45,
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
    return tensor_dir
