import json

import numpy as np
from typer.testing import CliRunner

from babappa.align import AlignmentConfig, align_simulation_directory
from babappa.cli import app
from babappa.simulate import SimulationConfig, simulate_families
from babappa.tensors import TensorBuildConfig, build_tensor_dataset


runner = CliRunner()


def test_build_tensor_dataset_writes_expected_files(tmp_path) -> None:
    sim_dir, align_dir = _prepare_alignment(tmp_path)
    tensor_dir = tmp_path / "tensors"

    build_tensor_dataset(
        TensorBuildConfig(
            sim_dir=str(sim_dir),
            align_dir=str(align_dir),
            outdir=str(tensor_dir),
            methods=["identity", "codon_dropout"],
        )
    )
    family_id = _family_ids(sim_dir)[0]
    family_dir = tensor_dir / "families" / family_id

    assert (tensor_dir / "tensor_manifest.json").exists()
    assert (tensor_dir / "tensor_audit.tsv").exists()
    assert (family_dir / f"{family_id}.identity.tensor.npz").exists()
    assert (family_dir / f"{family_id}.identity.tensor_meta.json").exists()
    assert (family_dir / f"{family_id}.labels.json").exists()


def test_npz_contains_three_dimensional_tensor_with_gap_channel(tmp_path) -> None:
    sim_dir, align_dir = _prepare_alignment(tmp_path)
    tensor_dir = tmp_path / "tensors_npz"
    build_tensor_dataset(
        TensorBuildConfig(
            sim_dir=str(sim_dir),
            align_dir=str(align_dir),
            outdir=str(tensor_dir),
        )
    )
    family_id = _family_ids(sim_dir)[0]
    tensor_path = tensor_dir / "families" / family_id / f"{family_id}.identity.tensor.npz"

    with np.load(tensor_path, allow_pickle=False) as shard:
        assert "X" in shard.files
        assert shard["X"].ndim == 3
        assert shard["X"].shape[2] == 2


def test_identity_tensor_has_zero_gap_indicators(tmp_path) -> None:
    sim_dir, align_dir = _prepare_alignment(tmp_path)
    tensor_dir = tmp_path / "tensors_identity"
    build_tensor_dataset(
        TensorBuildConfig(
            sim_dir=str(sim_dir),
            align_dir=str(align_dir),
            outdir=str(tensor_dir),
            methods=["identity"],
        )
    )
    family_id = _family_ids(sim_dir)[0]
    tensor_path = tensor_dir / "families" / family_id / f"{family_id}.identity.tensor.npz"

    with np.load(tensor_path, allow_pickle=False) as shard:
        assert int(shard["X"][:, :, 1].sum()) == 0


def test_codon_dropout_tensor_preserves_identity_shape(tmp_path) -> None:
    sim_dir, align_dir = _prepare_alignment(tmp_path)
    tensor_dir = tmp_path / "tensors_shapes"
    build_tensor_dataset(
        TensorBuildConfig(
            sim_dir=str(sim_dir),
            align_dir=str(align_dir),
            outdir=str(tensor_dir),
            methods=["identity", "codon_dropout"],
        )
    )
    family_id = _family_ids(sim_dir)[0]
    identity_path = tensor_dir / "families" / family_id / f"{family_id}.identity.tensor.npz"
    dropout_path = (
        tensor_dir / "families" / family_id / f"{family_id}.codon_dropout.tensor.npz"
    )

    with np.load(identity_path, allow_pickle=False) as identity:
        with np.load(dropout_path, allow_pickle=False) as dropout:
            assert dropout["X"].shape[:2] == identity["X"].shape[:2]


def test_labels_json_contains_truth_fields(tmp_path) -> None:
    sim_dir, align_dir = _prepare_alignment(tmp_path)
    tensor_dir = tmp_path / "tensors_labels"
    build_tensor_dataset(
        TensorBuildConfig(
            sim_dir=str(sim_dir),
            align_dir=str(align_dir),
            outdir=str(tensor_dir),
        )
    )
    family_id = _family_ids(sim_dir)[0]
    labels_path = tensor_dir / "families" / family_id / f"{family_id}.labels.json"
    labels = json.loads(labels_path.read_text(encoding="utf-8"))

    assert "gene_label" in labels
    assert "branch_labels" in labels
    assert "selected_sites_1based" in labels


def test_cli_build_tensors_exits_successfully(tmp_path) -> None:
    sim_dir, align_dir = _prepare_alignment(tmp_path)
    tensor_dir = tmp_path / "tensors_cli"

    result = runner.invoke(
        app,
        [
            "build-tensors",
            "--sim-dir",
            str(sim_dir),
            "--align-dir",
            str(align_dir),
            "--outdir",
            str(tensor_dir),
            "--methods",
            "identity,codon_dropout",
        ],
    )

    assert result.exit_code == 0
    assert "Tensor manifest path:" in result.output


def test_cli_validate_tensors_exits_successfully(tmp_path) -> None:
    sim_dir, align_dir = _prepare_alignment(tmp_path)
    tensor_dir = tmp_path / "tensors_cli_validate"
    build_tensor_dataset(
        TensorBuildConfig(
            sim_dir=str(sim_dir),
            align_dir=str(align_dir),
            outdir=str(tensor_dir),
        )
    )

    result = runner.invoke(app, ["validate-tensors", "--tensor-dir", str(tensor_dir)])

    assert result.exit_code == 0
    assert "Status" in result.output
    assert "ok" in result.output


def test_validate_tensors_fails_when_npz_missing(tmp_path) -> None:
    sim_dir, align_dir = _prepare_alignment(tmp_path)
    tensor_dir = tmp_path / "tensors_corrupt"
    build_tensor_dataset(
        TensorBuildConfig(
            sim_dir=str(sim_dir),
            align_dir=str(align_dir),
            outdir=str(tensor_dir),
            methods=["identity"],
        )
    )
    family_id = _family_ids(sim_dir)[0]
    tensor_path = tensor_dir / "families" / family_id / f"{family_id}.identity.tensor.npz"
    tensor_path.unlink()

    result = runner.invoke(app, ["validate-tensors", "--tensor-dir", str(tensor_dir)])

    assert result.exit_code != 0
    assert "missing tensor file" in result.output


def _prepare_alignment(tmp_path):
    sim_dir = tmp_path / "sim"
    align_dir = tmp_path / "align"
    simulate_families(
        SimulationConfig(
            outdir=str(sim_dir),
            n_families=1,
            n_taxa=4,
            n_codons=30,
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
            dropout_rate=0.1,
        )
    )
    return sim_dir, align_dir


def _family_ids(sim_dir):
    manifest = json.loads((sim_dir / "manifest.json").read_text(encoding="utf-8"))
    return manifest["family_ids"]
