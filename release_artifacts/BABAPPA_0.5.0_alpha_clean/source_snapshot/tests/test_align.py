import json

from typer.testing import CliRunner

from babappa.align import (
    AlignmentConfig,
    align_simulation_directory,
    validate_alignment_directory,
)
from babappa.cli import app
from babappa.simulate import SimulationConfig, simulate_families
from babappa.simulate.audit import read_fasta


runner = CliRunner()


def test_align_simulation_directory_writes_expected_files(tmp_path) -> None:
    sim_dir = _simulate_small(tmp_path / "sim")
    align_dir = tmp_path / "align"

    summary = align_simulation_directory(
        AlignmentConfig(sim_dir=str(sim_dir), outdir=str(align_dir))
    )
    family_id = summary["family_ids"][0] if "family_ids" in summary else "family_000001"
    family_dir = align_dir / "families" / family_id

    assert (align_dir / "alignment_manifest.json").exists()
    assert (family_dir / f"{family_id}.identity.codon.fasta").exists()
    assert (family_dir / f"{family_id}.codon_dropout.codon.fasta").exists()
    assert (family_dir / f"{family_id}.identity.qc.json").exists()
    assert (family_dir / f"{family_id}.identity.map.tsv").exists()


def test_validate_alignment_directory_succeeds(tmp_path) -> None:
    sim_dir = _simulate_small(tmp_path / "sim_validate")
    align_dir = tmp_path / "align_validate"
    align_simulation_directory(
        AlignmentConfig(sim_dir=str(sim_dir), outdir=str(align_dir))
    )

    summary = validate_alignment_directory(align_dir)

    assert summary["status"] == "ok"
    assert summary["n_fail"] == 0


def test_identity_alignment_matches_original_fasta(tmp_path) -> None:
    sim_dir = _simulate_small(tmp_path / "sim_identity")
    align_dir = tmp_path / "align_identity"
    align_simulation_directory(
        AlignmentConfig(
            sim_dir=str(sim_dir),
            outdir=str(align_dir),
            methods=["identity"],
        )
    )
    family_id = _family_ids(sim_dir)[0]

    source_fasta = sim_dir / "families" / family_id / f"{family_id}.fasta"
    identity_fasta = (
        align_dir / "families" / family_id / f"{family_id}.identity.codon.fasta"
    )

    assert identity_fasta.read_text(encoding="utf-8") == source_fasta.read_text(
        encoding="utf-8"
    )


def test_codon_dropout_preserves_sequence_length_and_frame(tmp_path) -> None:
    sim_dir = _simulate_small(tmp_path / "sim_dropout")
    align_dir = tmp_path / "align_dropout"
    align_simulation_directory(
        AlignmentConfig(
            sim_dir=str(sim_dir),
            outdir=str(align_dir),
            methods=["codon_dropout"],
            dropout_rate=0.5,
        )
    )
    family_id = _family_ids(sim_dir)[0]
    source_records = read_fasta(sim_dir / "families" / family_id / f"{family_id}.fasta")
    dropout_records = read_fasta(
        align_dir / "families" / family_id / f"{family_id}.codon_dropout.codon.fasta"
    )

    for record_id, source_sequence in source_records.items():
        dropout_sequence = dropout_records[record_id]
        assert len(dropout_sequence) == len(source_sequence)
        assert len(dropout_sequence) % 3 == 0


def test_cli_align_sim_exits_successfully(tmp_path) -> None:
    sim_dir = _simulate_small(tmp_path / "sim_cli_align")
    align_dir = tmp_path / "align_cli"

    result = runner.invoke(
        app,
        [
            "align-sim",
            "--sim-dir",
            str(sim_dir),
            "--outdir",
            str(align_dir),
            "--methods",
            "identity,codon_dropout",
            "--seed",
            "42",
            "--dropout-rate",
            "0.02",
        ],
    )

    assert result.exit_code == 0
    assert "Alignment manifest path:" in result.output


def test_cli_validate_align_exits_successfully(tmp_path) -> None:
    sim_dir = _simulate_small(tmp_path / "sim_cli_validate_align")
    align_dir = tmp_path / "align_cli_validate"
    align_simulation_directory(
        AlignmentConfig(sim_dir=str(sim_dir), outdir=str(align_dir))
    )

    result = runner.invoke(app, ["validate-align", "--align-dir", str(align_dir)])

    assert result.exit_code == 0
    assert "BABAPPA Alignment Validation Summary" in result.output


def test_invalid_alignment_method_exits_nonzero(tmp_path) -> None:
    sim_dir = _simulate_small(tmp_path / "sim_bad_method")

    result = runner.invoke(
        app,
        [
            "align-sim",
            "--sim-dir",
            str(sim_dir),
            "--outdir",
            str(tmp_path / "align_bad_method"),
            "--methods",
            "identity,bad_method",
        ],
    )

    assert result.exit_code != 0
    assert "unknown alignment method" in result.output


def test_validate_align_fails_when_qc_json_missing(tmp_path) -> None:
    sim_dir = _simulate_small(tmp_path / "sim_corrupt_align")
    align_dir = tmp_path / "align_corrupt"
    align_simulation_directory(
        AlignmentConfig(sim_dir=str(sim_dir), outdir=str(align_dir))
    )
    family_id = _family_ids(sim_dir)[0]
    qc_path = align_dir / "families" / family_id / f"{family_id}.identity.qc.json"
    qc_path.unlink()

    result = runner.invoke(app, ["validate-align", "--align-dir", str(align_dir)])

    assert result.exit_code != 0
    assert "missing QC JSON" in result.output


def _simulate_small(sim_dir):
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
    return sim_dir


def _family_ids(sim_dir):
    manifest = json.loads((sim_dir / "manifest.json").read_text(encoding="utf-8"))
    return manifest["family_ids"]
