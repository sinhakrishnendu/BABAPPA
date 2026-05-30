import json

from typer.testing import CliRunner

from babappa.cli import app
from babappa.simulate import SimulationConfig, simulate_families
from babappa.simulate.audit import audit_simulation_directory, compute_family_audit


runner = CliRunner()


def test_audit_simulation_directory_writes_outputs(tmp_path) -> None:
    sim_dir = tmp_path / "sim"
    simulate_families(
        SimulationConfig(
            outdir=str(sim_dir),
            n_families=2,
            n_taxa=4,
            n_codons=30,
            seed=42,
        )
    )

    summary = audit_simulation_directory(sim_dir)

    assert (sim_dir / "audit" / "family_audit.tsv").exists()
    assert (sim_dir / "audit" / "dataset_summary.json").exists()
    assert summary["n_families_audited"] == 2
    assert summary["n_fail"] == 0


def test_family_audit_contains_saturation_and_distance_fields(tmp_path) -> None:
    sim_dir = tmp_path / "sim_fields"
    simulate_families(
        SimulationConfig(
            outdir=str(sim_dir),
            n_families=1,
            n_taxa=4,
            n_codons=30,
            seed=43,
            saturation_tier="moderate",
        )
    )
    family_id = _family_ids(sim_dir)[0]

    audit = compute_family_audit(sim_dir / "families" / family_id)

    assert audit["saturation_tier"] == "moderate"
    assert "mean_pairwise_nt_pdist" in audit
    assert "codon_pos1_pdist" in audit
    assert "codon_pos2_pdist" in audit
    assert "codon_pos3_pdist" in audit
    assert "ti_tv_ratio" in audit


def test_cli_audit_sim_exits_successfully_on_valid_simulation(tmp_path) -> None:
    sim_dir = tmp_path / "sim_cli_audit"
    simulate_families(
        SimulationConfig(
            outdir=str(sim_dir),
            n_families=1,
            n_taxa=4,
            n_codons=30,
            seed=44,
        )
    )

    result = runner.invoke(
        app,
        [
            "audit-sim",
            "--sim-dir",
            str(sim_dir),
            "--outdir",
            str(sim_dir / "audit"),
        ],
    )

    assert result.exit_code == 0
    assert "BABAPPA Simulation Audit Summary" in result.output


def test_cli_audit_sim_exits_nonzero_on_missing_manifest(tmp_path) -> None:
    result = runner.invoke(app, ["audit-sim", "--sim-dir", str(tmp_path / "missing")])

    assert result.exit_code != 0
    assert "missing manifest.json" in result.output


def test_deleted_family_fasta_marks_audit_fail(tmp_path) -> None:
    sim_dir = tmp_path / "sim_corrupt"
    simulate_families(
        SimulationConfig(
            outdir=str(sim_dir),
            n_families=1,
            n_taxa=4,
            n_codons=30,
            seed=45,
        )
    )
    family_id = _family_ids(sim_dir)[0]
    fasta_path = sim_dir / "families" / family_id / f"{family_id}.fasta"
    fasta_path.unlink()

    summary = audit_simulation_directory(sim_dir)

    assert summary["n_fail"] == 1


def _family_ids(sim_dir):
    manifest = json.loads((sim_dir / "manifest.json").read_text(encoding="utf-8"))
    return manifest["family_ids"]
