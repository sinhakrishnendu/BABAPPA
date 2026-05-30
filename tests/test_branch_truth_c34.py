import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from babappa.align import AlignmentConfig, align_simulation_directory
from babappa.branch import (
    BranchSiteOracleLabelConfig,
    BranchTruthStatusAuditConfig,
    ExplicitBranchTruth1kPlanConfig,
    audit_branch_truth_status,
    extract_branch_site_labels,
    plan_explicit_branch_truth_1k,
)
from babappa.cli import app
from babappa.datasets import DatasetIndexConfig, build_dataset_index, read_tsv
from babappa.simulate import SimulationConfig, simulate_families
from babappa.tensors import TensorBuildConfig, build_tensor_dataset


runner = CliRunner()


def test_simulator_emits_branch_truth_json_for_tiny_simulation(tmp_path) -> None:
    sim_dir = tmp_path / "sim"
    summary = simulate_families(
        SimulationConfig(
            outdir=str(sim_dir),
            n_families=1,
            n_taxa=4,
            n_codons=30,
            seed=101,
            positive_rate=1.0,
        )
    )
    family_id = summary["family_ids"][0]
    branch_truth_path = sim_dir / "families" / family_id / f"{family_id}.branch_truth.json"
    branch_truth = json.loads(branch_truth_path.read_text("utf-8"))

    assert branch_truth_path.exists()
    assert branch_truth["truth_source"] == "explicit_simulator_branch_truth"
    assert branch_truth["branch_truth_version"] == "0.1"
    assert branch_truth["branch_site_records"]


def test_branch_site_truth_tsv_exists_and_has_required_columns(tmp_path) -> None:
    sim_dir = tmp_path / "sim"
    simulate_families(
        SimulationConfig(outdir=str(sim_dir), n_families=1, n_taxa=4, n_codons=30, seed=102)
    )
    rows = read_tsv(sim_dir / "branch_site_truth.tsv")

    assert rows
    assert {
        "family_id",
        "saturation_tier",
        "branch_id",
        "foreground_taxon",
        "branch_type",
        "site_index_zero",
        "site_index_one",
        "y_branch_site",
        "selection_event_id",
        "truth_source",
    } <= set(rows[0])
    assert {row["truth_source"] for row in rows} == {"explicit_simulator_branch_truth"}


def test_validate_sim_require_branch_truth_passes_on_new_simulation(tmp_path) -> None:
    sim_dir = tmp_path / "sim"
    simulate_families(
        SimulationConfig(outdir=str(sim_dir), n_families=1, n_taxa=4, n_codons=30, seed=103)
    )

    result = runner.invoke(app, ["validate-sim", "--sim-dir", str(sim_dir), "--require-branch-truth"])

    assert result.exit_code == 0, result.output
    assert "Branch truth status: explicit_truth_ok" in result.output


def test_validate_sim_require_branch_truth_fails_when_missing(tmp_path) -> None:
    sim_dir = tmp_path / "sim"
    summary = simulate_families(
        SimulationConfig(outdir=str(sim_dir), n_families=1, n_taxa=4, n_codons=30, seed=104)
    )
    family_id = summary["family_ids"][0]
    (sim_dir / "families" / family_id / f"{family_id}.branch_truth.json").unlink()
    (sim_dir / "branch_truth_manifest.json").unlink()
    (sim_dir / "branch_site_truth.tsv").unlink()

    result = runner.invoke(app, ["validate-sim", "--sim-dir", str(sim_dir), "--require-branch-truth"])

    assert result.exit_code != 0
    assert "missing branch truth file" in result.output


def test_extract_branch_site_labels_truth_mode_explicit_uses_explicit_truth(tmp_path) -> None:
    dataset_dir = _build_identity_dataset(tmp_path, seed=105)
    outdir = tmp_path / "branch_labels"

    summary = extract_branch_site_labels(
        BranchSiteOracleLabelConfig(
            dataset_dir=str(dataset_dir),
            outdir=str(outdir),
            truth_mode="explicit",
            aligned_site_mode="original",
            streaming_output=False,
        )
    )
    payload = json.loads((outdir / "branch_site_oracle_summary.json").read_text("utf-8"))
    rows = read_tsv(outdir / "branch_site_oracle_labels.tsv")

    assert summary["branch_site_labels_status"] == "explicit_simulator_branch_truth"
    assert payload["explicit_branch_site_truth_available"] is True
    assert payload["proxy_labels_used"] is False
    assert {row["branch_label_source"] for row in rows} == {"explicit_simulator_branch_truth"}


def test_extract_branch_site_labels_truth_mode_required_fails_if_absent(tmp_path) -> None:
    dataset_dir = _build_identity_dataset(tmp_path, seed=106)
    labels_path = next((tmp_path / "tensors" / "families").glob("*/*.labels.json"))
    labels = json.loads(labels_path.read_text("utf-8"))
    Path(labels["branch_truth_file"]).unlink()

    with pytest.raises(ValueError, match="explicit branch-site truth is required"):
        extract_branch_site_labels(
            BranchSiteOracleLabelConfig(
                dataset_dir=str(dataset_dir),
                outdir=str(tmp_path / "branch_labels_required"),
                truth_mode="required",
                aligned_site_mode="original",
            )
        )


def test_truth_audit_reports_explicit_truth_ok_when_explicit_used(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    _write_label_summary(tmp_path, "low", "explicit_simulator_branch_truth", explicit=True, proxy=False)

    audit = audit_branch_truth_status(
        BranchTruthStatusAuditConfig(tiers="low", outdir="truth_audit_explicit")
    )
    payload = json.loads((tmp_path / "truth_audit_explicit" / "branch_truth_status_audit.json").read_text("utf-8"))

    assert audit["n_warning"] == 0
    assert payload["tier_records"][0]["audit_status"] == "explicit_truth_ok"


def test_truth_audit_reports_proxy_warning_when_proxy_used(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    _write_label_summary(tmp_path, "low", "proxy_from_foreground_taxon", explicit=False, proxy=True)

    audit_branch_truth_status(
        BranchTruthStatusAuditConfig(tiers="low", outdir="truth_audit_proxy")
    )
    payload = json.loads((tmp_path / "truth_audit_proxy" / "branch_truth_status_audit.json").read_text("utf-8"))

    assert payload["tier_records"][0]["audit_status"] == "proxy_warning"
    assert payload["warnings"]


def test_explicit_branch_truth_1k_planner_generates_scripts_without_execution(tmp_path) -> None:
    outdir = tmp_path / "explicit_branch_truth_1k_plan"

    summary = plan_explicit_branch_truth_1k(
        ExplicitBranchTruth1kPlanConfig(
            outdir=str(outdir),
            n_families_per_tier=250,
            tiers="low,extreme",
            methods="identity,mafft",
        )
    )
    run_text = (outdir / "run_explicit_branch_truth_1k.sh").read_text("utf-8")
    expected = json.loads((outdir / "expected_outputs.json").read_text("utf-8"))

    assert summary["does_not_run_jobs"] is True
    assert "--require-branch-truth" in run_text
    assert "--truth-mode explicit" in run_text
    assert "build-site-map --sim-dir" in run_text
    assert "aligner-method-policy --align-dir" in run_text
    build_site_map_lines = [
        line for line in run_text.splitlines()
        if line.startswith("babappa build-site-map")
    ]
    assert build_site_map_lines
    assert all("--sim-dir" in line for line in build_site_map_lines)
    assert all("--align-dir" in line for line in build_site_map_lines)
    assert all("--outdir" in line for line in build_site_map_lines)
    assert all("--methods" in line for line in build_site_map_lines)
    assert all("babappa build-site-map --sim-dir" in line for line in build_site_map_lines)

    method_policy_lines = [
        line for line in run_text.splitlines()
        if line.startswith("babappa aligner-method-policy")
    ]
    assert method_policy_lines
    assert all("--align-dir" in line for line in method_policy_lines)
    assert all("--site-map-dir" in line for line in method_policy_lines)
    assert all("--outdir" in line for line in method_policy_lines)
    assert all("--max-conflict-fraction" in line for line in method_policy_lines)
    assert all("--max-frame-error-fraction" in line for line in method_policy_lines)
    assert all("--max-method-failure-fraction" in line for line in method_policy_lines)
    assert all("babappa aligner-method-policy --align-dir" in line for line in method_policy_lines)

    validate_method_policy_lines = [
        line for line in run_text.splitlines()
        if line.startswith("babappa validate-aligner-method-policy")
    ]
    assert validate_method_policy_lines
    assert all("--policy-dir method_policy_" in line for line in validate_method_policy_lines)

    build_tensor_lines = [
        line for line in run_text.splitlines()
        if line.startswith("babappa build-tensors")
    ]
    assert build_tensor_lines
    assert all("--sim-dir" in line for line in build_tensor_lines)
    assert all("--align-dir" in line for line in build_tensor_lines)
    assert all("--outdir" in line for line in build_tensor_lines)
    assert all("--methods" in line for line in build_tensor_lines)

    validate_tensor_lines = [
        line for line in run_text.splitlines()
        if line.startswith("babappa validate-tensors")
    ]
    assert validate_tensor_lines
    assert all("--tensor-dir tensors_" in line for line in validate_tensor_lines)

    index_lines = [
        line for line in run_text.splitlines()
        if line.startswith("babappa index-dataset")
    ]
    assert index_lines
    assert all("--tensor-dir" in line for line in index_lines)
    assert all("--outdir" in line for line in index_lines)

    label_lines = [
        line for line in run_text.splitlines()
        if line.startswith("babappa extract-branch-site-labels")
    ]
    assert label_lines
    assert all("--truth-mode explicit" in line for line in label_lines)

    branch_dataset_lines = [
        line for line in run_text.splitlines()
        if line.startswith("babappa build-branch-site-dataset")
    ]
    assert branch_dataset_lines
    assert all("--streaming" in line for line in branch_dataset_lines)
    assert all("--max-output-rows" in line for line in branch_dataset_lines)
    assert expected["plan_only"] is True


def test_branch_truth_schema_supports_branch_specific_labels(tmp_path) -> None:
    sim_dir = tmp_path / "sim"
    summary = simulate_families(
        SimulationConfig(
            outdir=str(sim_dir),
            n_families=1,
            n_taxa=4,
            n_codons=30,
            seed=107,
            positive_rate=1.0,
            selected_site_fraction=0.1,
        )
    )
    family_id = summary["family_ids"][0]
    branch_truth = json.loads(
        (sim_dir / "families" / family_id / f"{family_id}.branch_truth.json").read_text("utf-8")
    )
    foreground = branch_truth["foreground_branches"][0]["branch_id"]
    selected_site = branch_truth["foreground_branches"][0]["selected_sites_zero"][0]
    same_site = [
        row for row in branch_truth["branch_site_records"]
        if row["site_index_zero"] == selected_site
    ]

    assert any(row["branch_id"] == foreground and row["y_branch_site"] == 1 for row in same_site)
    assert any(row["branch_id"] != foreground and row["y_branch_site"] == 0 for row in same_site)


def _build_identity_dataset(tmp_path: Path, seed: int) -> Path:
    sim_dir = tmp_path / "sim"
    align_dir = tmp_path / "align"
    tensor_dir = tmp_path / "tensors"
    dataset_dir = tmp_path / "dataset"
    simulate_families(
        SimulationConfig(
            outdir=str(sim_dir),
            n_families=1,
            n_taxa=4,
            n_codons=30,
            seed=seed,
            positive_rate=1.0,
            selected_site_fraction=0.1,
        )
    )
    align_simulation_directory(AlignmentConfig(sim_dir=str(sim_dir), outdir=str(align_dir), methods=["identity"]))
    build_tensor_dataset(TensorBuildConfig(sim_dir=str(sim_dir), align_dir=str(align_dir), outdir=str(tensor_dir), methods=["identity"]))
    build_dataset_index(DatasetIndexConfig(tensor_dir=str(tensor_dir), outdir=str(dataset_dir), methods=["identity"], seed=seed))
    return dataset_dir


def _write_label_summary(tmp_path: Path, tier: str, status: str, explicit: bool, proxy: bool) -> None:
    label_dir = tmp_path / f"branch_site_oracle_fast_external_10k_{tier}"
    label_dir.mkdir(parents=True)
    rows = (
        "family_id\tmethod\tsplit\tsaturation_tier\tbranch_id\tforeground_taxon\tsite_index_zero\ty_branch_site\ty_site\tgene_label\tbranch_label_source\n"
        f"family_1\tidentity\ttest\t{tier}\ttaxon_001\ttaxon_001\t0\t1\t1\t1\t{status}\n"
    )
    (label_dir / "branch_site_oracle_labels.tsv").write_text(rows, encoding="utf-8")
    (label_dir / "branch_site_oracle_summary.json").write_text(
        json.dumps(
            {
                "branch_site_labels_status": status,
                "explicit_branch_site_truth_available": explicit,
                "proxy_labels_used": proxy,
                "n_branch_site_rows": 1,
                "n_positive_branch_sites": 1,
                "status_counts": {status: 1},
                "branch_label_source_counts": {status: 1},
                "generated_files": {
                    "labels_tsv": str(label_dir / "branch_site_oracle_labels.tsv")
                },
                "warnings": [],
            }
        ),
        encoding="utf-8",
    )
