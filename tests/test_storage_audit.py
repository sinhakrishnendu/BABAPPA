from pathlib import Path

from babappa.maintenance import StorageAuditConfig, audit_storage


def _write(path: Path, text: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_storage_audit_classifies_source_as_keep(tmp_path):
    _write(tmp_path / "src" / "babappa" / "__init__.py", "__version__ = 'x'\n")
    summary = audit_storage(StorageAuditConfig(root=str(tmp_path), outdir=str(tmp_path / "audit")))
    assert summary["n_keep"] >= 1
    inventory = (tmp_path / "audit" / "storage_inventory.tsv").read_text(encoding="utf-8")
    assert "src\t" in inventory
    assert "protected source, report, package, or evidence artifact" in inventory


def test_storage_audit_classifies_raw_simulation_dir_as_remove(tmp_path):
    _write(tmp_path / "sim_explicit_branch_truth_100k_mps_low" / "family_0001.truth.json", "{}")
    audit_storage(StorageAuditConfig(root=str(tmp_path), outdir=str(tmp_path / "audit")))
    remove = (tmp_path / "audit" / "remove_candidates.tsv").read_text(encoding="utf-8")
    assert "sim_explicit_branch_truth_100k_mps_low" in remove
    assert "reproducible generated output or raw download" in remove


def test_storage_audit_classifies_deployable_package_as_keep(tmp_path):
    _write(
        tmp_path
        / "deployable_model_conservative_branch_site_100k_mps"
        / "tier_models"
        / "moderate"
        / "branch_site_neural_checkpoint.pt",
        "tiny",
    )
    audit_storage(StorageAuditConfig(root=str(tmp_path), outdir=str(tmp_path / "audit")))
    keep = (tmp_path / "audit" / "keep_list.tsv").read_text(encoding="utf-8")
    assert "deployable_model_conservative_branch_site_100k_mps" in keep
    assert "branch_site_neural_checkpoint.pt" in keep


def test_quarantine_script_contains_no_permanent_delete_command(tmp_path):
    _write(tmp_path / "saturation_panel_100000" / "data.tsv", "x")
    audit_storage(StorageAuditConfig(root=str(tmp_path), outdir=str(tmp_path / "audit")))
    script = (tmp_path / "audit" / "quarantine_large_reproducible_outputs.sh").read_text(
        encoding="utf-8"
    )
    assert "MANUAL EXECUTION SCRIPT" in script
    assert "rm -rf" not in script
    assert " mv " in script or "\nmv " in script


def test_delete_script_requires_confirm_yes(tmp_path):
    audit_storage(StorageAuditConfig(root=str(tmp_path), outdir=str(tmp_path / "audit")))
    script = (tmp_path / "audit" / "delete_quarantine_after_review.sh").read_text(
        encoding="utf-8"
    )
    assert 'CONFIRM_DELETE="${CONFIRM_DELETE:-NO}"' in script
    assert 'CONFIRM_DELETE" != "YES"' in script
    assert "rm -rf" in script


def test_gitignore_contains_large_output_patterns():
    text = Path(".gitignore").read_text(encoding="utf-8")
    assert "sim_explicit_branch_truth_*/" in text
    assert "branch_site_dataset_explicit_branch_truth_*/" in text
    assert "real_empirical_pilot/input/raw_downloads/" in text
    assert "real_empirical_pilot/calibration_runs/" in text
    assert ".stage_partial_*" in text
    assert "!deployable_model_conservative_branch_site_100k_mps/**" in text
