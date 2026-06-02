import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from babappa.align import (
    ExternalAlignmentConfig,
    MethodPolicyConfig,
    SiteMapConfig,
    babappalign_model_status,
    build_alignment_site_maps,
    build_method_policy,
    build_site_map_for_alignment,
    detect_aligner_backends,
    run_alignment_ensemble,
    validate_method_policy_dir,
    validate_alignment_directory,
    validate_site_map_dir,
)
from babappa.benchmarks import (
    ExternalCompletedTierReportPlanConfig,
    ExternalAlignerValidationPlanConfig,
    ExternalExtremeRecoveryPlanConfig,
    plan_complete_external_tier_reports,
    plan_external_aligner_validation,
    plan_external_extreme_recovery,
)
from babappa.cli import app
from babappa.datasets import DatasetIndexConfig, build_dataset_index
from babappa.site import (
    OracleSiteLabelConfig,
    SiteDatasetConfig,
    build_site_dataset,
    extract_oracle_site_labels,
    validate_site_dataset_dir,
)
from babappa.simulate import SimulationConfig, simulate_families
from babappa.tensors import TensorBuildConfig, build_tensor_dataset

runner = CliRunner()


def test_repository_hardening_init_files() -> None:
    root = Path(__file__).resolve().parents[1]
    accidental = sorted((root / "src" / "babappa").rglob("init.py"))
    assert accidental == []
    assert (root / "src" / "babappa" / "__init__.py").exists()
    assert (root / "src" / "babappa" / "align" / "__init__.py").exists()


def test_detect_aligner_backends() -> None:
    backends = detect_aligner_backends()
    assert backends["identity"].available is True
    assert backends["codon_dropout"].available is True
    assert backends["identity"].kind == "internal"
    assert "mafft" in backends
    assert "prank" in backends
    assert "babappalign" in backends
    assert "muscle" in backends
    assert "tcoffee" in backends


def test_check_aligners_cli(tmp_path) -> None:
    json_out = tmp_path / "aligners.json"
    result = runner.invoke(app, ["check-aligners", "--json-out", str(json_out)])
    assert result.exit_code == 0
    payload = json.loads(json_out.read_text("utf-8"))
    assert payload["identity"]["available"] is True
    assert "runtime_class" in payload["identity"]
    assert "production_default" in payload["identity"]
    assert payload["prank"]["production_default"] is False
    assert payload["prank"]["default_role"] == "diagnostic_slow"


def test_babappalign_missing_model_detected(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    status = babappalign_model_status()
    assert status["model_status"] == "model_missing"
    assert status["model_present"] is False
    assert status["model_expected_path"].endswith(".cache/babappalign/models/babappascore.pt")
    assert "curl -L" in status["install_command"]


def test_check_aligners_reports_babappalign_model_missing(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    json_out = tmp_path / "aligners.json"

    result = runner.invoke(app, ["check-aligners", "--json-out", str(json_out)])

    assert result.exit_code == 0
    payload = json.loads(json_out.read_text("utf-8"))
    backend = payload["babappalign"]
    assert backend["model_status"] == "model_missing"
    assert backend["model_present"] is False
    assert "model_missing" in backend["notes"]
    assert "curl -L" in backend["install_command"]
    assert "model_missing" in result.output


def test_smoke_aligner_reports_babappalign_model_missing(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    outdir = tmp_path / "smoke"

    result = runner.invoke(
        app,
        ["smoke-aligner", "--method", "babappalign", "--outdir", str(outdir)],
    )

    assert result.exit_code == 1
    assert "babappalign_model_missing" in result.output
    payload = json.loads((outdir / "aligner_smoke_report.json").read_text("utf-8"))
    assert payload["reason"] == "babappalign_model_missing"
    assert payload["model_present"] is False


def test_muscle_detection_does_not_fail_if_absent() -> None:
    backend = detect_aligner_backends()["muscle"]
    assert backend.name == "muscle"
    assert backend.wrapper_status == "active"
    assert backend.runtime_class == "fast"
    if not backend.available:
        assert backend.production_default is False


def test_tcoffee_detection_does_not_fail_if_absent() -> None:
    backend = detect_aligner_backends()["tcoffee"]
    assert backend.name == "tcoffee"
    assert backend.wrapper_status == "active_optional_diagnostic"
    assert backend.production_default is False


def test_readme_mentions_research_alpha_and_oracle_limitation() -> None:
    root = Path(__file__).resolve().parents[1]
    text = (root / "README.md").read_text("utf-8").lower()
    assert "research-alpha" in text
    assert "oracle-supervised" in text


@pytest.mark.skipif(shutil.which("babappalign") is None, reason="babappalign executable unavailable")
def test_babappalign_detection_reports_active_if_available() -> None:
    backend = detect_aligner_backends()["babappalign"]
    assert backend.available is True
    assert backend.wrapper_status == "active"
    assert backend.command_template == "babappalign --mode codon --device cpu|cuda <input.fasta>"


def test_site_map_identity_tiny(tmp_path) -> None:
    sim_dir, align_dir = _tiny_sim_and_alignment(tmp_path, methods=["identity"])
    site_map_dir = tmp_path / "site_map"
    summary = build_alignment_site_maps(
        SiteMapConfig(sim_dir=str(sim_dir), align_dir=str(align_dir), outdir=str(site_map_dir))
    )
    assert summary["status"] == "ok"
    validation = validate_site_map_dir(site_map_dir)
    assert validation["status"] == "ok"
    map_path = site_map_dir / "families" / "family_000001" / "family_000001.identity.site_map.tsv"
    rows = _read_tsv(map_path)
    assert rows
    assert {row["mapping_status"] for row in rows} == {"unique"}
    assert {float(row["mapping_confidence"]) for row in rows} == {1.0}


def test_site_map_with_inserted_gap_synthetic(tmp_path) -> None:
    original = tmp_path / "original.fasta"
    aligned = tmp_path / "aligned.fasta"
    original.write_text(">taxon_a\nATGAAACCC\n>taxon_b\nATGAAACCC\n", encoding="utf-8")
    aligned.write_text(">taxon_a\nATG---AAACCC\n>taxon_b\nATG---AAACCC\n", encoding="utf-8")
    rows = build_site_map_for_alignment(original, aligned, family_id="f1", method="synthetic")
    assert [row["mapping_status"] for row in rows] == ["unique", "all_gap", "unique", "unique"]
    assert [row["original_site_index_zero"] for row in rows] == [0, "", 1, 2]


def test_site_label_extraction_with_site_map(tmp_path) -> None:
    sim_dir, align_dir = _tiny_sim_and_alignment(tmp_path, methods=["identity"])
    tensor_dir = tmp_path / "tensors"
    dataset_dir = tmp_path / "dataset"
    site_map_dir = tmp_path / "site_map"
    build_tensor_dataset(TensorBuildConfig(str(sim_dir), str(align_dir), str(tensor_dir)))
    build_dataset_index(DatasetIndexConfig(str(tensor_dir), str(dataset_dir)))
    build_alignment_site_maps(
        SiteMapConfig(sim_dir=str(sim_dir), align_dir=str(align_dir), outdir=str(site_map_dir))
    )
    outdir = tmp_path / "site_labels"
    summary = extract_oracle_site_labels(
        OracleSiteLabelConfig(
            dataset_dir=str(dataset_dir),
            outdir=str(outdir),
            site_map_dir=str(site_map_dir),
            aligned_site_mode="mapped",
        )
    )
    rows = _read_tsv(Path(summary["site_labels_tsv"]))
    assert rows
    assert "aligned_site_index_zero" in rows[0]
    assert "original_site_index_zero" in rows[0]
    assert {row["mapping_status"] for row in rows} == {"unique"}
    assert any(row["y_site"] == "1" for row in rows)


def test_build_site_dataset_with_mapped_labels(tmp_path) -> None:
    sim_dir, align_dir = _tiny_sim_and_alignment(tmp_path, methods=["identity"])
    tensor_dir = tmp_path / "tensors"
    dataset_dir = tmp_path / "dataset"
    site_map_dir = tmp_path / "site_map"
    labels_dir = tmp_path / "site_labels"
    site_dataset_dir = tmp_path / "site_dataset"
    build_tensor_dataset(TensorBuildConfig(str(sim_dir), str(align_dir), str(tensor_dir)))
    build_dataset_index(DatasetIndexConfig(str(tensor_dir), str(dataset_dir)))
    build_alignment_site_maps(
        SiteMapConfig(sim_dir=str(sim_dir), align_dir=str(align_dir), outdir=str(site_map_dir))
    )
    extract_oracle_site_labels(
        OracleSiteLabelConfig(
            dataset_dir=str(dataset_dir),
            outdir=str(labels_dir),
            site_map_dir=str(site_map_dir),
            aligned_site_mode="mapped",
        )
    )
    summary = build_site_dataset(
        SiteDatasetConfig(
            dataset_dir=str(dataset_dir),
            oracle_labels_tsv=str(labels_dir / "site_oracle_labels.tsv"),
            outdir=str(site_dataset_dir),
            require_mappable_sites=True,
        )
    )
    assert summary["n_site_rows"] > 0
    validation = validate_site_dataset_dir(site_dataset_dir)
    assert validation["status"] == "ok"
    rows = _read_tsv(site_dataset_dir / "site_features.tsv")
    assert "mapping_status" in rows[0]
    assert "mappable_site" in rows[0]


@pytest.mark.skipif(shutil.which("mafft") is None, reason="mafft executable unavailable")
def test_mafft_external_smoke_if_available(tmp_path) -> None:
    sim_dir = _tiny_sim(tmp_path)
    align_dir = tmp_path / "align_mafft"
    summary = run_alignment_ensemble(
        ExternalAlignmentConfig(
            sim_dir=str(sim_dir),
            outdir=str(align_dir),
            methods=["mafft"],
            require_available=True,
            timeout_seconds=60,
        )
    )
    assert summary["status"] == "ok"
    assert "mafft" in summary["methods_run"]


def test_babappalign_device_config_rejects_invalid_value(tmp_path) -> None:
    sim_dir = _tiny_sim(tmp_path)
    with pytest.raises(ValueError, match="babappalign_device"):
        ExternalAlignmentConfig(
            sim_dir=str(sim_dir),
            outdir=str(tmp_path / "align_bad_device"),
            methods=["identity"],
            babappalign_device="gpu",
        )


@pytest.mark.skipif(
    shutil.which("babappalign") is None or not babappalign_model_status()["model_present"],
    reason="babappalign executable or BABAPPAScore model unavailable",
)
def test_babappalign_tiny_smoke_if_available(tmp_path) -> None:
    sim_dir = _tiny_sim(tmp_path)
    align_dir = tmp_path / "align_babappalign"
    site_map_dir = tmp_path / "site_map_babappalign"
    summary = run_alignment_ensemble(
        ExternalAlignmentConfig(
            sim_dir=str(sim_dir),
            outdir=str(align_dir),
            methods=["babappalign"],
            require_available=True,
            timeout_seconds=120,
        )
    )
    assert summary["status"] == "ok"
    assert "babappalign" in summary["methods_run"]
    assert validate_alignment_directory(align_dir)["status"] == "ok"
    build_alignment_site_maps(
        SiteMapConfig(sim_dir=str(sim_dir), align_dir=str(align_dir), outdir=str(site_map_dir))
    )
    validation = validate_site_map_dir(
        site_map_dir,
        methods=["babappalign"],
        max_conflict_fraction=0.03,
    )
    assert validation["status"] == "ok"
    payload = json.loads((site_map_dir / "site_map_manifest.json").read_text("utf-8"))
    assert payload["frame_error_fraction"] == 0.0


def test_external_aligner_plan_includes_babappalign_by_default(tmp_path) -> None:
    outdir = tmp_path / "external_plan"
    summary = plan_external_aligner_validation(
        ExternalAlignerValidationPlanConfig(
            panel_dir="saturation_panel_external_1k",
            outdir=str(outdir),
        )
    )
    commands = (outdir / "external_aligner_validation_commands.sh").read_text("utf-8")
    expected = json.loads((outdir / "expected_external_outputs.json").read_text("utf-8"))
    assert summary["status"] == "ok"
    assert expected["effective_methods"] == ["identity", "mafft", "babappalign", "muscle"]
    lines = commands.splitlines()
    assert lines[:4] == [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "source /home/rajamosai/miniconda3/etc/profile.d/conda.sh",
        "conda activate molevo",
    ]
    assert "--methods identity,mafft,babappalign,muscle" in commands
    assert "--methods identity,mafft,prank,babappalign" not in commands
    assert "aligner-method-policy" in commands
    assert "codon_dropout" not in commands
    assert "MANUAL EXECUTION SCRIPT" in commands
    assert "prank" not in expected["effective_methods"]


def test_external_aligner_plan_default_methods_exclude_prank_include_fast_methods(tmp_path) -> None:
    outdir = tmp_path / "external_plan_fast"
    plan_external_aligner_validation(
        ExternalAlignerValidationPlanConfig(
            panel_dir="saturation_panel_external_1k",
            outdir=str(outdir),
        )
    )
    expected = json.loads((outdir / "expected_external_outputs.json").read_text("utf-8"))
    commands = (outdir / "external_aligner_validation_commands.sh").read_text("utf-8")
    assert "prank" not in expected["effective_methods"]
    assert expected["effective_methods"] == ["identity", "mafft", "babappalign", "muscle"]
    assert "identity,mafft,babappalign,muscle" in commands
    assert "MANUAL EXECUTION SCRIPT" in commands


def test_external_aligner_plan_accepts_conda_overrides(tmp_path) -> None:
    outdir = tmp_path / "external_plan_conda"
    plan_external_aligner_validation(
        ExternalAlignerValidationPlanConfig(
            panel_dir="saturation_panel_external_1k",
            outdir=str(outdir),
            conda_sh="/opt/miniconda/etc/profile.d/conda.sh",
            conda_env="babappa-test",
        )
    )
    commands = (outdir / "external_aligner_validation_commands.sh").read_text("utf-8")
    expected = json.loads((outdir / "expected_external_outputs.json").read_text("utf-8"))
    assert commands.splitlines()[:4] == [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "source /opt/miniconda/etc/profile.d/conda.sh",
        "conda activate babappa-test",
    ]
    assert expected["conda_sh"] == "/opt/miniconda/etc/profile.d/conda.sh"
    assert expected["conda_env"] == "babappa-test"


def test_completion_and_extreme_planners_write_user_run_scripts(tmp_path) -> None:
    complete_dir = tmp_path / "complete"
    extreme_dir = tmp_path / "extreme"
    complete = plan_complete_external_tier_reports(
        ExternalCompletedTierReportPlanConfig(
            tiers=["low", "moderate", "high"],
            outdir=str(complete_dir),
        )
    )
    extreme = plan_external_extreme_recovery(
        ExternalExtremeRecoveryPlanConfig(
            panel_dir="saturation_panel_external_1k",
            outdir=str(extreme_dir),
            methods=["identity", "mafft", "babappalign", "muscle"],
        )
    )

    complete_script = Path(complete["commands"]).read_text("utf-8")
    extreme_script = Path(extreme["commands"]).read_text("utf-8")
    assert "MANUAL EXECUTION SCRIPT" in complete_script
    assert "calibrate-site-neural" in complete_script
    assert "aggregation-threshold-policy" in complete_script
    assert "MANUAL EXECUTION SCRIPT" in extreme_script
    assert "--methods identity,mafft,babappalign,muscle" in extreme_script
    assert "prank" not in json.loads((extreme_dir / "expected_outputs.json").read_text("utf-8"))["effective_methods"]


def test_method_policy_quarantines_frame_error_method(tmp_path) -> None:
    align_dir, site_map_dir = _synthetic_policy_inputs(
        tmp_path,
        method="prank",
        failed=0,
        frame_error_fraction=0.01,
    )
    outdir = tmp_path / "policy"

    summary = build_method_policy(
        MethodPolicyConfig(
            align_dir=str(align_dir),
            site_map_dir=str(site_map_dir),
            outdir=str(outdir),
            max_frame_error_fraction=0.0,
        )
    )

    assert summary["status"] == "ok"
    assert summary["quarantined_methods"] == ["prank"]
    assert validate_method_policy_dir(outdir)["status"] == "ok"
    rows = _read_tsv(outdir / "method_policy.tsv")
    assert rows[0]["recommendation"] == "quarantine"
    assert "frame_error_fraction" in rows[0]["reason"]


def test_method_policy_allows_low_failure_fraction_without_site_map_errors(tmp_path) -> None:
    align_dir, site_map_dir = _synthetic_policy_inputs(
        tmp_path,
        method="babappalign",
        attempted=250,
        successful=249,
        failed=1,
        frame_error_fraction=0.0,
        conflict_fraction=0.0,
    )
    outdir = tmp_path / "policy_low_failure"

    summary = build_method_policy(
        MethodPolicyConfig(
            align_dir=str(align_dir),
            site_map_dir=str(site_map_dir),
            outdir=str(outdir),
            max_method_failure_fraction=0.01,
        )
    )

    assert summary["usable_methods"] == ["babappalign"]
    rows = _read_tsv(outdir / "method_policy.tsv")
    assert rows[0]["recommendation"] == "usable"
    assert rows[0]["reason"] == "low_failure_fraction_accepted"


def _tiny_sim_and_alignment(tmp_path, methods):
    sim_dir = _tiny_sim(tmp_path)
    align_dir = tmp_path / "align"
    summary = run_alignment_ensemble(
        ExternalAlignmentConfig(
            sim_dir=str(sim_dir),
            outdir=str(align_dir),
            methods=list(methods),
            require_available=True,
        )
    )
    assert summary["status"] == "ok"
    return sim_dir, align_dir


def _tiny_sim(tmp_path):
    sim_dir = tmp_path / "sim"
    simulate_families(
        SimulationConfig(
            outdir=str(sim_dir),
            n_families=1,
            n_taxa=4,
            n_codons=30,
            seed=42,
            positive_rate=1.0,
            selected_site_fraction=0.1,
            saturation_tier="moderate",
        )
    )
    return sim_dir


def _read_tsv(path: Path):
    text = path.read_text("utf-8").strip().splitlines()
    header = text[0].split("\t")
    return [dict(zip(header, line.split("\t"))) for line in text[1:]]


def _synthetic_policy_inputs(
    tmp_path,
    method: str,
    attempted: int = 250,
    successful: int = 250,
    failed: int = 0,
    frame_error_fraction: float = 0.0,
    conflict_fraction: float = 0.0,
):
    align_dir = tmp_path / f"align_{method}"
    site_map_dir = tmp_path / f"site_map_{method}"
    align_dir.mkdir()
    site_map_dir.mkdir()
    failure_fraction = 0.0 if attempted == 0 else failed / attempted
    (align_dir / "alignment_manifest.json").write_text(
        json.dumps(
            {
                "alignment_manifest_version": "test",
                "n_families": attempted,
                "family_ids": [f"family_{index:06d}" for index in range(attempted)],
                "methods": [method],
                "methods_requested": [method],
                "method_status": {
                    method: {
                        "attempted_families": attempted,
                        "successful_families": successful,
                        "failed_families": failed,
                        "failure_fraction": failure_fraction,
                        "failure_reasons": [],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    (site_map_dir / "site_map_manifest.json").write_text(
        json.dumps(
            {
                "site_map_version": "test",
                "method_summary": [
                    {
                        "method": method,
                        "n_family_method_maps": successful,
                        "total_aligned_sites": successful * 300,
                        "unique_fraction": 1.0 - frame_error_fraction - conflict_fraction,
                        "conflict_fraction": conflict_fraction,
                        "all_gap_fraction": 0.0,
                        "frame_error_fraction": frame_error_fraction,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return align_dir, site_map_dir
