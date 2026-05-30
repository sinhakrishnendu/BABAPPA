import json
from pathlib import Path

import pytest

from babappa.datasets.index import write_tsv
from babappa.deploy.package import (
    DeployableModelPackageConfig,
    DeployableModelPackageValidationConfig,
    DeployableModelSmokeConfig,
    package_deployable_model,
    smoke_load_deployable_model,
    validate_deployable_model_package,
)
from babappa.empirical.calibration import (
    EmpiricalScoringPlanConfig,
    SimulationMatchedCalibrationPlanConfig,
    plan_empirical_scoring,
    plan_simulation_matched_calibration,
)

TIERS = ["low", "moderate", "high", "extreme"]


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _synthetic_artifacts(base: Path) -> tuple[list[str], list[str], Path, Path]:
    model_dirs = []
    calibration_dirs = []
    for tier in TIERS:
        model_dir = base / f"branch_site_neural_explicit_branch_truth_100k_mps_{tier}_streamed"
        calibration_dir = base / f"branch_site_calibration_explicit_branch_truth_100k_mps_{tier}_streamed"
        model_dir.mkdir(parents=True)
        calibration_dir.mkdir(parents=True)
        (model_dir / "branch_site_neural_checkpoint.pt").write_bytes(b"fake torch checkpoint")
        _write_json(
            model_dir / "branch_site_neural_model_meta.json",
            {
                "feature_columns": ["site_relative_position", "n_taxa"],
                "feature_policy": "conservative_branch_site",
                "n_features": 2,
                "hidden_dim": 64,
                "dropout": 0.1,
                "device": "mps",
            },
        )
        _write_json(
            model_dir / "branch_site_neural_metrics.json",
            {
                "metrics_by_split": {
                    "test": {
                        "auroc": 0.99,
                        "f1": 0.95,
                        "mcc": 0.9,
                        "precision": 0.94,
                        "recall": 0.96,
                    }
                }
            },
        )
        write_tsv(model_dir / "branch_site_neural_history.tsv", [{"epoch": 1}], ["epoch"])
        _write_json(
            calibration_dir / "branch_site_calibration.json",
            {"temperature": 1.0, "selected_threshold": 0.1, "target_fdr": 0.1, "warnings": []},
        )
        _write_json(calibration_dir / "branch_site_calibrated_metrics.json", {"selected_threshold": 0.1})
        (calibration_dir / "branch_site_calibration.md").write_text("# Calibration\n", encoding="utf-8")
        model_dirs.append(str(model_dir))
        calibration_dirs.append(str(calibration_dir))

    truth = base / "branch_truth_status_audit_explicit_branch_truth_100k_mps"
    truth.mkdir()
    rows = [
        {
            "tier": tier,
            "audit_status": "explicit_truth_ok",
            "explicit_branch_site_truth_available": "True",
            "proxy_from_foreground_taxon": "False",
        }
        for tier in TIERS
    ]
    write_tsv(truth / "branch_truth_status_audit.tsv", rows, list(rows[0]))

    report = base / "explicit_branch_truth_100k_mps_final_validation_report.json"
    _write_json(
        report,
        {
            "decision": {"status": "CONDITIONAL PASS", "reason": "synthetic"},
            "run_identity": {"run_name": "explicit_branch_truth_100k_mps"},
            "neural_rows": [
                {"tier": tier, "split": "test", "auroc": "0.99", "f1": "0.95", "mcc": "0.9"}
                for tier in TIERS
            ],
            "branch_aggregation_rows": [
                {"tier": tier, "split": "all", "auroc": "0.999"} for tier in TIERS
            ],
            "gene_aggregation_rows": [
                {"tier": tier, "split": "all", "auroc": "0.999"} for tier in TIERS
            ],
            "controls_rows": [
                {"tier": tier, "control": "global_shuffled_branch_labels", "mean_auroc": "0.5"}
                for tier in TIERS
            ],
            "tier_summary": [{"tier": tier, "status": "complete"} for tier in TIERS],
            "scientific_cautions": ["simulation_supervised_only"],
        },
    )
    return model_dirs, calibration_dirs, truth, report


def _package_config(tmp_path: Path, outdir: Path) -> DeployableModelPackageConfig:
    model_dirs, calibration_dirs, truth, report = _synthetic_artifacts(tmp_path)
    return DeployableModelPackageConfig(
        run_name="explicit_branch_truth_100k_mps",
        model_dirs=",".join(model_dirs),
        calibration_dirs=",".join(calibration_dirs),
        truth_audit_dir=str(truth),
        validation_report=str(report),
        feature_policy="conservative_branch_site",
        truth_mode="explicit",
        methods="identity,mafft,babappalign,muscle",
        outdir=str(outdir),
    )


def test_package_deployable_model_blocks_missing_artifacts(tmp_path: Path) -> None:
    model_dirs, calibration_dirs, truth, report = _synthetic_artifacts(tmp_path)
    Path(model_dirs[0], "branch_site_neural_checkpoint.pt").unlink()
    result = package_deployable_model(
        DeployableModelPackageConfig(
            run_name="explicit_branch_truth_100k_mps",
            model_dirs=",".join(model_dirs),
            calibration_dirs=",".join(calibration_dirs),
            truth_audit_dir=str(truth),
            validation_report=str(report),
            feature_policy="conservative_branch_site",
            truth_mode="explicit",
            methods="identity,mafft,babappalign,muscle",
            outdir=str(tmp_path / "package"),
        )
    )
    assert result["status"] == "blocked"
    assert "missing_model_artifact:low:branch_site_neural_checkpoint.pt" in result["blockers"]


def test_package_deployable_model_writes_manifest_card_and_schema(tmp_path: Path) -> None:
    result = package_deployable_model(_package_config(tmp_path, tmp_path / "package"))
    assert result["status"] == "ok"
    assert (tmp_path / "package" / "model_manifest.json").exists()
    assert (tmp_path / "package" / "model_card.md").exists()
    assert (tmp_path / "package" / "feature_schema.json").exists()
    manifest = json.loads((tmp_path / "package" / "model_manifest.json").read_text())
    assert manifest["feature_policy"] == "conservative_branch_site"
    assert manifest["empirical_claim_status"] == "not_final_empirical_inference"


def test_validate_deployable_package_fails_if_warning_missing(tmp_path: Path) -> None:
    package_deployable_model(_package_config(tmp_path, tmp_path / "package"))
    card = tmp_path / "package" / "model_card.md"
    card.write_text("not final empirical inference\n", encoding="utf-8")
    result = validate_deployable_model_package(
        DeployableModelPackageValidationConfig(package_dir=str(tmp_path / "package"))
    )
    assert result["status"] == "fail"
    assert "model_card_missing_simulation_supervised_limitation" in result["failures"]


def test_validate_deployable_package_fails_if_raw_truth_files_included(tmp_path: Path) -> None:
    package_deployable_model(_package_config(tmp_path, tmp_path / "package"))
    (tmp_path / "package" / "branch_site_truth.tsv").write_text("truth\n", encoding="utf-8")
    result = validate_deployable_model_package(
        DeployableModelPackageValidationConfig(package_dir=str(tmp_path / "package"))
    )
    assert result["status"] == "fail"
    assert any(item.startswith("forbidden_raw_truth_file") for item in result["failures"])


def test_smoke_load_deployable_model_supports_metadata_only(tmp_path: Path, monkeypatch) -> None:
    package_deployable_model(_package_config(tmp_path, tmp_path / "package"))
    import babappa.deploy.package as package_module

    monkeypatch.setattr(package_module, "safe_import_torch", lambda: (None, "missing torch"))
    result = smoke_load_deployable_model(
        DeployableModelSmokeConfig(package_dir=str(tmp_path / "package"), outdir=str(tmp_path / "smoke"))
    )
    assert result["status"] == "ok"
    assert result["metadata_only"] is True
    assert (tmp_path / "smoke" / "deployable_model_load_smoke.json").exists()


def test_simulation_matched_calibration_planner_writes_commands_without_execution(tmp_path: Path) -> None:
    package_deployable_model(_package_config(tmp_path, tmp_path / "package"))
    empirical = tmp_path / "empirical_input_validation"
    empirical.mkdir()
    _write_json(
        empirical / "qc.json",
        {"n_taxa": 6, "n_codons": 120, "mean_pairwise_p_distance": 0.08, "gap_fraction": 0.01},
    )
    result = plan_simulation_matched_calibration(
        SimulationMatchedCalibrationPlanConfig(
            empirical_validation_dir=str(empirical),
            deployable_model_package=str(tmp_path / "package"),
            outdir=str(tmp_path / "calibration_plan"),
        )
    )
    assert result["heavy_jobs_executed"] is False
    commands = (tmp_path / "calibration_plan" / "proposed_null_simulation_commands.sh").read_text()
    assert "USER-RUN ONLY" in commands
    assert "# babappa simulate" in commands


def test_empirical_scoring_planner_refuses_truth_derived_columns(tmp_path: Path) -> None:
    package_deployable_model(_package_config(tmp_path, tmp_path / "package"))
    bad = tmp_path / "bad.tsv"
    bad.write_text("family_id\ty_branch_site\nf1\t1\n", encoding="utf-8")
    tree = tmp_path / "tiny.treefile"
    tree.write_text("(taxon1:0.1,taxon2:0.1);\n", encoding="utf-8")
    with pytest.raises(ValueError, match="truth-derived empirical input blocked"):
        plan_empirical_scoring(
            EmpiricalScoringPlanConfig(
                cds_fasta=str(bad),
                tree=str(tree),
                foreground="taxon1",
                deployable_model_package=str(tmp_path / "package"),
                outdir=str(tmp_path / "scoring"),
                methods="identity,mafft,babappalign,muscle",
            )
        )


def test_empirical_scoring_planner_creates_scripts(tmp_path: Path) -> None:
    package_deployable_model(_package_config(tmp_path, tmp_path / "package"))
    fasta = tmp_path / "tiny.cds.fasta"
    tree = tmp_path / "tiny.treefile"
    fasta.write_text(">taxon1\nATGGCTTAA\n>taxon2\nATGGCGTAA\n", encoding="utf-8")
    tree.write_text("(taxon1:0.1,taxon2:0.1);\n", encoding="utf-8")
    result = plan_empirical_scoring(
        EmpiricalScoringPlanConfig(
            cds_fasta=str(fasta),
            tree=str(tree),
            foreground="taxon1",
            deployable_model_package=str(tmp_path / "package"),
            outdir=str(tmp_path / "scoring"),
            methods="identity,mafft,babappalign,muscle",
        )
    )
    assert result["status"] == "planned"
    assert (tmp_path / "scoring" / "run_empirical_scoring.sh").exists()
    assert "empirical_feature_extraction" in result["current_stopping_point"]


def test_cycle40_docs_mention_not_final_empirical_inference() -> None:
    docs = [
        Path("docs/DEPLOYABLE_MODEL_PACKAGE.md").read_text(encoding="utf-8"),
        Path("docs/SIMULATION_MATCHED_EMPIRICAL_CALIBRATION.md").read_text(encoding="utf-8"),
        Path("docs/POST_100K_EMPIRICAL_TRANSITION_PLAN.md").read_text(encoding="utf-8"),
    ]
    assert all("not final empirical" in text.lower() for text in docs)
