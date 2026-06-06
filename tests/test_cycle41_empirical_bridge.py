import json
import os
from pathlib import Path

import numpy as np
import pytest

from babappa.datasets.index import write_tsv
from babappa.deploy.package import DeployableModelPackageConfig, package_deployable_model
from babappa.empirical.bridge import (
    EmpiricalAlignmentEnsembleConfig,
    EmpiricalApplicabilityConfig,
    EmpiricalBranchSiteReportConfig,
    EmpiricalBranchSiteScoringConfig,
    EmpiricalFeatureAuditConfig,
    EmpiricalFeatureExtractionConfig,
    EmpiricalInputValidationConfig,
    ExternalBenchmarkPanelPlanConfig,
    audit_empirical_features,
    extract_empirical_branch_site_features,
    make_empirical_branch_site_report,
    plan_external_benchmark_panel,
    run_empirical_alignment_ensemble,
    run_empirical_applicability,
    score_empirical_branch_sites,
    validate_empirical_input,
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


def _tiny_inputs(tmp_path: Path) -> tuple[Path, Path]:
    fasta = tmp_path / "tiny.cds.fasta"
    tree = tmp_path / "tiny.treefile"
    fasta.write_text(
        ">taxon1\nATGGCTGCTGCTTAA\n>taxon2\nATGGCTGCCGCTTAA\n>taxon3\nATGGCTGCTGCCTAA\n",
        encoding="utf-8",
    )
    tree.write_text("(taxon1:0.1,(taxon2:0.1,taxon3:0.1):0.1);\n", encoding="utf-8")
    return fasta, tree


def _mock_aligners(bin_dir: Path) -> None:
    bin_dir.mkdir()
    (bin_dir / "mafft").write_text("#!/usr/bin/env bash\ncat \"$2\"\n", encoding="utf-8")
    (bin_dir / "muscle").write_text(
        "#!/usr/bin/env bash\nin=''; out=''; while [ \"$#\" -gt 0 ]; do case \"$1\" in -align|-in) in=\"$2\"; shift 2;; -output|-out) out=\"$2\"; shift 2;; *) shift;; esac; done; cp \"$in\" \"$out\"\n",
        encoding="utf-8",
    )
    (bin_dir / "babappalign").write_text("#!/usr/bin/env bash\ncat \"${@: -1}\"\n", encoding="utf-8")
    for script in bin_dir.iterdir():
        script.chmod(0o755)


def _synthetic_package(tmp_path: Path) -> Path:
    model_dirs = []
    calibration_dirs = []
    feature_columns = [
        "site_index_zero",
        "aligned_site_index_zero",
        "original_site_index_zero",
        "site_relative_position",
        "n_taxa",
        "n_codons",
        "codon_id_mean",
        "codon_id_std",
        "codon_id_min",
        "codon_id_max",
        "codon_id_range",
        "codon_id_unique_count",
        "gap_fraction",
        "non_gap_fraction",
        "taxon_codon_variability",
        "foreground_codon_id",
        "foreground_gap",
        "branch_codon_id",
        "branch_gap",
        "background_mean_codon_id",
        "foreground_background_codon_delta",
        "branch_background_codon_delta",
    ]
    for tier in TIERS:
        model_dir = tmp_path / f"branch_site_neural_explicit_branch_truth_100k_mps_{tier}_streamed"
        calibration_dir = tmp_path / f"branch_site_calibration_explicit_branch_truth_100k_mps_{tier}_streamed"
        model_dir.mkdir()
        calibration_dir.mkdir()
        (model_dir / "branch_site_neural_checkpoint.pt").write_bytes(b"fake")
        _write_json(
            model_dir / "branch_site_neural_model_meta.json",
            {
                "feature_columns": feature_columns,
                "feature_policy": "conservative_branch_site",
                "n_features": len(feature_columns),
                "hidden_dim": 64,
                "dropout": 0.1,
                "device": "mps",
            },
        )
        _write_json(
            model_dir / "branch_site_neural_metrics.json",
            {"metrics_by_split": {"test": {"auroc": 0.99, "f1": 0.9, "mcc": 0.8}}},
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
    truth = tmp_path / "truth"
    truth.mkdir()
    truth_rows = [
        {
            "tier": tier,
            "audit_status": "explicit_truth_ok",
            "explicit_branch_site_truth_available": "True",
            "proxy_from_foreground_taxon": "False",
        }
        for tier in TIERS
    ]
    write_tsv(truth / "branch_truth_status_audit.tsv", truth_rows, list(truth_rows[0]))
    report = tmp_path / "report.json"
    _write_json(
        report,
        {
            "decision": {"status": "CONDITIONAL PASS"},
            "run_identity": {"run_name": "explicit_branch_truth_100k_mps"},
            "neural_rows": [{"tier": tier, "split": "test", "auroc": "0.99"} for tier in TIERS],
            "branch_aggregation_rows": [{"tier": tier, "split": "all", "auroc": "0.999"} for tier in TIERS],
            "gene_aggregation_rows": [{"tier": tier, "split": "all", "auroc": "0.999"} for tier in TIERS],
            "controls_rows": [],
            "tier_summary": [],
            "scientific_cautions": ["simulation_supervised_only"],
        },
    )
    outdir = tmp_path / "package"
    package_deployable_model(
        DeployableModelPackageConfig(
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
    )
    return outdir


def _feature_pipeline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, Path, Path, Path]:
    fasta, tree = _tiny_inputs(tmp_path)
    package = _synthetic_package(tmp_path)
    validate_empirical_input(
        EmpiricalInputValidationConfig(str(fasta), str(tree), "taxon1", str(tmp_path / "input"))
    )
    _mock_aligners(tmp_path / "bin")
    monkeypatch.setenv("PATH", str(tmp_path / "bin") + os.pathsep + os.environ["PATH"])
    alignment = run_empirical_alignment_ensemble(
        EmpiricalAlignmentEnsembleConfig(
            cds_fasta=str(fasta),
            tree=str(tree),
            foreground="taxon1",
            outdir=str(tmp_path / "alignment"),
            methods="identity,mafft,babappalign,muscle",
            require_babappalign=False,
        )
    )
    assert alignment["status"] == "ok"
    extract_empirical_branch_site_features(
        EmpiricalFeatureExtractionConfig(
            empirical_validation_dir=str(tmp_path / "input"),
            alignment_dir=str(tmp_path / "alignment"),
            deployable_model_package=str(package),
            outdir=str(tmp_path / "features"),
            foreground="taxon1",
        )
    )
    audit_empirical_features(
        EmpiricalFeatureAuditConfig(
            features=str(tmp_path / "features" / "empirical_branch_site_features.tsv"),
            deployable_model_package=str(package),
            outdir=str(tmp_path / "audit"),
        )
    )
    return package, tmp_path / "input", tmp_path / "alignment", tmp_path / "features", tmp_path / "audit"


def test_empirical_alignment_ensemble_smoke_with_mocked_external_methods(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fasta, tree = _tiny_inputs(tmp_path)
    _mock_aligners(tmp_path / "bin")
    monkeypatch.setenv("PATH", str(tmp_path / "bin") + os.pathsep + os.environ["PATH"])
    result = run_empirical_alignment_ensemble(
        EmpiricalAlignmentEnsembleConfig(
            cds_fasta=str(fasta),
            tree=str(tree),
            foreground="taxon1",
            outdir=str(tmp_path / "alignment"),
            methods="identity,mafft,babappalign,muscle",
            require_babappalign=False,
        )
    )
    assert result["status"] == "ok"
    assert set(result["methods_run"]) == {"identity", "mafft", "babappalign", "muscle"}
    assert (tmp_path / "alignment" / "site_map" / "identity.site_map.tsv").exists()


def test_empirical_input_accepts_unaligned_codon_valid_cds_with_warning(tmp_path: Path) -> None:
    fasta = tmp_path / "unaligned.cds.fasta"
    tree = tmp_path / "tiny.treefile"
    fasta.write_text(
        ">taxon1\nATGGCTGCTGCT\n>taxon2\nATGGCTGCTGCTGCT\n>taxon3\nATGGCTGCTGCT\n",
        encoding="utf-8",
    )
    tree.write_text("(taxon1:0.1,(taxon2:0.1,taxon3:0.1):0.1);\n", encoding="utf-8")
    result = validate_empirical_input(
        EmpiricalInputValidationConfig(str(fasta), str(tree), "taxon1", str(tmp_path / "input"))
    )
    assert result["status"] == "warning"
    assert "unequal_sequence_lengths_unaligned_input" in result["warnings"]


def test_empirical_feature_extractor_creates_schema_matching_table(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _package, _input, _alignment, features, _audit = _feature_pipeline(tmp_path, monkeypatch)
    check = json.loads((features / "empirical_feature_schema_check.json").read_text())
    rows = (features / "empirical_branch_site_features.tsv").read_text().splitlines()
    assert check["feature_schema_match"] == "pass"
    assert check["n_rows"] > 0
    assert "y_branch_site" not in rows[0]


def test_empirical_feature_audit_fails_on_forbidden_truth_columns(tmp_path: Path) -> None:
    bad = tmp_path / "bad_features.tsv"
    bad.write_text("family_id\ty_branch_site\nempirical\t1\n", encoding="utf-8")
    package = _synthetic_package(tmp_path)
    result = audit_empirical_features(
        EmpiricalFeatureAuditConfig(str(bad), str(package), str(tmp_path / "audit"))
    )
    assert result["status"] == "fail"
    assert "y_branch_site" in result["forbidden_columns"]


def test_applicability_returns_rule_based_status(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    package, input_dir, _alignment, features, _audit = _feature_pipeline(tmp_path, monkeypatch)
    result = run_empirical_applicability(
        EmpiricalApplicabilityConfig(str(input_dir), str(features), str(package), str(tmp_path / "applicability"))
    )
    assert result["status"] in {"in_domain", "borderline", "out_of_domain"}
    assert result["reasons"]


def test_applicability_marks_model_feature_envelope_overflow_ood(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    package, input_dir, _alignment, features, _audit = _feature_pipeline(tmp_path, monkeypatch)
    import babappa.empirical.bridge as bridge

    feature_columns = json.loads((package / "feature_schema.json").read_text())["expected_feature_columns"]
    mean = np.zeros(len(feature_columns), dtype=np.float32)
    std = np.ones(len(feature_columns), dtype=np.float32)
    mean[feature_columns.index("n_codons")] = 300.0
    monkeypatch.setattr(bridge, "safe_import_torch", lambda: (object(), None))
    monkeypatch.setattr(bridge, "_torch_load", lambda _torch, _path: {"feature_mean": mean, "feature_std": std})

    result = run_empirical_applicability(
        EmpiricalApplicabilityConfig(str(input_dir), str(features), str(package), str(tmp_path / "applicability"))
    )
    payload = json.loads((tmp_path / "applicability" / "empirical_applicability.json").read_text())

    assert result["status"] == "out_of_domain"
    assert payload["feature_distribution_range_check"] == "fail"
    assert payload["diagnostic_only_if_scored"] is True
    assert any(reason.startswith("model_feature_out_of_envelope:n_codons") for reason in result["reasons"])


def test_scoring_refuses_when_torch_unavailable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    package, input_dir, _alignment, features, _audit = _feature_pipeline(tmp_path, monkeypatch)
    run_empirical_applicability(
        EmpiricalApplicabilityConfig(str(input_dir), str(features), str(package), str(tmp_path / "applicability"))
    )
    import babappa.empirical.bridge as bridge

    monkeypatch.setattr(bridge, "safe_import_torch", lambda: (None, "missing torch"))
    with pytest.raises(RuntimeError, match="PyTorch is required"):
        score_empirical_branch_sites(
            EmpiricalBranchSiteScoringConfig(
                features=str(features / "empirical_branch_site_features.tsv"),
                deployable_model_package=str(package),
                applicability_dir=str(tmp_path / "applicability"),
                outdir=str(tmp_path / "scores"),
                device="cpu",
            )
        )


def test_scoring_marks_out_of_domain_as_diagnostic_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    package, _input, _alignment, features, _audit = _feature_pipeline(tmp_path, monkeypatch)
    app = tmp_path / "applicability"
    app.mkdir()
    _write_json(app / "empirical_applicability.json", {"applicability_status": "out_of_domain", "recommended_tier": "low"})

    class FakeTensor:
        def __init__(self, arr):
            self.arr = np.asarray(arr, dtype=float)

        def to(self, _device):
            return self

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self.arr

        def __truediv__(self, value):
            return FakeTensor(self.arr / value)

    class FakeCuda:
        @staticmethod
        def is_available():
            return False

    class FakeTorch:
        cuda = FakeCuda()

        @staticmethod
        def load(_path, **_kwargs):
            return {
                "model_state_dict": {},
                "feature_mean": np.zeros(22, dtype=np.float32),
                "feature_std": np.ones(22, dtype=np.float32),
            }

        @staticmethod
        def from_numpy(arr):
            return FakeTensor(arr)

        @staticmethod
        def sigmoid(tensor):
            return FakeTensor(1 / (1 + np.exp(-tensor.arr)))

        class no_grad:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

    class FakeModel:
        def __init__(self, input_dim, hidden_dim, dropout):
            self.input_dim = input_dim

        def load_state_dict(self, _state):
            return None

        def to(self, _device):
            return self

        def eval(self):
            return None

        def __call__(self, tensor):
            return FakeTensor(np.full(tensor.arr.shape[0], 0.25))

    import babappa.empirical.bridge as bridge
    import babappa.site.neural_model as neural_model

    monkeypatch.setattr(bridge, "safe_import_torch", lambda: (FakeTorch(), None))
    monkeypatch.setattr(neural_model, "SiteMLPClassifier", FakeModel)
    result = score_empirical_branch_sites(
        EmpiricalBranchSiteScoringConfig(
            features=str(features / "empirical_branch_site_features.tsv"),
            deployable_model_package=str(package),
            applicability_dir=str(app),
            outdir=str(tmp_path / "scores"),
            device="cpu",
        )
    )
    assert result["diagnostic_only"] is True


def test_simulation_matched_calibration_planner_uses_empirical_qc_values(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    _write_json(
        input_dir / "empirical_input_validation.json",
        {
            "n_taxa": 7,
            "n_codons": 123,
            "mean_pairwise_p_distance": 0.08,
            "gap_fraction": 0.02,
            "ambiguous_base_fraction": 0.01,
            "foreground_taxon": "taxon1",
        },
    )
    package = _synthetic_package(tmp_path)
    result = plan_simulation_matched_calibration(
        SimulationMatchedCalibrationPlanConfig(str(input_dir), str(package), str(tmp_path / "plan"))
    )
    payload = json.loads((tmp_path / "plan" / "simulation_matched_calibration_plan.json").read_text())
    assert result["status"] == "planned"
    assert payload["proposed_simulation_parameters"]["n_taxa"] == 7
    assert payload["proposed_simulation_parameters"]["n_codons"] == 123
    assert (tmp_path / "plan" / "proposed_alt_simulation_commands.sh").exists()


def test_empirical_report_includes_no_simulator_truth_used(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    package, input_dir, alignment, features, audit = _feature_pipeline(tmp_path, monkeypatch)
    app = tmp_path / "applicability"
    run_empirical_applicability(EmpiricalApplicabilityConfig(str(input_dir), str(features), str(package), str(app)))
    scores = tmp_path / "scores"
    scores.mkdir()
    _write_json(scores / "empirical_scoring_manifest.json", {"status": "fail", "reason": "torch_unavailable"})
    plan_simulation_matched_calibration(SimulationMatchedCalibrationPlanConfig(str(input_dir), str(package), str(tmp_path / "cal_plan")))
    result = make_empirical_branch_site_report(
        EmpiricalBranchSiteReportConfig(
            outdir=str(tmp_path / "report"),
            empirical_validation_dir=str(input_dir),
            alignment_dir=str(alignment),
            feature_dir=str(features),
            feature_audit_dir=str(audit),
            applicability_dir=str(app),
            scoring_dir=str(scores),
            simulation_matched_calibration_plan=str(tmp_path / "cal_plan"),
            deployable_model_package=str(package),
        )
    )
    text = (tmp_path / "report" / "empirical_branch_site_report.md").read_text()
    assert result["no_simulator_truth_used"] is True
    assert "No simulator truth used" in text


def test_empirical_scoring_planner_wires_all_steps(tmp_path: Path) -> None:
    fasta, tree = _tiny_inputs(tmp_path)
    package = _synthetic_package(tmp_path)
    result = plan_empirical_scoring(
        EmpiricalScoringPlanConfig(
            cds_fasta=str(fasta),
            tree=str(tree),
            foreground="taxon1",
            deployable_model_package=str(package),
            outdir=str(tmp_path / "scoring_plan"),
            methods="identity,mafft,babappalign,muscle",
            allow_diagnostic_out_of_domain=True,
        )
    )
    script = (tmp_path / "scoring_plan" / "run_empirical_scoring.sh").read_text()
    assert result["status"] == "planned"
    for command in [
        "validate-empirical-input",
        "run-empirical-alignment-ensemble",
        "extract-empirical-branch-site-features",
        "audit-empirical-features",
        "empirical-applicability",
        "score-empirical-branch-sites",
        "make-empirical-branch-site-report",
    ]:
        assert command in script


def test_external_benchmark_planner_generates_templates(tmp_path: Path) -> None:
    package = _synthetic_package(tmp_path)
    result = plan_external_benchmark_panel(
        ExternalBenchmarkPanelPlanConfig(
            panel_manifest=str(tmp_path / "missing_panel.tsv"),
            deployable_model_package=str(package),
            outdir=str(tmp_path / "benchmark"),
            methods="identity,mafft,babappalign,muscle",
            classical_tools="codeml,hyphy",
        )
    )
    assert result["status"] == "planned"
    assert (tmp_path / "benchmark" / "proposed_codeml_commands.sh").exists()
    assert (tmp_path / "benchmark" / "proposed_hyphy_commands.sh").exists()
    babappa_script = (tmp_path / "benchmark" / "proposed_babappa_commands.sh").read_text()
    assert "babappa predict-branch-sites" in babappa_script
    assert "--null-replicates 1000" in babappa_script
