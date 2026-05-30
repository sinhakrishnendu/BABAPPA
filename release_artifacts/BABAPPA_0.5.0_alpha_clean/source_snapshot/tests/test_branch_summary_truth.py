import json
from pathlib import Path

from babappa.branch import (
    BranchConditionedTierSummaryConfig,
    BranchTruthStatusAuditConfig,
    ExplicitBranchTruthPrototypePlanConfig,
    audit_branch_truth_status,
    plan_explicit_branch_truth_prototype,
    summarize_branch_conditioned_tiers,
    validate_branch_conditioned_tier_summary_dir,
    validate_branch_truth_status_audit_dir,
)


TIERS = ["low", "moderate", "high", "extreme"]


def test_summarize_branch_conditioned_tiers_synthetic_minimal_artifacts(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    for index, tier in enumerate(TIERS):
        _write_minimal_branch_tier(tmp_path, tier, auroc=0.99 - index * 0.01)

    summary = summarize_branch_conditioned_tiers(
        BranchConditionedTierSummaryConfig(
            tiers="low,moderate,high,extreme",
            run_name="fast_external_10k_streamed",
            outdir="branch_conditioned_10k_cross_tier_summary",
        )
    )

    outdir = tmp_path / "branch_conditioned_10k_cross_tier_summary"
    assert summary["status"] == "ok"
    assert summary["n_warning"] > 0
    for filename in [
        "branch_conditioned_tier_summary.json",
        "branch_conditioned_tier_summary.tsv",
        "branch_site_neural_performance.tsv",
        "branch_aggregation_performance.tsv",
        "branch_gene_aggregation_performance.tsv",
        "branch_calibration_summary.tsv",
        "branch_threshold_policy_summary.tsv",
        "branch_controls_summary.tsv",
        "branch_conditioned_tier_summary.md",
    ]:
        assert (outdir / filename).exists()


def test_branch_conditioned_summary_validator_passes_synthetic(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    for tier in TIERS:
        _write_minimal_branch_tier(tmp_path, tier)
    summarize_branch_conditioned_tiers(
        BranchConditionedTierSummaryConfig(
            tiers=TIERS,
            outdir="summary",
        )
    )

    validation = validate_branch_conditioned_tier_summary_dir(tmp_path / "summary")

    assert validation["status"] == "ok"
    assert validation["n_fail"] == 0
    assert validation["n_warning"] > 0


def test_truth_status_audit_detects_proxy_labels(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    for tier in TIERS:
        _write_proxy_label_dir(tmp_path, tier)

    audit = audit_branch_truth_status(
        BranchTruthStatusAuditConfig(
            tiers=TIERS,
            outdir="branch_truth_status_audit_fast_external_10k",
        )
    )
    payload = json.loads(
        (tmp_path / "branch_truth_status_audit_fast_external_10k" / "branch_truth_status_audit.json").read_text("utf-8")
    )

    assert audit["status"] == "ok"
    assert audit["explicit_truth_available"] is False
    assert set(audit["proxy_label_tiers"]) == set(TIERS)
    assert "branch-conditioned proxy validation" in payload["interpretation"]


def test_truth_status_audit_recommends_larger_validation_when_all_tiers_explicit(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    for tier in TIERS:
        _write_explicit_label_dir(tmp_path, tier)

    audit = audit_branch_truth_status(
        BranchTruthStatusAuditConfig(tiers=TIERS, outdir="truth_audit_explicit")
    )
    markdown = (tmp_path / "truth_audit_explicit" / "branch_truth_status_audit.md").read_text("utf-8")

    assert audit["explicit_truth_available"] is True
    assert "Explicit simulator branch-site truth is available for all audited tiers" in markdown
    assert "Upgrade the simulator to emit explicit branch-site selected-event truth" not in markdown


def test_truth_status_audit_validator_passes(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    for tier in TIERS:
        _write_proxy_label_dir(tmp_path, tier)
    audit_branch_truth_status(
        BranchTruthStatusAuditConfig(tiers=TIERS, outdir="truth_audit")
    )

    validation = validate_branch_truth_status_audit_dir(tmp_path / "truth_audit")

    assert validation["status"] == "ok"
    assert validation["n_fail"] == 0
    assert validation["n_warning"] > 0


def test_explicit_branch_truth_prototype_planner_generates_scripts_without_execution(tmp_path) -> None:
    outdir = tmp_path / "explicit_branch_truth_prototype_plan"

    summary = plan_explicit_branch_truth_prototype(
        ExplicitBranchTruthPrototypePlanConfig(
            outdir=str(outdir),
            n_families=1000,
            tiers="low,extreme",
            methods="identity,mafft",
        )
    )
    run_text = (outdir / "run_explicit_branch_truth_prototype.sh").read_text("utf-8")
    expected = json.loads((outdir / "expected_outputs.json").read_text("utf-8"))

    assert summary["does_not_run_jobs"] is True
    assert "TODO after simulator support exists" in run_text
    assert "# babappa simulate" in run_text
    assert expected["plan_only"] is True
    assert expected["defer_100k_until_passes"] is True


def test_summarize_branch_conditioned_tiers_supports_streamed_output_suffix(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    for tier in TIERS:
        _write_minimal_branch_tier(
            tmp_path,
            tier,
            run_name="explicit_branch_truth_1k",
            output_suffix="_streamed",
            label_status="explicit_simulator_branch_truth",
        )

    summary = summarize_branch_conditioned_tiers(
        BranchConditionedTierSummaryConfig(
            tiers=TIERS,
            run_name="explicit_branch_truth_1k",
            output_suffix="_streamed",
            outdir="explicit_summary",
        )
    )
    payload = json.loads((tmp_path / "explicit_summary" / "branch_conditioned_tier_summary.json").read_text("utf-8"))

    assert summary["status"] == "ok"
    assert payload["label_truth_status"]["explicit_branch_site_truth_available"] is True
    assert validate_branch_conditioned_tier_summary_dir(tmp_path / "explicit_summary")["status"] == "ok"


def test_readme_contains_explicit_branch_site_truth_warning() -> None:
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text("utf-8")

    assert "Branch-conditioned 10K streamed validation completed" in readme
    assert "Branch-conditioned labels may be proxy-derived" in readme
    assert "explicit branch-site simulator truth" in readme
    assert "Final 100K is deferred until explicit branch-truth validation passes" in readme


def test_missing_optional_artifacts_warn_not_crash(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    for tier in TIERS:
        _write_minimal_branch_tier(tmp_path, tier)

    summary = summarize_branch_conditioned_tiers(
        BranchConditionedTierSummaryConfig(tiers=TIERS, outdir="summary")
    )

    assert summary["status"] == "ok"
    assert any("optional branch_site_calibration missing" in warning for warning in summary["warnings"])
    assert validate_branch_conditioned_tier_summary_dir(tmp_path / "summary")["status"] == "ok"


def _write_minimal_branch_tier(
    tmp_path: Path,
    tier: str,
    auroc: float = 0.99,
    run_name: str = "fast_external_10k",
    output_suffix: str = "_streamed",
    label_status: str = "proxy_from_foreground_taxon",
) -> None:
    prefix = f"{run_name}_{tier}{output_suffix}"
    run_dir = tmp_path / f"branch_site_run_summary_{prefix}"
    neural_dir = tmp_path / f"branch_site_neural_{prefix}"
    aggregation_dir = tmp_path / f"branch_aggregation_{prefix}"
    label_dir = tmp_path / f"branch_site_oracle_{run_name}_{tier}"
    for directory in [run_dir, neural_dir, aggregation_dir, label_dir]:
        directory.mkdir(parents=True)
    (run_dir / "branch_site_run_summary.json").write_text(
        json.dumps(
            {
                "sections": {
                    "branch_site_labels": {"directory": str(label_dir), "status": "present"},
                    "branch_site_neural": {"directory": str(neural_dir), "status": "present"},
                    "branch_aggregation": {"directory": str(aggregation_dir), "status": "present"},
                },
                "warnings": [],
                "scientific_boundary": "Research-alpha simulation-supervised branch-conditioned validation.",
            }
        ),
        encoding="utf-8",
    )
    (label_dir / "branch_site_oracle_summary.json").write_text(
        json.dumps(
            {
                "branch_site_labels_status": label_status,
                "n_branch_site_rows": 4,
                "n_positive_branch_sites": 1,
                "status_counts": {label_status: 1},
            }
        ),
        encoding="utf-8",
    )
    (neural_dir / "branch_site_neural_metrics.json").write_text(
        json.dumps({"metrics_by_split": {"all": _metrics(20, auroc), "test": _metrics(5, auroc - 0.01)}}),
        encoding="utf-8",
    )
    (aggregation_dir / "branch_aggregation_metrics.json").write_text(
        json.dumps(
            {
                "branch_level_metrics_default": {
                    "all": _metrics(12, 0.999),
                    "by_split": {"test": _metrics(3, 0.998)},
                },
                "gene_level_metrics_default": {
                    "all": _metrics(8, 0.995),
                    "by_split": {"test": _metrics(2, 0.994)},
                },
            }
        ),
        encoding="utf-8",
    )


def _write_proxy_label_dir(tmp_path: Path, tier: str) -> None:
    label_dir = tmp_path / f"branch_site_oracle_fast_external_10k_{tier}"
    label_dir.mkdir(parents=True)
    labels_path = label_dir / "branch_site_oracle_labels.tsv"
    labels_path.write_text(
        "family_id\tmethod\tsplit\tsaturation_tier\tbranch_id\tforeground_taxon\tsite_index_zero\ty_branch_site\ty_site\tgene_label\tbranch_label_source\n"
        f"family_1\tidentity\ttest\t{tier}\ttaxon_001\ttaxon_001\t0\t1\t1\t1\tproxy_from_foreground_taxon:branch_labels_x_selected_sites\n"
        f"family_1\tidentity\ttest\t{tier}\ttaxon_002\ttaxon_001\t0\t0\t1\t1\tproxy_from_foreground_taxon:branch_labels_x_selected_sites\n",
        encoding="utf-8",
    )
    (label_dir / "branch_site_oracle_summary.json").write_text(
        json.dumps(
            {
                "branch_site_labels_status": "proxy_from_foreground_taxon",
                "n_branch_site_rows": 2,
                "n_positive_branch_sites": 1,
                "positive_branch_site_fraction": 0.5,
                "status_counts": {"proxy_from_foreground_taxon": 1},
                "generated_files": {"labels_tsv": str(labels_path)},
                "warnings": [],
            }
        ),
        encoding="utf-8",
    )


def _write_explicit_label_dir(tmp_path: Path, tier: str) -> None:
    label_dir = tmp_path / f"branch_site_oracle_fast_external_10k_{tier}"
    label_dir.mkdir(parents=True)
    labels_path = label_dir / "branch_site_oracle_labels.tsv"
    labels_path.write_text(
        "family_id\tmethod\tsplit\tsaturation_tier\tbranch_id\tforeground_taxon\tsite_index_zero\ty_branch_site\ty_site\tgene_label\tbranch_label_source\n"
        f"family_1\tidentity\ttest\t{tier}\ttaxon_001\ttaxon_001\t0\t1\t1\t1\texplicit_simulator_branch_truth\n"
        f"family_1\tidentity\ttest\t{tier}\ttaxon_002\ttaxon_001\t0\t0\t1\t1\texplicit_simulator_branch_truth\n",
        encoding="utf-8",
    )
    (label_dir / "branch_site_oracle_summary.json").write_text(
        json.dumps(
            {
                "branch_site_labels_status": "explicit_simulator_branch_truth",
                "explicit_branch_site_truth_available": True,
                "proxy_labels_used": False,
                "n_branch_site_rows": 2,
                "n_positive_branch_sites": 1,
                "positive_branch_site_fraction": 0.5,
                "status_counts": {"explicit_simulator_branch_truth": 1},
                "branch_label_source_counts": {"explicit_simulator_branch_truth": 2},
                "generated_files": {"labels_tsv": str(labels_path)},
                "warnings": [],
            }
        ),
        encoding="utf-8",
    )


def _metrics(n: int, auroc: float) -> dict:
    return {
        "n": n,
        "positives": n // 2,
        "negatives": n // 2,
        "auroc": auroc,
        "accuracy": 0.9,
        "precision": 0.9,
        "recall": 0.9,
        "f1": 0.9,
        "mcc": 0.8,
        "specificity": 0.9,
    }
