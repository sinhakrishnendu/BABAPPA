import json
from pathlib import Path

from typer.testing import CliRunner

from babappa.benchmarks import SaturationPanelConfig, build_saturation_panel
from babappa.cli import app
from babappa.datasets import read_tsv
from babappa.site import (
    OracleSiteLabelConfig,
    SiteBaselineConfig,
    SiteDatasetConfig,
    audit_site_dataset_leakage,
    build_site_dataset,
    extract_oracle_site_labels,
    normalize_site_indices,
    train_site_baseline,
    validate_site_baseline_dir,
    validate_site_dataset_dir,
    validate_site_label_dir,
)
from babappa.reports import (
    ReportConfig,
    RunSummaryConfig,
    generate_report,
    summarize_run,
    validate_report_dir,
    validate_run_summary_dir,
)


runner = CliRunner()


def test_repository_hardening_init_files() -> None:
    root = Path(__file__).resolve().parents[1]
    for relative in [
        "src/babappa/init.py",
        "src/babappa/calibration/init.py",
        "src/babappa/reports/init.py",
        "src/babappa/training/init.py",
        "src/babappa/benchmarks/init.py",
        "src/babappa/datasets/init.py",
        "src/babappa/site/init.py",
    ]:
        assert not (root / relative).exists()
    for relative in [
        "src/babappa/__init__.py",
        "src/babappa/calibration/__init__.py",
        "src/babappa/reports/__init__.py",
        "src/babappa/training/__init__.py",
        "src/babappa/benchmarks/__init__.py",
        "src/babappa/datasets/__init__.py",
        "src/babappa/site/__init__.py",
    ]:
        assert (root / relative).exists()


def test_normalize_site_indices() -> None:
    zero, zero_warnings = normalize_site_indices([0, 2, 5], 10, "zero")
    one, one_warnings = normalize_site_indices([1, 3, 6], 10, "one")
    csv_values, _ = normalize_site_indices("1,3,6", 10, "one")
    out_of_range, warnings = normalize_site_indices([1, 99], 10, "one")

    assert zero == [0, 2, 5]
    assert one == [0, 2, 5]
    assert csv_values == [0, 2, 5]
    assert out_of_range == [0]
    assert zero_warnings == []
    assert one_warnings == []
    assert any("out_of_range_site_index_ignored" in warning for warning in warnings)


def test_extract_site_labels_tiny(tmp_path) -> None:
    dataset_dir = _build_tiny_dataset(tmp_path)
    outdir = tmp_path / "site_oracle"

    summary = extract_oracle_site_labels(
        OracleSiteLabelConfig(dataset_dir=str(dataset_dir), outdir=str(outdir))
    )
    payload = json.loads((outdir / "site_oracle_summary.json").read_text("utf-8"))
    rows = read_tsv(outdir / "site_oracle_labels.tsv")

    assert summary["status"] == "ok"
    assert (outdir / "site_oracle_labels.md").exists()
    assert rows
    assert "y_site" in rows[0]
    assert payload["n_positive_sites"] > 0
    assert validate_site_label_dir(outdir)["status"] == "ok"


def test_build_site_dataset_tiny(tmp_path) -> None:
    site_dataset_dir = _build_tiny_site_dataset(tmp_path)
    rows = read_tsv(site_dataset_dir / "site_features.tsv")

    assert validate_site_dataset_dir(site_dataset_dir)["status"] == "ok"
    assert rows
    assert "codon_id_mean" in rows[0]
    assert "y_site" in rows[0]
    assert "n_selected_sites" not in rows[0]


def test_site_leakage_audit_flags_bad_columns(tmp_path) -> None:
    site_dataset = tmp_path / "site_dataset_bad"
    site_dataset.mkdir()
    (site_dataset / "site_dataset_index.json").write_text(
        json.dumps({"n_site_rows": 2}) + "\n",
        encoding="utf-8",
    )
    (site_dataset / "site_features.tsv").write_text(
        "site_id\tfamily_id\tmethod\tsplit\tsaturation_tier\tsite_index_zero\ty_site\tselected_sites\ttruth_label\n"
        "s1\tf1\tidentity\ttrain\tlow\t0\t1\t0\t1\n"
        "s2\tf1\tidentity\ttrain\tlow\t1\t0\t0\t0\n",
        encoding="utf-8",
    )

    summary = audit_site_dataset_leakage(site_dataset, tmp_path / "site_leakage")
    payload = json.loads(
        (tmp_path / "site_leakage" / "site_leakage_audit.json").read_text("utf-8")
    )

    assert summary["status"] == "ok"
    assert payload["status"] == "warning"
    assert "selected_sites" in payload["forbidden_columns_present"]
    assert "truth_label" in payload["forbidden_columns_present"]


def test_train_site_baseline_tiny(tmp_path) -> None:
    site_dataset_dir = _build_tiny_site_dataset(tmp_path)
    outdir = tmp_path / "site_baseline"

    summary = train_site_baseline(
        SiteBaselineConfig(
            site_dataset_dir=str(site_dataset_dir),
            outdir=str(outdir),
            epochs=20,
            learning_rate=0.05,
        )
    )
    metrics = json.loads((outdir / "site_baseline_metrics.json").read_text("utf-8"))

    assert summary["status"] == "ok"
    assert validate_site_baseline_dir(outdir)["status"] == "ok"
    assert (outdir / "site_baseline_predictions.tsv").exists()
    assert "metrics_by_split" in metrics


def test_cli_extract_site_labels_exits_0(tmp_path) -> None:
    dataset_dir = _build_tiny_dataset(tmp_path)
    result = runner.invoke(
        app,
        [
            "extract-site-labels",
            "--dataset-dir",
            str(dataset_dir),
            "--outdir",
            str(tmp_path / "site_oracle_cli"),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Oracle Site Labels" in result.output


def test_cli_build_site_dataset_exits_0(tmp_path) -> None:
    dataset_dir = _build_tiny_dataset(tmp_path)
    oracle_dir = tmp_path / "site_oracle"
    extract_oracle_site_labels(
        OracleSiteLabelConfig(dataset_dir=str(dataset_dir), outdir=str(oracle_dir))
    )
    result = runner.invoke(
        app,
        [
            "build-site-dataset",
            "--dataset-dir",
            str(dataset_dir),
            "--oracle-labels",
            str(oracle_dir / "site_oracle_labels.tsv"),
            "--outdir",
            str(tmp_path / "site_dataset_cli"),
            "--negative-downsample-ratio",
            "20",
            "--seed",
            "42",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Site Dataset" in result.output


def test_cli_train_site_baseline_exits_0(tmp_path) -> None:
    site_dataset_dir = _build_tiny_site_dataset(tmp_path)
    result = runner.invoke(
        app,
        [
            "train-site-baseline",
            "--site-dataset-dir",
            str(site_dataset_dir),
            "--outdir",
            str(tmp_path / "site_baseline_cli"),
            "--epochs",
            "10",
            "--learning-rate",
            "0.05",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Site Baseline" in result.output


def test_report_and_summary_accept_site_outputs(tmp_path) -> None:
    dataset_dir = _build_tiny_dataset(tmp_path)
    oracle_dir = tmp_path / "site_oracle"
    extract_oracle_site_labels(
        OracleSiteLabelConfig(dataset_dir=str(dataset_dir), outdir=str(oracle_dir))
    )
    site_dataset_dir = tmp_path / "site_dataset"
    build_site_dataset(
        SiteDatasetConfig(
            dataset_dir=str(dataset_dir),
            oracle_labels_tsv=str(oracle_dir / "site_oracle_labels.tsv"),
            outdir=str(site_dataset_dir),
            negative_downsample_ratio=20,
        )
    )
    leakage_dir = tmp_path / "site_leakage"
    audit_site_dataset_leakage(site_dataset_dir, leakage_dir)
    baseline_dir = tmp_path / "site_baseline"
    train_site_baseline(
        SiteBaselineConfig(
            site_dataset_dir=str(site_dataset_dir),
            outdir=str(baseline_dir),
            epochs=10,
        )
    )

    report_dir = tmp_path / "report"
    generate_report(
        ReportConfig(
            outdir=str(report_dir),
            site_label_dir=str(oracle_dir),
            site_dataset_dir=str(site_dataset_dir),
            site_leakage_audit_dir=str(leakage_dir),
            site_baseline_dir=str(baseline_dir),
        )
    )
    summary_dir = tmp_path / "summary"
    summarize_run(
        RunSummaryConfig(
            outdir=str(summary_dir),
            site_label_dir=str(oracle_dir),
            site_dataset_dir=str(site_dataset_dir),
            site_leakage_audit_dir=str(leakage_dir),
            site_baseline_dir=str(baseline_dir),
        )
    )

    report_payload = json.loads((report_dir / "report_summary.json").read_text("utf-8"))
    summary_payload = json.loads((summary_dir / "run_summary.json").read_text("utf-8"))
    report_md = (report_dir / "report.md").read_text("utf-8")
    summary_md = (summary_dir / "run_summary.md").read_text("utf-8")

    assert validate_report_dir(report_dir)["status"] == "ok"
    assert validate_run_summary_dir(summary_dir)["status"] == "ok"
    assert "site_labels" in report_payload["sections"]
    assert summary_payload["status_overview"]["site_baseline_present"] is True
    assert "Site-label overview" in report_md
    assert "Site baseline overview" in summary_md


def _build_tiny_dataset(tmp_path) -> Path:
    panel_dir = tmp_path / "panel"
    build_saturation_panel(
        SaturationPanelConfig(
            outdir=str(panel_dir),
            n_families_per_tier=4,
            tiers=["low"],
            n_taxa=4,
            n_codons=30,
            seed=42,
            positive_rate=1.0,
            selected_site_fraction=0.1,
            methods=["identity"],
            build_tensors=True,
            index_datasets=True,
        )
    )
    return panel_dir / "tiers" / "low" / "dataset"


def _build_tiny_site_dataset(tmp_path) -> Path:
    dataset_dir = _build_tiny_dataset(tmp_path)
    oracle_dir = tmp_path / "site_oracle"
    extract_oracle_site_labels(
        OracleSiteLabelConfig(dataset_dir=str(dataset_dir), outdir=str(oracle_dir))
    )
    site_dataset_dir = tmp_path / "site_dataset"
    build_site_dataset(
        SiteDatasetConfig(
            dataset_dir=str(dataset_dir),
            oracle_labels_tsv=str(oracle_dir / "site_oracle_labels.tsv"),
            outdir=str(site_dataset_dir),
            negative_downsample_ratio=20,
            seed=42,
        )
    )
    return site_dataset_dir
