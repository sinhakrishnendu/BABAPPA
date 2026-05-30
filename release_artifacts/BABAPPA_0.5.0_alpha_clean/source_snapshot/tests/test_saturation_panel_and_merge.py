import json
from pathlib import Path

from typer.testing import CliRunner

from babappa.benchmarks import (
    SaturationPanelConfig,
    build_saturation_panel,
    validate_saturation_panel_dir,
)
from babappa.cli import app
from babappa.datasets import (
    DatasetMergeConfig,
    merge_dataset_indexes,
    read_tsv,
    validate_merged_dataset_dir,
)
from babappa.reports import (
    ReportConfig,
    RunSummaryConfig,
    generate_report,
    summarize_run,
    validate_report_dir,
    validate_run_summary_dir,
)
from babappa.training.neural_data import resolve_tensor_file


runner = CliRunner()


def test_repository_hardening_init_files() -> None:
    root = Path(__file__).resolve().parents[1]

    assert not (root / "src" / "babappa" / "init.py").exists()
    assert not (root / "src" / "babappa" / "calibration" / "init.py").exists()
    assert not (root / "src" / "babappa" / "reports" / "init.py").exists()
    assert not (root / "src" / "babappa" / "training" / "init.py").exists()
    assert not (root / "src" / "babappa" / "benchmarks" / "init.py").exists()
    assert not (root / "src" / "babappa" / "datasets" / "init.py").exists()
    assert (root / "src" / "babappa" / "__init__.py").exists()
    assert (root / "src" / "babappa" / "calibration" / "__init__.py").exists()
    assert (root / "src" / "babappa" / "reports" / "__init__.py").exists()
    assert (root / "src" / "babappa" / "training" / "__init__.py").exists()
    assert (root / "src" / "babappa" / "benchmarks" / "__init__.py").exists()
    assert (root / "src" / "babappa" / "datasets" / "__init__.py").exists()


def test_make_saturation_panel_small(tmp_path) -> None:
    panel_dir = _build_panel(tmp_path, n_families_per_tier=2)

    assert (panel_dir / "saturation_panel.json").exists()
    assert (panel_dir / "saturation_panel.md").exists()
    assert validate_saturation_panel_dir(panel_dir)["status"] == "ok"


def test_merge_dataset_indexes(tmp_path) -> None:
    panel_dir = _build_panel(tmp_path, n_families_per_tier=2)
    merged_dir = _merge_panel_datasets(panel_dir, tmp_path / "merged")
    payload = json.loads((merged_dir / "dataset_index.json").read_text("utf-8"))

    assert (merged_dir / "features.tsv").exists()
    assert (merged_dir / "splits.tsv").exists()
    assert payload["saturation_tier_counts"]["low"] == 2
    assert payload["saturation_tier_counts"]["moderate"] == 2
    assert validate_merged_dataset_dir(merged_dir)["status"] == "ok"


def test_merged_dataset_family_disjoint_splits(tmp_path) -> None:
    panel_dir = _build_panel(tmp_path, n_families_per_tier=2)
    merged_dir = _merge_panel_datasets(panel_dir, tmp_path / "merged")
    family_to_splits = {}
    for row in read_tsv(merged_dir / "splits.tsv"):
        family_to_splits.setdefault(row["family_id"], set()).add(row["split"])

    assert all(len(splits) == 1 for splits in family_to_splits.values())


def test_merged_dataset_tensor_paths_resolve(tmp_path) -> None:
    panel_dir = _build_panel(tmp_path, n_families_per_tier=1)
    merged_dir = _merge_panel_datasets(panel_dir, tmp_path / "merged")

    for row in read_tsv(merged_dir / "splits.tsv"):
        assert resolve_tensor_file(row["tensor_file"], merged_dir).exists()


def test_cli_make_saturation_panel_exits_0(tmp_path) -> None:
    result = runner.invoke(
        app,
        [
            "make-saturation-panel",
            "--outdir",
            str(tmp_path / "panel_cli"),
            "--n-families-per-tier",
            "1",
            "--tiers",
            "low,moderate",
            "--n-taxa",
            "4",
            "--n-codons",
            "30",
            "--seed",
            "42",
            "--positive-rate",
            "0.5",
            "--methods",
            "identity,codon_dropout",
        ],
    )

    assert result.exit_code == 0
    assert "Saturation Panel" in result.output


def test_cli_merge_datasets_exits_0(tmp_path) -> None:
    panel_dir = _build_panel(tmp_path, n_families_per_tier=1)
    dataset_dirs = [
        panel_dir / "tiers" / "low" / "dataset",
        panel_dir / "tiers" / "moderate" / "dataset",
    ]
    result = runner.invoke(
        app,
        [
            "merge-datasets",
            "--dataset-dirs",
            ",".join(str(path) for path in dataset_dirs),
            "--names",
            "low,moderate",
            "--outdir",
            str(tmp_path / "merged_cli"),
            "--seed",
            "42",
            "--resplit",
        ],
    )

    assert result.exit_code == 0
    assert "Dataset Merge" in result.output


def test_cli_validate_saturation_panel_exits_0(tmp_path) -> None:
    panel_dir = _build_panel(tmp_path, n_families_per_tier=1)

    result = runner.invoke(
        app,
        ["validate-saturation-panel", "--panel-dir", str(panel_dir)],
    )

    assert result.exit_code == 0
    assert "ok" in result.output


def test_cli_validate_merged_dataset_exits_0(tmp_path) -> None:
    panel_dir = _build_panel(tmp_path, n_families_per_tier=1)
    merged_dir = _merge_panel_datasets(panel_dir, tmp_path / "merged")

    result = runner.invoke(
        app,
        ["validate-merged-dataset", "--dataset-dir", str(merged_dir)],
    )

    assert result.exit_code == 0
    assert "ok" in result.output


def test_run_summary_accepts_saturation_panel_and_merged_dataset(tmp_path) -> None:
    panel_dir = _build_panel(tmp_path, n_families_per_tier=1)
    merged_dir = _merge_panel_datasets(panel_dir, tmp_path / "merged")
    summary_dir = tmp_path / "summary"

    summary = summarize_run(
        RunSummaryConfig(
            outdir=str(summary_dir),
            saturation_panel_dir=str(panel_dir),
            merged_dataset_dir=str(merged_dir),
            title="Saturation summary",
        )
    )
    payload = json.loads((summary_dir / "run_summary.json").read_text("utf-8"))
    markdown = (summary_dir / "run_summary.md").read_text("utf-8")

    assert summary["status"] == "ok"
    assert validate_run_summary_dir(summary_dir)["status"] == "ok"
    assert payload["status_overview"]["saturation_panel_present"] is True
    assert payload["status_overview"]["merged_dataset_present"] is True
    assert payload["merged_dataset_overview"]["saturation_tier_counts"]
    assert "Saturation panel overview" in markdown
    assert "Merged dataset overview" in markdown


def test_make_report_accepts_saturation_panel_and_merged_dataset(tmp_path) -> None:
    panel_dir = _build_panel(tmp_path, n_families_per_tier=1)
    merged_dir = _merge_panel_datasets(panel_dir, tmp_path / "merged")
    report_dir = tmp_path / "report"

    generate_report(
        ReportConfig(
            outdir=str(report_dir),
            saturation_panel_dir=str(panel_dir),
            merged_dataset_dir=str(merged_dir),
            title="Saturation report",
        )
    )
    payload = json.loads((report_dir / "report_summary.json").read_text("utf-8"))
    markdown = (report_dir / "report.md").read_text("utf-8")

    assert validate_report_dir(report_dir)["status"] == "ok"
    assert "saturation_panel" in payload["sections"]
    assert "merged_dataset" in payload["sections"]
    assert "Saturation panel" in markdown
    assert "Merged dataset" in markdown


def _build_panel(tmp_path, n_families_per_tier: int) -> Path:
    panel_dir = tmp_path / f"panel_{n_families_per_tier}"
    build_saturation_panel(
        SaturationPanelConfig(
            outdir=str(panel_dir),
            n_families_per_tier=n_families_per_tier,
            tiers=["low", "moderate"],
            n_taxa=4,
            n_codons=30,
            seed=42,
            positive_rate=0.5,
            methods=["identity", "codon_dropout"],
            dropout_rate=0.02,
        )
    )
    return panel_dir


def _merge_panel_datasets(panel_dir: Path, outdir: Path) -> Path:
    merge_dataset_indexes(
        DatasetMergeConfig(
            dataset_dirs=[
                str(panel_dir / "tiers" / "low" / "dataset"),
                str(panel_dir / "tiers" / "moderate" / "dataset"),
            ],
            names=["low", "moderate"],
            outdir=str(outdir),
            seed=42,
            resplit=True,
        )
    )
    return outdir
