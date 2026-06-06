import csv
import math

import pytest

from babappa.branch.dataset import (
    BRANCH_FEATURE_FIELDNAMES,
    BRANCH_SPLIT_FIELDNAMES,
    BranchSiteDatasetMergeConfig,
    merge_branch_site_datasets,
    validate_branch_site_dataset_dir,
)
from babappa.branch.feature_policy import columns_for_policy
from babappa.branch.retrain_plan import (
    VariableLength100KRetrainingPlanConfig,
    plan_variable_length_100k_retraining,
)
from babappa.empirical.bridge import _feature_row_from_site


def test_normalized_v2_policy_excludes_raw_length_and_site_columns() -> None:
    columns = [
        "site_index_zero",
        "aligned_site_index_zero",
        "original_site_index_zero",
        "site_relative_position",
        "site_centered_position",
        "site_terminal_distance",
        "n_taxa",
        "n_codons",
        "log_n_taxa",
        "log_n_codons",
        "codon_id_mean",
        "foreground_taxon_index",
    ]

    selected = columns_for_policy(columns, "conservative_branch_site_normalized_v2")

    assert "site_index_zero" not in selected
    assert "aligned_site_index_zero" not in selected
    assert "original_site_index_zero" not in selected
    assert "n_taxa" not in selected
    assert "n_codons" not in selected
    assert "site_relative_position" in selected
    assert "site_centered_position" in selected
    assert "site_terminal_distance" in selected
    assert "log_n_taxa" in selected
    assert "log_n_codons" in selected
    assert "foreground_taxon_index" not in selected


def test_empirical_feature_row_provides_normalized_length_site_features() -> None:
    records = {
        "taxon1": "ATGAAAACCGGG",
        "taxon2": "ATGAAAACCGGA",
    }
    expected = [
        "site_relative_position",
        "site_centered_position",
        "site_terminal_distance",
        "log_n_taxa",
        "log_n_codons",
    ]

    row = _feature_row_from_site(
        records=records,
        site_row={"aligned_site_index_zero": "2", "original_site_index_zero": "2"},
        branch_id="taxon1",
        foreground="taxon1",
        validation={"n_codons": 4},
        expected_features=expected,
    )

    assert row["site_relative_position"] == 2 / 3
    assert row["site_centered_position"] == (2 / 3) - 0.5
    assert row["site_terminal_distance"] == pytest.approx(min(2 / 3, 1 / 3))
    assert row["log_n_taxa"] == math.log1p(2)
    assert row["log_n_codons"] == math.log1p(4)


def test_merge_branch_site_datasets_creates_trainable_feature_dir(tmp_path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    _write_branch_dataset(first, "row1", "0")
    _write_branch_dataset(second, "row2", "1")

    summary = merge_branch_site_datasets(
        BranchSiteDatasetMergeConfig(dataset_dirs=f"{first},{second}", outdir=str(tmp_path / "merged"))
    )
    validation = validate_branch_site_dataset_dir(tmp_path / "merged")

    assert summary["status"] == "ok"
    assert summary["n_branch_site_rows"] == 2
    assert validation["status"] == "ok"


def test_variable_length_retraining_plan_writes_storage_safe_scripts(tmp_path) -> None:
    outdir = tmp_path / "plan"

    summary = plan_variable_length_100k_retraining(
        VariableLength100KRetrainingPlanConfig(
            outdir=str(outdir),
            workspace="branch_site_v2_100k_workspace",
            n_families_per_tier=10,
            tiers="low",
            threads=2,
            min_free_gb=1,
            max_train_items=100,
            max_eval_items=20,
        )
    )
    run_text = (outdir / "run_variable_length_100k_retraining.sh").read_text()
    package_text = (outdir / "package_variable_length_deployable.sh").read_text()

    assert summary["does_not_run_jobs"] is True
    assert "conservative_branch_site_normalized_v2" in run_text
    assert "BABAPPA_RETRAIN_DELETE_INTERMEDIATES=YES" in run_text
    assert "cleanup_path" in run_text
    assert "merge-branch-site-datasets" in run_text
    assert "deployable_model_conservative_branch_site_v2_100k_mps" in package_text


def _write_branch_dataset(path, branch_site_id: str, label: str) -> None:
    path.mkdir(parents=True)
    feature_row = {field: "0" for field in BRANCH_FEATURE_FIELDNAMES}
    feature_row.update(
        {
            "branch_site_id": branch_site_id,
            "family_id": f"family_{branch_site_id}",
            "method": "identity",
            "saturation_tier": "low",
            "split": "train",
            "branch_id": "taxon1",
            "foreground_taxon": "taxon1",
            "mapping_status": "mapped",
            "mapping_confidence": "1",
            "mappable_site": "True",
            "y_branch_site": label,
            "y_site": label,
            "gene_label": label,
            "tensor_file": "",
            "labels_file": "",
            "site_relative_position": "0.5",
            "site_centered_position": "0",
            "site_terminal_distance": "0.5",
            "n_taxa": "4",
            "n_codons": "120",
            "log_n_taxa": str(math.log1p(4)),
            "log_n_codons": str(math.log1p(120)),
        }
    )
    split_row = {field: feature_row.get(field, "0") for field in BRANCH_SPLIT_FIELDNAMES}
    _write_tsv(path / "branch_site_features.tsv", [feature_row], BRANCH_FEATURE_FIELDNAMES)
    _write_tsv(path / "branch_site_splits.tsv", [split_row], BRANCH_SPLIT_FIELDNAMES)


def _write_tsv(path, rows, fieldnames) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
