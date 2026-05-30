"""Validation for BABAPPA site-level datasets."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List

from babappa.training.neural_data import resolve_tensor_file

REQUIRED_FEATURE_COLUMNS = {
    "site_id",
    "family_id",
    "method",
    "split",
    "saturation_tier",
    "site_index_zero",
    "y_site",
    "tensor_file",
}
REQUIRED_SPLIT_COLUMNS = {
    "site_id",
    "family_id",
    "method",
    "split",
    "saturation_tier",
    "site_index_zero",
    "y_site",
}
FORBIDDEN_FEATURE_COLUMNS = {
    "selected_sites",
    "n_selected_sites",
    "positive_sites",
    "oracle_selected_sites",
    "site_labels",
    "gene_label",
    "truth_label",
    "true_label",
    "positive_family",
    "is_positive",
    "y_gene",
    "y",
    "label",
}


def validate_site_dataset_dir(site_dataset_dir: str | Path) -> dict:
    """Validate site-level dataset artifacts."""
    dataset_dir = Path(site_dataset_dir)
    failures: List[str] = []
    warnings: List[str] = []
    index_path = dataset_dir / "site_dataset_index.json"
    features_path = dataset_dir / "site_features.tsv"
    splits_path = dataset_dir / "site_splits.tsv"
    n_rows = 0
    n_positive = 0
    family_splits: Dict[str, set] = {}
    site_splits: Dict[str, set] = {}
    saturation_tiers = set()

    index_payload = _load_json(index_path, failures)
    feature_rows = _read_tsv(features_path, failures)
    split_rows = _read_tsv(splits_path, failures)
    _check_header(features_path, feature_rows, REQUIRED_FEATURE_COLUMNS, failures)
    _check_header(splits_path, split_rows, REQUIRED_SPLIT_COLUMNS, failures)

    seen_site_ids = set()
    for row in feature_rows:
        n_rows += 1
        site_id = row.get("site_id", "")
        if site_id in seen_site_ids:
            failures.append(f"duplicate_site_id:{site_id}")
        seen_site_ids.add(site_id)
        if row.get("y_site") not in {"0", "1", 0, 1}:
            failures.append(f"invalid_y_site:{site_id}:{row.get('y_site')}")
        if str(row.get("y_site")) == "1":
            n_positive += 1
        family_splits.setdefault(row.get("family_id", ""), set()).add(row.get("split", ""))
        site_splits.setdefault(site_id, set()).add(row.get("split", ""))
        saturation_tiers.add(row.get("saturation_tier", "unknown") or "unknown")

    for row in split_rows:
        site_splits.setdefault(row.get("site_id", ""), set()).add(row.get("split", ""))

    for site_id, splits in site_splits.items():
        if len(splits) > 1:
            failures.append(f"site_id_multiple_splits:{site_id}:{sorted(splits)}")
    for family_id, splits in family_splits.items():
        if family_id and len(splits) > 1:
            failures.append(f"family_id_multiple_splits:{family_id}:{sorted(splits)}")

    if feature_rows:
        forbidden = sorted(FORBIDDEN_FEATURE_COLUMNS & set(feature_rows[0].keys()))
        forbidden = [column for column in forbidden if column != "y_site"]
        if forbidden:
            failures.append("forbidden_site_feature_columns:" + ",".join(forbidden))

    source_dataset_dir = Path(str(index_payload.get("dataset_dir", dataset_dir)))
    _check_tensor_paths(feature_rows[:25], source_dataset_dir, failures, warnings)
    if not saturation_tiers:
        warnings.append("no_saturation_tier_present")
    positive_fraction = None if n_rows == 0 else n_positive / n_rows
    if n_rows == 0:
        failures.append("no_site_rows")
    elif n_positive == 0:
        warnings.append("no_positive_sites")
    elif positive_fraction is not None and positive_fraction < 0.001:
        warnings.append("positive_fraction_below_0_001")
    elif positive_fraction is not None and positive_fraction > 0.5:
        warnings.append("positive_fraction_above_0_5")

    if index_payload and index_payload.get("n_site_rows") != n_rows:
        warnings.append("index_site_row_count_mismatch")

    return {
        "status": "fail" if failures else "ok",
        "n_rows": n_rows,
        "n_positive_sites": n_positive,
        "n_fail": len(failures),
        "n_warning": len(warnings),
        "failures": failures,
        "warnings": warnings,
    }


def _load_json(path: Path, failures: List[str]) -> dict:
    if not path.exists():
        failures.append(f"missing_file:{path}")
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        failures.append(f"could_not_parse_json:{path}:{exc}")
        return {}
    if not isinstance(payload, dict):
        failures.append(f"json_not_object:{path}")
        return {}
    return payload


def _read_tsv(path: Path, failures: List[str]) -> List[dict]:
    if not path.exists():
        failures.append(f"missing_file:{path}")
        return []
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle, delimiter="\t"))
    except OSError as exc:
        failures.append(f"could_not_read_tsv:{path}:{exc}")
        return []


def _check_header(
    path: Path, rows: List[dict], required_columns: set, failures: List[str]
) -> None:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        missing = sorted(required_columns - set(reader.fieldnames or []))
    if missing:
        failures.append(f"missing_required_columns:{path}:{','.join(missing)}")


def _check_tensor_paths(
    rows: List[dict], dataset_dir: Path, failures: List[str], warnings: List[str]
) -> None:
    for row in rows:
        tensor_file = row.get("tensor_file", "")
        try:
            resolve_tensor_file(tensor_file, dataset_dir)
        except FileNotFoundError:
            failures.append(f"tensor_file_unresolved:{tensor_file}")
        except Exception as exc:
            warnings.append(f"tensor_file_resolution_warning:{tensor_file}:{exc}")
