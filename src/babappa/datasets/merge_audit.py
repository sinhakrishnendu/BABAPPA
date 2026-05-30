"""Validation utilities for merged BABAPPA datasets."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Union

from babappa.datasets.index import SPLIT_NAMES
from babappa.training.neural_data import resolve_tensor_file

REQUIRED_FEATURE_COLUMNS = [
    "family_id",
    "original_family_id",
    "source_dataset",
    "method",
    "tensor_file",
    "gene_label",
    "saturation_tier",
]
REQUIRED_SPLIT_COLUMNS = REQUIRED_FEATURE_COLUMNS + ["split"]


def validate_merged_dataset_dir(dataset_dir: Union[str, Path]) -> dict:
    """Validate a merged BABAPPA dataset directory."""
    path = Path(dataset_dir)
    failures: List[str] = []
    warnings: List[str] = []
    index_path = path / "dataset_index.json"
    features_path = path / "features.tsv"
    splits_path = path / "splits.tsv"

    dataset_index = _read_json_if_possible(index_path, failures)
    feature_rows = _read_required_tsv(
        features_path, REQUIRED_FEATURE_COLUMNS, "features.tsv", failures
    )
    split_rows = _read_required_tsv(
        splits_path, REQUIRED_SPLIT_COLUMNS, "splits.tsv", failures
    )
    _validate_split_values(split_rows, failures)
    if isinstance(dataset_index, dict) and dataset_index.get("split_by_family") is True:
        _validate_family_disjoint_splits(split_rows, failures)
    _validate_tensor_paths(path, split_rows, failures)
    saturation_tiers = {
        row.get("saturation_tier", "") for row in split_rows if row.get("saturation_tier")
    }
    if not saturation_tiers:
        failures.append("merged dataset has no saturation_tier values")
    elif len(saturation_tiers) == 1:
        warnings.append(
            f"merged dataset contains only one saturation_tier: {sorted(saturation_tiers)[0]}"
        )

    return {
        "status": "fail" if failures else "ok",
        "n_rows": len(split_rows),
        "n_families": len({row.get("family_id", "") for row in split_rows if row.get("family_id")}),
        "n_fail": len(failures),
        "n_warning": len(warnings),
        "failures": failures,
        "warnings": warnings,
    }


def _read_json_if_possible(path: Path, failures: List[str]) -> object:
    if not path.exists():
        failures.append(f"missing dataset_index.json: {path}")
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        failures.append(f"unreadable dataset_index.json: {exc}")
        return None
    if not isinstance(payload, dict):
        failures.append("dataset_index.json is not a JSON object")
        return None
    return payload


def _read_required_tsv(
    path: Path, required_columns: List[str], label: str, failures: List[str]
) -> List[Dict[str, str]]:
    if not path.exists():
        failures.append(f"missing {label}: {path}")
        return []
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            fieldnames = reader.fieldnames or []
            missing = [column for column in required_columns if column not in fieldnames]
            if missing:
                failures.append(f"{label} missing required columns: {', '.join(missing)}")
            return list(reader)
    except OSError as exc:
        failures.append(f"unreadable {label}: {exc}")
        return []


def _validate_split_values(rows: List[Dict[str, str]], failures: List[str]) -> None:
    allowed = set(SPLIT_NAMES)
    for row in rows:
        split = row.get("split", "")
        if split not in allowed:
            failures.append(f"invalid split value in splits.tsv: {split}")


def _validate_family_disjoint_splits(
    rows: List[Dict[str, str]], failures: List[str]
) -> None:
    family_to_splits: Dict[str, set] = {}
    for row in rows:
        family_id = row.get("family_id", "")
        if family_id:
            family_to_splits.setdefault(family_id, set()).add(row.get("split", ""))
    for family_id, splits in family_to_splits.items():
        if len(splits) > 1:
            failures.append(
                f"family appears in multiple splits: {family_id} -> {sorted(splits)}"
            )


def _validate_tensor_paths(
    dataset_dir: Path, rows: List[Dict[str, str]], failures: List[str]
) -> None:
    for row in rows:
        tensor_file = row.get("tensor_file", "")
        if not tensor_file:
            failures.append("splits.tsv row missing tensor_file")
            continue
        try:
            resolve_tensor_file(tensor_file, dataset_dir)
        except FileNotFoundError as exc:
            failures.append(str(exc))
