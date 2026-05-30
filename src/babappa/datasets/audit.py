"""Validation utilities for BABAPPA dataset indexes."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Union

from babappa.datasets.index import FEATURE_FIELDNAMES, SPLIT_FIELDNAMES, SPLIT_NAMES


def validate_dataset_index(index_dir: Union[str, Path]) -> dict:
    """Validate dataset index metadata, feature table, and split assignments."""
    index_path = Path(index_dir)
    failures: List[str] = []
    warnings: List[str] = []
    dataset_index_path = index_path / "dataset_index.json"
    features_path = index_path / "features.tsv"
    splits_path = index_path / "splits.tsv"

    if not dataset_index_path.exists():
        return _summary("fail", 0, 0, 1, 0, [f"missing dataset_index.json: {dataset_index_path}"], warnings)

    try:
        dataset_index = _read_json(dataset_index_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return _summary("fail", 0, 0, 1, 0, [f"unreadable dataset_index.json: {exc}"], warnings)

    features_rows = _read_required_tsv(
        features_path, FEATURE_FIELDNAMES, "features.tsv", failures
    )
    splits_rows = _read_required_tsv(
        splits_path, SPLIT_FIELDNAMES, "splits.tsv", failures
    )

    tensor_dir = Path(str(dataset_index.get("tensor_dir", "")))
    if not tensor_dir.exists():
        failures.append(f"tensor_dir in dataset_index.json does not exist: {tensor_dir}")

    methods = dataset_index.get("methods", [])
    if not isinstance(methods, list):
        failures.append("dataset_index.json methods is not a list")
        methods = []

    _validate_tensor_file_references(tensor_dir, splits_rows, failures)
    _validate_split_values(splits_rows, failures)
    if dataset_index.get("split_by_family") is True:
        _validate_family_disjoint_splits(splits_rows, failures)
    _validate_method_coverage(methods, splits_rows, failures, warnings)

    n_rows = len(features_rows)
    n_families = len({row.get("family_id", "") for row in splits_rows if row.get("family_id")})
    return _summary(
        status="fail" if failures else "ok",
        n_rows=n_rows,
        n_families=n_families,
        n_fail=len(failures),
        n_warning=len(warnings),
        failures=failures,
        warnings=warnings,
    )


def _read_required_tsv(
    path: Path,
    required_columns: List[str],
    label: str,
    failures: List[str],
) -> List[Dict[str, str]]:
    if not path.exists():
        failures.append(f"missing {label}: {path}")
        return []
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            fieldnames = reader.fieldnames or []
            missing_columns = [
                column for column in required_columns if column not in fieldnames
            ]
            if missing_columns:
                failures.append(
                    f"{label} missing required columns: {', '.join(missing_columns)}"
                )
            return list(reader)
    except OSError as exc:
        failures.append(f"unreadable {label}: {exc}")
        return []


def _validate_tensor_file_references(
    tensor_dir: Path, splits_rows: List[Dict[str, str]], failures: List[str]
) -> None:
    for row in splits_rows:
        tensor_file = row.get("tensor_file", "")
        if not tensor_file:
            failures.append("splits.tsv row missing tensor_file")
            continue
        tensor_path = tensor_dir / tensor_file
        if not tensor_path.exists():
            failures.append(f"missing tensor_file referenced by splits.tsv: {tensor_path}")


def _validate_split_values(
    splits_rows: List[Dict[str, str]], failures: List[str]
) -> None:
    allowed = set(SPLIT_NAMES)
    for row in splits_rows:
        split = row.get("split", "")
        if split not in allowed:
            failures.append(f"invalid split value in splits.tsv: {split}")


def _validate_family_disjoint_splits(
    splits_rows: List[Dict[str, str]], failures: List[str]
) -> None:
    family_to_splits: Dict[str, set] = {}
    for row in splits_rows:
        family_id = row.get("family_id", "")
        split = row.get("split", "")
        if family_id:
            family_to_splits.setdefault(family_id, set()).add(split)
    for family_id, splits in family_to_splits.items():
        if len(splits) > 1:
            failures.append(
                f"family appears in multiple splits: {family_id} -> {sorted(splits)}"
            )


def _validate_method_coverage(
    methods: List[object],
    splits_rows: List[Dict[str, str]],
    failures: List[str],
    warnings: List[str],
) -> None:
    represented_methods = {row.get("method", "") for row in splits_rows}
    for method in methods:
        if not isinstance(method, str):
            failures.append("dataset_index.json methods contains a non-string method")
            continue
        if method not in represented_methods:
            message = f"method listed in dataset_index.json is absent from splits.tsv: {method}"
            if len(splits_rows) < len(methods):
                warnings.append(message)
            else:
                failures.append(message)


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON file is not an object: {path}")
    return payload


def _summary(
    status: str,
    n_rows: int,
    n_families: int,
    n_fail: int,
    n_warning: int,
    failures: List[str],
    warnings: List[str],
) -> dict:
    return {
        "status": status,
        "n_rows": n_rows,
        "n_families": n_families,
        "n_fail": n_fail,
        "n_warning": n_warning,
        "failures": failures,
        "warnings": warnings,
    }
