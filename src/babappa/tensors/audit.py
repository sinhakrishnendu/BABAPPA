"""Validation utilities for BABAPPA tensor datasets."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import List, Union

import numpy as np


def validate_tensor_directory(tensor_dir: Union[str, Path]) -> dict:
    """Validate tensor manifest, audit rows, labels, and tensor shard files."""
    tensor_path = Path(tensor_dir)
    failures: List[str] = []
    warnings: List[str] = []
    manifest_path = tensor_path / "tensor_manifest.json"
    audit_path = tensor_path / "tensor_audit.tsv"

    if not manifest_path.exists():
        return _summary("fail", 0, 1, 0, [f"missing tensor_manifest.json: {manifest_path}"], warnings)
    try:
        manifest = _read_json(manifest_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return _summary("fail", 0, 1, 0, [f"unreadable tensor_manifest.json: {exc}"], warnings)

    if not audit_path.exists():
        failures.append(f"missing tensor_audit.tsv: {audit_path}")
        audit_rows = []
    else:
        try:
            audit_rows = _read_audit_rows(audit_path)
        except (OSError, ValueError) as exc:
            failures.append(f"unreadable tensor_audit.tsv: {exc}")
            audit_rows = []

    family_ids = manifest.get("family_ids", [])
    if not isinstance(family_ids, list):
        failures.append("tensor_manifest.json family_ids is not a list")
        family_ids = []

    for family_id in family_ids:
        if not isinstance(family_id, str):
            failures.append("tensor_manifest.json contains a non-string family id")
            continue
        labels_path = tensor_path / "families" / family_id / f"{family_id}.labels.json"
        if not labels_path.exists():
            failures.append(f"missing labels JSON: {labels_path}")
        else:
            try:
                _read_json(labels_path)
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                failures.append(f"unreadable labels JSON {labels_path}: {exc}")

    tensor_files_checked = 0
    for row in audit_rows:
        tensor_file = row.get("tensor_file", "")
        if not tensor_file:
            failures.append("tensor_audit.tsv row is missing tensor_file")
            continue
        tensor_file_path = tensor_path / tensor_file
        if not tensor_file_path.exists():
            failures.append(f"missing tensor file: {tensor_file_path}")
            continue
        try:
            with np.load(tensor_file_path, allow_pickle=False) as shard:
                tensor_files_checked += 1
                if "X" not in shard.files:
                    failures.append(f"tensor file missing X array: {tensor_file_path}")
                    continue
                if shard["X"].ndim != 3:
                    failures.append(f"X array is not 3-dimensional: {tensor_file_path}")
        except (OSError, ValueError) as exc:
            failures.append(f"unreadable tensor file {tensor_file_path}: {exc}")

        if row.get("status") != "ok":
            warnings.append(
                f"tensor audit row status is {row.get('status')}: {tensor_file}"
            )

    return _summary(
        status="fail" if failures else "ok",
        n_tensor_files_checked=tensor_files_checked,
        n_fail=len(failures),
        n_warning=len(warnings),
        failures=failures,
        warnings=warnings,
    )


def _read_audit_rows(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if "tensor_file" not in (reader.fieldnames or []):
            raise ValueError("tensor_audit.tsv does not contain tensor_file column")
        return list(reader)


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON file is not an object: {path}")
    return payload


def _summary(
    status: str,
    n_tensor_files_checked: int,
    n_fail: int,
    n_warning: int,
    failures: List[str],
    warnings: List[str],
) -> dict:
    return {
        "status": status,
        "n_tensor_files_checked": n_tensor_files_checked,
        "n_fail": n_fail,
        "n_warning": n_warning,
        "failures": failures,
        "warnings": warnings,
    }
