"""Validation for resplit BABAPPA dataset directories."""

from __future__ import annotations

import json
from pathlib import Path

from babappa.datasets.index import read_tsv
from babappa.training.neural_data import resolve_tensor_file


def validate_resplit_dataset_dir(dataset_dir: str | Path) -> dict:
    """Validate a resplit dataset directory."""
    path = Path(dataset_dir)
    failures = []
    warnings = []
    for filename in ["dataset_index.json", "features.tsv", "splits.tsv"]:
        if not (path / filename).exists():
            failures.append(f"missing:{path / filename}")
    payload = None
    if (path / "dataset_index.json").exists():
        try:
            payload = json.loads((path / "dataset_index.json").read_text("utf-8"))
            if not isinstance(payload, dict):
                failures.append("dataset_index_not_object")
        except json.JSONDecodeError as exc:
            failures.append(f"dataset_index_parse_error:{exc}")

    rows = []
    if (path / "splits.tsv").exists():
        rows = read_tsv(path / "splits.tsv")
        required = {
            "family_id",
            "method",
            "tensor_file",
            "gene_label",
            "saturation_tier",
            "split",
        }
        fieldnames = set(rows[0].keys()) if rows else set()
        missing = sorted(required - fieldnames)
        if missing:
            failures.append("splits_missing_columns:" + ",".join(missing))
        family_to_splits = {}
        for row in rows:
            family_to_splits.setdefault(row.get("family_id", ""), set()).add(
                row.get("split", "")
            )
            try:
                resolve_tensor_file(row.get("tensor_file", ""), path)
            except FileNotFoundError:
                warnings.append(f"tensor_file_unresolved:{row.get('tensor_file', '')}")
        multi_split = [
            family for family, splits in family_to_splits.items() if len(splits) > 1
        ]
        if multi_split:
            failures.append("family_in_multiple_splits:" + ",".join(sorted(multi_split)[:5]))

    if isinstance(payload, dict) and not payload.get("saturation_tier_counts"):
        warnings.append("missing_saturation_tier_counts")

    return {
        "status": "fail" if failures else "ok",
        "n_rows": len(rows),
        "n_families": len({row.get("family_id", "") for row in rows}),
        "n_fail": len(failures),
        "n_warning": len(warnings),
        "failures": failures,
        "warnings": warnings,
    }
