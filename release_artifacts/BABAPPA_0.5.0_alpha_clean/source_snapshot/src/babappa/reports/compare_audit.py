"""Validation utilities for BABAPPA model comparison reports."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import List, Union


def validate_model_comparison_dir(compare_dir: Union[str, Path]) -> dict:
    """Validate a BABAPPA model comparison output directory."""
    compare_path = Path(compare_dir)
    failures: List[str] = []
    warnings: List[str] = []
    json_path = compare_path / "model_comparison.json"
    tsv_path = compare_path / "model_comparison.tsv"
    markdown_path = compare_path / "model_comparison.md"

    payload = _validate_json(json_path, failures)
    _validate_tsv(tsv_path, failures)
    _validate_markdown(markdown_path, failures)

    if isinstance(payload, dict):
        if "comparison_by_split" not in payload:
            failures.append("model_comparison.json missing comparison_by_split")
        if "generated_files" not in payload:
            failures.append("model_comparison.json missing generated_files")

    return {
        "status": "fail" if failures else "ok",
        "n_fail": len(failures),
        "n_warning": len(warnings),
        "failures": failures,
        "warnings": warnings,
    }


def _validate_json(path: Path, failures: List[str]) -> object:
    if not path.exists():
        failures.append(f"missing model_comparison.json: {path}")
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        failures.append(f"unreadable model_comparison.json: {exc}")
        return None
    if not isinstance(payload, dict):
        failures.append("model_comparison.json is not a JSON object")
        return None
    return payload


def _validate_tsv(path: Path, failures: List[str]) -> None:
    if not path.exists():
        failures.append(f"missing model_comparison.tsv: {path}")
        return
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if not reader.fieldnames:
                failures.append("model_comparison.tsv missing header")
    except OSError as exc:
        failures.append(f"unreadable model_comparison.tsv: {exc}")


def _validate_markdown(path: Path, failures: List[str]) -> None:
    if not path.exists():
        failures.append(f"missing model_comparison.md: {path}")
        return
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        failures.append(f"unreadable model_comparison.md: {exc}")
        return
    if not text.strip():
        failures.append("model_comparison.md is empty")
    if "Interpretation caveats" not in text:
        failures.append("model_comparison.md missing Interpretation caveats section")
