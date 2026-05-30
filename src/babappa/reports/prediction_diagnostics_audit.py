"""Validation for BABAPPA prediction diagnostics artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import List, Union


def validate_prediction_diagnostics_dir(diag_dir: Union[str, Path]) -> dict:
    """Validate a prediction diagnostics output directory."""
    diag_path = Path(diag_dir)
    failures: List[str] = []
    warnings: List[str] = []
    json_path = diag_path / "prediction_diagnostics.json"
    summary_path = diag_path / "prediction_score_summary.tsv"
    threshold_path = diag_path / "threshold_curve.tsv"
    markdown_path = diag_path / "prediction_diagnostics.md"

    _validate_json(json_path, failures)
    _validate_tsv(summary_path, "prediction_score_summary.tsv", failures)
    _validate_tsv(threshold_path, "threshold_curve.tsv", failures)
    _validate_markdown(markdown_path, failures)

    return {
        "status": "fail" if failures else "ok",
        "n_fail": len(failures),
        "n_warning": len(warnings),
        "failures": failures,
        "warnings": warnings,
    }


def _validate_json(path: Path, failures: List[str]) -> None:
    if not path.exists():
        failures.append(f"missing prediction_diagnostics.json: {path}")
        return
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        failures.append(f"unreadable prediction_diagnostics.json: {exc}")
        return
    if not isinstance(payload, dict):
        failures.append("prediction_diagnostics.json is not a JSON object")
        return
    if "warnings" not in payload:
        failures.append("prediction_diagnostics.json missing warnings")


def _validate_tsv(path: Path, label: str, failures: List[str]) -> None:
    if not path.exists():
        failures.append(f"missing {label}: {path}")
        return
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if not reader.fieldnames:
                failures.append(f"{label} missing header")
    except OSError as exc:
        failures.append(f"unreadable {label}: {exc}")


def _validate_markdown(path: Path, failures: List[str]) -> None:
    if not path.exists():
        failures.append(f"missing prediction_diagnostics.md: {path}")
        return
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        failures.append(f"unreadable prediction_diagnostics.md: {exc}")
        return
    if not text.strip():
        failures.append("prediction_diagnostics.md is empty")
    if "Interpretation" not in text:
        failures.append("prediction_diagnostics.md missing Interpretation section")
