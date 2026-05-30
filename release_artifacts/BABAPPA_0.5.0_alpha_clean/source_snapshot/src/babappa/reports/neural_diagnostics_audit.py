"""Validation utilities for neural diagnostics outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Union

from babappa.datasets.index import read_tsv


def validate_neural_diagnostics_dir(diag_dir: Union[str, Path]) -> dict:
    """Validate a BABAPPA neural diagnostics directory."""
    path = Path(diag_dir)
    failures: List[str] = []
    warnings: List[str] = []
    json_path = path / "neural_diagnostics.json"
    tsv_path = path / "neural_probability_summary.tsv"
    markdown_path = path / "neural_diagnostics.md"

    payload = _validate_json(json_path, "neural_diagnostics.json", failures)
    _validate_tsv(tsv_path, failures)
    _validate_markdown(markdown_path, failures)
    if isinstance(payload, dict):
        for key in ["metadata_summary", "history_summary", "probability_summary_by_split"]:
            if key not in payload:
                failures.append(f"neural_diagnostics.json missing {key}")

    return {
        "status": "fail" if failures else "ok",
        "n_fail": len(failures),
        "n_warning": len(warnings),
        "failures": failures,
        "warnings": warnings,
    }


def _validate_json(path: Path, label: str, failures: List[str]) -> object:
    if not path.exists():
        failures.append(f"missing {label}: {path}")
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        failures.append(f"unreadable {label}: {exc}")
        return None
    if not isinstance(payload, dict):
        failures.append(f"{label} is not a JSON object")
        return None
    return payload


def _validate_tsv(path: Path, failures: List[str]) -> None:
    if not path.exists():
        failures.append(f"missing neural_probability_summary.tsv: {path}")
        return
    try:
        rows = read_tsv(path)
    except OSError as exc:
        failures.append(f"unreadable neural_probability_summary.tsv: {exc}")
        return
    if not rows:
        failures.append("neural_probability_summary.tsv has no rows")


def _validate_markdown(path: Path, failures: List[str]) -> None:
    if not path.exists():
        failures.append(f"missing neural_diagnostics.md: {path}")
        return
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        failures.append(f"unreadable neural_diagnostics.md: {exc}")
        return
    if not text.strip():
        failures.append("neural_diagnostics.md is empty")
    if "Interpretation" not in text:
        failures.append("neural_diagnostics.md missing Interpretation")
