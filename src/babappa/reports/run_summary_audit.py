"""Validation utilities for BABAPPA run summaries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Union


def validate_run_summary_dir(summary_dir: Union[str, Path]) -> dict:
    """Validate a BABAPPA run summary output directory."""
    summary_path = Path(summary_dir)
    failures: List[str] = []
    warnings: List[str] = []
    json_path = summary_path / "run_summary.json"
    markdown_path = summary_path / "run_summary.md"

    payload = _validate_json(json_path, failures)
    _validate_markdown(markdown_path, failures)

    if isinstance(payload, dict):
        if "recommended_next_action" not in payload:
            failures.append("run_summary.json missing recommended_next_action")
        if "generated_files" not in payload:
            failures.append("run_summary.json missing generated_files")

    return {
        "status": "fail" if failures else "ok",
        "n_fail": len(failures),
        "n_warning": len(warnings),
        "failures": failures,
        "warnings": warnings,
    }


def _validate_json(path: Path, failures: List[str]) -> object:
    if not path.exists():
        failures.append(f"missing run_summary.json: {path}")
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        failures.append(f"unreadable run_summary.json: {exc}")
        return None
    if not isinstance(payload, dict):
        failures.append("run_summary.json is not a JSON object")
        return None
    return payload


def _validate_markdown(path: Path, failures: List[str]) -> None:
    if not path.exists():
        failures.append(f"missing run_summary.md: {path}")
        return
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        failures.append(f"unreadable run_summary.md: {exc}")
        return
    if not text.strip():
        failures.append("run_summary.md is empty")
    if "Limitations" not in text:
        failures.append("run_summary.md missing Limitations section")
