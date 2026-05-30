"""Validation utilities for BABAPPA consolidated reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Union


def validate_report_dir(report_dir: Union[str, Path]) -> dict:
    """Validate a BABAPPA report output directory."""
    report_path = Path(report_dir)
    failures: List[str] = []
    warnings: List[str] = []
    json_path = report_path / "report_summary.json"
    markdown_path = report_path / "report.md"

    summary = _validate_json(json_path, failures)
    _validate_markdown(markdown_path, failures)

    if isinstance(summary, dict):
        if "sections" not in summary:
            failures.append("report_summary.json missing sections")
        if "generated_files" not in summary:
            failures.append("report_summary.json missing generated_files")

    return {
        "status": "fail" if failures else "ok",
        "n_fail": len(failures),
        "n_warning": len(warnings),
        "failures": failures,
        "warnings": warnings,
    }


def _validate_json(path: Path, failures: List[str]) -> object:
    if not path.exists():
        failures.append(f"missing report_summary.json: {path}")
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            failures.append("report_summary.json is not a JSON object")
            return None
        return payload
    except (OSError, json.JSONDecodeError) as exc:
        failures.append(f"unreadable report_summary.json: {exc}")
        return None


def _validate_markdown(path: Path, failures: List[str]) -> None:
    if not path.exists():
        failures.append(f"missing report.md: {path}")
        return
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        failures.append(f"unreadable report.md: {exc}")
        return
    if not text.strip():
        failures.append("report.md is empty")
    if "Limitations" not in text:
        failures.append("report.md missing Limitations section")
