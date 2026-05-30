"""Validation utilities for neural ablation comparison outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Union

from babappa.datasets.index import read_tsv


def validate_ablation_comparison_dir(compare_dir: Union[str, Path]) -> dict:
    """Validate a BABAPPA neural ablation comparison directory."""
    path = Path(compare_dir)
    failures: List[str] = []
    warnings: List[str] = []
    json_path = path / "ablation_comparison.json"
    tsv_path = path / "ablation_comparison.tsv"
    markdown_path = path / "ablation_comparison.md"

    payload = _validate_json(json_path, "ablation_comparison.json", failures)
    _validate_tsv(tsv_path, failures)
    _validate_markdown(markdown_path, failures)
    if isinstance(payload, dict):
        for key in ["models", "recommendation", "generated_files"]:
            if key not in payload:
                failures.append(f"ablation_comparison.json missing {key}")

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
        failures.append(f"missing ablation_comparison.tsv: {path}")
        return
    try:
        rows = read_tsv(path)
    except OSError as exc:
        failures.append(f"unreadable ablation_comparison.tsv: {exc}")
        return
    if not rows:
        failures.append("ablation_comparison.tsv has no rows")


def _validate_markdown(path: Path, failures: List[str]) -> None:
    if not path.exists():
        failures.append(f"missing ablation_comparison.md: {path}")
        return
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        failures.append(f"unreadable ablation_comparison.md: {exc}")
        return
    if not text.strip():
        failures.append("ablation_comparison.md is empty")
    if "Recommendation" not in text:
        failures.append("ablation_comparison.md missing Recommendation")
