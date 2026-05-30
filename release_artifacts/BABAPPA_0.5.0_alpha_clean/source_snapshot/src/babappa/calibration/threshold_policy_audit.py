"""Validation utilities for threshold-policy outputs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import List, Union


def validate_threshold_policy_dir(policy_dir: Union[str, Path]) -> dict:
    """Validate a BABAPPA threshold-policy output directory."""
    path = Path(policy_dir)
    failures: List[str] = []
    warnings: List[str] = []

    payload = _validate_json(path / "threshold_profiles.json", failures)
    _validate_tsv(path / "threshold_profiles.tsv", failures)
    _validate_tsv(path / "threshold_profile_metrics.tsv", failures)
    _validate_tsv(path / "threshold_policy_curve.tsv", failures)
    _validate_markdown(path / "threshold_policy.md", failures)

    if isinstance(payload, dict):
        profiles = payload.get("profiles")
        if not isinstance(profiles, dict) or not profiles:
            failures.append("threshold_profiles.json missing profiles")

    return {
        "status": "fail" if failures else "ok",
        "n_fail": len(failures),
        "n_warning": len(warnings),
        "failures": failures,
        "warnings": warnings,
    }


def _validate_json(path: Path, failures: List[str]) -> object:
    if not path.exists():
        failures.append(f"missing threshold_profiles.json: {path}")
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        failures.append(f"unreadable threshold_profiles.json: {exc}")
        return None
    if not isinstance(payload, dict):
        failures.append("threshold_profiles.json is not a JSON object")
        return None
    return payload


def _validate_tsv(path: Path, failures: List[str]) -> None:
    if not path.exists():
        failures.append(f"missing TSV: {path}")
        return
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle, delimiter="\t")
            header = next(reader, None)
    except OSError as exc:
        failures.append(f"unreadable TSV {path}: {exc}")
        return
    if not header:
        failures.append(f"TSV missing header: {path}")


def _validate_markdown(path: Path, failures: List[str]) -> None:
    if not path.exists():
        failures.append(f"missing threshold_policy.md: {path}")
        return
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        failures.append(f"unreadable threshold_policy.md: {exc}")
        return
    if not text.strip():
        failures.append("threshold_policy.md is empty")
    if "Recommended interpretation" not in text:
        failures.append("threshold_policy.md missing Recommended interpretation")
