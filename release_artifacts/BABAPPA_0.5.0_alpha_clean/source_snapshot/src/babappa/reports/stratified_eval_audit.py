"""Validation utilities for stratified evaluation outputs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import List, Union


def validate_stratified_eval_dir(eval_dir: Union[str, Path]) -> dict:
    """Validate a BABAPPA stratified evaluation directory."""
    path = Path(eval_dir)
    failures: List[str] = []
    warnings: List[str] = []

    payload = _validate_json(path / "stratified_eval.json", failures)
    _validate_tsv(path / "stratified_metrics.tsv", failures)
    _validate_markdown(path / "stratified_eval.md", failures)

    if isinstance(payload, dict):
        if "key_findings" not in payload:
            failures.append("stratified_eval.json missing key_findings")
        if "generated_files" not in payload:
            failures.append("stratified_eval.json missing generated_files")

    return {
        "status": "fail" if failures else "ok",
        "n_fail": len(failures),
        "n_warning": len(warnings),
        "failures": failures,
        "warnings": warnings,
    }


def _validate_json(path: Path, failures: List[str]) -> object:
    if not path.exists():
        failures.append(f"missing stratified_eval.json: {path}")
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        failures.append(f"unreadable stratified_eval.json: {exc}")
        return None
    if not isinstance(payload, dict):
        failures.append("stratified_eval.json is not a JSON object")
        return None
    return payload


def _validate_tsv(path: Path, failures: List[str]) -> None:
    if not path.exists():
        failures.append(f"missing stratified_metrics.tsv: {path}")
        return
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle, delimiter="\t")
            header = next(reader, None)
    except OSError as exc:
        failures.append(f"unreadable stratified_metrics.tsv: {exc}")
        return
    if not header:
        failures.append("stratified_metrics.tsv missing header")


def _validate_markdown(path: Path, failures: List[str]) -> None:
    if not path.exists():
        failures.append(f"missing stratified_eval.md: {path}")
        return
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        failures.append(f"unreadable stratified_eval.md: {exc}")
        return
    if not text.strip():
        failures.append("stratified_eval.md is empty")
    if "Interpretation caveats" not in text:
        failures.append("stratified_eval.md missing Interpretation caveats")
