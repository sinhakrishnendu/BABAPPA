"""Validation utilities for stratified calibration outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Union

from babappa.calibration.stratified_calibration import (
    STRATIFIED_CALIBRATED_PREDICTION_FIELDNAMES,
)
from babappa.datasets.index import read_tsv


def validate_stratified_calibration_dir(calibration_dir: Union[str, Path]) -> dict:
    """Validate stratified calibration output artifacts."""
    path = Path(calibration_dir)
    failures: List[str] = []
    warnings: List[str] = []
    calibration_json = path / "stratified_calibration.json"
    predictions_tsv = path / "stratified_calibrated_predictions.tsv"
    metrics_json = path / "stratified_calibrated_metrics.json"
    markdown_path = path / "stratified_calibration.md"

    calibration = _validate_json(calibration_json, "stratified_calibration.json", failures)
    metrics = _validate_json(metrics_json, "stratified_calibrated_metrics.json", failures)
    rows = _validate_predictions(predictions_tsv, failures)
    _validate_markdown(markdown_path, failures)

    if isinstance(calibration, dict) and "groups" not in calibration:
        failures.append("stratified_calibration.json missing groups")
    if isinstance(metrics, dict):
        for key in ["metrics_by_group", "metrics_by_split", "metrics_by_group_and_split"]:
            if key not in metrics:
                failures.append(f"stratified_calibrated_metrics.json missing {key}")

    return {
        "status": "fail" if failures else "ok",
        "n_predictions": len(rows),
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


def _validate_predictions(path: Path, failures: List[str]) -> List[dict]:
    if not path.exists():
        failures.append(f"missing stratified_calibrated_predictions.tsv: {path}")
        return []
    try:
        rows = read_tsv(path)
    except OSError as exc:
        failures.append(f"unreadable stratified_calibrated_predictions.tsv: {exc}")
        return []
    if rows:
        missing = [
            column
            for column in STRATIFIED_CALIBRATED_PREDICTION_FIELDNAMES
            if column not in rows[0]
        ]
        if missing:
            failures.append(
                "stratified_calibrated_predictions.tsv missing columns: "
                + ", ".join(missing)
            )
    for row in rows:
        _validate_probability(row, failures)
        _validate_threshold(row, failures)
        if row.get("pred_label_group_calibrated") not in {"0", "1", 0, 1}:
            failures.append(
                "invalid pred_label_group_calibrated: "
                f"{row.get('pred_label_group_calibrated')}"
            )
    return rows


def _validate_probability(row: dict, failures: List[str]) -> None:
    try:
        value = float(row.get("prob_positive_group_calibrated", ""))
    except ValueError:
        failures.append(
            "prob_positive_group_calibrated is not numeric: "
            f"{row.get('prob_positive_group_calibrated')}"
        )
        return
    if not 0 <= value <= 1:
        failures.append(f"prob_positive_group_calibrated out of range: {value}")


def _validate_threshold(row: dict, failures: List[str]) -> None:
    try:
        value = float(row.get("group_selected_threshold", ""))
    except ValueError:
        failures.append(
            f"group_selected_threshold is not numeric: {row.get('group_selected_threshold')}"
        )
        return
    if not 0 <= value <= 1:
        failures.append(f"group_selected_threshold out of range: {value}")


def _validate_markdown(path: Path, failures: List[str]) -> None:
    if not path.exists():
        failures.append(f"missing stratified_calibration.md: {path}")
        return
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        failures.append(f"unreadable stratified_calibration.md: {exc}")
        return
    if not text.strip():
        failures.append("stratified_calibration.md is empty")
    if "Interpretation" not in text and "limitations" not in text:
        failures.append("stratified_calibration.md missing interpretation or limitations")
