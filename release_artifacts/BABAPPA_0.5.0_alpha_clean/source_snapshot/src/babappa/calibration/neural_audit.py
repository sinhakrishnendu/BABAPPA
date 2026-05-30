"""Validation utilities for BABAPPA neural calibration artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Union

from babappa.calibration.neural import NEURAL_CALIBRATED_PREDICTION_FIELDNAMES
from babappa.datasets.index import read_tsv

VALID_SPLITS = {"train", "val", "calib", "test"}


def validate_neural_calibration_dir(calibration_dir: Union[str, Path]) -> dict:
    """Validate neural calibration output artifacts."""
    calibration_path = Path(calibration_dir)
    failures: List[str] = []
    warnings: List[str] = []
    calibration_json = calibration_path / "neural_calibration.json"
    predictions_tsv = calibration_path / "neural_calibrated_predictions.tsv"
    metrics_json = calibration_path / "neural_calibrated_metrics.json"

    calibration = _validate_json(calibration_json, "neural_calibration.json", failures)
    metrics = _validate_json(metrics_json, "neural_calibrated_metrics.json", failures)
    prediction_rows = _validate_predictions(predictions_tsv, failures)

    if isinstance(calibration, dict):
        _validate_threshold_and_temperature(calibration, failures)
    if isinstance(metrics, dict) and "metrics_by_split_calibrated" not in metrics:
        failures.append("neural_calibrated_metrics.json missing metrics_by_split_calibrated")

    return {
        "status": "fail" if failures else "ok",
        "n_predictions": len(prediction_rows),
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
        if not isinstance(payload, dict):
            failures.append(f"{label} is not a JSON object")
            return None
        return payload
    except (OSError, json.JSONDecodeError) as exc:
        failures.append(f"unreadable {label}: {exc}")
        return None


def _validate_predictions(path: Path, failures: List[str]) -> List[dict]:
    if not path.exists():
        failures.append(f"missing neural_calibrated_predictions.tsv: {path}")
        return []
    try:
        rows = read_tsv(path)
    except OSError as exc:
        failures.append(f"unreadable neural_calibrated_predictions.tsv: {exc}")
        return []

    if rows:
        missing_columns = [
            column
            for column in NEURAL_CALIBRATED_PREDICTION_FIELDNAMES
            if column not in rows[0]
        ]
        if missing_columns:
            failures.append(
                "neural_calibrated_predictions.tsv missing columns: "
                + ", ".join(missing_columns)
            )

    for row in rows:
        try:
            calibrated_prob = float(row.get("prob_positive_calibrated", ""))
            if not 0 <= calibrated_prob <= 1:
                failures.append(
                    f"prob_positive_calibrated out of range: {calibrated_prob}"
                )
        except ValueError:
            failures.append(
                "prob_positive_calibrated is not numeric: "
                f"{row.get('prob_positive_calibrated')}"
            )

        if row.get("pred_label_calibrated") not in {"0", "1", 0, 1}:
            failures.append(
                f"invalid pred_label_calibrated: {row.get('pred_label_calibrated')}"
            )
        if row.get("split") not in VALID_SPLITS:
            failures.append(f"invalid split value: {row.get('split')}")

    return rows


def _validate_threshold_and_temperature(calibration: dict, failures: List[str]) -> None:
    try:
        selected_threshold = float(calibration.get("selected_threshold", ""))
        if not 0 <= selected_threshold <= 1:
            failures.append(f"selected_threshold out of range: {selected_threshold}")
    except (TypeError, ValueError):
        failures.append(
            f"selected_threshold is not numeric: {calibration.get('selected_threshold')}"
        )

    try:
        temperature = float(calibration.get("temperature", ""))
        if temperature <= 0:
            failures.append(f"temperature must be > 0: {temperature}")
    except (TypeError, ValueError):
        failures.append(f"temperature is not numeric: {calibration.get('temperature')}")
