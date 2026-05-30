"""Validation for site-level calibration outputs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import List


def validate_site_calibration_dir(calibration_dir: str | Path) -> dict:
    """Validate site calibration artifacts."""
    path = Path(calibration_dir)
    failures: List[str] = []
    warnings: List[str] = []
    required = {
        "calibration": path / "site_calibration.json",
        "predictions": path / "site_calibrated_predictions.tsv",
        "metrics": path / "site_calibrated_metrics.json",
        "markdown": path / "site_calibration.md",
    }
    for label, file_path in required.items():
        if not file_path.exists():
            failures.append(f"missing_{label}:{file_path}")
    calibration = _load_json(required["calibration"], failures)
    _load_json(required["metrics"], failures)
    threshold = calibration.get("selected_threshold")
    temperature = calibration.get("temperature")
    if threshold is not None and not 0 <= float(threshold) <= 1:
        failures.append("selected_threshold_out_of_range")
    if temperature is not None and float(temperature) <= 0:
        failures.append("temperature_not_positive")
    n_predictions = 0
    if required["predictions"].exists():
        with required["predictions"].open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for row in reader:
                n_predictions += 1
                prob = float(row.get("prob_positive_calibrated", "nan"))
                if not 0 <= prob <= 1:
                    failures.append(f"calibrated_probability_out_of_range:{row.get('site_id')}")
                if row.get("y_site") not in {"0", "1"}:
                    failures.append(f"invalid_y_site:{row.get('site_id')}")
    if n_predictions == 0:
        failures.append("no_predictions")
    if required["markdown"].exists() and "Interpretation" not in required["markdown"].read_text(encoding="utf-8"):
        warnings.append("markdown_missing_interpretation")
    return {
        "status": "fail" if failures else "ok",
        "n_predictions": n_predictions,
        "n_fail": len(failures),
        "n_warning": len(warnings),
        "failures": failures,
        "warnings": warnings,
    }


def _load_json(path: Path, failures: List[str]) -> dict:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        failures.append(f"could_not_parse_json:{path}:{exc}")
        return {}
    if not isinstance(payload, dict):
        failures.append(f"json_not_object:{path}")
        return {}
    return payload
