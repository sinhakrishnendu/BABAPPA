"""Validation for site aggregation control outputs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import List


def validate_site_aggregation_controls_dir(controls_dir: str | Path) -> dict:
    """Validate aggregation control artifacts."""
    path = Path(controls_dir)
    failures: List[str] = []
    required = {
        "json": path / "site_aggregation_controls.json",
        "tsv": path / "site_aggregation_controls.tsv",
        "markdown": path / "site_aggregation_controls.md",
    }
    payload = _load_json(required["json"], failures)
    if payload and "controls" not in payload:
        failures.append("missing_controls")
    if not required["tsv"].exists():
        failures.append(f"missing_file:{required['tsv']}")
    else:
        with required["tsv"].open("r", encoding="utf-8", newline="") as handle:
            if "empirical_p_value" not in next(csv.reader(handle, delimiter="\t"), []):
                failures.append("missing_empirical_p_value_header")
    if not required["markdown"].exists():
        failures.append(f"missing_file:{required['markdown']}")
    elif "Interpretation" not in required["markdown"].read_text(encoding="utf-8"):
        failures.append("markdown_missing_interpretation")
    return {
        "status": "fail" if failures else "ok",
        "n_fail": len(failures),
        "n_warning": 0,
        "failures": failures,
        "warnings": [],
    }


def _load_json(path: Path, failures: List[str]) -> dict:
    if not path.exists():
        failures.append(f"missing_file:{path}")
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        failures.append(f"could_not_parse_json:{path}:{exc}")
        return {}
    return payload if isinstance(payload, dict) else {}
