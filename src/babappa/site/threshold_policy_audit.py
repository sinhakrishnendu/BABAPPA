"""Validation for site-level threshold-policy artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import List


def validate_site_threshold_policy_dir(policy_dir: str | Path) -> dict:
    """Validate site threshold-policy artifacts."""
    path = Path(policy_dir)
    failures: List[str] = []
    warnings: List[str] = []
    required = {
        "json": path / "site_threshold_profiles.json",
        "profiles": path / "site_threshold_profiles.tsv",
        "metrics": path / "site_threshold_profile_metrics.tsv",
        "curve": path / "site_threshold_policy_curve.tsv",
        "markdown": path / "site_threshold_policy.md",
    }
    for label, file_path in required.items():
        if not file_path.exists():
            failures.append(f"missing_{label}:{file_path}")
    payload = _load_json(required["json"], failures)
    if payload and not payload.get("profiles"):
        failures.append("missing_profiles")
    for key in ("profiles", "metrics", "curve"):
        if required[key].exists():
            with required[key].open("r", encoding="utf-8", newline="") as handle:
                reader = csv.reader(handle, delimiter="\t")
                header = next(reader, [])
                if not header:
                    failures.append(f"empty_header:{required[key]}")
    if required["markdown"].exists() and not required["markdown"].read_text(encoding="utf-8").strip():
        failures.append("empty_markdown")
    return {
        "status": "fail" if failures else "ok",
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
