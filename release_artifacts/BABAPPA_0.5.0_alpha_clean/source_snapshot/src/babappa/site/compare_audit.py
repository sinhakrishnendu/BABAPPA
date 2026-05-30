"""Validation for site model comparison outputs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import List


def validate_site_model_comparison_dir(compare_dir: str | Path) -> dict:
    """Validate site model comparison artifacts."""
    path = Path(compare_dir)
    failures: List[str] = []
    required = {
        "json": path / "site_model_comparison.json",
        "tsv": path / "site_model_comparison.tsv",
        "markdown": path / "site_model_comparison.md",
    }
    payload = _load_json(required["json"], failures)
    if payload and "comparison" not in payload:
        failures.append("missing_comparison")
    if not required["tsv"].exists():
        failures.append(f"missing_file:{required['tsv']}")
    else:
        with required["tsv"].open("r", encoding="utf-8", newline="") as handle:
            if not next(csv.reader(handle, delimiter="\t"), []):
                failures.append("empty_tsv_header")
    if not required["markdown"].exists():
        failures.append(f"missing_file:{required['markdown']}")
    elif "Recommendation" not in required["markdown"].read_text(encoding="utf-8"):
        failures.append("markdown_missing_recommendation")
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
