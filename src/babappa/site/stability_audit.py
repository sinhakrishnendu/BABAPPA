"""Validation for site-neural stability benchmark outputs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import List


def validate_site_stability_dir(benchmark_dir: str | Path) -> dict:
    """Validate site stability benchmark artifacts."""
    path = Path(benchmark_dir)
    failures: List[str] = []
    required = {
        "json": path / "site_stability_benchmark.json",
        "tsv": path / "site_stability_results.tsv",
        "markdown": path / "site_stability_benchmark.md",
    }
    payload = _load_json(required["json"], failures)
    if payload and "aggregate_summary" not in payload:
        failures.append("missing_aggregate_summary")
    if not required["tsv"].exists():
        failures.append(f"missing_file:{required['tsv']}")
    else:
        with required["tsv"].open("r", encoding="utf-8", newline="") as handle:
            if not next(csv.reader(handle, delimiter="\t"), []):
                failures.append("empty_tsv_header")
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
