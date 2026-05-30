"""Validation for BABAPPA stability benchmark artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path


def validate_stability_benchmark_dir(benchmark_dir: str | Path) -> dict:
    """Validate stability benchmark outputs."""
    path = Path(benchmark_dir)
    failures = []
    warnings = []
    json_path = path / "stability_benchmark.json"
    tsv_path = path / "stability_results.tsv"
    markdown_path = path / "stability_benchmark.md"
    payload = None

    if not json_path.exists():
        failures.append(f"missing:{json_path}")
    else:
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                failures.append("json_not_object")
        except json.JSONDecodeError as exc:
            failures.append(f"json_parse_error:{exc}")

    if not tsv_path.exists():
        failures.append(f"missing:{tsv_path}")
    else:
        with tsv_path.open("r", encoding="utf-8", newline="") as handle:
            header = next(csv.reader(handle, delimiter="\t"), [])
        if not header:
            failures.append("empty_tsv_header")

    if not markdown_path.exists():
        failures.append(f"missing:{markdown_path}")
    else:
        markdown = markdown_path.read_text(encoding="utf-8")
        if not markdown.strip():
            failures.append("empty_markdown")
        if "Interpretation" not in markdown:
            failures.append("markdown_missing_interpretation")

    if isinstance(payload, dict):
        if "aggregate_summary" not in payload:
            failures.append("json_missing_aggregate_summary")
        warnings.extend(str(warning) for warning in payload.get("warnings", []))
        warnings.extend(
            str(warning)
            for warning in (payload.get("aggregate_summary") or {}).get(
                "instability_warnings", []
            )
        )

    return {
        "status": "fail" if failures else "ok",
        "n_fail": len(failures),
        "n_warning": len(warnings),
        "failures": failures,
        "warnings": warnings,
    }
