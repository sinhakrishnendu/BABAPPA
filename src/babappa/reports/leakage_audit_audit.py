"""Validation for BABAPPA leakage audit artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path


def validate_leakage_audit_dir(audit_dir: str | Path) -> dict:
    """Validate leakage audit outputs."""
    path = Path(audit_dir)
    failures = []
    warnings = []
    json_path = path / "leakage_audit.json"
    columns_path = path / "leakage_columns.tsv"
    markdown_path = path / "leakage_audit.md"
    payload = None

    if not json_path.exists():
        failures.append(f"missing:{json_path}")
    else:
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                failures.append(f"json_not_object:{json_path}")
        except json.JSONDecodeError as exc:
            failures.append(f"json_parse_error:{json_path}:{exc}")

    if not columns_path.exists():
        failures.append(f"missing:{columns_path}")
    else:
        with columns_path.open("r", encoding="utf-8", newline="") as handle:
            header = next(csv.reader(handle, delimiter="\t"), [])
        if not header:
            failures.append(f"empty_header:{columns_path}")

    if not markdown_path.exists():
        failures.append(f"missing:{markdown_path}")
    else:
        markdown = markdown_path.read_text(encoding="utf-8")
        if not markdown.strip():
            failures.append(f"empty_markdown:{markdown_path}")
        if "Interpretation" not in markdown:
            failures.append("markdown_missing_interpretation")

    if isinstance(payload, dict):
        if "recommended_excluded_columns" not in payload:
            failures.append("json_missing_recommended_excluded_columns")
        if payload.get("status") == "warning":
            warnings.append("leakage_audit_status_warning")
        warnings.extend(str(warning) for warning in payload.get("warnings", []))

    return {
        "status": "fail" if failures else "ok",
        "n_fail": len(failures),
        "n_warning": len(warnings),
        "failures": failures,
        "warnings": warnings,
    }
