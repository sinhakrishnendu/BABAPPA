"""Validation for BABAPPA label-signal audit artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path


def validate_label_signal_audit_dir(audit_dir: str | Path) -> dict:
    """Validate label-signal audit outputs."""
    path = Path(audit_dir)
    failures = []
    warnings = []
    json_path = path / "label_signal_audit.json"
    features_path = path / "label_signal_features.tsv"
    markdown_path = path / "label_signal_audit.md"

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

    if not features_path.exists():
        failures.append(f"missing:{features_path}")
    else:
        with features_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle, delimiter="\t")
            header = next(reader, [])
        if not header:
            failures.append(f"empty_header:{features_path}")

    if not markdown_path.exists():
        failures.append(f"missing:{markdown_path}")
    else:
        markdown = markdown_path.read_text(encoding="utf-8")
        if not markdown.strip():
            failures.append(f"empty_markdown:{markdown_path}")
        if "Interpretation" not in markdown:
            failures.append("markdown_missing_interpretation")

    if isinstance(payload, dict):
        if "top_features_by_auroc_distance" not in payload:
            failures.append("json_missing_top_features_by_auroc_distance")
        if payload.get("warnings"):
            warnings.extend(str(warning) for warning in payload["warnings"])

    return {
        "status": "fail" if failures else "ok",
        "n_fail": len(failures),
        "n_warning": len(warnings),
        "failures": failures,
        "warnings": warnings,
    }
