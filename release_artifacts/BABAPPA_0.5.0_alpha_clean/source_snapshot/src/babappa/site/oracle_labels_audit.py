"""Validation for oracle site-label extraction outputs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import List

REQUIRED_COLUMNS = {
    "family_id",
    "method",
    "split",
    "saturation_tier",
    "tensor_file",
    "n_taxa",
    "n_codons",
    "site_index_zero",
    "site_index_one",
    "y_site",
    "oracle_label_source",
}


def validate_site_label_dir(site_label_dir: str | Path) -> dict:
    """Validate oracle site-label extraction artifacts."""
    label_dir = Path(site_label_dir)
    failures: List[str] = []
    warnings: List[str] = []
    summary_path = label_dir / "site_oracle_summary.json"
    labels_path = label_dir / "site_oracle_labels.tsv"
    markdown_path = label_dir / "site_oracle_labels.md"
    n_rows = 0
    n_positive = 0

    summary = _load_json(summary_path, failures)
    if not labels_path.exists():
        failures.append(f"missing_file:{labels_path}")
    else:
        try:
            with labels_path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle, delimiter="\t")
                fieldnames = set(reader.fieldnames or [])
                missing = sorted(REQUIRED_COLUMNS - fieldnames)
                if missing:
                    failures.append("missing_required_columns:" + ",".join(missing))
                for row in reader:
                    n_rows += 1
                    if row.get("y_site") not in {"0", "1", 0, 1}:
                        failures.append(f"invalid_y_site:{row.get('y_site')}")
                    if str(row.get("y_site")) == "1":
                        n_positive += 1
        except OSError as exc:
            failures.append(f"could_not_read_tsv:{labels_path}:{exc}")

    if not markdown_path.exists():
        failures.append(f"missing_file:{markdown_path}")
    else:
        text = markdown_path.read_text(encoding="utf-8")
        if not text.strip():
            failures.append(f"empty_markdown:{markdown_path}")
        if "Leakage note" not in text:
            warnings.append("markdown_missing_leakage_note")

    if summary and summary.get("n_site_records") != n_rows:
        warnings.append("summary_site_record_count_mismatch")
    if n_rows > 0 and n_positive == 0:
        warnings.append("no_positive_site_labels")

    return {
        "status": "fail" if failures else "ok",
        "n_rows": n_rows,
        "n_positive_sites": n_positive,
        "n_fail": len(failures),
        "n_warning": len(warnings),
        "failures": failures,
        "warnings": warnings,
    }


def _load_json(path: Path, failures: List[str]) -> dict:
    if not path.exists():
        failures.append(f"missing_file:{path}")
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
