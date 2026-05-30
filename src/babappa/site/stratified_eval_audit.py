"""Validation for site stratified evaluation outputs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import List


def validate_site_stratified_eval_dir(eval_dir: str | Path) -> dict:
    """Validate site stratified evaluation artifacts."""
    path = Path(eval_dir)
    failures: List[str] = []
    warnings: List[str] = []
    json_path = path / "site_stratified_eval.json"
    tsv_path = path / "site_stratified_metrics.tsv"
    md_path = path / "site_stratified_eval.md"
    _load_json(json_path, failures)
    if not tsv_path.exists():
        failures.append(f"missing_file:{tsv_path}")
    else:
        with tsv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle, delimiter="\t")
            if not next(reader, []):
                failures.append("empty_tsv_header")
    if not md_path.exists():
        failures.append(f"missing_file:{md_path}")
    elif "Interpretation caveats" not in md_path.read_text(encoding="utf-8"):
        warnings.append("markdown_missing_interpretation_caveats")
    return {
        "status": "fail" if failures else "ok",
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
