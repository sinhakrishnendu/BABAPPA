"""Validation for site-to-gene aggregation outputs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import List


def validate_site_aggregation_dir(aggregation_dir: str | Path) -> dict:
    """Validate site-to-gene aggregation artifacts."""
    path = Path(aggregation_dir)
    failures: List[str] = []
    warnings: List[str] = []
    predictions_path = path / "site_to_gene_predictions.tsv"
    metrics_path = path / "site_to_gene_metrics.json"
    markdown_path = path / "site_to_gene_aggregation.md"
    _load_json(metrics_path, failures)
    n_rows = 0
    if not predictions_path.exists():
        failures.append(f"missing_file:{predictions_path}")
    else:
        with predictions_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for row in reader:
                n_rows += 1
                if row.get("gene_label") not in {"", "0", "1"}:
                    failures.append(f"invalid_gene_label:{row.get('family_id')}:{row.get('gene_label')}")
                for column in [
                    "max_site_probability",
                    "mean_site_probability",
                    "top5_mean_site_probability",
                    "top10_mean_site_probability",
                    "fraction_sites_prob_ge_0_5",
                    "fraction_sites_prob_ge_0_8",
                ]:
                    value = float(row.get(column, "nan"))
                    if not 0 <= value <= 1:
                        failures.append(f"score_out_of_range:{column}:{value}")
    if n_rows == 0:
        failures.append("no_family_method_rows")
    if not markdown_path.exists():
        failures.append(f"missing_file:{markdown_path}")
    elif not markdown_path.read_text(encoding="utf-8").strip():
        failures.append("empty_markdown")
    return {
        "status": "fail" if failures else "ok",
        "n_rows": n_rows,
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
