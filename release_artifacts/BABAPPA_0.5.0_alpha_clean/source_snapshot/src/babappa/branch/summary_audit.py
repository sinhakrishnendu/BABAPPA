"""Validation for branch-conditioned cross-tier summaries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Union

from babappa.branch.summary import MARKDOWN_SECTIONS, REQUIRED_TSV_FILES, SUMMARY_FILES
from babappa.datasets.index import read_tsv

REQUIRED_TIERS = {"low", "moderate", "high", "extreme"}


def validate_branch_conditioned_tier_summary_dir(summary_dir: Union[str, Path]) -> Dict[str, Any]:
    """Validate branch-conditioned cross-tier summary artifacts."""

    path = Path(summary_dir)
    failures: List[str] = []
    warnings: List[str] = []

    payload = _load_summary_json(path / SUMMARY_FILES["json"], failures)
    markdown_path = path / SUMMARY_FILES["markdown"]
    if not markdown_path.exists():
        failures.append(f"missing_file:{markdown_path}")
        markdown_text = ""
    else:
        markdown_text = markdown_path.read_text(encoding="utf-8")
        if not markdown_text.strip():
            failures.append(f"empty_file:{markdown_path}")

    for filename in REQUIRED_TSV_FILES:
        candidate = path / filename
        if not candidate.exists():
            failures.append(f"missing_file:{candidate}")

    if payload:
        tiers = payload.get("tiers")
        if not isinstance(tiers, list):
            failures.append("summary_json_missing_tiers")
            tier_names = set()
        else:
            tier_names = {str(row.get("tier")) for row in tiers if isinstance(row, dict)}
            missing_tiers = sorted(REQUIRED_TIERS - tier_names)
            if missing_tiers:
                failures.append("missing_required_tiers:" + ",".join(missing_tiers))
        warnings.extend(str(warning) for warning in payload.get("warnings", []))
        _check_metric_rows(
            payload.get("branch_site_neural_rows"),
            "branch_site_neural",
            "branch_site_neural_missing_tier",
            failures,
        )
        _check_metric_rows(
            payload.get("branch_aggregation_rows"),
            "branch_level_aggregation",
            "branch_aggregation_missing_tier",
            failures,
        )
        _check_metric_rows(
            payload.get("branch_gene_aggregation_rows"),
            "branch_to_gene_aggregation",
            "branch_gene_aggregation_missing_tier",
            failures,
        )

    if markdown_text:
        expected_title = str(payload.get("title", "")).strip() if payload else ""
        expected_heading = f"# {expected_title}" if expected_title else ""
        title_ok = bool(expected_heading and expected_heading in markdown_text)
        if not title_ok and "# BABAPPA branch-conditioned 10K cross-tier summary" in markdown_text:
            title_ok = True
        if not title_ok:
            failures.append("markdown_missing_title")
        for section in MARKDOWN_SECTIONS:
            if section.startswith("# BABAPPA "):
                continue
            if section not in markdown_text:
                failures.append(f"markdown_missing_section:{section}")
        if "## Label-truth status" not in markdown_text:
            failures.append("markdown_missing_label_truth_status_section")
        if "## Scientific boundary" not in markdown_text:
            failures.append("markdown_missing_scientific_boundary_section")

    for filename in REQUIRED_TSV_FILES:
        candidate = path / filename
        if candidate.exists() and filename in {
            SUMMARY_FILES["neural_tsv"],
            SUMMARY_FILES["branch_aggregation_tsv"],
            SUMMARY_FILES["gene_aggregation_tsv"],
        }:
            rows = read_tsv(candidate)
            if not rows:
                failures.append(f"no_rows:{candidate}")

    return {
        "status": "fail" if failures else "ok",
        "n_fail": len(failures),
        "n_warning": len(warnings),
        "failures": failures,
        "warnings": warnings,
    }


def _load_summary_json(path: Path, failures: List[str]) -> Dict[str, Any]:
    if not path.exists():
        failures.append(f"missing_file:{path}")
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        failures.append(f"could_not_parse_json:{path}:{exc}")
        return {}
    if not isinstance(payload, dict):
        failures.append("summary_json_root_not_object")
        return {}
    return payload


def _check_metric_rows(rows: Any, level: str, failure_prefix: str, failures: List[str]) -> None:
    if not isinstance(rows, list):
        failures.append(f"{level}_rows_not_list")
        return
    seen = {
        str(row.get("tier"))
        for row in rows
        if isinstance(row, dict)
        and row.get("level") == level
        and row.get("split") in {"test", "all"}
        and row.get("auroc") not in (None, "")
    }
    missing = sorted(REQUIRED_TIERS - seen)
    if missing:
        failures.append(f"{failure_prefix}:" + ",".join(missing))
