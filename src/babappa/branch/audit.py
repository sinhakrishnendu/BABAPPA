"""Branch-site run summarization utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from babappa import __version__

BRANCH_RUN_SUMMARY_VERSION = __version__


@dataclass(frozen=True)
class BranchSiteRunSummaryConfig:
    outdir: str
    title: str = "BABAPPA branch-conditioned validation summary"
    branch_site_label_dir: Optional[str] = None
    branch_site_dataset_dir: Optional[str] = None
    branch_site_leakage_dir: Optional[str] = None
    branch_site_baseline_dir: Optional[str] = None
    branch_site_neural_dir: Optional[str] = None
    branch_site_calibration_dir: Optional[str] = None
    branch_aggregation_dir: Optional[str] = None
    branch_aggregation_controls_dir: Optional[str] = None
    branch_site_threshold_policy_dir: Optional[str] = None
    branch_aggregation_threshold_policy_dir: Optional[str] = None

    def __post_init__(self) -> None:
        Path(self.outdir).mkdir(parents=True, exist_ok=True)


def summarize_branch_site_run(config: BranchSiteRunSummaryConfig) -> dict:
    outdir = Path(config.outdir)
    sections = {
        "branch_site_labels": _artifact_status(config.branch_site_label_dir, ["branch_site_oracle_summary.json", "branch_site_oracle_labels.tsv"]),
        "branch_site_dataset": _artifact_status(config.branch_site_dataset_dir, ["branch_site_dataset_index.json", "branch_site_features.tsv"]),
        "branch_site_leakage": _artifact_status(config.branch_site_leakage_dir, ["branch_site_leakage_audit.json"]),
        "branch_site_baseline": _artifact_status(config.branch_site_baseline_dir, ["branch_site_baseline_metrics.json", "branch_site_baseline_predictions.tsv"]),
        "branch_site_neural": _artifact_status(config.branch_site_neural_dir, ["branch_site_neural_metrics.json", "branch_site_neural_predictions.tsv"]),
        "branch_site_calibration": _artifact_status(config.branch_site_calibration_dir, ["branch_site_calibration.json", "branch_site_calibrated_predictions.tsv"]),
        "branch_aggregation": _artifact_status(config.branch_aggregation_dir, ["branch_aggregation_metrics.json", "branch_to_gene_predictions.tsv"]),
        "branch_aggregation_controls": _artifact_status(config.branch_aggregation_controls_dir, ["branch_aggregation_controls.json"]),
        "branch_site_threshold_policy": _artifact_status(config.branch_site_threshold_policy_dir, ["branch_site_threshold_profiles.json"]),
        "branch_aggregation_threshold_policy": _artifact_status(config.branch_aggregation_threshold_policy_dir, ["branch_aggregation_threshold_profiles.json"]),
    }
    warnings = [f"missing_or_incomplete:{name}" for name, payload in sections.items() if payload["status"] != "present"]
    payload = {
        "branch_site_run_summary_version": BRANCH_RUN_SUMMARY_VERSION,
        "title": config.title,
        "sections": sections,
        "warnings": warnings,
        "recommended_next_action": "Run branch-conditioned 10K validation before considering final 100K.",
        "scientific_boundary": "Research-alpha simulation-supervised branch-conditioned validation; not empirical branch-site inference.",
    }
    json_path = outdir / "branch_site_run_summary.json"
    markdown_path = outdir / "branch_site_run_summary.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path.write_text(_render_markdown(payload), encoding="utf-8")
    return {"status": "ok", "outdir": str(outdir), "json": str(json_path), "markdown": str(markdown_path), "warnings": warnings}


def validate_branch_site_run_summary_dir(summary_dir: str | Path) -> dict:
    path = Path(summary_dir)
    failures = []
    warnings = []
    json_path = path / "branch_site_run_summary.json"
    markdown_path = path / "branch_site_run_summary.md"
    if not json_path.exists():
        failures.append(f"missing_file:{json_path}")
        payload = {}
    else:
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            failures.append(f"could_not_parse_json:{json_path}:{exc}")
            payload = {}
    if not markdown_path.exists():
        failures.append(f"missing_file:{markdown_path}")
    elif not markdown_path.read_text(encoding="utf-8").strip():
        failures.append("empty_markdown")
    warnings.extend(payload.get("warnings", []) if isinstance(payload, dict) else [])
    return {"status": "fail" if failures else "ok", "n_fail": len(failures), "n_warning": len(warnings), "failures": failures, "warnings": warnings}


def _artifact_status(directory: Optional[str], required_files: list[str]) -> dict:
    if not directory:
        return {"status": "missing", "directory": None, "missing_files": required_files}
    path = Path(directory)
    missing = [filename for filename in required_files if not (path / filename).exists()]
    return {"status": "present" if path.exists() and not missing else "incomplete", "directory": str(path), "missing_files": missing}


def _render_markdown(payload: dict) -> str:
    lines = ["# " + str(payload.get("title")), ""]
    lines.extend(["## Scientific boundary", "", str(payload.get("scientific_boundary")), ""])
    lines.extend(["## Artifacts", ""])
    for name, section in payload.get("sections", {}).items():
        lines.append(f"- {name}: {section.get('status')}")
    lines.extend(["", "## Recommended next action", "", str(payload.get("recommended_next_action")), ""])
    return "\n".join(lines)
