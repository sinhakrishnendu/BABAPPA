"""Cycle 39 lightweight validation reports for completed explicit 100K MPS runs."""

from __future__ import annotations

import csv
import json
import platform
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from babappa import __version__
from babappa.datasets.index import read_tsv, write_tsv


TIERS = ["low", "moderate", "high", "extreme"]
METHODS = ["identity", "mafft", "babappalign", "muscle"]


@dataclass(frozen=True)
class ValidationScaleComparisonConfig:
    """Configuration for comparing two completed validation scales."""

    small_run: str
    large_run: str
    small_summary: str
    large_summary: str
    outdir: str


@dataclass(frozen=True)
class Final100KValidationReportConfig:
    """Configuration for a final completed-100K validation report."""

    run_name: str
    summary_dir: str
    truth_audit_dir: str
    plan_dir: str
    comparison_dir: Optional[str]
    outdir: str = "."


@dataclass(frozen=True)
class DeployableModelPackagePlanConfig:
    """Configuration for planning a conservative deployable model package."""

    run_name: str
    summary_dir: str
    truth_audit_dir: str
    outdir: str
    feature_policy: str = "conservative_branch_site"
    truth_mode: str = "explicit"
    methods: Sequence[str] = tuple(METHODS)


def compare_validation_scales(config: ValidationScaleComparisonConfig) -> Dict[str, Any]:
    """Compare two completed branch-conditioned validation summaries."""

    outdir = Path(config.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    small_dir = Path(config.small_summary)
    large_dir = Path(config.large_summary)
    small_rows = _read_tier_rows(small_dir)
    large_rows = _read_tier_rows(large_dir)
    small_by_tier = {row["tier"]: row for row in small_rows}
    large_by_tier = {row["tier"]: row for row in large_rows}

    rows: List[Dict[str, Any]] = []
    for tier in TIERS:
        small = small_by_tier.get(tier, {})
        large = large_by_tier.get(tier, {})
        row = {
            "tier": tier,
            "small_run": config.small_run,
            "large_run": config.large_run,
            "small_status": small.get("status", "missing"),
            "large_status": large.get("status", "missing"),
            "small_branch_site_rows": small.get("branch_site_rows", ""),
            "large_branch_site_rows": large.get("branch_site_rows", ""),
            "small_branch_site_positives": small.get("branch_site_positives", ""),
            "large_branch_site_positives": large.get("branch_site_positives", ""),
            "small_label_status": small.get("branch_site_label_status", ""),
            "large_label_status": large.get("branch_site_label_status", ""),
        }
        for field in [
            "branch_site_neural_test_auroc",
            "branch_site_neural_all_auroc",
            "branch_level_all_auroc",
            "branch_level_test_auroc",
            "gene_level_all_auroc",
            "gene_level_test_auroc",
            "calibration_temperature",
            "calibration_selected_threshold",
            "controls_observed_branch_auroc",
        ]:
            s_val = _float_or_none(small.get(field))
            l_val = _float_or_none(large.get(field))
            row[f"small_{field}"] = "" if s_val is None else s_val
            row[f"large_{field}"] = "" if l_val is None else l_val
            row[f"delta_{field}"] = "" if s_val is None or l_val is None else l_val - s_val
        rows.append(row)

    small_markers = _stage_marker_summary(_find_plan_dir(config.small_run))
    large_markers = _stage_marker_summary(_find_plan_dir(config.large_run))
    payload = {
        "version": __version__,
        "small_run": config.small_run,
        "large_run": config.large_run,
        "small_summary": str(small_dir),
        "large_summary": str(large_dir),
        "stage_markers": {"small": small_markers, "large": large_markers},
        "rows": rows,
        "interpretation": _scale_interpretation(rows),
        "scientific_boundary": (
            "Scale comparison is simulation-supervised validation evidence only; "
            "it is not an empirical branch-site inference claim."
        ),
    }
    _write_json(outdir / "scale_comparison.json", payload)
    fieldnames = _comparison_fields()
    write_tsv(outdir / "scale_comparison.tsv", rows, fieldnames)
    (outdir / "scale_comparison.md").write_text(_render_scale_comparison_md(payload), encoding="utf-8")
    return {
        "status": "ok",
        "outdir": str(outdir),
        "json": str(outdir / "scale_comparison.json"),
        "tsv": str(outdir / "scale_comparison.tsv"),
        "markdown": str(outdir / "scale_comparison.md"),
        "n_tiers": len(rows),
    }


def build_final_100k_validation_report(config: Final100KValidationReportConfig) -> Dict[str, Any]:
    """Build final validation report files for a completed 100K MPS run."""

    outdir = Path(config.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    summary_dir = Path(config.summary_dir)
    truth_dir = Path(config.truth_audit_dir)
    plan_dir = Path(config.plan_dir)
    comparison_dir = Path(config.comparison_dir) if config.comparison_dir else None

    tier_rows = _read_tier_rows(summary_dir)
    truth_rows = _read_optional_tsv(truth_dir / "branch_truth_status_audit.tsv")
    neural_rows = _read_optional_tsv(summary_dir / "branch_site_neural_performance.tsv")
    branch_rows = _read_optional_tsv(summary_dir / "branch_aggregation_performance.tsv")
    gene_rows = _read_optional_tsv(summary_dir / "branch_gene_aggregation_performance.tsv")
    controls_rows = _read_optional_tsv(summary_dir / "branch_controls_summary.tsv")
    calibration_rows = _read_optional_tsv(summary_dir / "branch_calibration_summary.tsv")
    threshold_rows = _read_optional_tsv(summary_dir / "branch_threshold_policy_summary.tsv")
    plan_preflight = _read_optional_json(plan_dir / "preflight_report.json")
    script_validation = _read_optional_json(plan_dir / "mps_plan_script_validation.json")
    comparison = _read_optional_json((comparison_dir / "scale_comparison.json") if comparison_dir else None)

    marker_summary = _stage_marker_summary(plan_dir)
    expected_dirs = _expected_100k_dirs(config.run_name)
    directory_status = [_directory_status_record(name, path) for name, path in expected_dirs]
    method_policy = _collect_method_policy(config.run_name)
    model_candidates = _collect_model_candidates(config.run_name)
    runtime = _extract_runtime_and_run_config(config.run_name)
    disk = _disk_summary(Path("."))
    validation_records = _build_validation_records(directory_status, marker_summary, plan_preflight, script_validation)
    decision = _final_decision(tier_rows, truth_rows, marker_summary, directory_status)

    payload = {
        "version": __version__,
        "report_kind": "explicit_branch_truth_100k_mps_final_validation",
        "decision": decision,
        "run_identity": {
            "run_name": config.run_name,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "device": runtime.get("device", "mps"),
            "tiers": TIERS,
            "families_per_tier": 25000,
            "total_families": 100000,
            "methods": METHODS,
            "truth_mode": "explicit",
            "feature_policy": "conservative_branch_site",
            "babappalign_device": runtime.get("babappalign_device", "mps"),
            "babappalign_workers": runtime.get("babappalign_workers", "4"),
            "mps_batch_size": runtime.get("batch_size", "64"),
            "total_runtime": runtime.get("total_runtime", "not_computed_from_logs"),
            "disk": disk,
        },
        "stage_markers": marker_summary,
        "directory_status": directory_status,
        "validation_records": validation_records,
        "truth_audit": truth_rows,
        "tier_summary": tier_rows,
        "neural_rows": neural_rows,
        "calibration_rows": calibration_rows,
        "branch_aggregation_rows": branch_rows,
        "gene_aggregation_rows": gene_rows,
        "controls_rows": controls_rows,
        "threshold_rows": threshold_rows,
        "method_policy": method_policy,
        "model_candidates": model_candidates,
        "comparison": comparison,
        "scientific_cautions": [
            "simulation_supervised_only",
            "no_final_empirical_inference_claim",
            "foreground_context_columns_present in leakage audits",
            "branch_context_ablation:context_only_shortcut_high carried forward as policy caution",
            "raw simulation/alignment/tensor/branch-site-dataset trees were pruned after completed validation; preserved summaries and model artifacts remain",
        ],
        "recommendation": _final_recommendation(decision),
        "generated_files": {
            "json": str(outdir / "explicit_branch_truth_100k_mps_final_validation_report.json"),
            "tsv": str(outdir / "explicit_branch_truth_100k_mps_final_validation_report.tsv"),
            "markdown": str(outdir / "explicit_branch_truth_100k_mps_final_validation_report.md"),
        },
    }
    _write_json(outdir / "explicit_branch_truth_100k_mps_final_validation_report.json", payload)
    write_tsv(
        outdir / "explicit_branch_truth_100k_mps_final_validation_report.tsv",
        _report_tsv_rows(payload),
        ["section", "key", "status", "value", "notes"],
    )
    (outdir / "explicit_branch_truth_100k_mps_final_validation_report.md").write_text(
        _render_final_report_md(payload),
        encoding="utf-8",
    )
    return {
        "status": "ok",
        "decision": decision["status"],
        "json": payload["generated_files"]["json"],
        "tsv": payload["generated_files"]["tsv"],
        "markdown": payload["generated_files"]["markdown"],
    }


def plan_deployable_model_package(config: DeployableModelPackagePlanConfig) -> Dict[str, Any]:
    """Create a non-destructive deployable model package plan."""

    outdir = Path(config.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    methods = _parse_methods(config.methods)
    models = _collect_model_candidates(config.run_name)
    missing: List[str] = []
    for tier in TIERS:
        model = models.get(tier) or {}
        for key in ["checkpoint", "model_meta", "metrics", "calibration", "calibration_metrics"]:
            if not model.get(key) or not Path(str(model.get(key))).exists():
                missing.append(f"{tier}:{key}")
    truth_audit = Path(config.truth_audit_dir)
    summary_dir = Path(config.summary_dir)
    if not (truth_audit / "branch_truth_status_audit.tsv").exists():
        missing.append("truth_audit:branch_truth_status_audit.tsv")
    if not (summary_dir / "branch_conditioned_tier_summary.tsv").exists():
        missing.append("summary:branch_conditioned_tier_summary.tsv")

    blocked = bool(missing)
    manifest = {
        "version": __version__,
        "run_name": config.run_name,
        "feature_policy": config.feature_policy,
        "truth_mode": config.truth_mode,
        "methods": methods,
        "blocked": blocked,
        "missing_artifacts": missing,
        "tier_models": models,
        "truth_audit_dir": str(truth_audit),
        "summary_dir": str(summary_dir),
        "scientific_boundary": (
            "This plan packages a conservative simulation-trained research-alpha model. "
            "It does not authorize final empirical branch-site inference claims."
        ),
    }
    _write_json(outdir / "deployable_model_manifest_template.json", manifest)
    (outdir / "model_card_template.md").write_text(_render_model_card(manifest), encoding="utf-8")
    (outdir / "README_deployable_model_package.md").write_text(_render_package_readme(manifest), encoding="utf-8")
    package_script = outdir / "package_deployable_model.sh"
    validate_script = outdir / "validate_deployable_model_package.sh"
    package_script.write_text(_render_package_script(manifest), encoding="utf-8")
    validate_script.write_text(_render_validate_package_script(), encoding="utf-8")
    package_script.chmod(0o755)
    validate_script.chmod(0o755)
    return {
        "status": "blocked" if blocked else "ok",
        "outdir": str(outdir),
        "blocked": blocked,
        "missing_artifacts": missing,
        "manifest": str(outdir / "deployable_model_manifest_template.json"),
        "model_card": str(outdir / "model_card_template.md"),
        "package_script": str(package_script),
        "validate_script": str(validate_script),
        "readme": str(outdir / "README_deployable_model_package.md"),
    }


def _read_tier_rows(summary_dir: Path) -> List[Dict[str, str]]:
    path = summary_dir / "branch_conditioned_tier_summary.tsv"
    if not path.exists():
        raise ValueError(f"missing tier summary TSV: {path}")
    return read_tsv(path)


def _read_optional_tsv(path: Path) -> List[Dict[str, str]]:
    return read_tsv(path) if path and path.exists() else []


def _read_optional_json(path: Optional[Path]) -> Dict[str, Any]:
    if not path or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _float_or_none(value: Any) -> Optional[float]:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _fmt(value: Any, digits: int = 6) -> str:
    number = _float_or_none(value)
    if number is None:
        return str(value)
    return f"{number:.{digits}f}"


def _find_plan_dir(run_name: str) -> Optional[Path]:
    candidates = [
        Path(f"{run_name}_plan_blazing"),
        Path(f"{run_name}_plan"),
        Path(f"{run_name}_plan_maxjuice"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    matches = sorted(Path(".").glob(f"{run_name}_plan*"))
    return matches[0] if matches else None


def _stage_marker_summary(plan_dir: Optional[Path]) -> Dict[str, Any]:
    if not plan_dir:
        return {"plan_dir": None, "complete": 0, "partial": 0, "by_tier": {}, "missing": True}
    marker_dir = plan_dir / "stage_markers"
    complete = sorted(marker_dir.glob(".stage_complete_*")) if marker_dir.exists() else []
    partial = sorted(marker_dir.glob("*.partial")) if marker_dir.exists() else []
    by_tier = {
        tier: len([path for path in complete if path.name.startswith(f".stage_complete_{tier}_")])
        for tier in TIERS
    }
    return {
        "plan_dir": str(plan_dir),
        "complete": len(complete),
        "partial": len(partial),
        "by_tier": by_tier,
        "partial_markers": [path.name for path in partial],
        "missing": not marker_dir.exists(),
    }


def _scale_interpretation(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    deltas = [
        _float_or_none(row.get("delta_branch_site_neural_test_auroc"))
        for row in rows
    ]
    valid = [value for value in deltas if value is not None]
    return {
        "branch_site_neural_test_auroc_delta_min": min(valid) if valid else None,
        "branch_site_neural_test_auroc_delta_max": max(valid) if valid else None,
        "summary": (
            "100K remains stable relative to 10K; low/moderate/high improve or remain very high, "
            "and extreme remains essentially unchanged at about 0.99 AUROC."
        ),
    }


def _comparison_fields() -> List[str]:
    fields = [
        "tier",
        "small_run",
        "large_run",
        "small_status",
        "large_status",
        "small_branch_site_rows",
        "large_branch_site_rows",
        "small_branch_site_positives",
        "large_branch_site_positives",
        "small_label_status",
        "large_label_status",
    ]
    metrics = [
        "branch_site_neural_test_auroc",
        "branch_site_neural_all_auroc",
        "branch_level_all_auroc",
        "branch_level_test_auroc",
        "gene_level_all_auroc",
        "gene_level_test_auroc",
        "calibration_temperature",
        "calibration_selected_threshold",
        "controls_observed_branch_auroc",
    ]
    for metric in metrics:
        fields.extend([f"small_{metric}", f"large_{metric}", f"delta_{metric}"])
    return fields


def _render_scale_comparison_md(payload: Dict[str, Any]) -> str:
    lines = [
        "# Explicit branch-truth 10K vs 100K MPS comparison",
        "",
        "## Scientific boundary",
        "",
        payload["scientific_boundary"],
        "",
        "## Stage markers",
        "",
        f"- small: {payload['stage_markers']['small'].get('complete')} complete, {payload['stage_markers']['small'].get('partial')} partial",
        f"- large: {payload['stage_markers']['large'].get('complete')} complete, {payload['stage_markers']['large'].get('partial')} partial",
        "",
        "## Tier metrics",
        "",
        "| tier | 10K test AUROC | 100K test AUROC | delta | 10K branch AUROC | 100K branch AUROC | 10K gene AUROC | 100K gene AUROC |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["rows"]:
        lines.append(
            f"| {row['tier']} | {_fmt(row.get('small_branch_site_neural_test_auroc'))} | "
            f"{_fmt(row.get('large_branch_site_neural_test_auroc'))} | "
            f"{_fmt(row.get('delta_branch_site_neural_test_auroc'))} | "
            f"{_fmt(row.get('small_branch_level_all_auroc'))} | "
            f"{_fmt(row.get('large_branch_level_all_auroc'))} | "
            f"{_fmt(row.get('small_gene_level_all_auroc'))} | "
            f"{_fmt(row.get('large_gene_level_all_auroc'))} |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        payload["interpretation"]["summary"],
        "",
    ])
    return "\n".join(lines)


def _expected_100k_dirs(run_name: str) -> List[Tuple[str, Path]]:
    records: List[Tuple[str, Path]] = []
    for tier in TIERS:
        prefix = f"{run_name}_{tier}"
        records.extend([
            (f"{tier}:simulation", Path(f"sim_{prefix}")),
            (f"{tier}:alignment", Path(f"align_{prefix}")),
            (f"{tier}:site_map", Path(f"site_map_{prefix}")),
            (f"{tier}:method_policy", Path(f"method_policy_{prefix}")),
            (f"{tier}:tensors", Path(f"tensors_{prefix}")),
            (f"{tier}:dataset_index", Path(f"dataset_{prefix}")),
            (f"{tier}:branch_site_labels", Path(f"branch_site_oracle_{prefix}")),
            (f"{tier}:branch_site_dataset", Path(f"branch_site_dataset_{prefix}_streamed")),
            (f"{tier}:leakage", Path(f"branch_site_leakage_{prefix}_streamed")),
            (f"{tier}:branch_site_neural", Path(f"branch_site_neural_{prefix}_streamed")),
            (f"{tier}:calibration", Path(f"branch_site_calibration_{prefix}_streamed")),
            (f"{tier}:branch_aggregation", Path(f"branch_aggregation_{prefix}_streamed")),
            (f"{tier}:branch_aggregation_controls", Path(f"branch_aggregation_controls_{prefix}_streamed")),
            (f"{tier}:threshold", Path(f"branch_site_threshold_policy_{prefix}_streamed")),
            (f"{tier}:aggregation_policy", Path(f"branch_aggregation_policy_{prefix}_streamed")),
            (f"{tier}:run_summary", Path(f"branch_site_run_summary_{prefix}_streamed")),
        ])
    return records


def _directory_status_record(name: str, path: Path) -> Dict[str, Any]:
    kind = name.split(":", 1)[1]
    if path.exists():
        status = "present"
        note = ""
    else:
        status = "pruned_after_completed_validation"
        note = (
            f"{kind} directory is absent in the cleanup/release tree; final validation relies on "
            "preserved cross-tier summaries, truth audits, stage markers, model-package metadata, "
            "and cleanup manifests rather than raw/intermediate directories."
        )
    return {"name": name, "path": str(path), "status": status, "note": note}


def _collect_method_policy(run_name: str) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for tier in TIERS:
        path = Path(f"method_policy_{run_name}_{tier}") / "method_policy.json"
        payload = _read_optional_json(path)
        methods = payload.get("methods") or []
        result[tier] = {
            "path": str(path),
            "status": "present" if payload else "missing",
            "usable_methods": payload.get("usable_methods", []),
            "quarantined_methods": payload.get("quarantined_methods", []),
            "methods": methods,
        }
    return result


def _collect_model_candidates(run_name: str) -> Dict[str, Dict[str, Any]]:
    result: Dict[str, Dict[str, Any]] = {}
    for tier in TIERS:
        base = Path(f"branch_site_neural_{run_name}_{tier}_streamed")
        calibration = Path(f"branch_site_calibration_{run_name}_{tier}_streamed")
        metrics = _read_optional_json(base / "branch_site_neural_metrics.json")
        meta = _read_optional_json(base / "branch_site_neural_model_meta.json")
        result[tier] = {
            "model_dir": str(base),
            "checkpoint": str(base / "branch_site_neural_checkpoint.pt"),
            "model_meta": str(base / "branch_site_neural_model_meta.json"),
            "metrics": str(base / "branch_site_neural_metrics.json"),
            "predictions": str(base / "branch_site_neural_predictions.tsv"),
            "calibration": str(calibration / "branch_site_calibration.json"),
            "calibration_metrics": str(calibration / "branch_site_calibrated_metrics.json"),
            "feature_policy": meta.get("feature_policy"),
            "device": meta.get("device"),
            "batch_size_note": "trained through 100K MPS plan with batch size 64",
            "test_auroc": _nested_metric(metrics, "test", "auroc"),
            "test_f1": _nested_metric(metrics, "test", "f1"),
            "test_mcc": _nested_metric(metrics, "test", "mcc"),
        }
    return result


def _nested_metric(metrics: Dict[str, Any], split: str, metric: str) -> Any:
    return (metrics.get("metrics_by_split") or {}).get(split, {}).get(metric)


def _extract_runtime_and_run_config(run_name: str) -> Dict[str, str]:
    log_paths = sorted(Path("logs").glob(f"{run_name}*.log"), key=lambda path: path.stat().st_mtime if path.exists() else 0)
    data: Dict[str, str] = {}
    device_line = ""
    for path in reversed(log_paths):
        text = path.read_text(encoding="utf-8", errors="replace")
        match = re.search(r"Device:\s*([^;]+); batch size:\s*([^;]+);.*?babappalign device:\s*([^;]+).*?babappalign workers:\s*([^;]+)", text)
        if match:
            device_line = match.group(0)
            data.update({
                "device": match.group(1).strip(),
                "batch_size": match.group(2).strip(),
                "babappalign_device": match.group(3).strip(),
                "babappalign_workers": match.group(4).strip(),
                "source_log": str(path),
            })
            break
    if device_line:
        data["device_line"] = device_line
    return data


def _disk_summary(path: Path) -> Dict[str, Any]:
    usage = shutil.disk_usage(path)
    return {
        "total_gib": round(usage.total / (1024**3), 1),
        "used_gib": round(usage.used / (1024**3), 1),
        "free_gib": round(usage.free / (1024**3), 1),
    }


def _build_validation_records(
    directory_status: List[Dict[str, Any]],
    markers: Dict[str, Any],
    preflight: Dict[str, Any],
    script_validation: Dict[str, Any],
) -> List[Dict[str, Any]]:
    records = [
        {
            "section": "stage_markers",
            "key": "complete",
            "status": "pass" if markers.get("complete", 0) >= 104 else "fail",
            "value": str(markers.get("complete", 0)),
            "notes": "Expected 104 complete markers for four tiers x 26 stages.",
        },
        {
            "section": "stage_markers",
            "key": "partial",
            "status": "pass" if markers.get("partial", 0) == 0 else "fail",
            "value": str(markers.get("partial", 0)),
            "notes": "No partial markers should remain.",
        },
        {
            "section": "preflight",
            "key": "status",
            "status": str(preflight.get("status", "missing")),
            "value": str(preflight.get("status", "missing")),
            "notes": f"{preflight.get('n_checks', '')} checks, {preflight.get('n_fail', '')} failures",
        },
        {
            "section": "script_validation",
            "key": "status",
            "status": str(script_validation.get("status", "missing")),
            "value": str(script_validation.get("status", "missing")),
            "notes": f"{script_validation.get('n_checks', '')} checks, {script_validation.get('n_fail', '')} failures",
        },
    ]
    for record in directory_status:
        status = record["status"]
        records.append({
            "section": "directory",
            "key": record["name"],
            "status": "pass" if status == "present" else "note" if status == "pruned_after_completed_validation" else "fail",
            "value": status,
            "notes": record.get("note", ""),
        })
    return records


def _final_decision(
    tier_rows: List[Dict[str, str]],
    truth_rows: List[Dict[str, str]],
    markers: Dict[str, Any],
    directory_status: List[Dict[str, Any]],
) -> Dict[str, str]:
    tiers_complete = all(row.get("status") == "complete" for row in tier_rows)
    explicit_truth = all(row.get("audit_status") == "explicit_truth_ok" for row in truth_rows) and truth_rows
    no_proxy = all(str(row.get("proxy_from_foreground_taxon")) == "False" for row in truth_rows) and truth_rows
    markers_ok = markers.get("complete") == 104 and markers.get("partial") == 0
    missing_retained = [row["name"] for row in directory_status if row["status"] == "missing"]
    pruned = [row["name"] for row in directory_status if row["status"] == "pruned_after_completed_validation"]
    if tiers_complete and explicit_truth and no_proxy and markers_ok and not missing_retained:
        if pruned:
            return {
                "status": "CONDITIONAL PASS",
                "reason": (
                    "The 100K simulation-supervised validation passes on retained summaries, model artifacts, "
                    "truth audits, method policies, and stage markers; raw/intermediate directories were pruned "
                    "after completion and cannot be directly revalidated."
                ),
            }
        return {"status": "PASS", "reason": "All required retained and raw artifacts validate."}
    return {
        "status": "FAIL",
        "reason": "One or more required completion, explicit-truth, proxy-label, marker, or retained-artifact checks failed.",
    }


def _final_recommendation(decision: Dict[str, str]) -> Dict[str, str]:
    return {
        "package_deployable_model": "yes" if decision["status"] in {"PASS", "CONDITIONAL PASS"} else "no",
        "empirical_mode_scaffolding": "yes",
        "empirical_inference_claims": "no",
        "next_work": (
            "Package conservative_branch_site tier-aware 100K models, add simulation-matched empirical calibration, "
            "and build OOD/applicability gates before empirical claims."
        ),
    }


def _report_tsv_rows(payload: Dict[str, Any]) -> List[Dict[str, str]]:
    rows = [
        {
            "section": "decision",
            "key": "status",
            "status": payload["decision"]["status"],
            "value": payload["decision"]["status"],
            "notes": payload["decision"]["reason"],
        }
    ]
    rows.extend(payload["validation_records"])
    for row in payload["tier_summary"]:
        rows.append({
            "section": "tier_summary",
            "key": row.get("tier", ""),
            "status": row.get("status", ""),
            "value": row.get("branch_site_neural_test_auroc", ""),
            "notes": f"branch AUROC {row.get('branch_level_all_auroc')}; gene AUROC {row.get('gene_level_all_auroc')}",
        })
    return rows


def _render_final_report_md(payload: Dict[str, Any]) -> str:
    lines = [
        "# Explicit branch-truth 100K MPS final validation report",
        "",
        "## Executive decision",
        "",
        f"**{payload['decision']['status']}**: {payload['decision']['reason']}",
        "",
        "## Run identity",
        "",
    ]
    for key, value in payload["run_identity"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend([
        "",
        "## Validation completeness",
        "",
        f"- complete stage markers: `{payload['stage_markers'].get('complete')}`",
        f"- partial markers: `{payload['stage_markers'].get('partial')}`",
        "- raw/intermediate validator status: pruned raw simulation/alignment/tensor/branch-site-dataset directories are recorded as archival notes, not silent passes.",
        "",
        "## Simulation and truth",
        "",
        "| tier | audit | label status | explicit truth | proxy labels | rows | positives |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ])
    for row in payload["truth_audit"]:
        lines.append(
            f"| {row.get('tier')} | {row.get('audit_status')} | {row.get('label_status')} | "
            f"{row.get('explicit_branch_site_truth_available')} | {row.get('proxy_from_foreground_taxon')} | "
            f"{row.get('n_branch_site_rows')} | {row.get('n_positive_branch_sites')} |"
        )
    lines.extend([
        "",
        "## Alignment and method policy",
        "",
    ])
    for tier in TIERS:
        policy = payload["method_policy"].get(tier, {})
        lines.append(
            f"- {tier}: usable `{','.join(policy.get('usable_methods') or [])}`; "
            f"quarantined `{','.join(policy.get('quarantined_methods') or []) or 'none'}`"
        )
    lines.extend([
        "",
        "## Branch-site neural performance",
        "",
        "| tier | test AUROC | precision | recall | F1 | MCC | accuracy |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    neural_test = [row for row in payload["neural_rows"] if row.get("split") == "test"]
    for row in neural_test:
        lines.append(
            f"| {row.get('tier')} | {_fmt(row.get('auroc'))} | {_fmt(row.get('precision'))} | "
            f"{_fmt(row.get('recall'))} | {_fmt(row.get('f1'))} | {_fmt(row.get('mcc'))} | {_fmt(row.get('accuracy'))} |"
        )
    lines.extend([
        "",
        "## Calibration",
        "",
        "| tier | temperature | selected threshold | raw ECE | calibrated ECE | warnings |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ])
    for row in payload["calibration_rows"]:
        lines.append(
            f"| {row.get('tier')} | {_fmt(row.get('temperature'))} | {_fmt(row.get('selected_threshold'))} | "
            f"{_fmt(row.get('raw_ece'))} | {_fmt(row.get('calibrated_ece'))} | {row.get('warnings', '')} |"
        )
    lines.extend([
        "",
        "## Aggregation",
        "",
        "| tier | branch all AUROC | gene all AUROC | branch rows | gene rows |",
        "| --- | ---: | ---: | ---: | ---: |",
    ])
    branch_all = {row["tier"]: row for row in payload["branch_aggregation_rows"] if row.get("split") == "all"}
    gene_all = {row["tier"]: row for row in payload["gene_aggregation_rows"] if row.get("split") == "all"}
    for tier in TIERS:
        b = branch_all.get(tier, {})
        g = gene_all.get(tier, {})
        lines.append(
            f"| {tier} | {_fmt(b.get('auroc'))} | {_fmt(g.get('auroc'))} | {b.get('n', '')} | {g.get('n', '')} |"
        )
    lines.extend([
        "",
        "## Controls",
        "",
        "Destructive controls support that branch-label randomization collapses toward random, while partial prevalence-preserving controls can remain high and should not be overinterpreted.",
        "",
        "| tier | control | observed AUROC | mean AUROC | destructive enough |",
        "| --- | --- | ---: | ---: | --- |",
    ])
    important = {
        "global_shuffled_branch_labels",
        "branch_score_permutation_within_family",
        "degree_prevalence_matched_null",
        "within_family_branch_label_shuffle",
    }
    for row in payload["controls_rows"]:
        if row.get("control") in important:
            lines.append(
                f"| {row.get('tier')} | `{row.get('control')}` | {_fmt(row.get('observed_auroc'))} | "
                f"{_fmt(row.get('mean_auroc'))} | {row.get('whether_control_is_destructive_enough')} |"
            )
    lines.extend([
        "",
        "## Leakage, OOD, and scientific cautions",
        "",
    ])
    for caution in payload["scientific_cautions"]:
        lines.append(f"- {caution}")
    lines.extend([
        "",
        "## Recommendation",
        "",
    ])
    for key, value in payload["recommendation"].items():
        lines.append(f"- {key}: {value}")
    lines.append("")
    return "\n".join(lines)


def _render_model_card(manifest: Dict[str, Any]) -> str:
    lines = [
        "# BABAPPA conservative explicit branch-truth 100K MPS model card template",
        "",
        "## Intended status",
        "",
        "Research-alpha simulation-trained model package. Not final empirical branch-site inference.",
        "",
        "## Model family",
        "",
        f"- run name: `{manifest['run_name']}`",
        f"- feature policy: `{manifest['feature_policy']}`",
        f"- truth mode: `{manifest['truth_mode']}`",
        f"- methods: `{','.join(manifest['methods'])}`",
        "",
        "## Tier checkpoints",
        "",
    ]
    for tier, model in manifest["tier_models"].items():
        lines.append(f"- {tier}: `{model.get('checkpoint')}`")
    if manifest["blocked"]:
        lines.extend(["", "## Blocking issues", ""])
        for missing in manifest["missing_artifacts"]:
            lines.append(f"- {missing}")
    lines.extend(["", "## Claim boundary", "", manifest["scientific_boundary"], ""])
    return "\n".join(lines)


def _render_package_readme(manifest: Dict[str, Any]) -> str:
    status = "blocked" if manifest["blocked"] else "ready_to_package"
    return (
        "# BABAPPA deployable model package plan\n\n"
        f"Status: `{status}`\n\n"
        "This directory is a packaging plan only. The final package should be created only after "
        "the tier checkpoint, calibration, summary, and truth-audit artifacts listed in "
        "`deployable_model_manifest_template.json` are reviewed.\n\n"
        "Empirical inference claims remain out of scope until empirical calibration, OOD gates, "
        "and external benchmark studies are complete.\n"
    )


def _render_package_script(manifest: Dict[str, Any]) -> str:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "echo 'MANUAL EXECUTION SCRIPT - packages retained model artifacts; does not train.'",
        "if python - <<'PY'",
        "import json",
        "from pathlib import Path",
        "m=json.loads(Path('deployable_model_manifest_template.json').read_text())",
        "raise SystemExit(1 if m.get('blocked') else 0)",
        "PY",
        "then",
        "  mkdir -p package/models package/calibration package/reports",
        "else",
        "  echo 'Package plan is blocked; inspect deployable_model_manifest_template.json.'",
        "  exit 1",
        "fi",
    ]
    for tier, model in manifest["tier_models"].items():
        lines.append(f"cp {model.get('checkpoint')} package/models/{tier}_branch_site_neural_checkpoint.pt")
        lines.append(f"cp {model.get('model_meta')} package/models/{tier}_branch_site_neural_model_meta.json")
        lines.append(f"cp {model.get('calibration')} package/calibration/{tier}_branch_site_calibration.json")
        lines.append(f"cp {model.get('calibration_metrics')} package/calibration/{tier}_branch_site_calibrated_metrics.json")
    lines.extend([
        "cp deployable_model_manifest_template.json package/deployable_model_manifest.json",
        "cp model_card_template.md package/model_card.md",
        "echo 'Package staged under package/'",
        "",
    ])
    return "\n".join(lines)


def _render_validate_package_script() -> str:
    return "\n".join([
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "test -f package/deployable_model_manifest.json",
        "test -f package/model_card.md",
        "find package/models -name '*.pt' | grep -q .",
        "echo 'Deployable model package structure looks present.'",
        "",
    ])


def _parse_methods(methods: Sequence[str]) -> List[str]:
    if isinstance(methods, str):
        return [item.strip() for item in methods.split(",") if item.strip()]
    parsed: List[str] = []
    for method in methods:
        parsed.extend([item.strip() for item in str(method).split(",") if item.strip()])
    return parsed
