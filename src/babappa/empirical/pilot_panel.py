"""Curated empirical pilot-panel framework for BABAPPA."""

from __future__ import annotations

import json
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from babappa import __version__
from babappa.datasets.index import read_tsv, write_tsv
from babappa.empirical.bridge import (
    EmpiricalAlignmentEnsembleConfig,
    EmpiricalApplicabilityConfig,
    EmpiricalBranchSiteReportConfig,
    EmpiricalBranchSiteScoringConfig,
    EmpiricalFeatureAuditConfig,
    EmpiricalFeatureExtractionConfig,
    EmpiricalInputValidationConfig,
    audit_empirical_features,
    extract_empirical_branch_site_features,
    make_empirical_branch_site_report,
    run_empirical_alignment_ensemble,
    run_empirical_applicability,
    score_empirical_branch_sites,
    validate_empirical_input,
)
from babappa.empirical.calibration import (
    SimulationMatchedCalibrationPlanConfig,
    plan_simulation_matched_calibration,
)

PANEL_REQUIRED_COLUMNS = [
    "panel_id",
    "gene_family",
    "species_group",
    "cds_fasta",
    "tree_file",
    "foreground",
    "expected_category",
    "reference_status",
    "notes",
]
VALID_EXPECTED_CATEGORIES = {
    "known_positive",
    "likely_positive",
    "likely_negative",
    "alignment_sensitive",
    "saturated",
    "short_low_information",
    "paralogy_risk",
    "unknown",
}
VALID_REFERENCE_STATUS = {
    "codeml_available",
    "hyphy_available",
    "both_available",
    "unavailable",
    "planned",
}
REFERENCE_RESULT_FIELDS = [
    "panel_id",
    "tool",
    "test_name",
    "p_value",
    "q_value",
    "selected_branch",
    "selected_sites",
    "result_class",
    "notes",
]
VALID_REFERENCE_RESULT_CLASSES = {"positive", "negative", "inconclusive", "failed", "pending", "pending_tool_missing", "pending_not_run"}
CLAIM_BOUNDARY_TEXT = (
    "BABAPPA model is simulation-trained. No simulator truth was used for empirical inference. "
    "Scores are diagnostic until simulation-matched calibration and external benchmark interpretation are complete. "
    "Out-of-domain cases are not positive-selection calls. Reference-tool disagreement must be interpreted biologically, "
    "not automatically treated as BABAPPA failure."
)
FORBIDDEN_DISCOVERY_LANGUAGE = [
    "positive selection discovered",
    "empirical branch-site inference proven",
    "babappa confirms adaptation",
]
REAL_PILOT_DIRS = [
    "input",
    "manifest",
    "babappa_run",
    "reference_plan",
    "reference_results",
    "comparison",
    "summary",
    "logs",
]


@dataclass(frozen=True)
class EmpiricalPilotPanelValidationConfig:
    """Configuration for empirical pilot-panel manifest validation."""

    panel_manifest: str
    outdir: str


@dataclass(frozen=True)
class EmpiricalPilotPanelRunConfig:
    """Configuration for running a small empirical pilot panel."""

    panel_manifest: str
    deployable_model_package: str
    outdir: str
    methods: Sequence[str] | str = "identity,mafft,babappalign,muscle"
    device: str = "auto"
    max_families: int = 5
    fail_fast: bool = False


@dataclass(frozen=True)
class ClassicalReferenceWorkflowPlanConfig:
    """Configuration for classical codeml/HyPhy reference workflow planning."""

    panel_manifest: str
    outdir: str
    tools: Sequence[str] | str = "codeml,hyphy"


@dataclass(frozen=True)
class EmpiricalReferenceComparisonConfig:
    """Configuration for comparing BABAPPA panel outputs to reference results."""

    babappa_panel_run: str
    reference_results: str
    outdir: str


@dataclass(frozen=True)
class EmpiricalPilotPanelSummaryConfig:
    """Configuration for empirical pilot panel summary."""

    panel_run: str
    outdir: str
    reference_comparison: Optional[str] = None


@dataclass(frozen=True)
class EmpiricalPilotSummaryValidationConfig:
    """Configuration for validating pilot summary claim boundaries."""

    summary_dir: str


@dataclass(frozen=True)
class RealEmpiricalPilotWorkspaceConfig:
    """Configuration for a guarded real empirical pilot workspace."""

    workspace: str
    max_families: int = 12


@dataclass(frozen=True)
class RealEmpiricalPilotDecisionReportConfig:
    """Configuration for a real empirical pilot decision report."""

    workspace: str
    outdir: Optional[str] = None


def prepare_real_empirical_pilot_workspace(config: RealEmpiricalPilotWorkspaceConfig) -> Dict[str, Any]:
    """Create a real empirical pilot workspace and manifest template without fabricating data."""

    workspace = Path(config.workspace)
    for name in REAL_PILOT_DIRS:
        (workspace / name).mkdir(parents=True, exist_ok=True)
    manifest_path = workspace / "manifest" / "real_empirical_pilot_panel.tsv"
    rows = _real_pilot_template_rows(max(1, min(int(config.max_families), 12)))
    manifest_created = False
    if not manifest_path.exists():
        write_tsv(manifest_path, rows, PANEL_REQUIRED_COLUMNS)
        manifest_created = True
    validation = validate_empirical_pilot_panel(
        EmpiricalPilotPanelValidationConfig(
            panel_manifest=str(manifest_path),
            outdir=str(workspace / "panel_validation"),
        )
    )
    readiness = _real_pilot_readiness_payload(workspace, manifest_path, validation, manifest_created)
    _write_json(workspace / "summary" / "real_empirical_pilot_readiness_report.json", readiness)
    (workspace / "summary" / "real_empirical_pilot_readiness_report.md").write_text(
        _render_real_pilot_readiness_md(readiness),
        encoding="utf-8",
    )
    return {
        "status": readiness["status"],
        "workspace": str(workspace),
        "manifest": str(manifest_path),
        "manifest_created": manifest_created,
        "families": readiness["manifest_rows"],
        "validation_status": validation["status"],
        "missing_inputs": readiness["missing_inputs"],
        "readiness_report": str(workspace / "summary" / "real_empirical_pilot_readiness_report.md"),
    }


def make_real_empirical_pilot_decision_report(config: RealEmpiricalPilotDecisionReportConfig) -> Dict[str, Any]:
    """Write a guarded decision report for a real empirical pilot workspace."""

    workspace = Path(config.workspace)
    outdir = Path(config.outdir) if config.outdir else workspace / "summary"
    outdir.mkdir(parents=True, exist_ok=True)
    panel_validation = _read_optional_json_local(workspace / "panel_validation" / "empirical_pilot_panel_validation.json")
    panel_run = _read_optional_json_local(workspace / "babappa_run" / "panel_run_manifest.json")
    run_rows = _safe_read_tsv(workspace / "babappa_run" / "panel_run_summary.tsv")
    comparison = _read_optional_json_local(workspace / "comparison" / "empirical_reference_comparison.json")
    reference_results_path = workspace / "reference_results" / "reference_results.tsv"
    decision, reasons = _real_pilot_decision(panel_validation, panel_run, run_rows, comparison, reference_results_path)
    payload = {
        "real_empirical_pilot_decision_report_version": __version__,
        "status": "ok",
        "workspace": str(workspace),
        "decision": decision,
        "not_ready_for_claims": True,
        "reasons": reasons,
        "panel_validation_status": panel_validation.get("status", "missing"),
        "families_in_manifest": panel_validation.get("n_rows", 0),
        "babappa_run_status": panel_run.get("status", "not_run"),
        "families_processed": panel_run.get("n_families_processed", 0),
        "applicability_counts": dict(Counter(row.get("applicability_status", "") for row in run_rows)) if run_rows else {},
        "family_status_counts": dict(Counter(row.get("family_status", "") for row in run_rows)) if run_rows else {},
        "reference_comparison_status": comparison.get("status", "pending"),
        "reference_concordance_counts": comparison.get("concordance_counts", {}),
        "reference_results_present": reference_results_path.exists(),
        "claim_boundary": CLAIM_BOUNDARY_TEXT,
        "forbidden_discovery_language_absent": True,
        "recommended_next_action": _real_pilot_next_action(decision, reference_results_path.exists()),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(outdir / "real_empirical_pilot_decision_report.json", payload)
    (outdir / "real_empirical_pilot_decision_report.md").write_text(
        _render_real_pilot_decision_md(payload),
        encoding="utf-8",
    )
    return {
        "status": "ok",
        "outdir": str(outdir),
        "decision": decision,
        "not_ready_for_claims": True,
        "json": str(outdir / "real_empirical_pilot_decision_report.json"),
        "markdown": str(outdir / "real_empirical_pilot_decision_report.md"),
        "reference_comparison_status": payload["reference_comparison_status"],
    }


def validate_empirical_pilot_panel(config: EmpiricalPilotPanelValidationConfig) -> Dict[str, Any]:
    """Validate a curated empirical pilot-panel manifest."""

    manifest_path = Path(config.panel_manifest)
    outdir = Path(config.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    failures: List[str] = []
    warnings: List[str] = []
    rows = _read_panel_rows(manifest_path, failures)
    row_summaries: List[Dict[str, Any]] = []
    seen: set[str] = set()
    duplicates: List[str] = []
    if rows:
        missing_columns = sorted(set(PANEL_REQUIRED_COLUMNS) - set(rows[0]))
        if missing_columns:
            failures.append("missing_required_columns:" + ",".join(missing_columns))
    for index, row in enumerate(rows, start=1):
        panel_id = str(row.get("panel_id", "")).strip()
        row_failures: List[str] = []
        row_warnings: List[str] = []
        if not panel_id:
            row_failures.append(f"row_{index}:missing_panel_id")
        elif panel_id in seen:
            row_failures.append(f"duplicate_panel_id:{panel_id}")
            duplicates.append(panel_id)
        seen.add(panel_id)
        category = str(row.get("expected_category", "")).strip()
        if category not in VALID_EXPECTED_CATEGORIES:
            row_failures.append(f"invalid_expected_category:{panel_id}:{category}")
        reference_status = str(row.get("reference_status", "")).strip()
        if reference_status not in VALID_REFERENCE_STATUS:
            row_failures.append(f"invalid_reference_status:{panel_id}:{reference_status}")
        if not str(row.get("foreground", "")).strip():
            row_failures.append(f"missing_foreground:{panel_id}")
        cds_path = _resolve_panel_path(manifest_path, row.get("cds_fasta", ""))
        tree_path = _resolve_panel_path(manifest_path, row.get("tree_file", ""))
        if not cds_path.exists():
            row_failures.append(f"missing_cds_fasta:{panel_id}:{cds_path}")
        if not tree_path.exists():
            row_failures.append(f"missing_tree_file:{panel_id}:{tree_path}")
        input_status = "not_run"
        input_failures: List[str] = []
        input_warnings: List[str] = []
        if cds_path.exists() and tree_path.exists() and row.get("foreground"):
            try:
                input_result = validate_empirical_input(
                    EmpiricalInputValidationConfig(
                        cds_fasta=str(cds_path),
                        tree=str(tree_path),
                        foreground=str(row.get("foreground")),
                        outdir=str(outdir / "per_family_input_validation" / panel_id),
                    )
                )
                input_status = str(input_result["status"])
                input_failures = list(input_result.get("failures") or [])
                input_warnings = list(input_result.get("warnings") or [])
                if input_status == "fail":
                    row_failures.append(f"empirical_input_failed:{panel_id}")
                elif input_status == "warning":
                    row_warnings.append(f"empirical_input_warning:{panel_id}")
            except Exception as exc:
                input_status = "fail"
                input_failures = [str(exc)]
                row_failures.append(f"empirical_input_exception:{panel_id}:{exc}")
        failures.extend(row_failures)
        warnings.extend(row_warnings)
        row_summaries.append({
            "panel_id": panel_id,
            "gene_family": row.get("gene_family", ""),
            "expected_category": category,
            "reference_status": reference_status,
            "cds_fasta": str(cds_path),
            "tree_file": str(tree_path),
            "foreground": row.get("foreground", ""),
            "row_status": "fail" if row_failures else ("warning" if row_warnings or input_warnings else "ok"),
            "input_status": input_status,
            "failures": ";".join(row_failures + input_failures),
            "warnings": ";".join(row_warnings + input_warnings),
        })
    if duplicates:
        failures.append("duplicate_panel_id_values:" + ",".join(sorted(set(duplicates))))
    category_counts = Counter(row.get("expected_category", "") for row in rows)
    payload = {
        "empirical_pilot_panel_validation_version": __version__,
        "status": "fail" if failures else ("warning" if warnings else "ok"),
        "panel_manifest": str(manifest_path),
        "n_rows": len(rows),
        "required_columns": PANEL_REQUIRED_COLUMNS,
        "category_counts": dict(sorted(category_counts.items())),
        "row_summaries": row_summaries,
        "failures": failures,
        "warnings": warnings,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(outdir / "empirical_pilot_panel_validation.json", payload)
    write_tsv(outdir / "empirical_pilot_panel_validation.tsv", row_summaries, _validation_tsv_fields())
    (outdir / "empirical_pilot_panel_validation.md").write_text(_render_panel_validation_md(payload), encoding="utf-8")
    return {
        "status": payload["status"],
        "outdir": str(outdir),
        "n_rows": len(rows),
        "n_fail": len(failures),
        "n_warning": len(warnings),
        "json": str(outdir / "empirical_pilot_panel_validation.json"),
        "tsv": str(outdir / "empirical_pilot_panel_validation.tsv"),
        "markdown": str(outdir / "empirical_pilot_panel_validation.md"),
        "failures": failures,
        "warnings": warnings,
    }


def run_empirical_pilot_panel(config: EmpiricalPilotPanelRunConfig) -> Dict[str, Any]:
    """Run a small diagnostic empirical pilot panel, continuing across family failures."""

    panel_path = Path(config.panel_manifest)
    outdir = Path(config.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rows = _read_panel_rows(panel_path, [])
    max_families = max(1, int(config.max_families))
    methods = _parse_csv(config.methods)
    selected = rows[:max_families]
    summaries: List[Dict[str, Any]] = []
    per_family: Dict[str, Any] = {}
    for row in selected:
        panel_id = str(row.get("panel_id", "")).strip()
        family_dir = outdir / "per_family" / panel_id
        family_dir.mkdir(parents=True, exist_ok=True)
        summary = _run_one_panel_family(row, panel_path, family_dir, config, methods)
        summaries.append(summary)
        per_family[panel_id] = summary
        if config.fail_fast and summary["family_status"] == "fail":
            break
    payload = {
        "empirical_pilot_panel_run_version": __version__,
        "status": "fail" if any(row["family_status"] == "fail" for row in summaries) else "ok",
        "panel_manifest": str(panel_path),
        "deployable_model_package": config.deployable_model_package,
        "methods": methods,
        "device": config.device,
        "max_families": max_families,
        "n_families_requested": len(selected),
        "n_families_processed": len(summaries),
        "per_family": per_family,
        "claim_boundary": CLAIM_BOUNDARY_TEXT,
        "heavy_jobs_executed": False,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(outdir / "panel_run_manifest.json", payload)
    write_tsv(outdir / "panel_run_summary.tsv", summaries, _panel_run_fields())
    (outdir / "panel_run_report.md").write_text(_render_panel_run_md(payload, summaries), encoding="utf-8")
    return {
        "status": payload["status"],
        "outdir": str(outdir),
        "families_processed": len(summaries),
        "qc_pass": sum(1 for row in summaries if row.get("input_status") in {"pass", "warning"}),
        "qc_fail": sum(1 for row in summaries if row.get("input_status") == "fail"),
        "scoring_ok": sum(1 for row in summaries if row.get("scoring_status") == "ok"),
        "manifest": str(outdir / "panel_run_manifest.json"),
        "summary": str(outdir / "panel_run_summary.tsv"),
        "report": str(outdir / "panel_run_report.md"),
    }


def plan_classical_reference_workflows(config: ClassicalReferenceWorkflowPlanConfig) -> Dict[str, Any]:
    """Generate codeml/HyPhy reference workflow templates without running them."""

    panel_path = Path(config.panel_manifest)
    outdir = Path(config.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rows = _read_panel_rows(panel_path, [])
    tools = _parse_csv(config.tools)
    codeml_dir = outdir / "codeml"
    hyphy_dir = outdir / "hyphy"
    codeml_dir.mkdir(exist_ok=True)
    hyphy_dir.mkdir(exist_ok=True)
    for row in rows:
        panel_id = row.get("panel_id", "")
        family_codeml = codeml_dir / panel_id
        family_codeml.mkdir(parents=True, exist_ok=True)
        (family_codeml / "branch_site_model_A.ctl").write_text(_codeml_ctl(row, null=False), encoding="utf-8")
        (family_codeml / "branch_site_null.ctl").write_text(_codeml_ctl(row, null=True), encoding="utf-8")
        (family_codeml / "README_foreground_branch_marking.md").write_text(_foreground_marking_readme(row), encoding="utf-8")
    (outdir / "codeml_commands.sh").write_text(_render_codeml_commands(rows), encoding="utf-8")
    (outdir / "hyphy_commands.sh").write_text(_render_hyphy_commands(rows), encoding="utf-8")
    for script in ["codeml_commands.sh", "hyphy_commands.sh"]:
        (outdir / script).chmod(0o755)
    expected = {
        "status": "planned",
        "tools": tools,
        "expected_outputs": {
            "codeml": ["model_A.out", "null.out", "lrt_summary.tsv", "bh_corrected_results.tsv"],
            "hyphy": ["meme.json", "absrel.json", "hyphy_summary.tsv"],
        },
        "executed": False,
    }
    _write_json(outdir / "expected_reference_outputs.json", expected)
    write_tsv(outdir / "reference_result_schema.tsv", [_empty_reference_schema_row()], REFERENCE_RESULT_FIELDS)
    (outdir / "classical_reference_plan.md").write_text(_render_classical_plan_md(rows, tools), encoding="utf-8")
    return {
        "status": "planned",
        "outdir": str(outdir),
        "tools": tools,
        "codeml_script": str(outdir / "codeml_commands.sh"),
        "hyphy_script": str(outdir / "hyphy_commands.sh"),
        "executed": False,
    }


def compare_empirical_reference_results(config: EmpiricalReferenceComparisonConfig) -> Dict[str, Any]:
    """Compare BABAPPA panel-run diagnostics to codeml/HyPhy-style reference results."""

    outdir = Path(config.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    panel_run = _read_json(Path(config.babappa_panel_run) / "panel_run_manifest.json")
    babappa_rows = {row["panel_id"]: row for row in read_tsv(Path(config.babappa_panel_run) / "panel_run_summary.tsv")}
    reference_rows = read_tsv(Path(config.reference_results))
    failures: List[str] = []
    comparison_rows: List[Dict[str, Any]] = []
    for row in reference_rows:
        result_class = row.get("result_class", "")
        if result_class not in VALID_REFERENCE_RESULT_CLASSES:
            failures.append(f"invalid_reference_result_class:{row.get('panel_id')}:{result_class}")
        panel_id = row.get("panel_id", "")
        babappa = babappa_rows.get(panel_id, {})
        comparison_rows.append(_comparison_row(panel_id, babappa, row))
    pending_classes = {"pending", "pending_tool_missing", "pending_not_run"}
    if failures:
        status = "fail"
    elif reference_rows and all(row.get("result_class") in pending_classes for row in reference_rows):
        status = "pending_tool_missing" if any(row.get("result_class") == "pending_tool_missing" for row in reference_rows) else "pending_reference_results"
    else:
        status = "ok"
    payload = {
        "empirical_reference_comparison_version": __version__,
        "status": status,
        "babappa_panel_run": config.babappa_panel_run,
        "reference_results": config.reference_results,
        "rows": comparison_rows,
        "concordance_counts": dict(Counter(row["concordance_class"] for row in comparison_rows)),
        "failures": failures,
        "claim_boundary": (
            "Reference tools are external comparison and failure-mode probes. BABAPPA is not tuned to mimic codeml/HyPhy."
        ),
        "panel_claim_boundary": panel_run.get("claim_boundary", CLAIM_BOUNDARY_TEXT),
    }
    _write_json(outdir / "empirical_reference_comparison.json", payload)
    write_tsv(outdir / "empirical_reference_comparison.tsv", comparison_rows, _comparison_fields())
    (outdir / "empirical_reference_comparison.md").write_text(_render_reference_comparison_md(payload), encoding="utf-8")
    return {
        "status": payload["status"],
        "outdir": str(outdir),
        "n_rows": len(comparison_rows),
        "concordance_classes": sorted(payload["concordance_counts"]),
        "json": str(outdir / "empirical_reference_comparison.json"),
    }


def summarize_empirical_pilot_panel(config: EmpiricalPilotPanelSummaryConfig) -> Dict[str, Any]:
    """Summarize an empirical pilot panel with optional reference comparison."""

    panel_dir = Path(config.panel_run)
    outdir = Path(config.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    run_manifest = _read_json(panel_dir / "panel_run_manifest.json")
    rows = read_tsv(panel_dir / "panel_run_summary.tsv")
    comparison_payload = None
    comparison_rows: List[Dict[str, str]] = []
    if config.reference_comparison:
        comparison_dir = Path(config.reference_comparison)
        comparison_payload = _read_json(comparison_dir / "empirical_reference_comparison.json")
        comparison_rows = read_tsv(comparison_dir / "empirical_reference_comparison.tsv")
    qc_counts = Counter(row.get("input_status", "") for row in rows)
    applicability_counts = Counter(row.get("applicability_status", "") for row in rows)
    scoring_counts = Counter(row.get("scoring_status", "") for row in rows)
    diagnostic_count = sum(1 for row in rows if str(row.get("diagnostic_only", "")).lower() == "true")
    payload = {
        "empirical_pilot_panel_summary_version": __version__,
        "status": "ok",
        "panel_run": config.panel_run,
        "n_families": len(rows),
        "input_qc_counts": dict(qc_counts),
        "applicability_counts": dict(applicability_counts),
        "scoring_counts": dict(scoring_counts),
        "diagnostic_only_cases": diagnostic_count,
        "simulation_matched_calibration": "planned_not_run",
        "reference_comparison": comparison_payload,
        "reference_concordance_counts": (
            dict(Counter(row.get("concordance_class", "") for row in comparison_rows))
            if comparison_rows
            else {}
        ),
        "recommended_next_empirical_panel_size": _recommended_next_panel_size(rows),
        "claim_boundary": CLAIM_BOUNDARY_TEXT,
        "no_empirical_discovery_claim": True,
        "run_manifest_status": run_manifest.get("status"),
    }
    _write_json(outdir / "empirical_pilot_panel_summary.json", payload)
    write_tsv(outdir / "empirical_pilot_panel_summary.tsv", [_summary_tsv_row(payload)], _summary_fields())
    (outdir / "empirical_pilot_panel_summary.md").write_text(_render_pilot_summary_md(payload), encoding="utf-8")
    return {
        "status": "ok",
        "outdir": str(outdir),
        "n_families": len(rows),
        "claim_boundary_present": True,
        "json": str(outdir / "empirical_pilot_panel_summary.json"),
        "markdown": str(outdir / "empirical_pilot_panel_summary.md"),
    }


def validate_empirical_pilot_summary(config: EmpiricalPilotSummaryValidationConfig) -> Dict[str, Any]:
    """Validate that pilot summary artifacts preserve empirical claim boundaries."""

    summary_dir = Path(config.summary_dir)
    failures: List[str] = []
    json_path = summary_dir / "empirical_pilot_panel_summary.json"
    md_path = summary_dir / "empirical_pilot_panel_summary.md"
    payload = _read_json(json_path) if json_path.exists() else {}
    if not json_path.exists():
        failures.append(f"missing_file:{json_path}")
    if not md_path.exists():
        failures.append(f"missing_file:{md_path}")
        markdown = ""
    else:
        markdown = md_path.read_text(encoding="utf-8")
    required_phrases = [
        "simulation-trained",
        "No simulator truth was used for empirical inference",
        "Scores are diagnostic",
        "Out-of-domain cases are not positive-selection calls",
        "Reference-tool disagreement",
    ]
    haystack = json.dumps(payload) + "\n" + markdown
    for phrase in required_phrases:
        if phrase not in haystack:
            failures.append(f"missing_claim_boundary_phrase:{phrase}")
    lower_haystack = haystack.lower()
    for phrase in FORBIDDEN_DISCOVERY_LANGUAGE:
        if phrase in lower_haystack:
            failures.append(f"forbidden_discovery_language:{phrase}")
    result = {
        "status": "fail" if failures else "ok",
        "n_fail": len(failures),
        "failures": failures,
        "summary_dir": str(summary_dir),
    }
    _write_json(summary_dir / "empirical_pilot_summary_validation.json", result)
    return result


def _run_one_panel_family(
    row: Dict[str, str],
    panel_path: Path,
    family_dir: Path,
    config: EmpiricalPilotPanelRunConfig,
    methods: List[str],
) -> Dict[str, Any]:
    panel_id = str(row.get("panel_id", "")).strip()
    cds_path = _resolve_panel_path(panel_path, row.get("cds_fasta", ""))
    tree_path = _resolve_panel_path(panel_path, row.get("tree_file", ""))
    foreground = str(row.get("foreground", "")).strip()
    summary: Dict[str, Any] = {
        "panel_id": panel_id,
        "gene_family": row.get("gene_family", ""),
        "expected_category": row.get("expected_category", ""),
        "reference_status": row.get("reference_status", ""),
        "family_status": "ok",
        "input_status": "not_run",
        "alignment_status": "not_run",
        "feature_status": "not_run",
        "feature_audit_status": "not_run",
        "applicability_status": "not_run",
        "scoring_status": "not_run",
        "simulation_matched_calibration_status": "not_run",
        "report_status": "not_run",
        "diagnostic_only": "",
        "babappa_result_class": "inconclusive",
        "max_gene_support": "",
        "n_called_positive": "",
        "warnings": "",
        "failures": "",
        "family_dir": str(family_dir),
    }
    failures: List[str] = []
    warnings: List[str] = []
    try:
        input_result = validate_empirical_input(
            EmpiricalInputValidationConfig(str(cds_path), str(tree_path), foreground, str(family_dir / "empirical_input"))
        )
        summary["input_status"] = input_result["status"]
        if input_result["status"] == "fail":
            failures.extend(input_result.get("failures") or [])
            summary["family_status"] = "fail"
            summary["failures"] = ";".join(failures)
            return summary
        if input_result.get("warnings"):
            warnings.extend(input_result["warnings"])
        alignment_result = run_empirical_alignment_ensemble(
            EmpiricalAlignmentEnsembleConfig(
                cds_fasta=str(cds_path),
                tree=str(tree_path),
                foreground=foreground,
                outdir=str(family_dir / "empirical_alignment"),
                methods=methods,
                require_babappalign=False,
            )
        )
        summary["alignment_status"] = alignment_result["status"]
        if alignment_result["status"] == "fail":
            failures.extend(alignment_result.get("failures") or [])
            summary["family_status"] = "fail"
            summary["failures"] = ";".join(failures)
            return summary
        if alignment_result.get("warnings"):
            warnings.extend(alignment_result["warnings"])
        feature_result = extract_empirical_branch_site_features(
            EmpiricalFeatureExtractionConfig(
                empirical_validation_dir=str(family_dir / "empirical_input"),
                alignment_dir=str(family_dir / "empirical_alignment"),
                deployable_model_package=config.deployable_model_package,
                outdir=str(family_dir / "empirical_features"),
                foreground=foreground,
            )
        )
        summary["feature_status"] = feature_result["status"]
        audit_result = audit_empirical_features(
            EmpiricalFeatureAuditConfig(
                features=str(family_dir / "empirical_features" / "empirical_branch_site_features.tsv"),
                deployable_model_package=config.deployable_model_package,
                outdir=str(family_dir / "empirical_feature_audit"),
            )
        )
        summary["feature_audit_status"] = audit_result["status"]
        if audit_result["status"] == "fail":
            failures.extend(audit_result["forbidden_columns"])
            summary["family_status"] = "fail"
            summary["failures"] = ";".join(failures)
            return summary
        applicability_result = run_empirical_applicability(
            EmpiricalApplicabilityConfig(
                empirical_validation_dir=str(family_dir / "empirical_input"),
                empirical_feature_dir=str(family_dir / "empirical_features"),
                deployable_model_package=config.deployable_model_package,
                outdir=str(family_dir / "empirical_applicability"),
            )
        )
        summary["applicability_status"] = applicability_result["status"]
        try:
            scoring_result = score_empirical_branch_sites(
                EmpiricalBranchSiteScoringConfig(
                    features=str(family_dir / "empirical_features" / "empirical_branch_site_features.tsv"),
                    deployable_model_package=config.deployable_model_package,
                    applicability_dir=str(family_dir / "empirical_applicability"),
                    outdir=str(family_dir / "empirical_scores"),
                    device=config.device,
                )
            )
            summary["scoring_status"] = scoring_result["status"]
            summary["diagnostic_only"] = str(scoring_result.get("diagnostic_only"))
            _fill_score_summary(summary, family_dir / "empirical_scores")
        except Exception as exc:
            summary["scoring_status"] = "fail"
            warnings.append(f"scoring_failed:{exc}")
        calibration_result = plan_simulation_matched_calibration(
            SimulationMatchedCalibrationPlanConfig(
                empirical_validation_dir=str(family_dir / "empirical_input"),
                deployable_model_package=config.deployable_model_package,
                outdir=str(family_dir / "simulation_matched_calibration_plan"),
            )
        )
        summary["simulation_matched_calibration_status"] = calibration_result["status"]
        report_result = make_empirical_branch_site_report(
            EmpiricalBranchSiteReportConfig(
                outdir=str(family_dir / "empirical_report"),
                empirical_validation_dir=str(family_dir / "empirical_input"),
                alignment_dir=str(family_dir / "empirical_alignment"),
                feature_dir=str(family_dir / "empirical_features"),
                feature_audit_dir=str(family_dir / "empirical_feature_audit"),
                applicability_dir=str(family_dir / "empirical_applicability"),
                scoring_dir=str(family_dir / "empirical_scores"),
                simulation_matched_calibration_plan=str(family_dir / "simulation_matched_calibration_plan"),
                deployable_model_package=config.deployable_model_package,
            )
        )
        summary["report_status"] = report_result["status"]
    except Exception as exc:
        failures.append(str(exc))
        summary["family_status"] = "fail"
    summary["warnings"] = ";".join(warnings)
    summary["failures"] = ";".join(failures)
    if failures:
        summary["family_status"] = "fail"
    elif warnings:
        summary["family_status"] = "warning"
    return summary


def _fill_score_summary(summary: Dict[str, Any], scoring_dir: Path) -> None:
    gene_path = scoring_dir / "empirical_gene_support.tsv"
    if not gene_path.exists():
        summary["babappa_result_class"] = "inconclusive"
        return
    rows = read_tsv(gene_path)
    if not rows:
        summary["babappa_result_class"] = "inconclusive"
        return
    max_support = max(_float(row.get("max_prob_positive"), 0.0) for row in rows)
    n_called = sum(_int(row.get("n_called_positive"), 0) for row in rows)
    summary["max_gene_support"] = f"{max_support:.6g}"
    summary["n_called_positive"] = str(n_called)
    summary["babappa_result_class"] = "positive" if n_called > 0 else "negative"


def _comparison_row(panel_id: str, babappa: Dict[str, str], reference: Dict[str, str]) -> Dict[str, Any]:
    babappa_class = babappa.get("babappa_result_class") or "inconclusive"
    reference_class = reference.get("result_class") or "inconclusive"
    diagnostic_only = str(babappa.get("diagnostic_only", "")).lower() == "true"
    applicability = babappa.get("applicability_status", "")
    if reference_class in {"pending", "pending_tool_missing", "pending_not_run"}:
        concordance = "pending_tool_missing" if reference_class == "pending_tool_missing" else "pending_reference_results"
    elif reference_class == "failed":
        concordance = "reference_failed"
    elif diagnostic_only or applicability == "out_of_domain":
        concordance = "BABAPPA_abstained"
    elif babappa_class == "positive" and reference_class == "positive":
        concordance = "concordant_positive"
    elif babappa_class == "negative" and reference_class == "negative":
        concordance = "concordant_negative"
    elif babappa_class == "positive" and reference_class in {"negative", "inconclusive"}:
        concordance = "BABAPPA_only"
    elif babappa_class in {"negative", "inconclusive"} and reference_class == "positive":
        concordance = "reference_only"
    else:
        concordance = "both_inconclusive"
    return {
        "panel_id": panel_id,
        "tool": reference.get("tool", ""),
        "test_name": reference.get("test_name", ""),
        "babappa_applicability_status": applicability,
        "babappa_result_class": babappa_class,
        "babappa_max_gene_support": babappa.get("max_gene_support", ""),
        "babappa_diagnostic_only": babappa.get("diagnostic_only", ""),
        "reference_result_class": reference_class,
        "reference_p_value": reference.get("p_value", ""),
        "reference_q_value": reference.get("q_value", ""),
        "concordance_class": concordance,
        "notes": reference.get("notes", ""),
    }


def _resolve_panel_path(panel_manifest: Path, raw: str | None) -> Path:
    path = Path(str(raw or "")).expanduser()
    if path.is_absolute():
        return path
    return (panel_manifest.parent / path).resolve()


def _read_panel_rows(path: Path, failures: List[str]) -> List[Dict[str, str]]:
    if not path.exists():
        failures.append(f"missing_manifest:{path}")
        return []
    try:
        return read_tsv(path)
    except Exception as exc:
        failures.append(f"could_not_read_manifest:{path}:{exc}")
        return []


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"JSON root is not object: {path}")
    return data


def _read_optional_json_local(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return _read_json(path)
    except (OSError, json.JSONDecodeError, ValueError):
        return {}


def _safe_read_tsv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    try:
        return read_tsv(path)
    except (OSError, ValueError):
        return []


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _parse_csv(value: Sequence[str] | str) -> List[str]:
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    parsed: List[str] = []
    for item in value:
        parsed.extend(part.strip() for part in str(item).split(",") if part.strip())
    return parsed


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _validation_tsv_fields() -> List[str]:
    return [
        "panel_id",
        "gene_family",
        "expected_category",
        "reference_status",
        "cds_fasta",
        "tree_file",
        "foreground",
        "row_status",
        "input_status",
        "failures",
        "warnings",
    ]


def _panel_run_fields() -> List[str]:
    return [
        "panel_id",
        "gene_family",
        "expected_category",
        "reference_status",
        "family_status",
        "input_status",
        "alignment_status",
        "feature_status",
        "feature_audit_status",
        "applicability_status",
        "scoring_status",
        "simulation_matched_calibration_status",
        "report_status",
        "diagnostic_only",
        "babappa_result_class",
        "max_gene_support",
        "n_called_positive",
        "warnings",
        "failures",
        "family_dir",
    ]


def _comparison_fields() -> List[str]:
    return [
        "panel_id",
        "tool",
        "test_name",
        "babappa_applicability_status",
        "babappa_result_class",
        "babappa_max_gene_support",
        "babappa_diagnostic_only",
        "reference_result_class",
        "reference_p_value",
        "reference_q_value",
        "concordance_class",
        "notes",
    ]


def _summary_fields() -> List[str]:
    return [
        "status",
        "n_families",
        "input_qc_counts",
        "applicability_counts",
        "scoring_counts",
        "diagnostic_only_cases",
        "simulation_matched_calibration",
        "reference_concordance_counts",
        "recommended_next_empirical_panel_size",
        "no_empirical_discovery_claim",
    ]


def _summary_tsv_row(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "status": payload["status"],
        "n_families": payload["n_families"],
        "input_qc_counts": json.dumps(payload["input_qc_counts"], sort_keys=True),
        "applicability_counts": json.dumps(payload["applicability_counts"], sort_keys=True),
        "scoring_counts": json.dumps(payload["scoring_counts"], sort_keys=True),
        "diagnostic_only_cases": payload["diagnostic_only_cases"],
        "simulation_matched_calibration": payload["simulation_matched_calibration"],
        "reference_concordance_counts": json.dumps(payload["reference_concordance_counts"], sort_keys=True),
        "recommended_next_empirical_panel_size": payload["recommended_next_empirical_panel_size"],
        "no_empirical_discovery_claim": payload["no_empirical_discovery_claim"],
    }


def _empty_reference_schema_row() -> Dict[str, str]:
    return {field: "" for field in REFERENCE_RESULT_FIELDS}


def _real_pilot_template_rows(max_families: int) -> List[Dict[str, str]]:
    rows = [
        ("wrky_candidate_01", "WRKY_transcription_factor_candidate_01", "replace_with_species_group", "../input/wrky_candidate_01.cds.fasta", "../input/wrky_candidate_01.treefile", "replace_with_foreground_taxon", "likely_positive", "planned", "Candidate adaptive/evolutionary-interest family; provide real CDS and tree before running."),
        ("constans_like_candidate_01", "CONSTANS_like_transcription_factor_candidate_01", "replace_with_species_group", "../input/constans_like_candidate_01.cds.fasta", "../input/constans_like_candidate_01.treefile", "replace_with_foreground_taxon", "likely_positive", "planned", "Candidate transcription-factor family; diagnostic only."),
        ("immune_candidate_01", "immune_or_host_interaction_candidate_01", "replace_with_species_group", "../input/immune_candidate_01.cds.fasta", "../input/immune_candidate_01.treefile", "replace_with_foreground_taxon", "likely_positive", "planned", "Candidate immune/host-interaction family; document source and foreground rationale."),
        ("housekeeping_negative_01", "housekeeping_conserved_gene_01", "replace_with_species_group", "../input/housekeeping_negative_01.cds.fasta", "../input/housekeeping_negative_01.treefile", "replace_with_foreground_taxon", "likely_negative", "planned", "Likely negative/conserved control."),
        ("housekeeping_negative_02", "housekeeping_conserved_gene_02", "replace_with_species_group", "../input/housekeeping_negative_02.cds.fasta", "../input/housekeeping_negative_02.treefile", "replace_with_foreground_taxon", "likely_negative", "planned", "Likely negative/conserved control."),
        ("gst_detox_candidate_01", "GST_detoxification_candidate_01", "replace_with_species_group", "../input/gst_detox_candidate_01.cds.fasta", "../input/gst_detox_candidate_01.treefile", "replace_with_foreground_taxon", "likely_positive", "planned", "Candidate detoxification family; diagnostic only."),
        ("alignment_sensitive_01", "alignment_sensitive_family_01", "replace_with_species_group", "../input/alignment_sensitive_01.cds.fasta", "../input/alignment_sensitive_01.treefile", "replace_with_foreground_taxon", "alignment_sensitive", "planned", "Moderate alignment difficulty but not unusable."),
        ("alignment_sensitive_02", "alignment_sensitive_family_02", "replace_with_species_group", "../input/alignment_sensitive_02.cds.fasta", "../input/alignment_sensitive_02.treefile", "replace_with_foreground_taxon", "alignment_sensitive", "planned", "Second alignment sensitivity probe if available."),
        ("saturated_01", "high_divergence_family_01", "replace_with_species_group", "../input/saturated_01.cds.fasta", "../input/saturated_01.treefile", "replace_with_foreground_taxon", "saturated", "planned", "High-divergence/OOD probe."),
        ("saturated_02", "high_divergence_family_02", "replace_with_species_group", "../input/saturated_02.cds.fasta", "../input/saturated_02.treefile", "replace_with_foreground_taxon", "saturated", "planned", "Second high-divergence/OOD probe if available."),
        ("short_low_information_01", "short_low_information_family_01", "replace_with_species_group", "../input/short_low_information_01.cds.fasta", "../input/short_low_information_01.treefile", "replace_with_foreground_taxon", "short_low_information", "planned", "Short/low-information probe."),
        ("paralogy_risk_01", "possible_paralogy_duplication_risk_01", "replace_with_species_group", "../input/paralogy_risk_01.cds.fasta", "../input/paralogy_risk_01.treefile", "replace_with_foreground_taxon", "paralogy_risk", "planned", "Possible paralogy/duplication risk; use only with orthology QC."),
    ]
    return [
        {
            "panel_id": row[0],
            "gene_family": row[1],
            "species_group": row[2],
            "cds_fasta": row[3],
            "tree_file": row[4],
            "foreground": row[5],
            "expected_category": row[6],
            "reference_status": row[7],
            "notes": row[8],
        }
        for row in rows[:max_families]
    ]


def _real_pilot_readiness_payload(
    workspace: Path,
    manifest_path: Path,
    validation: Dict[str, Any],
    manifest_created: bool,
) -> Dict[str, Any]:
    missing_inputs = [item for item in validation.get("failures", []) if item.startswith(("missing_cds_fasta:", "missing_tree_file:"))]
    status = "READY_TO_RUN_BABAPPA_PILOT" if validation.get("status") in {"ok", "warning"} and not missing_inputs else "NEED_INPUT_REPAIR"
    return {
        "real_empirical_pilot_readiness_version": __version__,
        "status": status,
        "workspace": str(workspace),
        "manifest": str(manifest_path),
        "manifest_created": manifest_created,
        "manifest_rows": validation.get("n_rows", 0),
        "panel_validation_status": validation.get("status"),
        "missing_inputs": missing_inputs,
        "n_missing_inputs": len(missing_inputs),
        "run_babappa_pilot": status == "READY_TO_RUN_BABAPPA_PILOT",
        "claim_boundary": CLAIM_BOUNDARY_TEXT,
        "next_action": (
            "Populate real_empirical_pilot/input with real CDS FASTA/tree files and set real foreground taxa."
            if status == "NEED_INPUT_REPAIR"
            else "Run babappa run-empirical-pilot-panel on the validated small panel."
        ),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def _render_real_pilot_readiness_md(payload: Dict[str, Any]) -> str:
    lines = [
        "# Real Empirical Pilot Readiness",
        "",
        f"- status: `{payload['status']}`",
        f"- manifest: `{payload['manifest']}`",
        f"- manifest rows: `{payload['manifest_rows']}`",
        f"- missing inputs: `{payload['n_missing_inputs']}`",
        "",
        "## Claim Boundary",
        "",
        payload["claim_boundary"],
        "",
    ]
    if payload["missing_inputs"]:
        lines.extend(["## Missing Inputs", ""])
        lines.extend(f"- {item}" for item in payload["missing_inputs"][:50])
        if len(payload["missing_inputs"]) > 50:
            lines.append(f"- ... {len(payload['missing_inputs']) - 50} more")
        lines.append("")
    lines.extend(["## Next Action", "", payload["next_action"], ""])
    return "\n".join(lines)


def _real_pilot_decision(
    validation: Dict[str, Any],
    run: Dict[str, Any],
    run_rows: List[Dict[str, str]],
    comparison: Dict[str, Any],
    reference_results_path: Path,
) -> tuple[str, List[str]]:
    reasons: List[str] = []
    if not validation:
        return "NEED_INPUT_REPAIR", ["panel_validation_missing"]
    if validation.get("status") == "fail":
        reasons.extend(validation.get("failures") or ["panel_validation_failed"])
        return "NEED_INPUT_REPAIR", reasons
    if not run:
        return "NEED_INPUT_REPAIR", ["babappa_pilot_not_run_yet"]
    failed = sum(1 for row in run_rows if row.get("family_status") == "fail")
    out_of_domain = sum(1 for row in run_rows if row.get("applicability_status") == "out_of_domain")
    processed = len(run_rows)
    if failed:
        reasons.append(f"family_failures:{failed}")
    if processed and out_of_domain / processed > 0.5:
        reasons.append(f"most_families_out_of_domain:{out_of_domain}/{processed}")
        return "NEED_OOD_REDESIGN", reasons
    if comparison and comparison.get("status") == "ok":
        reference_failed = int(comparison.get("concordance_counts", {}).get("reference_failed", 0))
        if failed == 0 and reference_failed == 0 and processed:
            reasons.append("babappa_pilot_completed_and_reference_comparison_available")
            return "READY_FOR_LARGER_EMPIRICAL_PANEL", reasons
    if run.get("status") in {"ok", "warning"} and processed and failed == 0:
        reasons.append("babappa_pilot_completed_reference_workflows_pending" if not reference_results_path.exists() else "reference_comparison_pending")
        return "READY_FOR_REFERENCE_RUNS", reasons
    reasons.append("pilot_completed_with_failures_or_missing_summary")
    return "NEED_INPUT_REPAIR", reasons


def _real_pilot_next_action(decision: str, reference_results_present: bool) -> str:
    if decision == "READY_FOR_REFERENCE_RUNS":
        return "Run codeml/HyPhy reference workflows for accepted pilot families, then ingest reference_results.tsv."
    if decision == "NEED_INPUT_REPAIR":
        return "Repair missing/invalid CDS FASTA, tree, foreground, codon, or manifest entries before running BABAPPA."
    if decision == "NEED_OOD_REDESIGN":
        return "Redesign the pilot with more in-domain/borderline families before expanding."
    if decision == "READY_FOR_LARGER_EMPIRICAL_PANEL":
        return "Review concordance/failure modes, then consider a slightly larger curated panel."
    return "Keep outputs diagnostic and do not make empirical claims."


def _render_real_pilot_decision_md(payload: Dict[str, Any]) -> str:
    lines = [
        "# Real Empirical Pilot Decision Report",
        "",
        f"- decision: `{payload['decision']}`",
        f"- not ready for claims: `{payload['not_ready_for_claims']}`",
        f"- panel validation: `{payload['panel_validation_status']}`",
        f"- BABAPPA run: `{payload['babappa_run_status']}`",
        f"- families processed: `{payload['families_processed']}`",
        f"- reference comparison: `{payload['reference_comparison_status']}`",
        "",
        "## Claim Boundary",
        "",
        payload["claim_boundary"],
        "",
        "This report uses diagnostic branch-site support and simulation-trained scores only. It requires simulation-matched calibration and reference-workflow comparison before biological interpretation.",
        "",
        "## Reasons",
        "",
    ]
    lines.extend(f"- {item}" for item in payload["reasons"])
    lines.extend([
        "",
        "## Recommended Next Action",
        "",
        payload["recommended_next_action"],
        "",
        "## Forbidden Claim Check",
        "",
        "The report does not claim empirical discovery, proven empirical branch-site inference, or confirmed adaptation.",
        "",
    ])
    return "\n".join(lines)


def _codeml_ctl(row: Dict[str, str], null: bool) -> str:
    omega = "1" if null else "estimated"
    return "\n".join([
        f"* USER-RUN ONLY - DO NOT EXECUTE IN CODEX",
        f"* panel_id: {row.get('panel_id')}",
        f"* foreground branch/taxon: {row.get('foreground')}",
        "seqfile = alignment.phy",
        "treefile = foreground_marked.tree",
        "outfile = null.out" if null else "outfile = model_A.out",
        "runmode = 0",
        "model = 2",
        "NSsites = 2",
        "fix_omega = 1" if null else "fix_omega = 0",
        f"omega = {omega}",
        "",
    ])


def _foreground_marking_readme(row: Dict[str, str]) -> str:
    return "\n".join([
        "# Foreground Branch Marking",
        "",
        "USER-RUN ONLY - DO NOT EXECUTE IN CODEX",
        "",
        f"Panel ID: `{row.get('panel_id')}`",
        f"Foreground: `{row.get('foreground')}`",
        "",
        "Create `foreground_marked.tree` by marking the intended foreground branch according to codeml syntax.",
        "",
    ])


def _render_codeml_commands(rows: List[Dict[str, str]]) -> str:
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", "echo 'USER-RUN ONLY - DO NOT EXECUTE IN CODEX'", ""]
    for row in rows:
        panel_id = row.get("panel_id", "")
        lines.append(f"# (cd codeml/{panel_id} && codeml branch_site_model_A.ctl && codeml branch_site_null.ctl)")
    lines.extend(["# After runs: compute LRT statistics and BH correction into reference_results.tsv.", ""])
    return "\n".join(lines)


def _render_hyphy_commands(rows: List[Dict[str, str]]) -> str:
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", "echo 'USER-RUN ONLY - DO NOT EXECUTE IN CODEX'", ""]
    lines.append("# Install HyPhy separately if `hyphy` is not available on PATH.")
    for row in rows:
        lines.append(f"# hyphy meme --alignment {row.get('cds_fasta')} --tree {row.get('tree_file')}")
        lines.append(f"# hyphy absrel --alignment {row.get('cds_fasta')} --tree {row.get('tree_file')}")
    lines.append("")
    return "\n".join(lines)


def _render_classical_plan_md(rows: List[Dict[str, str]], tools: List[str]) -> str:
    return "\n".join([
        "# Classical Reference Workflow Plan",
        "",
        "USER-RUN ONLY - DO NOT EXECUTE IN CODEX.",
        "",
        f"- tools: `{','.join(tools)}`",
        f"- families: `{len(rows)}`",
        "- codeml templates include branch-site model A and null controls.",
        "- HyPhy templates include MEME/aBSREL-style placeholders.",
        "- Reference results should be written to `reference_results.tsv` using the provided schema.",
        "",
    ])


def _render_panel_validation_md(payload: Dict[str, Any]) -> str:
    lines = ["# Empirical Pilot Panel Validation", "", f"- status: `{payload['status']}`", f"- rows: `{payload['n_rows']}`", "", "## Category Balance", ""]
    for category, count in payload["category_counts"].items():
        lines.append(f"- {category}: {count}")
    if payload["failures"]:
        lines.extend(["", "## Failures", *[f"- {item}" for item in payload["failures"]]])
    lines.append("")
    return "\n".join(lines)


def _render_panel_run_md(payload: Dict[str, Any], rows: List[Dict[str, Any]]) -> str:
    lines = [
        "# Empirical Pilot Panel Run",
        "",
        f"- status: `{payload['status']}`",
        f"- families processed: `{payload['n_families_processed']}`",
        "- heavy jobs executed: `False`",
        "",
        "## Claim Boundary",
        "",
        payload["claim_boundary"],
        "",
        "## Families",
        "",
    ]
    for row in rows:
        lines.append(
            f"- {row['panel_id']}: family={row['family_status']}, input={row['input_status']}, "
            f"applicability={row['applicability_status']}, scoring={row['scoring_status']}"
        )
    lines.append("")
    return "\n".join(lines)


def _render_reference_comparison_md(payload: Dict[str, Any]) -> str:
    lines = ["# Empirical Reference Comparison", "", f"- status: `{payload['status']}`", "", "## Concordance Counts", ""]
    for key, value in payload["concordance_counts"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Boundary", "", payload["claim_boundary"], ""])
    return "\n".join(lines)


def _render_pilot_summary_md(payload: Dict[str, Any]) -> str:
    lines = [
        "# Empirical Pilot Panel Summary",
        "",
        f"- status: `{payload['status']}`",
        f"- families: `{payload['n_families']}`",
        f"- simulation-matched calibration: `{payload['simulation_matched_calibration']}`",
        f"- recommended next empirical panel size: `{payload['recommended_next_empirical_panel_size']}`",
        "",
        "## Claim Boundary",
        "",
        payload["claim_boundary"],
        "",
        "No empirical discovery claim is made by this pilot summary.",
        "",
    ]
    return "\n".join(lines)


def _recommended_next_panel_size(rows: List[Dict[str, str]]) -> str:
    n = len(rows)
    if n < 5:
        return "5_to_10_families"
    if n < 20:
        return "20_to_50_families_after_review"
    return "hold_and_review_before_expansion"
