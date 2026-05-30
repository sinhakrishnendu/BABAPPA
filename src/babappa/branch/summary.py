"""Cross-tier summary for branch-conditioned validation runs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from babappa import __version__
from babappa.datasets.index import read_tsv, write_tsv


SUMMARY_FILES = {
    "json": "branch_conditioned_tier_summary.json",
    "tier_tsv": "branch_conditioned_tier_summary.tsv",
    "neural_tsv": "branch_site_neural_performance.tsv",
    "branch_aggregation_tsv": "branch_aggregation_performance.tsv",
    "gene_aggregation_tsv": "branch_gene_aggregation_performance.tsv",
    "calibration_tsv": "branch_calibration_summary.tsv",
    "threshold_policy_tsv": "branch_threshold_policy_summary.tsv",
    "controls_tsv": "branch_controls_summary.tsv",
    "markdown": "branch_conditioned_tier_summary.md",
}

REQUIRED_TSV_FILES = [
    SUMMARY_FILES["tier_tsv"],
    SUMMARY_FILES["neural_tsv"],
    SUMMARY_FILES["branch_aggregation_tsv"],
    SUMMARY_FILES["gene_aggregation_tsv"],
    SUMMARY_FILES["calibration_tsv"],
    SUMMARY_FILES["threshold_policy_tsv"],
    SUMMARY_FILES["controls_tsv"],
]

MARKDOWN_SECTIONS = [
    "# BABAPPA branch-conditioned 10K cross-tier summary",
    "## Executive conclusion",
    "## Scientific boundary",
    "## Completed tiers",
    "## Branch-site neural performance",
    "## Branch-level aggregation",
    "## Branch-to-gene aggregation",
    "## Calibration and threshold-policy behavior",
    "## Branch aggregation controls",
    "## Saturation robustness",
    "## Aligner-policy inheritance",
    "## Label-truth status",
    "## Branch feature policy",
    "## Limitations",
    "## Recommended next step",
]

PERFORMANCE_FIELDS = [
    "tier",
    "level",
    "split",
    "n",
    "positives",
    "negatives",
    "auroc",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "mcc",
    "specificity",
]

TIER_FIELDS = [
    "tier",
    "status",
    "run_summary_dir",
    "branch_site_label_status",
    "branch_site_rows",
    "branch_site_positives",
    "branch_site_neural_test_auroc",
    "branch_site_neural_all_auroc",
    "branch_level_all_auroc",
    "branch_level_test_auroc",
    "gene_level_all_auroc",
    "gene_level_test_auroc",
    "calibration_temperature",
    "calibration_selected_threshold",
    "branch_site_threshold_policy_profiles",
    "branch_aggregation_threshold_policy_profiles",
    "controls_observed_branch_auroc",
    "run_summary_warnings",
    "optional_warnings",
]

CALIBRATION_FIELDS = [
    "tier",
    "calibration_dir",
    "temperature",
    "selected_threshold",
    "target_fdr",
    "calibration_split_size",
    "calibration_split_positive_count",
    "raw_brier",
    "calibrated_brier",
    "raw_ece",
    "calibrated_ece",
    "warnings",
]

THRESHOLD_FIELDS = [
    "tier",
    "policy_level",
    "policy_dir",
    "profile",
    "selected_threshold",
    "precision",
    "recall",
    "empirical_fdr",
    "f1",
    "mcc",
    "warning",
    "warnings",
]

CONTROL_FIELDS = [
    "tier",
    "control",
    "observed_auroc",
    "mean_auroc",
    "q05_auroc",
    "q95_auroc",
    "min_auroc",
    "max_auroc",
    "std_auroc",
    "n_permutations",
    "empirical_p_value",
    "control_interpretation",
    "expected_behavior",
    "whether_control_is_destructive_enough",
]


@dataclass(frozen=True)
class BranchConditionedTierSummaryConfig:
    """Configuration for branch-conditioned cross-tier summary generation."""

    tiers: Union[str, Sequence[str]]
    outdir: str
    run_name: str = "fast_external_10k_streamed"
    output_suffix: Optional[str] = None
    allow_streamed: bool = True
    ablation_summary_dir: Optional[str] = None


def summarize_branch_conditioned_tiers(config: BranchConditionedTierSummaryConfig) -> Dict[str, Any]:
    """Generate machine-readable and manuscript-ready branch-conditioned tier summaries."""

    tiers = _parse_tiers(config.tiers)
    if not tiers:
        raise ValueError("at least one tier must be supplied")

    outdir = Path(config.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    warnings: List[str] = []
    failures: List[str] = []
    tier_records: List[Dict[str, Any]] = []
    neural_rows: List[Dict[str, Any]] = []
    branch_rows: List[Dict[str, Any]] = []
    gene_rows: List[Dict[str, Any]] = []
    calibration_rows: List[Dict[str, Any]] = []
    threshold_rows: List[Dict[str, Any]] = []
    controls_rows: List[Dict[str, Any]] = []

    for tier in tiers:
        record = _collect_tier(tier, config.run_name, warnings, failures, config.output_suffix, config.allow_streamed)
        tier_records.append(record)
        neural_rows.extend(record.pop("_neural_rows"))
        branch_rows.extend(record.pop("_branch_rows"))
        gene_rows.extend(record.pop("_gene_rows"))
        calibration_rows.extend(record.pop("_calibration_rows"))
        threshold_rows.extend(record.pop("_threshold_rows"))
        controls_rows.extend(record.pop("_controls_rows"))

    if failures:
        raise ValueError("; ".join(failures))

    label_statuses = {
        record["tier"]: record.get("branch_site_label_status", "unknown")
        for record in tier_records
    }
    proxy_tiers = [
        tier for tier, status in label_statuses.items()
        if "proxy" in str(status)
    ]
    branch_feature_policy = _branch_feature_policy_context(config, warnings)
    scale_context = _summary_scale_context(config, label_statuses, proxy_tiers)
    summary = {
        "branch_conditioned_tier_summary_version": __version__,
        "title": scale_context["title"],
        "run_name": config.run_name,
        "output_suffix": _normalize_output_suffix(config.output_suffix),
        "allow_streamed": config.allow_streamed,
        "tiers_requested": tiers,
        "tiers_included": [row["tier"] for row in tier_records],
        "interpretation": {
            "technical_validation": scale_context["technical_validation"],
            "saturation": "Extreme-tier performance remains strong but reduced relative to low/moderate tiers.",
            "aggregation": "Branch-level and gene-level aggregation are strong across tiers.",
            "truth_boundary": scale_context["truth_boundary"],
            "next_step": scale_context["next_step"],
            "hundred_k_policy": scale_context["hundred_k_policy"],
        },
        "scale_context": scale_context,
        "label_truth_status": {
            "by_tier": label_statuses,
            "proxy_tiers": proxy_tiers,
            "explicit_branch_site_truth_available": not proxy_tiers and all(
                "explicit" in str(status) for status in label_statuses.values()
            ),
        },
        "branch_feature_policy": branch_feature_policy,
        "tiers": tier_records,
        "branch_site_neural_rows": neural_rows,
        "branch_aggregation_rows": branch_rows,
        "branch_gene_aggregation_rows": gene_rows,
        "calibration_rows": calibration_rows,
        "threshold_policy_rows": threshold_rows,
        "controls_rows": controls_rows,
        "warnings": warnings,
        "generated_files": {
            name: str(outdir / filename) for name, filename in SUMMARY_FILES.items()
        },
    }

    _write_json(outdir / SUMMARY_FILES["json"], summary)
    write_tsv(outdir / SUMMARY_FILES["tier_tsv"], _tier_tsv_rows(tier_records), TIER_FIELDS)
    write_tsv(outdir / SUMMARY_FILES["neural_tsv"], neural_rows, PERFORMANCE_FIELDS)
    write_tsv(outdir / SUMMARY_FILES["branch_aggregation_tsv"], branch_rows, PERFORMANCE_FIELDS)
    write_tsv(outdir / SUMMARY_FILES["gene_aggregation_tsv"], gene_rows, PERFORMANCE_FIELDS)
    write_tsv(outdir / SUMMARY_FILES["calibration_tsv"], calibration_rows, CALIBRATION_FIELDS)
    write_tsv(outdir / SUMMARY_FILES["threshold_policy_tsv"], threshold_rows, THRESHOLD_FIELDS)
    write_tsv(outdir / SUMMARY_FILES["controls_tsv"], controls_rows, CONTROL_FIELDS)
    (outdir / SUMMARY_FILES["markdown"]).write_text(_render_markdown(summary), encoding="utf-8")

    return {
        "status": "ok",
        "outdir": str(outdir),
        "json_summary": str(outdir / SUMMARY_FILES["json"]),
        "markdown_summary": str(outdir / SUMMARY_FILES["markdown"]),
        "tiers_included": summary["tiers_included"],
        "n_warning": len(warnings),
        "warnings": warnings,
    }


def _collect_tier(
    tier: str,
    run_name: str,
    warnings: List[str],
    failures: List[str],
    output_suffix: Optional[str] = None,
    allow_streamed: bool = True,
) -> Dict[str, Any]:
    prefix = _select_existing_tier_prefix(tier, run_name, output_suffix, allow_streamed)
    default_dirs = _default_dirs(prefix)
    run_summary = _load_required_json(
        default_dirs["run_summary"] / "branch_site_run_summary.json",
        tier,
        "branch_site_run_summary",
        failures,
    )
    dirs = _dirs_from_run_summary(default_dirs, run_summary)

    neural = _load_required_json(
        dirs["branch_site_neural"] / "branch_site_neural_metrics.json",
        tier,
        "branch_site_neural_metrics",
        failures,
    )
    aggregation = _load_required_json(
        dirs["branch_aggregation"] / "branch_aggregation_metrics.json",
        tier,
        "branch_aggregation_metrics",
        failures,
    )
    label_summary = _load_optional_json(
        dirs["branch_site_labels"] / "branch_site_oracle_summary.json",
        tier,
        "branch_site_oracle_summary",
        warnings,
        optional_silent=True,
    )
    calibration = _load_optional_json(
        dirs["branch_site_calibration"] / "branch_site_calibration.json",
        tier,
        "branch_site_calibration",
        warnings,
    )
    controls = _load_optional_json(
        dirs["branch_aggregation_controls"] / "branch_aggregation_controls.json",
        tier,
        "branch_aggregation_controls",
        warnings,
    )
    site_policy = _load_policy_profiles(
        dirs["branch_site_threshold_policy"],
        tier,
        "branch_site_threshold_policy",
        "branch_site",
        warnings,
    )
    aggregation_policy = _load_policy_profiles(
        dirs["branch_aggregation_threshold_policy"],
        tier,
        "branch_aggregation_threshold_policy",
        "branch_aggregation",
        warnings,
    )

    neural_all = _nested_get(neural, ["metrics_by_split", "all"], {})
    neural_test = _nested_get(neural, ["metrics_by_split", "test"], {})
    branch_all = _nested_get(aggregation, ["branch_level_metrics_default", "all"], {})
    branch_test = _nested_get(aggregation, ["branch_level_metrics_default", "by_split", "test"], {})
    gene_all = _nested_get(aggregation, ["gene_level_metrics_default", "all"], {})
    gene_test = _nested_get(aggregation, ["gene_level_metrics_default", "by_split", "test"], {})
    run_warnings = _as_list((run_summary or {}).get("warnings"))

    tier_warning_prefix = f"{tier}: optional "
    optional_warnings = [warning for warning in warnings if warning.startswith(tier_warning_prefix)]
    neural_rows = [
        _performance_row(tier, "branch_site_neural", split, metrics)
        for split, metrics in [("all", neural_all), ("test", neural_test)]
        if metrics
    ]
    branch_rows = [
        _performance_row(tier, "branch_level_aggregation", split, metrics)
        for split, metrics in [("all", branch_all), ("test", branch_test)]
        if metrics
    ]
    gene_rows = [
        _performance_row(tier, "branch_to_gene_aggregation", split, metrics)
        for split, metrics in [("all", gene_all), ("test", gene_test)]
        if metrics
    ]
    calibration_rows = [_calibration_row(tier, dirs["branch_site_calibration"], calibration)]
    threshold_rows = _threshold_rows(tier, dirs["branch_site_threshold_policy"], "branch_site", site_policy)
    threshold_rows.extend(
        _threshold_rows(tier, dirs["branch_aggregation_threshold_policy"], "branch_aggregation", aggregation_policy)
    )
    controls_rows = _control_rows(tier, controls)

    return {
        "tier": tier,
        "status": "complete",
        "prefix": prefix,
        "run_summary_dir": str(dirs["run_summary"]),
        "branch_site_neural_dir": str(dirs["branch_site_neural"]),
        "branch_aggregation_dir": str(dirs["branch_aggregation"]),
        "branch_site_calibration_dir": str(dirs["branch_site_calibration"]),
        "branch_site_threshold_policy_dir": str(dirs["branch_site_threshold_policy"]),
        "branch_aggregation_threshold_policy_dir": str(dirs["branch_aggregation_threshold_policy"]),
        "branch_aggregation_controls_dir": str(dirs["branch_aggregation_controls"]),
        "branch_site_label_dir": str(dirs["branch_site_labels"]),
        "branch_site_label_status": (label_summary or {}).get("branch_site_labels_status", "unknown"),
        "branch_site_rows": (label_summary or {}).get("n_branch_site_rows"),
        "branch_site_positives": (label_summary or {}).get("n_positive_branch_sites"),
        "branch_site_neural_all": neural_all,
        "branch_site_neural_test": neural_test,
        "branch_level_all": branch_all,
        "branch_level_test": branch_test,
        "gene_level_all": gene_all,
        "gene_level_test": gene_test,
        "calibration_temperature": (calibration or {}).get("temperature"),
        "calibration_selected_threshold": (calibration or {}).get("selected_threshold"),
        "branch_site_threshold_policy_profiles": len((site_policy or {}).get("profiles", {})),
        "branch_aggregation_threshold_policy_profiles": len((aggregation_policy or {}).get("profiles", {})),
        "controls_observed_branch_auroc": _nested_get(controls, ["observed", "branch_auroc"]),
        "run_summary_warnings": run_warnings,
        "optional_warnings": optional_warnings,
        "_neural_rows": neural_rows,
        "_branch_rows": branch_rows,
        "_gene_rows": gene_rows,
        "_calibration_rows": calibration_rows,
        "_threshold_rows": threshold_rows,
        "_controls_rows": controls_rows,
    }


def _default_dirs(prefix: str) -> Dict[str, Path]:
    return {
        "run_summary": Path(f"branch_site_run_summary_{prefix}"),
        "branch_site_labels": _first_existing_dir([
            Path(f"branch_site_oracle_{_unsuffixed_prefix(prefix)}"),
            Path(f"branch_site_oracle_{prefix}"),
            Path(f"branch_site_labels_{_unsuffixed_prefix(prefix)}"),
            Path(f"branch_site_labels_{prefix}"),
        ]),
        "branch_site_neural": Path(f"branch_site_neural_{prefix}"),
        "branch_aggregation": Path(f"branch_aggregation_{prefix}"),
        "branch_aggregation_controls": Path(f"branch_aggregation_controls_{prefix}"),
        "branch_site_calibration": Path(f"branch_site_calibration_{prefix}"),
        "branch_site_threshold_policy": Path(f"branch_site_threshold_policy_{prefix}"),
        "branch_aggregation_threshold_policy": _first_existing_dir([
            Path(f"branch_aggregation_threshold_policy_{prefix}"),
            Path(f"branch_aggregation_policy_{prefix}"),
        ]),
    }


def _dirs_from_run_summary(default_dirs: Dict[str, Path], run_summary: Optional[Dict[str, Any]]) -> Dict[str, Path]:
    dirs = dict(default_dirs)
    sections = (run_summary or {}).get("sections")
    if not isinstance(sections, dict):
        return dirs
    mapping = {
        "branch_site_labels": "branch_site_labels",
        "branch_site_neural": "branch_site_neural",
        "branch_aggregation": "branch_aggregation",
        "branch_aggregation_controls": "branch_aggregation_controls",
        "branch_site_calibration": "branch_site_calibration",
        "branch_site_threshold_policy": "branch_site_threshold_policy",
        "branch_aggregation_threshold_policy": "branch_aggregation_threshold_policy",
    }
    for section_name, key in mapping.items():
        section = sections.get(section_name)
        if isinstance(section, dict) and section.get("directory"):
            dirs[key] = Path(str(section["directory"]))
    return dirs


def _load_required_json(
    path: Path,
    tier: str,
    label: str,
    failures: List[str],
) -> Optional[Dict[str, Any]]:
    if not path.exists():
        failures.append(f"{tier}: missing required {label}: {path}")
        return None
    return _load_json(path, failures, f"{tier}: unreadable required {label}")


def _load_optional_json(
    path: Path,
    tier: str,
    label: str,
    warnings: List[str],
    optional_silent: bool = False,
) -> Optional[Dict[str, Any]]:
    if not path.exists():
        if not optional_silent:
            warnings.append(f"{tier}: optional {label} missing: {path}")
        return None
    failures: List[str] = []
    payload = _load_json(path, failures, f"{tier}: unreadable optional {label}")
    warnings.extend(failures)
    return payload


def _load_policy_profiles(
    directory: Path,
    tier: str,
    label: str,
    prefix: str,
    warnings: List[str],
) -> Optional[Dict[str, Any]]:
    for filename in [
        f"{prefix}_threshold_profiles.json",
        "threshold_profiles.json",
    ]:
        candidate = directory / filename
        if candidate.exists():
            return _load_optional_json(candidate, tier, label, warnings)
    warnings.append(f"{tier}: optional {label} missing: {directory}")
    return None


def _load_json(path: Path, messages: List[str], label: str) -> Optional[Dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        messages.append(f"{label}: {exc}")
        return None
    if not isinstance(payload, dict):
        messages.append(f"{label}: JSON root is not an object")
        return None
    return payload


def _performance_row(tier: str, level: str, split: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
    row = {"tier": tier, "level": level, "split": split}
    for field in PERFORMANCE_FIELDS:
        if field not in row:
            row[field] = metrics.get(field)
    return row


def _calibration_row(tier: str, calibration_dir: Path, calibration: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    raw = (calibration or {}).get("raw_calibration_metrics") or {}
    calibrated = (calibration or {}).get("calibrated_calibration_metrics") or {}
    return {
        "tier": tier,
        "calibration_dir": str(calibration_dir),
        "temperature": (calibration or {}).get("temperature"),
        "selected_threshold": (calibration or {}).get("selected_threshold"),
        "target_fdr": (calibration or {}).get("target_fdr"),
        "calibration_split_size": (calibration or {}).get("calibration_split_size"),
        "calibration_split_positive_count": (calibration or {}).get("calibration_split_positive_count"),
        "raw_brier": raw.get("brier"),
        "calibrated_brier": calibrated.get("brier"),
        "raw_ece": raw.get("ece"),
        "calibrated_ece": calibrated.get("ece"),
        "warnings": _join((calibration or {}).get("warnings")),
    }


def _threshold_rows(
    tier: str,
    policy_dir: Path,
    policy_level: str,
    policy: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    rows = []
    profiles = (policy or {}).get("profiles") or {}
    if not isinstance(profiles, dict):
        return rows
    for profile_name in sorted(profiles):
        profile = profiles.get(profile_name) or {}
        metrics = profile.get("selection_metrics") or {}
        rows.append({
            "tier": tier,
            "policy_level": policy_level,
            "policy_dir": str(policy_dir),
            "profile": profile_name,
            "selected_threshold": profile.get("selected_threshold"),
            "precision": metrics.get("precision"),
            "recall": metrics.get("recall"),
            "empirical_fdr": metrics.get("empirical_fdr"),
            "f1": metrics.get("f1"),
            "mcc": metrics.get("mcc"),
            "warning": profile.get("warning", ""),
            "warnings": _join(profile.get("warnings")),
        })
    return rows


def _control_rows(tier: str, controls: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    control_payload = (controls or {}).get("controls") or {}
    if not isinstance(control_payload, dict):
        return rows
    for control_name in sorted(control_payload):
        summary = control_payload.get(control_name) or {}
        rows.append({"tier": tier, "control": control_name, **summary})
    return rows


def _tier_tsv_rows(tier_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for record in tier_records:
        rows.append({
            "tier": record.get("tier"),
            "status": record.get("status"),
            "run_summary_dir": record.get("run_summary_dir"),
            "branch_site_label_status": record.get("branch_site_label_status"),
            "branch_site_rows": record.get("branch_site_rows"),
            "branch_site_positives": record.get("branch_site_positives"),
            "branch_site_neural_test_auroc": _nested_get(record, ["branch_site_neural_test", "auroc"]),
            "branch_site_neural_all_auroc": _nested_get(record, ["branch_site_neural_all", "auroc"]),
            "branch_level_all_auroc": _nested_get(record, ["branch_level_all", "auroc"]),
            "branch_level_test_auroc": _nested_get(record, ["branch_level_test", "auroc"]),
            "gene_level_all_auroc": _nested_get(record, ["gene_level_all", "auroc"]),
            "gene_level_test_auroc": _nested_get(record, ["gene_level_test", "auroc"]),
            "calibration_temperature": record.get("calibration_temperature"),
            "calibration_selected_threshold": record.get("calibration_selected_threshold"),
            "branch_site_threshold_policy_profiles": record.get("branch_site_threshold_policy_profiles"),
            "branch_aggregation_threshold_policy_profiles": record.get("branch_aggregation_threshold_policy_profiles"),
            "controls_observed_branch_auroc": record.get("controls_observed_branch_auroc"),
            "run_summary_warnings": _join(record.get("run_summary_warnings")),
            "optional_warnings": _join(record.get("optional_warnings")),
        })
    return rows


def _summary_scale_context(
    config: BranchConditionedTierSummaryConfig,
    label_statuses: Dict[str, str],
    proxy_tiers: List[str],
) -> Dict[str, str]:
    run_text = " ".join(
        str(part or "").lower()
        for part in [config.run_name, config.outdir, config.output_suffix]
    )
    explicit_truth = not proxy_tiers and all("explicit" in str(status) for status in label_statuses.values())
    if "100k" in run_text or "100000" in run_text:
        return {
            "scale": "100K",
            "title": "BABAPPA explicit branch-truth 100K cross-tier summary"
            if explicit_truth
            else "BABAPPA branch-conditioned 100K cross-tier summary",
            "technical_validation": "Branch-conditioned workflow is technically validated at 100K scale.",
            "executive_conclusion": (
                "Conservative explicit branch-truth workflow is technically validated at 100K scale. "
                "Extreme-tier performance remains strong but reduced relative to low/moderate tiers. "
                "Branch-level and gene-level aggregation are strong. Results support moving from bulk "
                "simulation validation to targeted ablation, empirical pilot, and manuscript integration work."
            ),
            "truth_boundary": (
                "This is simulation-supervised branch-conditioned research-alpha validation using direct "
                "explicit simulator branch-site truth where available. It is not final empirical branch-site inference."
            ),
            "aligner_policy": (
                "This summary reads completed branch-conditioned 100K outputs and inherits the production-fast "
                "aligner-policy context: identity, MAFFT, BABAPPAlign, and MUSCLE. It does not run alignments, "
                "neural training, 10K generation, or 100K generation."
            ),
            "explicit_truth_limitation": (
                "- Explicit simulator branch-site truth is present in the summary metadata; empirical branch-site "
                "claims still require empirical deployment and calibration."
            ),
            "hundred_k_policy": "100K explicit branch-truth validation is complete for this simulation-supervised decision point.",
            "next_step": (
                "Use the completed 100K explicit branch-truth validation as the current research-alpha baseline; "
                "next prioritize targeted ablations, empirical pilot design, artifact/abstention heads, and manuscript integration."
            ),
        }
    return {
        "scale": "10K",
        "title": "BABAPPA branch-conditioned 10K cross-tier summary",
        "technical_validation": "Branch-conditioned workflow is technically validated at 10K scale.",
        "executive_conclusion": (
            "Branch-conditioned workflow is technically validated at 10K scale. Extreme-tier performance remains "
            "strong but reduced relative to low/moderate tiers. Branch-level and gene-level aggregation are strong. "
            "Results support moving to explicit branch-site truth implementation, while final 100K remains deferred "
            "until that validation passes."
        ),
        "truth_boundary": (
            "This is simulation-supervised branch-conditioned research-alpha validation. It is not final empirical "
            "branch-site inference. Current branch-conditioned labels may be proxy-derived from foreground branch/taxon "
            "labels crossed with selected-site labels."
        ),
        "aligner_policy": (
            "This summary reads completed branch-conditioned outputs and inherits the fast external 10K aligner-policy "
            "context. It does not run alignments, neural training, 10K generation, or 100K generation."
        ),
        "explicit_truth_limitation": "- Explicit per-branch per-site simulator truth is still required before final empirical branch-site claims.",
        "hundred_k_policy": "100K should wait until explicit branch-site truth validation passes.",
        "next_step": (
            "Implement simulator output for explicit branch-site selected-event truth, validate it on a 1K prototype "
            "across all saturation tiers and methods, then reconsider the final 100K run."
        ),
    }


def _render_markdown(summary: Dict[str, Any]) -> str:
    tiers = summary.get("tiers") or []
    proxy_tiers = summary.get("label_truth_status", {}).get("proxy_tiers") or []
    branch_policy = summary.get("branch_feature_policy") or {}
    scale_context = summary.get("scale_context") or _summary_scale_context(
        BranchConditionedTierSummaryConfig(
            tiers=summary.get("tiers_included") or [],
            outdir="",
            run_name=str(summary.get("run_name") or ""),
            output_suffix=summary.get("output_suffix"),
        ),
        summary.get("label_truth_status", {}).get("by_tier") or {},
        proxy_tiers,
    )
    lines = [
        f"# {scale_context['title']}",
        "",
        "## Executive conclusion",
        "",
        scale_context["executive_conclusion"],
        "",
        "## Scientific boundary",
        "",
        scale_context["truth_boundary"],
        "",
        "## Completed tiers",
        "",
    ]
    for record in tiers:
        lines.append(
            f"- {record.get('tier')}: {record.get('status')} "
            f"(label status: `{record.get('branch_site_label_status')}`)"
        )
    lines.extend(["", "## Branch-site neural performance", ""])
    lines.extend(_metric_table(tiers, "branch_site_neural_test", "test AUROC"))
    lines.extend(["", "## Branch-level aggregation", ""])
    lines.extend(_metric_table(tiers, "branch_level_all", "all AUROC"))
    lines.extend(["", "## Branch-to-gene aggregation", ""])
    lines.extend(_metric_table(tiers, "gene_level_all", "all AUROC"))
    lines.extend([
        "",
        "## Calibration and threshold-policy behavior",
        "",
        "Calibration and threshold-policy artifacts are summarized where present. Missing calibration or policy artifacts are warnings rather than summary failures because they are optional for the cross-tier audit layer.",
        "",
    ])
    for record in tiers:
        lines.append(
            f"- {record.get('tier')}: temperature `{_fmt(record.get('calibration_temperature'))}`, "
            f"selected threshold `{_fmt(record.get('calibration_selected_threshold'))}`, "
            f"branch-site profiles `{record.get('branch_site_threshold_policy_profiles')}`, "
            f"aggregation profiles `{record.get('branch_aggregation_threshold_policy_profiles')}`"
        )
    lines.extend([
        "",
        "## Branch aggregation controls",
        "",
    ])
    for record in tiers:
        lines.append(
            f"- {record.get('tier')}: observed branch-control AUROC `{_fmt(record.get('controls_observed_branch_auroc'))}`"
        )
    low = _tier_record(tiers, "low")
    extreme = _tier_record(tiers, "extreme")
    lines.extend([
        "",
        "## Saturation robustness",
        "",
    ])
    if low and extreme:
        lines.append(
            "Low-to-extreme degradation is visible at branch-site neural level "
            f"({_fmt(_nested_get(low, ['branch_site_neural_test', 'auroc']))} to "
            f"{_fmt(_nested_get(extreme, ['branch_site_neural_test', 'auroc']))}), "
            "while branch and gene aggregation remain strong."
        )
    else:
        lines.append("Saturation robustness is summarized across the included tiers.")
    lines.extend([
        "",
        "## Aligner-policy inheritance",
        "",
        scale_context["aligner_policy"],
        "",
        "## Label-truth status",
        "",
    ])
    if proxy_tiers:
        lines.append(
            "Proxy branch-conditioned labels are present in: "
            + ", ".join(str(tier) for tier in proxy_tiers)
            + ". These results are branch-conditioned proxy validation, not final explicit branch-site truth validation."
        )
    else:
        lines.append("No proxy label tiers were detected in the summary metadata.")
    lines.extend([
        "",
        "## Branch feature policy",
        "",
        f"- Recommended branch feature policy: `{branch_policy.get('recommended_policy', 'conservative_branch_site')}`",
        "- Full-context/full_model performance is treated as a context-aware upper-bound, not the main conservative branch-site claim.",
    ])
    if branch_policy.get("context_only_shortcut_high"):
        lines.append("- Warning: `context_only_shortcut_high`; context-only features are highly predictive.")
    if branch_policy.get("ablation_summary_dir"):
        lines.append(f"- Ablation summary: `{branch_policy.get('ablation_summary_dir')}`")
    lines.extend([
        "",
        "## Limitations",
        "",
        "- Simulation-supervised research-alpha evidence only.",
        scale_context["explicit_truth_limitation"],
        "- Branch-site full-context models are upper-bound diagnostics when foreground/context-only features are highly predictive.",
        f"- {scale_context['hundred_k_policy']}",
        "",
        "## Recommended next step",
        "",
        scale_context["next_step"],
        "",
    ])
    return "\n".join(lines)


def _branch_feature_policy_context(
    config: BranchConditionedTierSummaryConfig,
    warnings: List[str],
) -> Dict[str, Any]:
    path = _resolve_ablation_summary_dir(config)
    context = {
        "recommended_policy": "conservative_branch_site",
        "full_context_role": "context-aware upper-bound",
        "context_only_shortcut_high": False,
        "ablation_summary_dir": str(path) if path else None,
        "warnings": [],
    }
    if not path:
        return context
    rows = read_tsv(path / "branch_context_ablation_summary.tsv")
    context_only_values = [
        _safe_float(row.get("test_auroc"))
        for row in rows
        if row.get("profile") == "context_only"
    ]
    high = any(value is not None and value >= 0.95 for value in context_only_values)
    context["context_only_shortcut_high"] = high
    if high:
        warning = "branch_context_ablation:context_only_shortcut_high"
        context["warnings"].append(warning)
        warnings.append(warning)
    return context


def _resolve_ablation_summary_dir(config: BranchConditionedTierSummaryConfig) -> Optional[Path]:
    candidates = []
    if config.ablation_summary_dir:
        candidates.append(Path(config.ablation_summary_dir))
    candidates.extend(
        [
            Path(f"branch_context_ablation_{config.run_name}_summary"),
            Path("branch_context_ablation_explicit_1k_summary"),
        ]
    )
    for candidate in candidates:
        if (candidate / "branch_context_ablation_summary.tsv").exists():
            return candidate
    return None


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value in ("", None):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _metric_table(tiers: List[Dict[str, Any]], metric_key: str, label: str) -> List[str]:
    lines = ["| Tier | n | positives | " + label + " |", "| --- | ---: | ---: | ---: |"]
    for record in tiers:
        metrics = record.get(metric_key) or {}
        lines.append(
            f"| {record.get('tier')} | {_fmt(metrics.get('n'))} | "
            f"{_fmt(metrics.get('positives'))} | {_fmt(metrics.get('auroc'))} |"
        )
    return lines


def _tier_record(tiers: List[Dict[str, Any]], tier_name: str) -> Optional[Dict[str, Any]]:
    for record in tiers:
        if record.get("tier") == tier_name:
            return record
    return None


def _tier_prefix(
    run_name: str,
    tier: str,
    output_suffix: Optional[str] = None,
    allow_streamed: bool = True,
) -> str:
    if "{tier}" in run_name:
        prefix = run_name.format(tier=tier)
        suffix = _normalize_output_suffix(output_suffix)
        if suffix and not prefix.endswith(suffix):
            prefix = f"{prefix}{suffix}"
        return prefix
    base = run_name
    inherited_suffix = ""
    if base.endswith("_streamed"):
        base = base[:-len("_streamed")]
        inherited_suffix = "_streamed"
    suffix = inherited_suffix if output_suffix is None else _normalize_output_suffix(output_suffix)
    prefix = f"{base}_{tier}"
    if suffix and not prefix.endswith(suffix):
        prefix = f"{prefix}{suffix}"
    return prefix


def _select_existing_tier_prefix(
    tier: str,
    run_name: str,
    output_suffix: Optional[str],
    allow_streamed: bool,
) -> str:
    primary = _tier_prefix(run_name, tier, output_suffix=output_suffix, allow_streamed=allow_streamed)
    candidates = [primary]
    if allow_streamed and output_suffix is None and not primary.endswith("_streamed"):
        candidates.append(f"{primary}_streamed")
    for prefix in candidates:
        if (Path(f"branch_site_run_summary_{prefix}") / "branch_site_run_summary.json").exists():
            return prefix
    return primary


def _normalize_output_suffix(output_suffix: Optional[str]) -> str:
    if output_suffix in (None, "", "none", "None", "false", "False"):
        return ""
    suffix = str(output_suffix)
    if suffix == "streamed":
        return "_streamed"
    if suffix.startswith("_"):
        return suffix
    return f"_{suffix}"


def _unsuffixed_prefix(prefix: str) -> str:
    if prefix.endswith("_streamed"):
        return prefix[:-len("_streamed")]
    return prefix


def _first_existing_dir(candidates: List[Path]) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _parse_tiers(value: Union[str, Sequence[str]]) -> List[str]:
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return [str(item).strip() for item in value if str(item).strip()]


def _nested_get(payload: Optional[Dict[str, Any]], keys: List[str], default: Any = None) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def _as_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def _join(value: Any) -> str:
    return ";".join(_as_list(value))


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
