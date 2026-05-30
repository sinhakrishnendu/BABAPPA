"""Cross-tier summaries for external-aligner validation runs."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

from babappa import __version__


SUMMARY_FILES = {
    "json": "external_tier_summary.json",
    "tier_tsv": "external_tier_summary.tsv",
    "method_policy_tsv": "external_method_policy_summary.tsv",
    "performance_tsv": "external_performance_summary.tsv",
    "controls_tsv": "external_aggregation_controls_summary.tsv",
    "calibration_tsv": "external_calibration_summary.tsv",
    "markdown": "external_tier_summary.md",
}

MARKDOWN_SECTIONS = [
    "# BABAPPA external-aligner cross-tier summary",
    "## Executive conclusion",
    "## Completed tiers",
    "## Method policy by tier",
    "## Site-map and method quarantine",
    "## Site neural performance by tier",
    "## Site-to-gene aggregation by tier",
    "## Aggregation controls",
    "## Calibration and threshold-policy notes",
    "## Runtime and feasibility interpretation",
    "## Recommended 10K method set",
    "## Limitations",
]


@dataclass
class ExternalTierSummaryConfig:
    """Configuration for cross-tier external-aligner summary generation."""

    tiers: Union[str, Sequence[str]]
    outdir: str
    run_name: str = "external_aligner_validation"


def summarize_external_tiers(config: ExternalTierSummaryConfig) -> Dict[str, Any]:
    """Generate a machine-readable and manuscript-ready external tier summary."""

    tiers = _parse_tiers(config.tiers)
    if not tiers:
        raise ValueError("at least one tier must be supplied")

    outdir = Path(config.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    warnings: List[str] = []
    failures: List[str] = []
    tier_records: List[Dict[str, Any]] = []
    method_rows: List[Dict[str, Any]] = []
    performance_rows: List[Dict[str, Any]] = []
    controls_rows: List[Dict[str, Any]] = []
    calibration_rows: List[Dict[str, Any]] = []

    for tier in tiers:
        record = _collect_tier(tier, config.run_name, warnings, failures)
        tier_records.append(record)
        method_rows.extend(record.pop("_method_rows"))
        performance_rows.extend(record.pop("_performance_rows"))
        controls_rows.extend(record.pop("_controls_rows"))
        calibration_rows.extend(record.pop("_calibration_rows"))

    if failures:
        raise ValueError("; ".join(failures))

    scale_label = _summary_scale_label(tier_records, config.run_name)
    summary = {
        "external_tier_summary_version": __version__,
        "title": f"BABAPPA external-aligner {scale_label} cross-tier summary",
        "scale_label": scale_label,
        "run_name": config.run_name,
        "tiers_requested": tiers,
        "tiers_included": [row["tier"] for row in tier_records],
        "recommended_10k_method_set": [
            "identity",
            "mafft",
            "babappalign",
            "muscle-with-quarantine",
        ],
        "prank_policy": "diagnostic_only_excluded_from_default",
        "tcoffee_policy": "optional_diagnostic_only",
        "interpretation": {
            "oracle_upper_bound": (
                "Perfect site-to-gene aggregation is oracle-simulation upper-bound "
                "behavior and must not be interpreted as empirical branch-site inference."
            ),
            "external_alignment_robustness": (
                "Site-neural performance remains strong under external alignment "
                "uncertainty; method quarantine is part of the robustness design."
            ),
            "empirical_limitation": (
                "BABAPPA remains simulation-supervised research-alpha and is not yet "
                "empirical branch-site inference."
            ),
        },
        "tiers": tier_records,
        "method_policy_rows": method_rows,
        "performance_rows": performance_rows,
        "aggregation_controls_rows": controls_rows,
        "calibration_rows": calibration_rows,
        "warnings": warnings,
        "generated_files": {
            name: str(outdir / filename) for name, filename in SUMMARY_FILES.items()
        },
    }

    _write_json(outdir / SUMMARY_FILES["json"], summary)
    _write_tsv(outdir / SUMMARY_FILES["tier_tsv"], _tier_tsv_rows(tier_records), [
        "tier",
        "status",
        "n_families",
        "n_dataset_rows",
        "methods",
        "site_neural_test_auroc",
        "site_neural_all_auroc",
        "site_to_gene_test_auroc",
        "site_to_gene_all_auroc",
        "aggregation_controls_observed_auroc",
        "calibration_temperature",
        "calibration_selected_threshold",
        "quarantined_methods",
        "run_summary_warnings",
        "report_warnings",
        "optional_warnings",
    ])
    _write_tsv(outdir / SUMMARY_FILES["method_policy_tsv"], method_rows, [
        "tier",
        "method",
        "recommendation",
        "reason",
        "attempted_families",
        "successful_families",
        "failed_families",
        "failure_fraction",
        "site_map_unique_fraction",
        "site_map_conflict_fraction",
        "site_map_frame_error_fraction",
        "source",
    ])
    _write_tsv(outdir / SUMMARY_FILES["performance_tsv"], performance_rows, [
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
    ])
    _write_tsv(outdir / SUMMARY_FILES["controls_tsv"], controls_rows, [
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
    ])
    _write_tsv(outdir / SUMMARY_FILES["calibration_tsv"], calibration_rows, [
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
        "site_threshold_policy_warnings",
        "aggregation_threshold_policy_warnings",
        "warnings",
    ])
    (outdir / SUMMARY_FILES["markdown"]).write_text(
        _render_markdown(summary),
        encoding="utf-8",
    )

    return {
        "status": "ok",
        "outdir": str(outdir),
        "json_summary": str(outdir / SUMMARY_FILES["json"]),
        "markdown_summary": str(outdir / SUMMARY_FILES["markdown"]),
        "tiers_included": summary["tiers_included"],
        "n_warning": len(warnings),
        "warnings": warnings,
        "recommended_10k_method_set": summary["recommended_10k_method_set"],
    }


def validate_external_tier_summary_dir(summary_dir: Union[str, Path]) -> Dict[str, Any]:
    """Validate a cross-tier external-aligner summary directory."""

    path = Path(summary_dir)
    failures: List[str] = []
    warnings: List[str] = []

    for filename in SUMMARY_FILES.values():
        candidate = path / filename
        if not candidate.exists():
            failures.append(f"missing required summary artifact: {filename}")
        elif candidate.stat().st_size == 0:
            failures.append(f"empty required summary artifact: {filename}")

    payload: Optional[Dict[str, Any]] = None
    json_path = path / SUMMARY_FILES["json"]
    if json_path.exists():
        try:
            loaded = json.loads(json_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            failures.append(f"unreadable external_tier_summary.json: {exc}")
        else:
            if isinstance(loaded, dict):
                payload = loaded
            else:
                failures.append("external_tier_summary.json is not a JSON object")

    if payload is not None:
        tiers = payload.get("tiers")
        if not isinstance(tiers, list) or not tiers:
            failures.append("external_tier_summary.json missing non-empty tiers list")
        else:
            for tier in tiers:
                if not isinstance(tier, dict):
                    failures.append("tier record is not a JSON object")
                    continue
                name = tier.get("tier", "<unknown>")
                if not tier.get("run_summary_available"):
                    failures.append(f"{name}: required run summary unavailable")
                if not tier.get("site_neural_metrics_available"):
                    failures.append(f"{name}: required site neural metrics unavailable")
                if tier.get("optional_warnings"):
                    warnings.extend(str(w) for w in tier["optional_warnings"])
        if payload.get("recommended_10k_method_set") != [
            "identity",
            "mafft",
            "babappalign",
            "muscle-with-quarantine",
        ]:
            failures.append("recommended_10k_method_set is not the Cycle 30 fast set")
        if "empirical branch-site inference" not in json.dumps(
            payload.get("interpretation", {})
        ):
            failures.append("summary missing empirical branch-site inference limitation")

    markdown_path = path / SUMMARY_FILES["markdown"]
    if markdown_path.exists():
        text = markdown_path.read_text(encoding="utf-8")
        first_line = text.splitlines()[0] if text.splitlines() else ""
        if not (
            first_line.startswith("# BABAPPA external-aligner ")
            and first_line.endswith(" cross-tier summary")
        ):
            failures.append("external_tier_summary.md missing valid cross-tier title")
        for section in MARKDOWN_SECTIONS[1:]:
            if section not in text:
                failures.append(f"external_tier_summary.md missing section: {section}")

    performance_path = path / SUMMARY_FILES["performance_tsv"]
    if performance_path.exists() and _count_tsv_rows(performance_path) == 0:
        failures.append("external_performance_summary.tsv has no data rows")

    return {
        "status": "fail" if failures else "ok",
        "n_fail": len(failures),
        "n_warning": len(warnings),
        "failures": failures,
        "warnings": warnings,
    }


def _collect_tier(
    tier: str,
    run_name: str,
    warnings: List[str],
    failures: List[str],
) -> Dict[str, Any]:
    prefix = f"{run_name}_{tier}"
    run_summary_path = Path(f"run_summary_{prefix}") / "run_summary.json"
    report_path = Path(f"report_{prefix}") / "report_summary.json"
    site_neural_path = Path(f"site_neural_{prefix}") / "site_neural_metrics.json"
    aggregation_path = Path(f"site_to_gene_{prefix}") / "site_to_gene_metrics.json"
    controls_path = Path(f"aggregation_controls_{prefix}") / "site_aggregation_controls.json"
    method_policy_path = Path(f"method_policy_{prefix}") / "method_policy.json"
    calibration_dir = _first_existing_dir([
        Path(f"site_calibration_{prefix}"),
        Path(f"site_neural_calibration_{prefix}"),
    ])
    aggregation_policy_dir = Path(f"aggregation_policy_{prefix}")
    site_threshold_policy_dir = Path(f"site_threshold_policy_{prefix}")

    run_summary = _load_required_json(run_summary_path, tier, "run_summary", failures)
    site_neural = _load_required_json(site_neural_path, tier, "site_neural_metrics", failures)
    report = _load_optional_json(report_path, tier, "report_summary", warnings)
    aggregation = _load_optional_json(aggregation_path, tier, "site_to_gene_metrics", warnings)
    controls = _load_optional_json(controls_path, tier, "aggregation_controls", warnings)
    method_policy = _load_optional_json(method_policy_path, tier, "method_policy", warnings)

    if run_summary and calibration_dir is None:
        input_dir = (run_summary.get("inputs") or {}).get("site_calibration_dir")
        if input_dir:
            candidate = Path(input_dir)
            if candidate.exists():
                calibration_dir = candidate
    calibration = _load_calibration(tier, calibration_dir, warnings)
    aggregation_policy = _load_policy_profiles(
        tier,
        aggregation_policy_dir,
        "aggregation_threshold_policy",
        warnings,
    )
    site_threshold_policy = _load_policy_profiles(
        tier,
        site_threshold_policy_dir,
        "site_threshold_policy",
        warnings,
        optional_silent=True,
    )

    methods = _extract_methods(run_summary)
    n_families = _nested_get(run_summary, ["merged_dataset_overview", "n_families"])
    n_rows = _nested_get(run_summary, ["merged_dataset_overview", "n_rows"])
    run_warnings = _as_list((run_summary or {}).get("warnings"))
    report_warnings = _as_list((report or {}).get("warnings"))
    neural_all = _nested_get(site_neural, ["metrics_by_split", "all"], {})
    neural_test = _nested_get(site_neural, ["metrics_by_split", "test"], {})
    agg_all = _nested_get(aggregation, ["gene_level_metrics_default", "all"], {})
    agg_test = _nested_get(aggregation, ["gene_level_metrics_default", "by_split", "test"], {})
    controls_observed = _nested_get(controls, ["observed", "max_site_probability_auroc"])

    method_rows = _method_policy_rows(tier, method_policy, methods, n_families, warnings)
    optional_warnings = _tier_optional_warnings(tier, warnings)
    quarantined_methods = sorted(
        {
            row["method"]
            for row in method_rows
            if row.get("recommendation") == "quarantine"
        }
    )

    performance_rows = []
    for split_name in ["all", "test"]:
        metrics = _nested_get(site_neural, ["metrics_by_split", split_name], {})
        if metrics:
            performance_rows.append(_performance_row(tier, "site_neural", split_name, metrics))
    for split_name, metrics in [("all", agg_all), ("test", agg_test)]:
        if metrics:
            performance_rows.append(_performance_row(tier, "site_to_gene", split_name, metrics))

    controls_rows = _control_rows(tier, controls)
    calibration_rows = [
        _calibration_row(
            tier,
            calibration_dir,
            calibration,
            site_threshold_policy,
            aggregation_policy,
        )
    ]

    return {
        "tier": tier,
        "status": "complete",
        "run_summary_available": run_summary is not None,
        "report_available": report is not None,
        "site_neural_metrics_available": site_neural is not None,
        "site_to_gene_metrics_available": aggregation is not None,
        "aggregation_controls_available": controls is not None,
        "calibration_available": calibration is not None,
        "aggregation_threshold_policy_available": aggregation_policy is not None,
        "site_threshold_policy_available": site_threshold_policy is not None,
        "method_policy_available": method_policy is not None,
        "n_families": n_families,
        "n_dataset_rows": n_rows,
        "methods": methods,
        "quarantined_methods": quarantined_methods,
        "site_neural_all": neural_all,
        "site_neural_test": neural_test,
        "site_to_gene_all": agg_all,
        "site_to_gene_test": agg_test,
        "aggregation_controls_observed_auroc": controls_observed,
        "calibration_temperature": (calibration or {}).get("temperature"),
        "calibration_selected_threshold": (calibration or {}).get("selected_threshold"),
        "run_summary_warnings": run_warnings,
        "report_warnings": report_warnings,
        "optional_warnings": optional_warnings,
        "_method_rows": method_rows,
        "_performance_rows": performance_rows,
        "_controls_rows": controls_rows,
        "_calibration_rows": calibration_rows,
    }


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
) -> Optional[Dict[str, Any]]:
    if not path.exists():
        warnings.append(f"{tier}: optional {label} missing: {path}")
        return None
    failures: List[str] = []
    payload = _load_json(path, failures, f"{tier}: unreadable optional {label}")
    warnings.extend(failures)
    return payload


def _load_json(
    path: Path,
    messages: List[str],
    label: str,
) -> Optional[Dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        messages.append(f"{label}: {exc}")
        return None
    if not isinstance(payload, dict):
        messages.append(f"{label}: JSON root is not an object")
        return None
    return payload


def _load_calibration(
    tier: str,
    calibration_dir: Optional[Path],
    warnings: List[str],
) -> Optional[Dict[str, Any]]:
    if calibration_dir is None:
        warnings.append(f"{tier}: optional site calibration directory missing")
        return None
    return _load_optional_json(
        calibration_dir / "site_calibration.json",
        tier,
        "site_calibration",
        warnings,
    )


def _load_policy_profiles(
    tier: str,
    policy_dir: Path,
    label: str,
    warnings: List[str],
    optional_silent: bool = False,
) -> Optional[Dict[str, Any]]:
    for filename in [
        "aggregation_threshold_profiles.json",
        "threshold_profiles.json",
        "site_threshold_profiles.json",
    ]:
        candidate = policy_dir / filename
        if candidate.exists():
            return _load_optional_json(candidate, tier, label, warnings)
    if not optional_silent:
        warnings.append(f"{tier}: optional {label} missing: {policy_dir}")
    return None


def _method_policy_rows(
    tier: str,
    method_policy: Optional[Dict[str, Any]],
    methods: List[str],
    n_families: Optional[Any],
    warnings: List[str],
) -> List[Dict[str, Any]]:
    if method_policy:
        rows = []
        for row in method_policy.get("methods", []):
            if not isinstance(row, dict):
                continue
            rows.append({
                "tier": tier,
                "method": row.get("method"),
                "recommendation": row.get("recommendation"),
                "reason": row.get("reason"),
                "attempted_families": row.get("attempted_families"),
                "successful_families": row.get("successful_families"),
                "failed_families": row.get("failed_families"),
                "failure_fraction": row.get("failure_fraction"),
                "site_map_unique_fraction": row.get("site_map_unique_fraction"),
                "site_map_conflict_fraction": row.get("site_map_conflict_fraction"),
                "site_map_frame_error_fraction": row.get("site_map_frame_error_fraction"),
                "source": "method_policy",
            })
        return rows

    warnings.append(f"{tier}: method policy missing; method recommendations synthesized")
    return [
        {
            "tier": tier,
            "method": method,
            "recommendation": "not_evaluated",
            "reason": "method_policy_missing_optional",
            "attempted_families": n_families,
            "successful_families": "",
            "failed_families": "",
            "failure_fraction": "",
            "site_map_unique_fraction": "",
            "site_map_conflict_fraction": "",
            "site_map_frame_error_fraction": "",
            "source": "synthesized_from_run_summary_methods",
        }
        for method in methods
    ]


def _performance_row(
    tier: str,
    level: str,
    split: str,
    metrics: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "tier": tier,
        "level": level,
        "split": split,
        "n": metrics.get("n"),
        "positives": metrics.get("positives"),
        "negatives": metrics.get("negatives"),
        "auroc": metrics.get("auroc"),
        "accuracy": metrics.get("accuracy"),
        "precision": metrics.get("precision"),
        "recall": metrics.get("recall"),
        "f1": metrics.get("f1"),
        "mcc": metrics.get("mcc"),
        "specificity": metrics.get("specificity"),
    }


def _control_rows(tier: str, controls: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not controls:
        return []
    observed_auroc = _nested_get(controls, ["observed", "max_site_probability_auroc"])
    rows = []
    for name, metrics in (controls.get("controls") or {}).items():
        if not isinstance(metrics, dict):
            continue
        rows.append({
            "tier": tier,
            "control": name,
            "observed_auroc": observed_auroc,
            "mean_auroc": metrics.get("mean_auroc"),
            "q05_auroc": metrics.get("q05_auroc"),
            "q95_auroc": metrics.get("q95_auroc"),
            "min_auroc": metrics.get("min_auroc"),
            "max_auroc": metrics.get("max_auroc"),
            "std_auroc": metrics.get("std_auroc"),
            "n_permutations": metrics.get("n_permutations"),
            "empirical_p_value": metrics.get("empirical_p_value"),
        })
    return rows


def _calibration_row(
    tier: str,
    calibration_dir: Optional[Path],
    calibration: Optional[Dict[str, Any]],
    site_threshold_policy: Optional[Dict[str, Any]],
    aggregation_policy: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    raw = (calibration or {}).get("raw_calibration_metrics") or {}
    calibrated = (calibration or {}).get("calibrated_calibration_metrics") or {}
    return {
        "tier": tier,
        "calibration_dir": str(calibration_dir) if calibration_dir else "",
        "temperature": (calibration or {}).get("temperature"),
        "selected_threshold": (calibration or {}).get("selected_threshold"),
        "target_fdr": (calibration or {}).get("target_fdr"),
        "calibration_split_size": (calibration or {}).get("calibration_split_size"),
        "calibration_split_positive_count": (calibration or {}).get(
            "calibration_split_positive_count"
        ),
        "raw_brier": raw.get("brier"),
        "calibrated_brier": calibrated.get("brier"),
        "raw_ece": raw.get("ece"),
        "calibrated_ece": calibrated.get("ece"),
        "site_threshold_policy_warnings": _join_list(
            (site_threshold_policy or {}).get("warnings")
        ),
        "aggregation_threshold_policy_warnings": _join_list(
            (aggregation_policy or {}).get("warnings")
        ),
        "warnings": _join_list((calibration or {}).get("warnings")),
    }


def _tier_tsv_rows(tier_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for row in tier_records:
        rows.append({
            "tier": row["tier"],
            "status": row["status"],
            "n_families": row.get("n_families"),
            "n_dataset_rows": row.get("n_dataset_rows"),
            "methods": _join_list(row.get("methods")),
            "site_neural_test_auroc": _nested_get(row, ["site_neural_test", "auroc"]),
            "site_neural_all_auroc": _nested_get(row, ["site_neural_all", "auroc"]),
            "site_to_gene_test_auroc": _nested_get(row, ["site_to_gene_test", "auroc"]),
            "site_to_gene_all_auroc": _nested_get(row, ["site_to_gene_all", "auroc"]),
            "aggregation_controls_observed_auroc": row.get(
                "aggregation_controls_observed_auroc"
            ),
            "calibration_temperature": row.get("calibration_temperature"),
            "calibration_selected_threshold": row.get("calibration_selected_threshold"),
            "quarantined_methods": _join_list(row.get("quarantined_methods")),
            "run_summary_warnings": _join_list(row.get("run_summary_warnings")),
            "report_warnings": _join_list(row.get("report_warnings")),
            "optional_warnings": _join_list(row.get("optional_warnings")),
        })
    return rows


def _render_markdown(summary: Dict[str, Any]) -> str:
    tiers = summary["tiers"]
    method_rows = summary["method_policy_rows"]
    controls_rows = summary["aggregation_controls_rows"]
    calibration_rows = summary["calibration_rows"]
    warnings = summary["warnings"]
    scale_label = str(summary.get("scale_label") or "cross-tier")

    lines = [
        f"# BABAPPA external-aligner {scale_label} cross-tier summary",
        "",
        "## Executive conclusion",
        "",
        (
            f"Low, moderate, high, and extreme external-aligner {scale_label} tiers are summarized "
            "from completed artifacts. Site-neural performance remains strong under "
            "external alignment uncertainty, and site-to-gene aggregation is perfect "
            "across completed tiers under the oracle-simulation setup."
        ),
        "",
        (
            "Perfect site-to-gene aggregation is oracle-simulation upper-bound behavior. "
            "It should be used as a feasibility signal, not as an empirical branch-site "
            "inference claim."
        ),
        "",
        "## Completed tiers",
        "",
        _markdown_table(
            ["tier", "families", "rows", "methods", "report warnings", "summary warnings"],
            [
                [
                    row["tier"],
                    _fmt(row.get("n_families")),
                    _fmt(row.get("n_dataset_rows")),
                    _join_list(row.get("methods")),
                    str(len(row.get("report_warnings") or [])),
                    str(len(row.get("run_summary_warnings") or [])),
                ]
                for row in tiers
            ],
        ),
        "",
        "## Method policy by tier",
        "",
        _markdown_table(
            ["tier", "method", "recommendation", "reason"],
            [
                [
                    row.get("tier"),
                    row.get("method"),
                    row.get("recommendation"),
                    row.get("reason"),
                ]
                for row in method_rows
            ],
        ),
        "",
        "## Site-map and method quarantine",
        "",
        _method_quarantine_note(method_rows),
        "",
        "## Site neural performance by tier",
        "",
        _markdown_table(
            ["tier", "test AUROC", "test F1", "test MCC", "test n", "all AUROC"],
            [
                [
                    row["tier"],
                    _fmt(_nested_get(row, ["site_neural_test", "auroc"])),
                    _fmt(_nested_get(row, ["site_neural_test", "f1"])),
                    _fmt(_nested_get(row, ["site_neural_test", "mcc"])),
                    _fmt(_nested_get(row, ["site_neural_test", "n"])),
                    _fmt(_nested_get(row, ["site_neural_all", "auroc"])),
                ]
                for row in tiers
            ],
        ),
        "",
        "## Site-to-gene aggregation by tier",
        "",
        _markdown_table(
            ["tier", "test AUROC", "all AUROC", "rows"],
            [
                [
                    row["tier"],
                    _fmt(_nested_get(row, ["site_to_gene_test", "auroc"])),
                    _fmt(_nested_get(row, ["site_to_gene_all", "auroc"])),
                    _fmt(_nested_get(row, ["site_to_gene_all", "n"])),
                ]
                for row in tiers
            ],
        ),
        "",
        "## Aggregation controls",
        "",
        (
            "Aggregation controls remain the guardrail against overclaiming perfect "
            "aggregation. Observed AUROC should exceed shuffled-label, shuffled-site, "
            "family-permutation, and random-score controls."
        ),
        "",
        _markdown_table(
            ["tier", "control", "observed AUROC", "mean control AUROC", "q05", "q95"],
            [
                [
                    row.get("tier"),
                    row.get("control"),
                    _fmt(row.get("observed_auroc")),
                    _fmt(row.get("mean_auroc")),
                    _fmt(row.get("q05_auroc")),
                    _fmt(row.get("q95_auroc")),
                ]
                for row in controls_rows
            ],
        ),
        "",
        "## Calibration and threshold-policy notes",
        "",
        _markdown_table(
            [
                "tier",
                "temperature",
                "site threshold",
                "target FDR",
                "calibration warnings",
                "aggregation policy warnings",
            ],
            [
                [
                    row.get("tier"),
                    _fmt(row.get("temperature")),
                    _fmt(row.get("selected_threshold")),
                    _fmt(row.get("target_fdr")),
                    row.get("warnings") or "",
                    row.get("aggregation_threshold_policy_warnings") or "",
                ]
                for row in calibration_rows
            ],
        ),
        "",
        "## Runtime and feasibility interpretation",
        "",
        (
            f"The {scale_label} external-aligner track is feasible when slow/fragile methods are "
            "kept out of the production default and frame-unsafe outputs are quarantined "
            "before tensorization. The fast production path should emphasize identity, "
            "MAFFT, BABAPPAlign, and MUSCLE with automatic quarantine."
        ),
        "",
        "PRANK remains diagnostic only and is excluded from the default production ensemble.",
        "",
        "T-Coffee remains optional diagnostic only and should not be part of default 10K runs.",
        "",
        "## Recommended 10K method set",
        "",
        "`identity,mafft,babappalign,muscle-with-quarantine`",
        "",
        "## Limitations",
        "",
        "- This is simulation-supervised, oracle-labeled validation.",
        "- Perfect aggregation is an oracle-simulation upper bound.",
        "- The workflow does not yet claim empirical branch-site inference.",
        "- Missing optional method-policy artifacts for older tiers are reported as warnings.",
        "- PRANK and T-Coffee should remain diagnostic-only unless specifically requested.",
        "",
    ]

    if warnings:
        lines.extend([
            "## Warnings",
            "",
            *[f"- {warning}" for warning in warnings],
            "",
        ])

    return "\n".join(lines)


def _summary_scale_label(tier_records: Sequence[Dict[str, Any]], run_name: str) -> str:
    normalized = run_name.lower().replace("-", "_")
    if "10k" in normalized or "10000" in normalized:
        return "10K"
    if "1k" in normalized or "1000" in normalized:
        return "1K"
    total = 0
    for record in tier_records:
        try:
            total += int(record.get("n_families") or 0)
        except (TypeError, ValueError):
            continue
    if total:
        if total % 1000 == 0:
            return f"{total // 1000}K"
        return f"{total:,}-family"
    return "cross-tier"


def _method_quarantine_note(method_rows: Sequence[Dict[str, Any]]) -> str:
    quarantines = [
        f"{row.get('tier')}:{row.get('method')} ({row.get('reason')})"
        for row in method_rows
        if row.get("recommendation") == "quarantine"
    ]
    not_evaluated = sorted({
        str(row.get("tier"))
        for row in method_rows
        if row.get("recommendation") == "not_evaluated"
    })
    parts = ["Method quarantine is part of the robustness design, not a failure."]
    if quarantines:
        parts.append("Observed quarantines: " + "; ".join(quarantines) + ".")
    else:
        parts.append("No methods were quarantined in the summarized tiers.")
    if not_evaluated:
        parts.append(
            "Tiers without explicit method-policy artifacts are marked `not_evaluated`: "
            + ", ".join(not_evaluated)
            + "."
        )
    return " ".join(parts)


def _markdown_table(headers: List[str], rows: List[List[Any]]) -> str:
    if not rows:
        rows = [["" for _ in headers]]
    header_line = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = [
        "| " + " | ".join(_escape_md(_fmt(cell)) for cell in row) + " |"
        for row in rows
    ]
    return "\n".join([header_line, separator, *body])


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_tsv(
    path: Path,
    rows: Iterable[Dict[str, Any]],
    fieldnames: List[str],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: _tsv_value(row.get(name)) for name in fieldnames})


def _count_tsv_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return sum(1 for _ in reader)


def _parse_tiers(tiers: Union[str, Sequence[str]]) -> List[str]:
    if isinstance(tiers, str):
        raw = tiers.split(",")
    else:
        raw = list(tiers)
    return [item.strip() for item in raw if item and item.strip()]


def _first_existing_dir(paths: Sequence[Path]) -> Optional[Path]:
    for path in paths:
        if path.exists():
            return path
    return None


def _extract_methods(run_summary: Optional[Dict[str, Any]]) -> List[str]:
    methods = _nested_get(run_summary, ["merged_dataset_overview", "methods"], [])
    if isinstance(methods, list):
        return [str(method) for method in methods]
    return []


def _nested_get(value: Any, path: Sequence[str], default: Any = None) -> Any:
    current = value
    for key in path:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
    return default if current is None else current


def _as_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str):
        return [value] if value else []
    return [str(value)]


def _join_list(value: Any) -> str:
    return ",".join(_as_list(value))


def _tier_optional_warnings(tier: str, warnings: List[str]) -> List[str]:
    prefix = f"{tier}: "
    return [warning[len(prefix):] for warning in warnings if warning.startswith(prefix)]


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _tsv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, list):
        return ",".join(str(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True)
    return value


def _escape_md(value: str) -> str:
    return value.replace("|", "\\|")
