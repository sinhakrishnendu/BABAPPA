"""Compact run summaries for BABAPPA workflow outputs."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from babappa import __version__

RUN_SUMMARY_VERSION = __version__


@dataclass(frozen=True)
class RunSummaryConfig:
    """Configuration for a compact BABAPPA run summary."""

    outdir: str
    sim_dir: Optional[str] = None
    sim_audit_dir: Optional[str] = None
    align_dir: Optional[str] = None
    tensor_dir: Optional[str] = None
    dataset_dir: Optional[str] = None
    baseline_dir: Optional[str] = None
    baseline_calibration_dir: Optional[str] = None
    saturation_panel_dir: Optional[str] = None
    merged_dataset_dir: Optional[str] = None
    neural_dir: Optional[str] = None
    neural_calibration_dir: Optional[str] = None
    stratified_calibration_dir: Optional[str] = None
    threshold_policy_dir: Optional[str] = None
    stratified_eval_dir: Optional[str] = None
    neural_diagnostics_dir: Optional[str] = None
    ablation_comparison_dir: Optional[str] = None
    label_signal_audit_dir: Optional[str] = None
    leakage_audit_dir: Optional[str] = None
    stability_benchmark_dir: Optional[str] = None
    site_label_dir: Optional[str] = None
    site_dataset_dir: Optional[str] = None
    site_leakage_audit_dir: Optional[str] = None
    site_baseline_dir: Optional[str] = None
    site_neural_dir: Optional[str] = None
    site_calibration_dir: Optional[str] = None
    site_threshold_policy_dir: Optional[str] = None
    site_stratified_eval_dir: Optional[str] = None
    site_aggregation_dir: Optional[str] = None
    site_stability_dir: Optional[str] = None
    site_model_comparison_dir: Optional[str] = None
    site_aggregation_controls_dir: Optional[str] = None
    site_aggregation_threshold_policy_dir: Optional[str] = None
    site_calibration_comparison_dir: Optional[str] = None
    report_dir: Optional[str] = None
    title: str = "BABAPPA run summary"

    def __post_init__(self) -> None:
        supplied = _supplied_inputs(self)
        if not supplied:
            raise ValueError("at least one input directory must be supplied")
        for label, directory in supplied.items():
            path = Path(directory)
            if not path.exists():
                raise ValueError(f"{label} does not exist: {path}")
            if not path.is_dir():
                raise ValueError(f"{label} is not a directory: {path}")
        Path(self.outdir).mkdir(parents=True, exist_ok=True)


def load_json_if_exists(path: Path) -> Optional[dict]:
    """Load a JSON object if it exists."""
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON file is not an object: {path}")
    return payload


def read_tsv_preview(path: Path, max_rows: int = 5) -> dict:
    """Return basic TSV metadata plus a small preview."""
    if not path.exists():
        return {
            "exists": False,
            "n_rows": 0,
            "fieldnames": [],
            "preview_rows": [],
        }
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fieldnames = list(reader.fieldnames or [])
        rows = []
        count = 0
        for row in reader:
            if count < max_rows:
                rows.append(dict(row))
            count += 1
    return {
        "exists": True,
        "n_rows": count,
        "fieldnames": fieldnames,
        "preview_rows": rows,
    }


def safe_get_nested(d: dict, path: List[str], default: Any = None) -> Any:
    """Safely fetch a nested dictionary value."""
    current: Any = d
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def summarize_run(config: RunSummaryConfig) -> dict:
    """Inspect available workflow directories and write run summary artifacts."""
    outdir = Path(config.outdir)
    json_summary = outdir / "run_summary.json"
    markdown_summary = outdir / "run_summary.md"
    warnings: List[str] = []

    dataset_index = _load_json_from_dir(
        config.dataset_dir, "dataset_index.json", warnings
    )
    baseline_meta = _load_json_from_dir(
        config.baseline_dir, "baseline_model_meta.json", warnings
    )
    baseline_metrics = _load_json_from_dir(
        config.baseline_dir, "baseline_metrics.json", warnings
    )
    baseline_calibration = _load_json_from_dir(
        config.baseline_calibration_dir, "baseline_calibration.json", warnings
    )
    baseline_calibrated_metrics = _load_json_from_dir(
        config.baseline_calibration_dir,
        "baseline_calibrated_metrics.json",
        warnings,
    )
    neural_meta = _load_json_from_dir(
        config.neural_dir, "neural_model_meta.json", warnings
    )
    neural_metrics = _load_json_from_dir(
        config.neural_dir, "neural_metrics.json", warnings
    )
    neural_calibration = _load_json_from_dir(
        config.neural_calibration_dir, "neural_calibration.json", warnings
    )
    neural_calibrated_metrics = _load_json_from_dir(
        config.neural_calibration_dir,
        "neural_calibrated_metrics.json",
        warnings,
    )
    stratified_calibration = _load_json_from_dir(
        config.stratified_calibration_dir,
        "stratified_calibration.json",
        warnings,
    )
    stratified_calibrated_metrics = _load_json_from_dir(
        config.stratified_calibration_dir,
        "stratified_calibrated_metrics.json",
        warnings,
    )
    saturation_panel = _load_json_from_dir(
        config.saturation_panel_dir, "saturation_panel.json", warnings
    )
    merged_dataset = _load_json_from_dir(
        config.merged_dataset_dir, "dataset_index.json", warnings
    )
    threshold_policy = _load_json_from_dir(
        config.threshold_policy_dir, "threshold_profiles.json", warnings
    )
    stratified_eval = _load_json_from_dir(
        config.stratified_eval_dir, "stratified_eval.json", warnings
    )
    neural_diagnostics = _load_json_from_dir(
        config.neural_diagnostics_dir, "neural_diagnostics.json", warnings
    )
    ablation_comparison = _load_json_from_dir(
        config.ablation_comparison_dir, "ablation_comparison.json", warnings
    )
    label_signal_audit = _load_json_from_dir(
        config.label_signal_audit_dir, "label_signal_audit.json", warnings
    )
    leakage_audit = _load_json_from_dir(
        config.leakage_audit_dir, "leakage_audit.json", warnings
    )
    stability_benchmark = _load_json_from_dir(
        config.stability_benchmark_dir, "stability_benchmark.json", warnings
    )
    site_labels = _load_json_from_dir(
        config.site_label_dir, "site_oracle_summary.json", warnings
    )
    site_dataset = _load_json_from_dir(
        config.site_dataset_dir, "site_dataset_index.json", warnings
    )
    site_leakage_audit = _load_json_from_dir(
        config.site_leakage_audit_dir, "site_leakage_audit.json", warnings
    )
    site_baseline_meta = _load_json_from_dir(
        config.site_baseline_dir, "site_baseline_model_meta.json", warnings
    )
    site_baseline_metrics = _load_json_from_dir(
        config.site_baseline_dir, "site_baseline_metrics.json", warnings
    )
    site_neural_meta = _load_json_from_dir(
        config.site_neural_dir, "site_neural_model_meta.json", warnings
    )
    site_neural_metrics = _load_json_from_dir(
        config.site_neural_dir, "site_neural_metrics.json", warnings
    )
    site_calibration = _load_json_from_dir(
        config.site_calibration_dir, "site_calibration.json", warnings
    )
    site_threshold_policy = _load_json_from_dir(
        config.site_threshold_policy_dir, "site_threshold_profiles.json", warnings
    )
    site_stratified_eval = _load_json_from_dir(
        config.site_stratified_eval_dir, "site_stratified_eval.json", warnings
    )
    site_aggregation = _load_json_from_dir(
        config.site_aggregation_dir, "site_to_gene_metrics.json", warnings
    )
    site_stability = _load_json_from_dir(
        config.site_stability_dir, "site_stability_benchmark.json", warnings
    )
    site_model_comparison = _load_json_from_dir(
        config.site_model_comparison_dir, "site_model_comparison.json", warnings
    )
    site_aggregation_controls = _load_json_from_dir(
        config.site_aggregation_controls_dir, "site_aggregation_controls.json", warnings
    )
    site_aggregation_threshold_policy = _load_json_from_dir(
        config.site_aggregation_threshold_policy_dir,
        "aggregation_threshold_profiles.json",
        warnings,
    )
    site_calibration_comparison = _load_json_from_dir(
        config.site_calibration_comparison_dir,
        "site_calibration_comparison.json",
        warnings,
    )
    report_summary = _load_json_from_dir(
        config.report_dir, "report_summary.json", warnings
    )

    payload = {
        "run_summary_version": RUN_SUMMARY_VERSION,
        "title": config.title,
        "inputs": _supplied_inputs(config),
        "status_overview": _status_overview(config),
        "dataset_overview": _dataset_overview(dataset_index),
        "baseline_overview": _baseline_overview(baseline_meta, baseline_metrics),
        "neural_overview": _neural_overview(neural_meta, neural_metrics),
        "calibration_overview": _calibration_overview(
            baseline_calibration,
            baseline_calibrated_metrics,
            neural_calibration,
            neural_calibrated_metrics,
        ),
        "stratified_calibration_overview": _stratified_calibration_overview(
            config.stratified_calibration_dir,
            stratified_calibration,
            stratified_calibrated_metrics,
        ),
        "saturation_panel_overview": _saturation_panel_overview(
            config.saturation_panel_dir, saturation_panel
        ),
        "merged_dataset_overview": _merged_dataset_overview(
            config.merged_dataset_dir, merged_dataset
        ),
        "threshold_policy_overview": _threshold_policy_overview(
            config.threshold_policy_dir, threshold_policy
        ),
        "stratified_eval_overview": _stratified_eval_overview(
            config.stratified_eval_dir, stratified_eval
        ),
        "neural_diagnostics_overview": _neural_diagnostics_overview(
            config.neural_diagnostics_dir, neural_diagnostics
        ),
        "ablation_comparison_overview": _ablation_comparison_overview(
            config.ablation_comparison_dir, ablation_comparison
        ),
        "label_signal_audit_overview": _label_signal_audit_overview(
            config.label_signal_audit_dir, label_signal_audit
        ),
        "leakage_audit_overview": _leakage_audit_overview(
            config.leakage_audit_dir, leakage_audit
        ),
        "stability_benchmark_overview": _stability_benchmark_overview(
            config.stability_benchmark_dir, stability_benchmark
        ),
        "site_label_overview": _site_label_overview(
            config.site_label_dir, site_labels
        ),
        "site_dataset_overview": _site_dataset_overview(
            config.site_dataset_dir, site_dataset
        ),
        "site_leakage_overview": _site_leakage_overview(
            config.site_leakage_audit_dir, site_leakage_audit
        ),
        "site_baseline_overview": _site_baseline_overview(
            config.site_baseline_dir, site_baseline_meta, site_baseline_metrics
        ),
        "site_neural_overview": _site_neural_overview(
            config.site_neural_dir, site_neural_meta, site_neural_metrics
        ),
        "site_calibration_overview": _site_calibration_overview(
            config.site_calibration_dir, site_calibration
        ),
        "site_threshold_policy_overview": _site_threshold_policy_overview(
            config.site_threshold_policy_dir, site_threshold_policy
        ),
        "site_stratified_eval_overview": _site_stratified_eval_overview(
            config.site_stratified_eval_dir, site_stratified_eval
        ),
        "site_aggregation_overview": _site_aggregation_overview(
            config.site_aggregation_dir, site_aggregation
        ),
        "site_stability_overview": _generic_overview(
            config.site_stability_dir, site_stability
        ),
        "site_model_comparison_overview": _generic_overview(
            config.site_model_comparison_dir, site_model_comparison
        ),
        "site_aggregation_controls_overview": _generic_overview(
            config.site_aggregation_controls_dir, site_aggregation_controls
        ),
        "site_aggregation_threshold_policy_overview": _generic_overview(
            config.site_aggregation_threshold_policy_dir,
            site_aggregation_threshold_policy,
        ),
        "site_calibration_comparison_overview": _generic_overview(
            config.site_calibration_comparison_dir,
            site_calibration_comparison,
        ),
        "report_overview": _report_overview(config.report_dir, report_summary),
        "warnings": sorted(set(warnings)),
        "recommended_next_action": _recommended_next_action(config),
        "generated_files": {
            "json_summary": str(json_summary),
            "markdown_summary": str(markdown_summary),
        },
    }
    _write_json(json_summary, payload)
    markdown_summary.write_text(_render_markdown(payload), encoding="utf-8")

    return {
        "status": "ok",
        "outdir": str(outdir),
        "json_summary": str(json_summary),
        "markdown_summary": str(markdown_summary),
        "recommended_next_action": payload["recommended_next_action"],
        "warnings": payload["warnings"],
    }


def _load_json_from_dir(
    directory: Optional[str], filename: str, warnings: List[str]
) -> Optional[dict]:
    if directory is None:
        return None
    path = Path(directory) / filename
    try:
        payload = load_json_if_exists(path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        warnings.append(f"could_not_load_json:{path}:{exc}")
        return None
    if payload is None:
        warnings.append(f"missing_json:{path}")
    return payload


def _status_overview(config: RunSummaryConfig) -> dict:
    return {
        "simulation_present": config.sim_dir is not None,
        "simulation_audit_present": config.sim_audit_dir is not None,
        "alignment_present": config.align_dir is not None,
        "tensor_present": config.tensor_dir is not None,
        "dataset_present": config.dataset_dir is not None,
        "baseline_present": config.baseline_dir is not None,
        "baseline_calibration_present": config.baseline_calibration_dir is not None,
        "saturation_panel_present": config.saturation_panel_dir is not None,
        "merged_dataset_present": config.merged_dataset_dir is not None,
        "neural_present": config.neural_dir is not None,
        "neural_calibration_present": config.neural_calibration_dir is not None,
        "stratified_calibration_present": config.stratified_calibration_dir is not None,
        "threshold_policy_present": config.threshold_policy_dir is not None,
        "stratified_eval_present": config.stratified_eval_dir is not None,
        "neural_diagnostics_present": config.neural_diagnostics_dir is not None,
        "ablation_comparison_present": config.ablation_comparison_dir is not None,
        "label_signal_audit_present": config.label_signal_audit_dir is not None,
        "leakage_audit_present": config.leakage_audit_dir is not None,
        "stability_benchmark_present": config.stability_benchmark_dir is not None,
        "site_labels_present": config.site_label_dir is not None,
        "site_dataset_present": config.site_dataset_dir is not None,
        "site_leakage_audit_present": config.site_leakage_audit_dir is not None,
        "site_baseline_present": config.site_baseline_dir is not None,
        "site_neural_present": config.site_neural_dir is not None,
        "site_calibration_present": config.site_calibration_dir is not None,
        "site_threshold_policy_present": config.site_threshold_policy_dir is not None,
        "site_stratified_eval_present": config.site_stratified_eval_dir is not None,
        "site_aggregation_present": config.site_aggregation_dir is not None,
        "site_stability_present": config.site_stability_dir is not None,
        "site_model_comparison_present": config.site_model_comparison_dir is not None,
        "site_aggregation_controls_present": config.site_aggregation_controls_dir is not None,
        "site_aggregation_threshold_policy_present": config.site_aggregation_threshold_policy_dir is not None,
        "site_calibration_comparison_present": config.site_calibration_comparison_dir is not None,
        "report_present": config.report_dir is not None,
    }


def _dataset_overview(dataset_index: Optional[dict]) -> dict:
    if not dataset_index:
        return {}
    return {
        "n_rows": dataset_index.get("n_rows"),
        "n_families": dataset_index.get("n_families"),
        "methods": dataset_index.get("methods"),
        "split_counts_rows": dataset_index.get("split_counts_rows"),
        "split_counts_families": dataset_index.get("split_counts_families"),
        "positive_counts_by_split": dataset_index.get("positive_counts_by_split"),
    }


def _baseline_overview(meta: Optional[dict], metrics: Optional[dict]) -> dict:
    if not meta and not metrics:
        return {}
    return {
        "feature_columns": (meta or {}).get("feature_columns"),
        "metrics_by_split": (metrics or {}).get("metrics_by_split"),
        "warnings": (meta or {}).get("warnings", []),
    }


def _neural_overview(meta: Optional[dict], metrics: Optional[dict]) -> dict:
    if not meta and not metrics:
        return {}
    return {
        "model_class": (meta or {}).get("model_class"),
        "device_used": (meta or {}).get("device_used"),
        "epochs_completed": (meta or {}).get("epochs_completed"),
        "best_epoch": (meta or {}).get("best_epoch"),
        "stopped_early": (meta or {}).get("stopped_early"),
        "train_rows": (meta or {}).get("train_rows"),
        "val_rows": (meta or {}).get("val_rows"),
        "calib_rows": (meta or {}).get("calib_rows"),
        "test_rows": (meta or {}).get("test_rows"),
        "metrics_by_split": (metrics or {}).get("metrics_by_split"),
        "warnings": (meta or {}).get("warnings", []),
    }


def _calibration_overview(
    baseline_calibration: Optional[dict],
    baseline_metrics: Optional[dict],
    neural_calibration: Optional[dict],
    neural_metrics: Optional[dict],
) -> dict:
    return {
        "baseline": _single_calibration_overview(
            baseline_calibration, baseline_metrics
        ),
        "neural": _single_calibration_overview(neural_calibration, neural_metrics),
    }


def _single_calibration_overview(
    calibration: Optional[dict], metrics: Optional[dict]
) -> dict:
    if not calibration and not metrics:
        return {}
    return {
        "temperature": (calibration or {}).get("temperature"),
        "selected_threshold": (calibration or {}).get("selected_threshold"),
        "target_fdr": (calibration or {}).get("target_fdr"),
        "warnings": (calibration or {}).get("warnings", []),
        "metrics_by_split_calibrated": (metrics or {}).get(
            "metrics_by_split_calibrated"
        ),
    }


def _report_overview(report_dir: Optional[str], report_summary: Optional[dict]) -> dict:
    if report_dir is None:
        return {}
    report_path = Path(report_dir)
    sections = []
    if isinstance(report_summary, dict):
        sections = sorted((report_summary.get("sections") or {}).keys())
    return {
        "report_summary_json_exists": (report_path / "report_summary.json").exists(),
        "report_md_exists": (report_path / "report.md").exists(),
        "sections_included": sections,
    }


def _threshold_policy_overview(
    threshold_policy_dir: Optional[str], threshold_policy: Optional[dict]
) -> dict:
    if threshold_policy_dir is None:
        return {}
    policy_path = Path(threshold_policy_dir)
    profiles = (threshold_policy or {}).get("profiles") or {}
    selected_profiles = {}
    for name, profile in profiles.items():
        if not isinstance(profile, dict):
            continue
        selected_profiles[name] = {
            "selected_threshold": profile.get("selected_threshold"),
            "selection_metrics": profile.get("selection_metrics"),
            "warning": profile.get("warning"),
        }
    return {
        "threshold_profiles_json_exists": (
            policy_path / "threshold_profiles.json"
        ).exists(),
        "threshold_policy_md_exists": (policy_path / "threshold_policy.md").exists(),
        "model_name": (threshold_policy or {}).get("model_name"),
        "probability_used": (threshold_policy or {}).get("probability_used"),
        "selection_split": (threshold_policy or {}).get("selection_split"),
        "selected_profiles": selected_profiles,
        "warnings": (threshold_policy or {}).get("warnings", []),
        "recommended_interpretation_present": (
            policy_path / "threshold_policy.md"
        ).exists(),
    }


def _saturation_panel_overview(
    saturation_panel_dir: Optional[str], saturation_panel: Optional[dict]
) -> dict:
    if saturation_panel_dir is None:
        return {}
    panel_path = Path(saturation_panel_dir)
    return {
        "saturation_panel_json_exists": (
            panel_path / "saturation_panel.json"
        ).exists(),
        "saturation_panel_md_exists": (panel_path / "saturation_panel.md").exists(),
        "tiers": (saturation_panel or {}).get("tiers"),
        "n_families_per_tier": (saturation_panel or {}).get("n_families_per_tier"),
        "total_families_expected": (saturation_panel or {}).get(
            "total_families_expected"
        ),
        "tier_outputs": (saturation_panel or {}).get("tier_outputs"),
        "warnings": (saturation_panel or {}).get("warnings", []),
    }


def _merged_dataset_overview(
    merged_dataset_dir: Optional[str], merged_dataset: Optional[dict]
) -> dict:
    if merged_dataset_dir is None:
        return {}
    dataset_path = Path(merged_dataset_dir)
    return {
        "dataset_index_json_exists": (dataset_path / "dataset_index.json").exists(),
        "features_tsv_exists": (dataset_path / "features.tsv").exists(),
        "splits_tsv_exists": (dataset_path / "splits.tsv").exists(),
        "n_rows": (merged_dataset or {}).get("n_rows"),
        "n_families": (merged_dataset or {}).get("n_families"),
        "methods": (merged_dataset or {}).get("methods"),
        "saturation_tier_counts": (merged_dataset or {}).get(
            "saturation_tier_counts"
        ),
        "split_counts_rows": (merged_dataset or {}).get("split_counts_rows"),
        "split_counts_families": (merged_dataset or {}).get(
            "split_counts_families"
        ),
    }


def _stratified_eval_overview(
    stratified_eval_dir: Optional[str], stratified_eval: Optional[dict]
) -> dict:
    if stratified_eval_dir is None:
        return {}
    eval_path = Path(stratified_eval_dir)
    return {
        "stratified_eval_json_exists": (eval_path / "stratified_eval.json").exists(),
        "stratified_metrics_tsv_exists": (
            eval_path / "stratified_metrics.tsv"
        ).exists(),
        "stratified_eval_md_exists": (eval_path / "stratified_eval.md").exists(),
        "model_name": (stratified_eval or {}).get("model_name"),
        "profiles_evaluated": (stratified_eval or {}).get("profiles_evaluated"),
        "key_findings": (stratified_eval or {}).get("key_findings"),
        "warnings": (stratified_eval or {}).get("warnings", []),
        "generated_files": (stratified_eval or {}).get("generated_files"),
    }


def _stratified_calibration_overview(
    stratified_calibration_dir: Optional[str],
    stratified_calibration: Optional[dict],
    metrics: Optional[dict],
) -> dict:
    if stratified_calibration_dir is None:
        return {}
    calibration_path = Path(stratified_calibration_dir)
    return {
        "stratified_calibration_json_exists": (
            calibration_path / "stratified_calibration.json"
        ).exists(),
        "stratified_calibrated_predictions_tsv_exists": (
            calibration_path / "stratified_calibrated_predictions.tsv"
        ).exists(),
        "stratified_calibrated_metrics_json_exists": (
            calibration_path / "stratified_calibrated_metrics.json"
        ).exists(),
        "stratified_calibration_md_exists": (
            calibration_path / "stratified_calibration.md"
        ).exists(),
        "group_column": (stratified_calibration or {}).get("group_column"),
        "groups": sorted(((stratified_calibration or {}).get("groups") or {}).keys()),
        "warnings": (stratified_calibration or {}).get("warnings", []),
        "metrics_by_group": (metrics or {}).get("metrics_by_group"),
        "generated_files": (stratified_calibration or {}).get("generated_files"),
    }


def _neural_diagnostics_overview(
    neural_diagnostics_dir: Optional[str], neural_diagnostics: Optional[dict]
) -> dict:
    if neural_diagnostics_dir is None:
        return {}
    diag_path = Path(neural_diagnostics_dir)
    return {
        "neural_diagnostics_json_exists": (
            diag_path / "neural_diagnostics.json"
        ).exists(),
        "neural_probability_summary_tsv_exists": (
            diag_path / "neural_probability_summary.tsv"
        ).exists(),
        "neural_diagnostics_md_exists": (
            diag_path / "neural_diagnostics.md"
        ).exists(),
        "model_name": (neural_diagnostics or {}).get("model_name"),
        "metadata_summary": (neural_diagnostics or {}).get("metadata_summary"),
        "history_summary": (neural_diagnostics or {}).get("history_summary"),
        "warnings": (neural_diagnostics or {}).get("warnings", []),
    }


def _ablation_comparison_overview(
    ablation_comparison_dir: Optional[str], ablation_comparison: Optional[dict]
) -> dict:
    if ablation_comparison_dir is None:
        return {}
    compare_path = Path(ablation_comparison_dir)
    return {
        "ablation_comparison_json_exists": (
            compare_path / "ablation_comparison.json"
        ).exists(),
        "ablation_comparison_tsv_exists": (
            compare_path / "ablation_comparison.tsv"
        ).exists(),
        "ablation_comparison_md_exists": (
            compare_path / "ablation_comparison.md"
        ).exists(),
        "models": [
            model.get("model_name")
            for model in (ablation_comparison or {}).get("models", [])
            if isinstance(model, dict)
        ],
        "recommendation": (ablation_comparison or {}).get("recommendation"),
        "warnings": (ablation_comparison or {}).get("warnings", []),
    }


def _label_signal_audit_overview(
    label_signal_audit_dir: Optional[str], label_signal_audit: Optional[dict]
) -> dict:
    if label_signal_audit_dir is None:
        return {}
    audit_path = Path(label_signal_audit_dir)
    return {
        "label_signal_audit_json_exists": (
            audit_path / "label_signal_audit.json"
        ).exists(),
        "label_signal_features_tsv_exists": (
            audit_path / "label_signal_features.tsv"
        ).exists(),
        "label_signal_audit_md_exists": (
            audit_path / "label_signal_audit.md"
        ).exists(),
        "n_rows": (label_signal_audit or {}).get("n_rows"),
        "n_numeric_features": (label_signal_audit or {}).get("n_numeric_features"),
        "top_features_by_auroc_distance": (
            label_signal_audit or {}
        ).get("top_features_by_auroc_distance", [])[:5],
        "interpretation": (label_signal_audit or {}).get("interpretation"),
        "warnings": (label_signal_audit or {}).get("warnings", []),
    }


def _leakage_audit_overview(
    leakage_audit_dir: Optional[str], leakage_audit: Optional[dict]
) -> dict:
    if leakage_audit_dir is None:
        return {}
    audit_path = Path(leakage_audit_dir)
    return {
        "leakage_audit_json_exists": (audit_path / "leakage_audit.json").exists(),
        "leakage_columns_tsv_exists": (audit_path / "leakage_columns.tsv").exists(),
        "leakage_audit_md_exists": (audit_path / "leakage_audit.md").exists(),
        "status": (leakage_audit or {}).get("status"),
        "strict_leakage_columns_present": (
            leakage_audit or {}
        ).get("strict_leakage_columns_present"),
        "recommended_excluded_columns": (
            leakage_audit or {}
        ).get("recommended_excluded_columns"),
        "warnings": (leakage_audit or {}).get("warnings", []),
    }


def _stability_benchmark_overview(
    stability_benchmark_dir: Optional[str], stability_benchmark: Optional[dict]
) -> dict:
    if stability_benchmark_dir is None:
        return {}
    benchmark_path = Path(stability_benchmark_dir)
    aggregate = (stability_benchmark or {}).get("aggregate_summary") or {}
    return {
        "stability_benchmark_json_exists": (
            benchmark_path / "stability_benchmark.json"
        ).exists(),
        "stability_results_tsv_exists": (
            benchmark_path / "stability_results.tsv"
        ).exists(),
        "stability_benchmark_md_exists": (
            benchmark_path / "stability_benchmark.md"
        ).exists(),
        "best_preset_by_mean_val_auroc": aggregate.get(
            "best_preset_by_mean_val_auroc"
        ),
        "best_preset_by_mean_test_auroc": aggregate.get(
            "best_preset_by_mean_test_auroc"
        ),
        "instability_warnings": aggregate.get("instability_warnings", []),
        "warnings": (stability_benchmark or {}).get("warnings", []),
    }


def _site_label_overview(site_label_dir: Optional[str], site_labels: Optional[dict]) -> dict:
    if site_label_dir is None:
        return {}
    path = Path(site_label_dir)
    return {
        "site_oracle_summary_json_exists": (path / "site_oracle_summary.json").exists(),
        "site_oracle_labels_tsv_exists": (path / "site_oracle_labels.tsv").exists(),
        "site_oracle_labels_md_exists": (path / "site_oracle_labels.md").exists(),
        "n_site_records": (site_labels or {}).get("n_site_records"),
        "n_positive_sites": (site_labels or {}).get("n_positive_sites"),
        "positive_site_fraction": (site_labels or {}).get("positive_site_fraction"),
        "warnings": (site_labels or {}).get("warnings", []),
    }


def _site_dataset_overview(site_dataset_dir: Optional[str], site_dataset: Optional[dict]) -> dict:
    if site_dataset_dir is None:
        return {}
    path = Path(site_dataset_dir)
    return {
        "site_dataset_index_json_exists": (path / "site_dataset_index.json").exists(),
        "site_features_tsv_exists": (path / "site_features.tsv").exists(),
        "site_splits_tsv_exists": (path / "site_splits.tsv").exists(),
        "n_site_rows": (site_dataset or {}).get("n_site_rows"),
        "n_positive_sites": (site_dataset or {}).get("n_positive_sites"),
        "n_negative_sites": (site_dataset or {}).get("n_negative_sites"),
        "saturation_tier_counts": (site_dataset or {}).get("saturation_tier_counts"),
        "method_counts": (site_dataset or {}).get("method_counts"),
        "warnings": (site_dataset or {}).get("warnings", []),
    }


def _site_leakage_overview(
    site_leakage_dir: Optional[str], site_leakage: Optional[dict]
) -> dict:
    if site_leakage_dir is None:
        return {}
    path = Path(site_leakage_dir)
    return {
        "site_leakage_audit_json_exists": (path / "site_leakage_audit.json").exists(),
        "site_leakage_columns_tsv_exists": (path / "site_leakage_columns.tsv").exists(),
        "site_leakage_audit_md_exists": (path / "site_leakage_audit.md").exists(),
        "status": (site_leakage or {}).get("status"),
        "forbidden_columns_present": (site_leakage or {}).get(
            "forbidden_columns_present"
        ),
        "near_perfect_univariate_columns": (site_leakage or {}).get(
            "near_perfect_univariate_columns"
        ),
        "warnings": (site_leakage or {}).get("warnings", []),
    }


def _site_baseline_overview(
    site_baseline_dir: Optional[str],
    site_baseline_meta: Optional[dict],
    site_baseline_metrics: Optional[dict],
) -> dict:
    if site_baseline_dir is None:
        return {}
    path = Path(site_baseline_dir)
    return {
        "site_baseline_model_exists": (path / "site_baseline_model.npz").exists(),
        "site_baseline_model_meta_json_exists": (
            path / "site_baseline_model_meta.json"
        ).exists(),
        "site_baseline_predictions_tsv_exists": (
            path / "site_baseline_predictions.tsv"
        ).exists(),
        "site_baseline_metrics_json_exists": (
            path / "site_baseline_metrics.json"
        ).exists(),
        "feature_columns": (site_baseline_meta or {}).get("feature_columns"),
        "metrics_by_split": (site_baseline_metrics or {}).get("metrics_by_split"),
        "metrics_by_saturation_tier": (site_baseline_metrics or {}).get(
            "metrics_by_saturation_tier"
        ),
        "warnings": (site_baseline_meta or {}).get("warnings", []),
    }


def _site_neural_overview(
    site_neural_dir: Optional[str], meta: Optional[dict], metrics: Optional[dict]
) -> dict:
    if site_neural_dir is None:
        return {}
    path = Path(site_neural_dir)
    return {
        "site_neural_checkpoint_exists": (path / "site_neural_checkpoint.pt").exists(),
        "site_neural_predictions_tsv_exists": (path / "site_neural_predictions.tsv").exists(),
        "n_features": (meta or {}).get("n_features"),
        "best_epoch": (meta or {}).get("best_epoch"),
        "metrics_by_split": (metrics or {}).get("metrics_by_split"),
    }


def _site_calibration_overview(site_calibration_dir: Optional[str], payload: Optional[dict]) -> dict:
    if site_calibration_dir is None:
        return {}
    return {
        "temperature": (payload or {}).get("temperature"),
        "selected_threshold": (payload or {}).get("selected_threshold"),
        "target_fdr": (payload or {}).get("target_fdr"),
        "warnings": (payload or {}).get("warnings", []),
    }


def _site_threshold_policy_overview(policy_dir: Optional[str], payload: Optional[dict]) -> dict:
    if policy_dir is None:
        return {}
    return {
        "selection_split": (payload or {}).get("selection_split"),
        "target_fdr": (payload or {}).get("target_fdr"),
        "profiles": sorted(((payload or {}).get("profiles") or {}).keys()),
        "warnings": (payload or {}).get("warnings", []),
    }


def _site_stratified_eval_overview(eval_dir: Optional[str], payload: Optional[dict]) -> dict:
    if eval_dir is None:
        return {}
    return {
        "profiles_evaluated": (payload or {}).get("profiles_evaluated"),
        "warnings": (payload or {}).get("warnings", []),
    }


def _site_aggregation_overview(aggregation_dir: Optional[str], payload: Optional[dict]) -> dict:
    if aggregation_dir is None:
        return {}
    default = ((payload or {}).get("gene_level_metrics_default") or {}).get("all", {})
    return {
        "n_family_method_rows": (payload or {}).get("n_family_method_rows"),
        "default_score": (payload or {}).get("default_score"),
        "default_auroc": default.get("auroc"),
        "interpretation": (payload or {}).get("interpretation"),
    }


def _generic_overview(directory: Optional[str], payload: Optional[dict]) -> dict:
    if directory is None:
        return {}
    return {
        "directory": directory,
        "available": payload is not None,
        "recommendation": (payload or {}).get("recommendation"),
        "warnings": (payload or {}).get("warnings", []),
        "generated_files": (payload or {}).get("generated_files"),
        "observed": (payload or {}).get("observed"),
    }


def _recommended_next_action(config: RunSummaryConfig) -> str:
    if (
        config.site_model_comparison_dir is not None
        and config.site_aggregation_controls_dir is not None
        and config.site_aggregation_threshold_policy_dir is not None
    ):
        return "Inspect site robustness summaries, aggregation null controls, and calibration comparison before research-alpha release claims."
    if config.site_neural_dir is not None and config.site_calibration_dir is None:
        return "Run babappa calibrate-site-neural."
    if config.site_neural_dir is not None and config.site_threshold_policy_dir is None:
        return "Run babappa site-threshold-policy."
    if config.site_neural_dir is not None and config.site_aggregation_dir is None:
        return "Run babappa aggregate-sites to derive gene-level support from site evidence."
    if config.site_aggregation_dir is not None:
        return "Inspect site-level neural calibration, threshold profiles, stratified metrics, and site-to-gene aggregation before full-scale training."
    if config.site_label_dir is not None and config.site_dataset_dir is None:
        return "Run babappa build-site-dataset."
    if config.site_dataset_dir is not None and config.site_leakage_audit_dir is None:
        return "Run babappa audit-site-leakage before training site-level models."
    if config.site_dataset_dir is not None and config.site_baseline_dir is None:
        return "Run babappa train-site-baseline."
    if config.site_baseline_dir is not None:
        return "Inspect site-level baseline metrics before neural branch-site modeling."
    if (
        config.merged_dataset_dir is not None
        and config.neural_dir is None
    ):
        return "Run babappa train-neural-saturation on the merged saturation dataset."
    if (
        config.saturation_panel_dir is not None
        and config.merged_dataset_dir is None
    ):
        return "Run babappa merge-datasets to create a unified saturation dataset."
    if (
        config.neural_dir is not None
        and config.stratified_calibration_dir is None
        and config.merged_dataset_dir is not None
    ):
        return "Run babappa calibrate-stratified for saturation-tier calibration."
    if (
        config.neural_dir is not None
        and config.stratified_calibration_dir is not None
        and config.threshold_policy_dir is None
    ):
        return "Run babappa threshold-policy to select saturation-aware operating points."
    if (
        config.neural_dir is not None
        and config.stratified_calibration_dir is not None
        and config.threshold_policy_dir is not None
        and config.stratified_eval_dir is None
    ):
        return "Run babappa stratified-eval to inspect saturation and method behavior."
    if (
        config.neural_dir is not None
        and config.stratified_calibration_dir is not None
        and config.threshold_policy_dir is not None
        and config.stratified_eval_dir is not None
    ):
        return (
            "Inspect saturation-aware calibration, operating points, and stratified "
            "performance before larger datasets."
        )
    if (
        config.neural_dir is not None
        and config.neural_calibration_dir is not None
        and config.threshold_policy_dir is not None
        and config.stratified_eval_dir is None
    ):
        return "Run babappa stratified-eval to inspect saturation and method behavior."
    if (
        config.neural_dir is not None
        and config.neural_calibration_dir is not None
        and config.threshold_policy_dir is None
    ):
        return "Run babappa threshold-policy to select operating-point profiles."
    if (
        config.neural_dir is not None
        and config.neural_calibration_dir is not None
        and config.threshold_policy_dir is not None
        and config.stratified_eval_dir is not None
    ):
        return (
            "Inspect stratified saturation/method performance before moving to larger datasets."
        )
    if config.neural_dir is not None and config.neural_calibration_dir is not None:
        return (
            "Inspect calibrated neural metrics and proceed to larger-scale run "
            "if validation is satisfactory."
        )
    if config.neural_dir is not None and config.neural_calibration_dir is None:
        return "Run babappa calibrate-neural."
    if config.dataset_dir is not None and config.neural_dir is None:
        return "Run babappa train-neural."
    if config.tensor_dir is not None and config.dataset_dir is None:
        return "Run babappa index-dataset."
    return "Complete the missing upstream workflow stages."


def _render_markdown(payload: dict) -> str:
    lines: List[str] = [f"# {payload['title']}", ""]
    _append_stage_status(lines, payload.get("status_overview", {}))
    _append_overview(lines, "Dataset overview", payload.get("dataset_overview", {}))
    _append_overview(lines, "Baseline overview", payload.get("baseline_overview", {}))
    _append_overview(lines, "Neural overview", payload.get("neural_overview", {}))
    _append_overview(
        lines, "Calibration overview", payload.get("calibration_overview", {})
    )
    _append_overview(
        lines,
        "Stratified calibration overview",
        payload.get("stratified_calibration_overview", {}),
    )
    _append_overview(
        lines,
        "Saturation panel overview",
        payload.get("saturation_panel_overview", {}),
    )
    _append_overview(
        lines,
        "Merged dataset overview",
        payload.get("merged_dataset_overview", {}),
    )
    _append_overview(
        lines,
        "Threshold policy overview",
        payload.get("threshold_policy_overview", {}),
    )
    _append_overview(
        lines,
        "Stratified evaluation overview",
        payload.get("stratified_eval_overview", {}),
    )
    _append_overview(
        lines,
        "Neural diagnostics overview",
        payload.get("neural_diagnostics_overview", {}),
    )
    _append_overview(
        lines,
        "Ablation comparison overview",
        payload.get("ablation_comparison_overview", {}),
    )
    _append_overview(
        lines,
        "Label-signal audit overview",
        payload.get("label_signal_audit_overview", {}),
    )
    _append_overview(
        lines,
        "Leakage audit overview",
        payload.get("leakage_audit_overview", {}),
    )
    _append_overview(
        lines,
        "Stability benchmark overview",
        payload.get("stability_benchmark_overview", {}),
    )
    _append_overview(
        lines,
        "Site-label overview",
        payload.get("site_label_overview", {}),
    )
    _append_overview(
        lines,
        "Site dataset overview",
        payload.get("site_dataset_overview", {}),
    )
    _append_overview(
        lines,
        "Site leakage overview",
        payload.get("site_leakage_overview", {}),
    )
    _append_overview(
        lines,
        "Site baseline overview",
        payload.get("site_baseline_overview", {}),
    )
    _append_overview(
        lines,
        "Site neural overview",
        payload.get("site_neural_overview", {}),
    )
    _append_overview(
        lines,
        "Site calibration overview",
        payload.get("site_calibration_overview", {}),
    )
    _append_overview(
        lines,
        "Site threshold policy overview",
        payload.get("site_threshold_policy_overview", {}),
    )
    _append_overview(
        lines,
        "Site stratified evaluation overview",
        payload.get("site_stratified_eval_overview", {}),
    )
    _append_overview(
        lines,
        "Site-to-gene aggregation overview",
        payload.get("site_aggregation_overview", {}),
    )
    _append_overview(
        lines,
        "Site stability overview",
        payload.get("site_stability_overview", {}),
    )
    _append_overview(
        lines,
        "Site model comparison overview",
        payload.get("site_model_comparison_overview", {}),
    )
    _append_overview(
        lines,
        "Aggregation controls overview",
        payload.get("site_aggregation_controls_overview", {}),
    )
    _append_overview(
        lines,
        "Aggregation threshold-policy overview",
        payload.get("site_aggregation_threshold_policy_overview", {}),
    )
    _append_overview(
        lines,
        "Site calibration comparison overview",
        payload.get("site_calibration_comparison_overview", {}),
    )
    lines.extend(["## Warnings", ""])
    warnings = payload.get("warnings") or []
    if warnings:
        lines.extend(f"- {warning}" for warning in warnings)
    else:
        lines.append("- none")
    lines.extend(["", "## Recommended next action", ""])
    lines.append(payload.get("recommended_next_action", "NA"))
    lines.append("")
    _append_limitations(lines)
    return "\n".join(lines).rstrip() + "\n"


def _append_stage_status(lines: List[str], status: dict) -> None:
    lines.extend(["## Stage status", ""])
    for key in sorted(status):
        lines.append(f"- `{key}`: {status[key]}")
    lines.append("")


def _append_overview(lines: List[str], title: str, overview: dict) -> None:
    lines.extend([f"## {title}", ""])
    if not overview:
        lines.extend(["No data supplied.", ""])
        return
    for key, value in overview.items():
        lines.append(f"- `{key}`: {_format_value(value)}")
    lines.append("")


def _append_limitations(lines: List[str]) -> None:
    lines.extend(
        [
            "## Limitations",
            "",
            "- Current neural model is gene-level, not branch-site.",
            "- Current internal alignment methods are identity/codon_dropout scaffolds.",
            "- Current simulator is lightweight and not yet final codon-likelihood/saturation simulator.",
            "- Calibration is only reliable when calibration split is sufficiently large and contains both classes.",
            "- Threshold-policy profiles are operating-point aids, not proof of final biological performance.",
            "- Stratified evaluation is required before interpreting saturation robustness.",
            "- Multi-saturation merged datasets are still gene-level benchmark substrates.",
            "- Site-level oracle learning is supervised method development, not real-data inference.",
            "",
        ]
    )


def _format_value(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, dict):
        if not value:
            return "none"
        return json.dumps(value, sort_keys=True)
    if isinstance(value, list):
        return ", ".join(str(item) for item in value) if value else "none"
    return str(value)


def _supplied_inputs(config: RunSummaryConfig) -> Dict[str, str]:
    labels = [
        "sim_dir",
        "sim_audit_dir",
        "align_dir",
        "tensor_dir",
        "dataset_dir",
        "baseline_dir",
        "baseline_calibration_dir",
        "saturation_panel_dir",
        "merged_dataset_dir",
        "neural_dir",
        "neural_calibration_dir",
        "stratified_calibration_dir",
        "threshold_policy_dir",
        "stratified_eval_dir",
        "neural_diagnostics_dir",
        "ablation_comparison_dir",
        "label_signal_audit_dir",
        "leakage_audit_dir",
        "stability_benchmark_dir",
        "site_label_dir",
        "site_dataset_dir",
        "site_leakage_audit_dir",
        "site_baseline_dir",
        "site_neural_dir",
        "site_calibration_dir",
        "site_threshold_policy_dir",
        "site_stratified_eval_dir",
        "site_aggregation_dir",
        "site_stability_dir",
        "site_model_comparison_dir",
        "site_aggregation_controls_dir",
        "site_aggregation_threshold_policy_dir",
        "site_calibration_comparison_dir",
        "report_dir",
    ]
    return {
        label: str(getattr(config, label))
        for label in labels
        if getattr(config, label) is not None
    }


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
