"""Consolidated human-readable and machine-readable BABAPPA reports."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from babappa import __version__

REPORT_VERSION = __version__


@dataclass(frozen=True)
class ReportConfig:
    """Configuration for a consolidated BABAPPA run report."""

    sim_dir: Optional[str] = None
    sim_audit_dir: Optional[str] = None
    align_dir: Optional[str] = None
    tensor_dir: Optional[str] = None
    dataset_dir: Optional[str] = None
    baseline_dir: Optional[str] = None
    calibration_dir: Optional[str] = None
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
    outdir: str = "babappa_report"
    title: str = "BABAPPA run report"

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
    """Load a JSON object if the path exists, otherwise return None."""
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON file is not an object: {path}")
    return payload


def summarize_tsv(path: Path, max_rows: int = 5) -> dict:
    """Summarize a TSV file without loading external dataframe libraries."""
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
        preview_rows = []
        n_rows = 0
        for row in reader:
            if n_rows < max_rows:
                preview_rows.append(dict(row))
            n_rows += 1

    return {
        "exists": True,
        "n_rows": n_rows,
        "fieldnames": fieldnames,
        "preview_rows": preview_rows,
    }


def build_report(config: ReportConfig) -> dict:
    """Collect available BABAPPA artifacts and write report_summary.json."""
    outdir = Path(config.outdir)
    json_report = outdir / "report_summary.json"
    markdown_report = outdir / "report.md"
    warnings: List[str] = []
    sections: Dict[str, dict] = {}

    if config.sim_dir is not None:
        sections["simulation"] = _simulation_section(Path(config.sim_dir), warnings)
    if config.sim_audit_dir is not None:
        sections["simulation_audit"] = _simulation_audit_section(
            Path(config.sim_audit_dir), warnings
        )
    if config.align_dir is not None:
        sections["alignment"] = _alignment_section(Path(config.align_dir), warnings)
    if config.tensor_dir is not None:
        sections["tensorization"] = _tensor_section(Path(config.tensor_dir), warnings)
    if config.dataset_dir is not None:
        sections["dataset_index"] = _dataset_section(Path(config.dataset_dir), warnings)
    if config.baseline_dir is not None:
        sections["baseline_model"] = _baseline_section(Path(config.baseline_dir), warnings)
    if config.calibration_dir is not None:
        sections["calibration"] = _calibration_section(
            Path(config.calibration_dir), warnings
        )
    if config.saturation_panel_dir is not None:
        sections["saturation_panel"] = _saturation_panel_section(
            Path(config.saturation_panel_dir), warnings
        )
    if config.merged_dataset_dir is not None:
        sections["merged_dataset"] = _merged_dataset_section(
            Path(config.merged_dataset_dir), warnings
        )
    if config.neural_dir is not None:
        sections["neural_model"] = _neural_section(Path(config.neural_dir), warnings)
    if config.neural_calibration_dir is not None:
        sections["neural_calibration"] = _neural_calibration_section(
            Path(config.neural_calibration_dir), warnings
        )
    if config.stratified_calibration_dir is not None:
        sections["stratified_calibration"] = _stratified_calibration_section(
            Path(config.stratified_calibration_dir), warnings
        )
    if config.threshold_policy_dir is not None:
        sections["threshold_policy"] = _threshold_policy_section(
            Path(config.threshold_policy_dir), warnings
        )
    if config.stratified_eval_dir is not None:
        sections["stratified_eval"] = _stratified_eval_section(
            Path(config.stratified_eval_dir), warnings
        )
    if config.neural_diagnostics_dir is not None:
        sections["neural_diagnostics"] = _neural_diagnostics_section(
            Path(config.neural_diagnostics_dir), warnings
        )
    if config.ablation_comparison_dir is not None:
        sections["ablation_comparison"] = _ablation_comparison_section(
            Path(config.ablation_comparison_dir), warnings
        )
    if config.label_signal_audit_dir is not None:
        sections["label_signal_audit"] = _label_signal_audit_section(
            Path(config.label_signal_audit_dir), warnings
        )
    if config.leakage_audit_dir is not None:
        sections["leakage_audit"] = _leakage_audit_section(
            Path(config.leakage_audit_dir), warnings
        )
    if config.stability_benchmark_dir is not None:
        sections["stability_benchmark"] = _stability_benchmark_section(
            Path(config.stability_benchmark_dir), warnings
        )
    if config.site_label_dir is not None:
        sections["site_labels"] = _site_label_section(
            Path(config.site_label_dir), warnings
        )
    if config.site_dataset_dir is not None:
        sections["site_dataset"] = _site_dataset_section(
            Path(config.site_dataset_dir), warnings
        )
    if config.site_leakage_audit_dir is not None:
        sections["site_leakage_audit"] = _site_leakage_audit_section(
            Path(config.site_leakage_audit_dir), warnings
        )
    if config.site_baseline_dir is not None:
        sections["site_baseline"] = _site_baseline_section(
            Path(config.site_baseline_dir), warnings
        )
    if config.site_neural_dir is not None:
        sections["site_neural"] = _site_neural_section(Path(config.site_neural_dir), warnings)
    if config.site_calibration_dir is not None:
        sections["site_calibration"] = _site_calibration_section(
            Path(config.site_calibration_dir), warnings
        )
    if config.site_threshold_policy_dir is not None:
        sections["site_threshold_policy"] = _site_threshold_policy_section(
            Path(config.site_threshold_policy_dir), warnings
        )
    if config.site_stratified_eval_dir is not None:
        sections["site_stratified_eval"] = _site_stratified_eval_section(
            Path(config.site_stratified_eval_dir), warnings
        )
    if config.site_aggregation_dir is not None:
        sections["site_aggregation"] = _site_aggregation_section(
            Path(config.site_aggregation_dir), warnings
        )
    if config.site_stability_dir is not None:
        sections["site_stability"] = _json_section(
            Path(config.site_stability_dir), "site_stability_benchmark.json", warnings
        )
    if config.site_model_comparison_dir is not None:
        sections["site_model_comparison"] = _json_section(
            Path(config.site_model_comparison_dir), "site_model_comparison.json", warnings
        )
    if config.site_aggregation_controls_dir is not None:
        sections["site_aggregation_controls"] = _json_section(
            Path(config.site_aggregation_controls_dir), "site_aggregation_controls.json", warnings
        )
    if config.site_aggregation_threshold_policy_dir is not None:
        sections["site_aggregation_threshold_policy"] = _json_section(
            Path(config.site_aggregation_threshold_policy_dir),
            "aggregation_threshold_profiles.json",
            warnings,
        )
    if config.site_calibration_comparison_dir is not None:
        sections["site_calibration_comparison"] = _json_section(
            Path(config.site_calibration_comparison_dir),
            "site_calibration_comparison.json",
            warnings,
        )

    payload = {
        "report_version": REPORT_VERSION,
        "title": config.title,
        "inputs": _supplied_inputs(config),
        "sections": sections,
        "warnings": sorted(set(warnings)),
        "generated_files": {
            "json_report": str(json_report),
            "markdown_report": str(markdown_report),
        },
    }
    _write_json(json_report, payload)
    return payload


def generate_report(config: ReportConfig) -> dict:
    """Generate machine-readable JSON and human-readable Markdown reports."""
    payload = build_report(config)
    outdir = Path(config.outdir)
    markdown_report = outdir / "report.md"
    markdown_report.write_text(_render_markdown(payload), encoding="utf-8")
    return {
        "status": "ok",
        "outdir": str(outdir),
        "json_report": str(outdir / "report_summary.json"),
        "markdown_report": str(markdown_report),
        "warnings": payload["warnings"],
    }


def _simulation_section(sim_dir: Path, warnings: List[str]) -> dict:
    manifest_path = sim_dir / "manifest.json"
    manifest = _load_json_with_warning(manifest_path, warnings)
    return {
        "directory": str(sim_dir),
        "manifest_path": str(manifest_path),
        "manifest": manifest,
    }


def _simulation_audit_section(sim_audit_dir: Path, warnings: List[str]) -> dict:
    summary_path = sim_audit_dir / "dataset_summary.json"
    family_audit_path = sim_audit_dir / "family_audit.tsv"
    return {
        "directory": str(sim_audit_dir),
        "dataset_summary_path": str(summary_path),
        "dataset_summary": _load_json_with_warning(summary_path, warnings),
        "family_audit_path": str(family_audit_path),
        "family_audit": summarize_tsv(family_audit_path),
    }


def _alignment_section(align_dir: Path, warnings: List[str]) -> dict:
    manifest_path = align_dir / "alignment_manifest.json"
    return {
        "directory": str(align_dir),
        "alignment_manifest_path": str(manifest_path),
        "alignment_manifest": _load_json_with_warning(manifest_path, warnings),
    }


def _tensor_section(tensor_dir: Path, warnings: List[str]) -> dict:
    manifest_path = tensor_dir / "tensor_manifest.json"
    audit_path = tensor_dir / "tensor_audit.tsv"
    return {
        "directory": str(tensor_dir),
        "tensor_manifest_path": str(manifest_path),
        "tensor_manifest": _load_json_with_warning(manifest_path, warnings),
        "tensor_audit_path": str(audit_path),
        "tensor_audit": summarize_tsv(audit_path),
    }


def _dataset_section(dataset_dir: Path, warnings: List[str]) -> dict:
    index_path = dataset_dir / "dataset_index.json"
    features_path = dataset_dir / "features.tsv"
    splits_path = dataset_dir / "splits.tsv"
    return {
        "directory": str(dataset_dir),
        "dataset_index_path": str(index_path),
        "dataset_index": _load_json_with_warning(index_path, warnings),
        "features_path": str(features_path),
        "features": summarize_tsv(features_path),
        "splits_path": str(splits_path),
        "splits": summarize_tsv(splits_path),
    }


def _baseline_section(baseline_dir: Path, warnings: List[str]) -> dict:
    meta_path = baseline_dir / "baseline_model_meta.json"
    metrics_path = baseline_dir / "baseline_metrics.json"
    predictions_path = baseline_dir / "baseline_predictions.tsv"
    return {
        "directory": str(baseline_dir),
        "model_meta_path": str(meta_path),
        "model_meta": _load_json_with_warning(meta_path, warnings),
        "metrics_path": str(metrics_path),
        "metrics": _load_json_with_warning(metrics_path, warnings),
        "predictions_path": str(predictions_path),
        "predictions": summarize_tsv(predictions_path),
    }


def _calibration_section(calibration_dir: Path, warnings: List[str]) -> dict:
    calibration_path = calibration_dir / "baseline_calibration.json"
    metrics_path = calibration_dir / "baseline_calibrated_metrics.json"
    predictions_path = calibration_dir / "baseline_calibrated_predictions.tsv"
    return {
        "directory": str(calibration_dir),
        "calibration_path": str(calibration_path),
        "calibration": _load_json_with_warning(calibration_path, warnings),
        "metrics_path": str(metrics_path),
        "metrics": _load_json_with_warning(metrics_path, warnings),
        "predictions_path": str(predictions_path),
        "predictions": summarize_tsv(predictions_path),
    }


def _saturation_panel_section(panel_dir: Path, warnings: List[str]) -> dict:
    panel_path = panel_dir / "saturation_panel.json"
    markdown_path = panel_dir / "saturation_panel.md"
    return {
        "directory": str(panel_dir),
        "saturation_panel_path": str(panel_path),
        "saturation_panel": _load_json_with_warning(panel_path, warnings),
        "markdown_path": str(markdown_path),
        "markdown_exists": markdown_path.exists(),
    }


def _merged_dataset_section(dataset_dir: Path, warnings: List[str]) -> dict:
    index_path = dataset_dir / "dataset_index.json"
    features_path = dataset_dir / "features.tsv"
    splits_path = dataset_dir / "splits.tsv"
    return {
        "directory": str(dataset_dir),
        "dataset_index_path": str(index_path),
        "dataset_index": _load_json_with_warning(index_path, warnings),
        "features_path": str(features_path),
        "features": summarize_tsv(features_path),
        "splits_path": str(splits_path),
        "splits": summarize_tsv(splits_path),
    }


def _neural_section(neural_dir: Path, warnings: List[str]) -> dict:
    meta_path = neural_dir / "neural_model_meta.json"
    metrics_path = neural_dir / "neural_metrics.json"
    history_path = neural_dir / "logs" / "neural_training_history.tsv"
    predictions_path = neural_dir / "predictions" / "neural_predictions.tsv"
    return {
        "directory": str(neural_dir),
        "model_meta_path": str(meta_path),
        "model_meta": _load_json_with_warning(meta_path, warnings),
        "metrics_path": str(metrics_path),
        "metrics": _load_json_with_warning(metrics_path, warnings),
        "history_path": str(history_path),
        "history": summarize_tsv(history_path),
        "predictions_path": str(predictions_path),
        "predictions": summarize_tsv(predictions_path),
    }


def _neural_calibration_section(
    neural_calibration_dir: Path, warnings: List[str]
) -> dict:
    calibration_path = neural_calibration_dir / "neural_calibration.json"
    metrics_path = neural_calibration_dir / "neural_calibrated_metrics.json"
    predictions_path = neural_calibration_dir / "neural_calibrated_predictions.tsv"
    return {
        "directory": str(neural_calibration_dir),
        "calibration_path": str(calibration_path),
        "calibration": _load_json_with_warning(calibration_path, warnings),
        "metrics_path": str(metrics_path),
        "metrics": _load_json_with_warning(metrics_path, warnings),
        "predictions_path": str(predictions_path),
        "predictions": summarize_tsv(predictions_path),
    }


def _stratified_calibration_section(
    stratified_calibration_dir: Path, warnings: List[str]
) -> dict:
    calibration_path = stratified_calibration_dir / "stratified_calibration.json"
    metrics_path = stratified_calibration_dir / "stratified_calibrated_metrics.json"
    predictions_path = stratified_calibration_dir / "stratified_calibrated_predictions.tsv"
    markdown_path = stratified_calibration_dir / "stratified_calibration.md"
    return {
        "directory": str(stratified_calibration_dir),
        "calibration_path": str(calibration_path),
        "calibration": _load_json_with_warning(calibration_path, warnings),
        "metrics_path": str(metrics_path),
        "metrics": _load_json_with_warning(metrics_path, warnings),
        "predictions_path": str(predictions_path),
        "predictions": summarize_tsv(predictions_path),
        "markdown_path": str(markdown_path),
        "markdown_exists": markdown_path.exists(),
    }


def _threshold_policy_section(
    threshold_policy_dir: Path, warnings: List[str]
) -> dict:
    profiles_path = threshold_policy_dir / "threshold_profiles.json"
    profiles_tsv_path = threshold_policy_dir / "threshold_profiles.tsv"
    metrics_tsv_path = threshold_policy_dir / "threshold_profile_metrics.tsv"
    curve_tsv_path = threshold_policy_dir / "threshold_policy_curve.tsv"
    markdown_path = threshold_policy_dir / "threshold_policy.md"
    return {
        "directory": str(threshold_policy_dir),
        "profiles_path": str(profiles_path),
        "profiles": _load_json_with_warning(profiles_path, warnings),
        "profiles_tsv_path": str(profiles_tsv_path),
        "profiles_tsv": summarize_tsv(profiles_tsv_path),
        "profile_metrics_tsv_path": str(metrics_tsv_path),
        "profile_metrics_tsv": summarize_tsv(metrics_tsv_path),
        "curve_tsv_path": str(curve_tsv_path),
        "curve_tsv": summarize_tsv(curve_tsv_path),
        "markdown_path": str(markdown_path),
        "markdown_exists": markdown_path.exists(),
    }


def _stratified_eval_section(stratified_eval_dir: Path, warnings: List[str]) -> dict:
    eval_path = stratified_eval_dir / "stratified_eval.json"
    metrics_path = stratified_eval_dir / "stratified_metrics.tsv"
    markdown_path = stratified_eval_dir / "stratified_eval.md"
    return {
        "directory": str(stratified_eval_dir),
        "stratified_eval_path": str(eval_path),
        "stratified_eval": _load_json_with_warning(eval_path, warnings),
        "stratified_metrics_path": str(metrics_path),
        "stratified_metrics": summarize_tsv(metrics_path),
        "markdown_path": str(markdown_path),
        "markdown_exists": markdown_path.exists(),
    }


def _neural_diagnostics_section(
    neural_diagnostics_dir: Path, warnings: List[str]
) -> dict:
    json_path = neural_diagnostics_dir / "neural_diagnostics.json"
    tsv_path = neural_diagnostics_dir / "neural_probability_summary.tsv"
    markdown_path = neural_diagnostics_dir / "neural_diagnostics.md"
    return {
        "directory": str(neural_diagnostics_dir),
        "diagnostics_path": str(json_path),
        "diagnostics": _load_json_with_warning(json_path, warnings),
        "summary_tsv_path": str(tsv_path),
        "summary_tsv": summarize_tsv(tsv_path),
        "markdown_path": str(markdown_path),
        "markdown_exists": markdown_path.exists(),
    }


def _ablation_comparison_section(
    ablation_comparison_dir: Path, warnings: List[str]
) -> dict:
    json_path = ablation_comparison_dir / "ablation_comparison.json"
    tsv_path = ablation_comparison_dir / "ablation_comparison.tsv"
    markdown_path = ablation_comparison_dir / "ablation_comparison.md"
    return {
        "directory": str(ablation_comparison_dir),
        "comparison_path": str(json_path),
        "comparison": _load_json_with_warning(json_path, warnings),
        "comparison_tsv_path": str(tsv_path),
        "comparison_tsv": summarize_tsv(tsv_path),
        "markdown_path": str(markdown_path),
        "markdown_exists": markdown_path.exists(),
    }


def _label_signal_audit_section(
    label_signal_audit_dir: Path, warnings: List[str]
) -> dict:
    json_path = label_signal_audit_dir / "label_signal_audit.json"
    tsv_path = label_signal_audit_dir / "label_signal_features.tsv"
    markdown_path = label_signal_audit_dir / "label_signal_audit.md"
    return {
        "directory": str(label_signal_audit_dir),
        "audit_path": str(json_path),
        "audit": _load_json_with_warning(json_path, warnings),
        "features_tsv_path": str(tsv_path),
        "features_tsv": summarize_tsv(tsv_path),
        "markdown_path": str(markdown_path),
        "markdown_exists": markdown_path.exists(),
    }


def _leakage_audit_section(leakage_audit_dir: Path, warnings: List[str]) -> dict:
    json_path = leakage_audit_dir / "leakage_audit.json"
    tsv_path = leakage_audit_dir / "leakage_columns.tsv"
    markdown_path = leakage_audit_dir / "leakage_audit.md"
    return {
        "directory": str(leakage_audit_dir),
        "audit_path": str(json_path),
        "audit": _load_json_with_warning(json_path, warnings),
        "columns_tsv_path": str(tsv_path),
        "columns_tsv": summarize_tsv(tsv_path),
        "markdown_path": str(markdown_path),
        "markdown_exists": markdown_path.exists(),
    }


def _stability_benchmark_section(
    stability_benchmark_dir: Path, warnings: List[str]
) -> dict:
    json_path = stability_benchmark_dir / "stability_benchmark.json"
    tsv_path = stability_benchmark_dir / "stability_results.tsv"
    markdown_path = stability_benchmark_dir / "stability_benchmark.md"
    return {
        "directory": str(stability_benchmark_dir),
        "benchmark_path": str(json_path),
        "benchmark": _load_json_with_warning(json_path, warnings),
        "results_tsv_path": str(tsv_path),
        "results_tsv": summarize_tsv(tsv_path),
        "markdown_path": str(markdown_path),
        "markdown_exists": markdown_path.exists(),
    }


def _site_label_section(site_label_dir: Path, warnings: List[str]) -> dict:
    summary_path = site_label_dir / "site_oracle_summary.json"
    labels_path = site_label_dir / "site_oracle_labels.tsv"
    markdown_path = site_label_dir / "site_oracle_labels.md"
    return {
        "directory": str(site_label_dir),
        "summary_path": str(summary_path),
        "summary": _load_json_with_warning(summary_path, warnings),
        "labels_path": str(labels_path),
        "labels": summarize_tsv(labels_path),
        "markdown_path": str(markdown_path),
        "markdown_exists": markdown_path.exists(),
    }


def _site_dataset_section(site_dataset_dir: Path, warnings: List[str]) -> dict:
    index_path = site_dataset_dir / "site_dataset_index.json"
    features_path = site_dataset_dir / "site_features.tsv"
    splits_path = site_dataset_dir / "site_splits.tsv"
    markdown_path = site_dataset_dir / "site_dataset.md"
    return {
        "directory": str(site_dataset_dir),
        "index_path": str(index_path),
        "index": _load_json_with_warning(index_path, warnings),
        "features_path": str(features_path),
        "features": summarize_tsv(features_path),
        "splits_path": str(splits_path),
        "splits": summarize_tsv(splits_path),
        "markdown_path": str(markdown_path),
        "markdown_exists": markdown_path.exists(),
    }


def _site_leakage_audit_section(site_leakage_dir: Path, warnings: List[str]) -> dict:
    json_path = site_leakage_dir / "site_leakage_audit.json"
    columns_path = site_leakage_dir / "site_leakage_columns.tsv"
    markdown_path = site_leakage_dir / "site_leakage_audit.md"
    return {
        "directory": str(site_leakage_dir),
        "audit_path": str(json_path),
        "audit": _load_json_with_warning(json_path, warnings),
        "columns_path": str(columns_path),
        "columns": summarize_tsv(columns_path),
        "markdown_path": str(markdown_path),
        "markdown_exists": markdown_path.exists(),
    }


def _site_baseline_section(site_baseline_dir: Path, warnings: List[str]) -> dict:
    meta_path = site_baseline_dir / "site_baseline_model_meta.json"
    metrics_path = site_baseline_dir / "site_baseline_metrics.json"
    predictions_path = site_baseline_dir / "site_baseline_predictions.tsv"
    return {
        "directory": str(site_baseline_dir),
        "model_meta_path": str(meta_path),
        "model_meta": _load_json_with_warning(meta_path, warnings),
        "metrics_path": str(metrics_path),
        "metrics": _load_json_with_warning(metrics_path, warnings),
        "predictions_path": str(predictions_path),
        "predictions": summarize_tsv(predictions_path),
    }


def _site_neural_section(site_neural_dir: Path, warnings: List[str]) -> dict:
    meta_path = site_neural_dir / "site_neural_model_meta.json"
    metrics_path = site_neural_dir / "site_neural_metrics.json"
    predictions_path = site_neural_dir / "site_neural_predictions.tsv"
    return {
        "directory": str(site_neural_dir),
        "model_meta": _load_json_with_warning(meta_path, warnings),
        "metrics": _load_json_with_warning(metrics_path, warnings),
        "predictions": summarize_tsv(predictions_path),
    }


def _site_calibration_section(site_calibration_dir: Path, warnings: List[str]) -> dict:
    return {
        "directory": str(site_calibration_dir),
        "calibration": _load_json_with_warning(
            site_calibration_dir / "site_calibration.json", warnings
        ),
        "metrics": _load_json_with_warning(
            site_calibration_dir / "site_calibrated_metrics.json", warnings
        ),
        "predictions": summarize_tsv(site_calibration_dir / "site_calibrated_predictions.tsv"),
    }


def _site_threshold_policy_section(policy_dir: Path, warnings: List[str]) -> dict:
    return {
        "directory": str(policy_dir),
        "profiles": _load_json_with_warning(policy_dir / "site_threshold_profiles.json", warnings),
        "profiles_tsv": summarize_tsv(policy_dir / "site_threshold_profiles.tsv"),
        "metrics_tsv": summarize_tsv(policy_dir / "site_threshold_profile_metrics.tsv"),
    }


def _site_stratified_eval_section(eval_dir: Path, warnings: List[str]) -> dict:
    return {
        "directory": str(eval_dir),
        "eval": _load_json_with_warning(eval_dir / "site_stratified_eval.json", warnings),
        "metrics_tsv": summarize_tsv(eval_dir / "site_stratified_metrics.tsv"),
    }


def _site_aggregation_section(aggregation_dir: Path, warnings: List[str]) -> dict:
    return {
        "directory": str(aggregation_dir),
        "metrics": _load_json_with_warning(
            aggregation_dir / "site_to_gene_metrics.json", warnings
        ),
        "predictions": summarize_tsv(aggregation_dir / "site_to_gene_predictions.tsv"),
    }


def _json_section(directory: Path, filename: str, warnings: List[str]) -> dict:
    return {
        "directory": str(directory),
        "payload": _load_json_with_warning(directory / filename, warnings),
    }


def _load_json_with_warning(path: Path, warnings: List[str]) -> Optional[dict]:
    try:
        payload = load_json_if_exists(path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        warnings.append(f"could_not_load_json:{path}:{exc}")
        return None
    if payload is None:
        warnings.append(f"missing_json:{path}")
    return payload


def _render_markdown(payload: dict) -> str:
    lines: List[str] = [f"# {payload['title']}", ""]
    sections = payload.get("sections", {})
    _append_inputs(lines, payload.get("inputs", {}))
    if "simulation" in sections or "simulation_audit" in sections:
        _append_simulation_summary(lines, sections)
    if "alignment" in sections:
        _append_alignment_summary(lines, sections["alignment"])
    if "tensorization" in sections:
        _append_tensor_summary(lines, sections["tensorization"])
    if "dataset_index" in sections:
        _append_dataset_summary(lines, sections["dataset_index"])
    if "baseline_model" in sections:
        _append_baseline_summary(lines, sections["baseline_model"])
    if "calibration" in sections:
        _append_calibration_summary(lines, sections["calibration"])
    if "saturation_panel" in sections:
        _append_saturation_panel_summary(lines, sections["saturation_panel"])
    if "merged_dataset" in sections:
        _append_merged_dataset_summary(lines, sections["merged_dataset"])
    if "neural_model" in sections:
        _append_neural_summary(lines, sections["neural_model"])
    if "neural_calibration" in sections:
        _append_neural_calibration_summary(lines, sections["neural_calibration"])
    if "stratified_calibration" in sections:
        _append_stratified_calibration_summary(
            lines, sections["stratified_calibration"]
        )
    if "threshold_policy" in sections:
        _append_threshold_policy_summary(lines, sections["threshold_policy"])
    if "stratified_eval" in sections:
        _append_stratified_eval_summary(lines, sections["stratified_eval"])
    if "neural_diagnostics" in sections:
        _append_neural_diagnostics_summary(lines, sections["neural_diagnostics"])
    if "ablation_comparison" in sections:
        _append_ablation_comparison_summary(lines, sections["ablation_comparison"])
    if "label_signal_audit" in sections:
        _append_label_signal_audit_summary(lines, sections["label_signal_audit"])
    if "leakage_audit" in sections:
        _append_leakage_audit_summary(lines, sections["leakage_audit"])
    if "stability_benchmark" in sections:
        _append_stability_benchmark_summary(lines, sections["stability_benchmark"])
    if "site_labels" in sections:
        _append_site_label_summary(lines, sections["site_labels"])
    if "site_dataset" in sections:
        _append_site_dataset_summary(lines, sections["site_dataset"])
    if "site_leakage_audit" in sections:
        _append_site_leakage_summary(lines, sections["site_leakage_audit"])
    if "site_baseline" in sections:
        _append_site_baseline_summary(lines, sections["site_baseline"])
    if "site_neural" in sections:
        _append_site_neural_summary(lines, sections["site_neural"])
    if "site_calibration" in sections:
        _append_site_calibration_summary(lines, sections["site_calibration"])
    if "site_threshold_policy" in sections:
        _append_site_threshold_policy_summary(lines, sections["site_threshold_policy"])
    if "site_stratified_eval" in sections:
        _append_site_stratified_eval_summary(lines, sections["site_stratified_eval"])
    if "site_aggregation" in sections:
        _append_site_aggregation_summary(lines, sections["site_aggregation"])
    if "site_stability" in sections:
        _append_generic_site_json_summary(lines, "Site stability overview", sections["site_stability"])
    if "site_model_comparison" in sections:
        _append_generic_site_json_summary(lines, "Site model comparison overview", sections["site_model_comparison"])
    if "site_aggregation_controls" in sections:
        _append_generic_site_json_summary(lines, "Aggregation controls overview", sections["site_aggregation_controls"])
    if "site_aggregation_threshold_policy" in sections:
        _append_generic_site_json_summary(lines, "Aggregation threshold-policy overview", sections["site_aggregation_threshold_policy"])
    if "site_calibration_comparison" in sections:
        _append_generic_site_json_summary(lines, "Site calibration comparison overview", sections["site_calibration_comparison"])
    _append_generated_files(lines, payload.get("generated_files", {}))
    _append_limitations(lines)
    return "\n".join(lines).rstrip() + "\n"


def _append_inputs(lines: List[str], inputs: dict) -> None:
    lines.extend(["## Inputs", ""])
    for label, directory in inputs.items():
        lines.append(f"- `{label}`: `{directory}`")
    lines.append("")


def _append_simulation_summary(lines: List[str], sections: dict) -> None:
    lines.extend(["## Simulation summary", ""])
    simulation = sections.get("simulation", {})
    manifest = simulation.get("manifest") or {}
    audit = sections.get("simulation_audit", {}).get("dataset_summary") or {}
    lines.append(f"- Simulator version: {_value(manifest.get('simulator_version'))}")
    lines.append(f"- Number of families: {_value(manifest.get('n_families'))}")
    lines.append(
        f"- Positive family count: {_value(audit.get('positive_family_count'))}"
    )
    lines.append(
        f"- Saturation tier counts: {_format_mapping(audit.get('saturation_tier_counts'))}"
    )
    lines.append("")


def _append_alignment_summary(lines: List[str], section: dict) -> None:
    lines.extend(["## Alignment summary", ""])
    manifest = section.get("alignment_manifest") or {}
    lines.append(f"- Methods: {_format_list(manifest.get('methods'))}")
    lines.append(f"- Number of families: {_value(manifest.get('n_families'))}")
    lines.append("")


def _append_tensor_summary(lines: List[str], section: dict) -> None:
    lines.extend(["## Tensorization summary", ""])
    manifest = section.get("tensor_manifest") or {}
    audit = section.get("tensor_audit") or {}
    lines.append(f"- Number of families: {_value(manifest.get('n_families'))}")
    lines.append(f"- Methods: {_format_list(manifest.get('methods'))}")
    lines.append(
        f"- Include gap channel: {_value(manifest.get('include_gap_channel'))}"
    )
    _append_tsv_preview(lines, "Tensor audit preview", audit)


def _append_dataset_summary(lines: List[str], section: dict) -> None:
    lines.extend(["## Dataset index and splits", ""])
    index = section.get("dataset_index") or {}
    lines.append(f"- Number of rows: {_value(index.get('n_rows'))}")
    lines.append(f"- Number of families: {_value(index.get('n_families'))}")
    lines.append(f"- Methods: {_format_list(index.get('methods'))}")
    lines.append(
        f"- Split counts: {_format_mapping(index.get('split_counts_rows'))}"
    )
    lines.append(
        f"- Positive counts by split: {_format_mapping(index.get('positive_counts_by_split'))}"
    )
    lines.append("")


def _append_baseline_summary(lines: List[str], section: dict) -> None:
    lines.extend(["## Baseline model", ""])
    meta = section.get("model_meta") or {}
    metrics = (section.get("metrics") or {}).get("metrics_by_split") or {}
    lines.append(f"- Feature columns: {_format_list(meta.get('feature_columns'))}")
    lines.append(f"- Training rows: {_value(meta.get('train_rows'))}")
    lines.append(f"- Warnings: {_format_list(meta.get('warnings'))}")
    _append_metrics_table(lines, metrics)


def _append_calibration_summary(lines: List[str], section: dict) -> None:
    lines.extend(["## Calibration", ""])
    calibration = section.get("calibration") or {}
    metrics = (section.get("metrics") or {}).get("metrics_by_split_calibrated") or {}
    lines.append(
        f"- Calibration method: {_value(calibration.get('calibration_method'))}"
    )
    lines.append(f"- Temperature: {_value(calibration.get('temperature'))}")
    lines.append(
        f"- Selected threshold: {_value(calibration.get('selected_threshold'))}"
    )
    lines.append(f"- Target FDR: {_value(calibration.get('target_fdr'))}")
    lines.append(f"- Calibration warnings: {_format_list(calibration.get('warnings'))}")
    _append_metrics_table(lines, metrics)


def _append_saturation_panel_summary(lines: List[str], section: dict) -> None:
    lines.extend(["## Saturation panel", ""])
    panel = section.get("saturation_panel") or {}
    lines.append(f"- Tiers: {_format_list(panel.get('tiers'))}")
    lines.append(
        f"- Families per tier: {_value(panel.get('n_families_per_tier'))}"
    )
    lines.append(
        f"- Total expected families: {_value(panel.get('total_families_expected'))}"
    )
    lines.append(f"- Warnings: {_format_list(panel.get('warnings'))}")
    lines.append("")


def _append_merged_dataset_summary(lines: List[str], section: dict) -> None:
    lines.extend(["## Merged dataset", ""])
    index = section.get("dataset_index") or {}
    lines.append(f"- Number of rows: {_value(index.get('n_rows'))}")
    lines.append(f"- Number of families: {_value(index.get('n_families'))}")
    lines.append(f"- Methods: {_format_list(index.get('methods'))}")
    lines.append(
        f"- Saturation tier counts: {_format_mapping(index.get('saturation_tier_counts'))}"
    )
    lines.append(
        f"- Split counts: {_format_mapping(index.get('split_counts_rows'))}"
    )
    lines.append("")


def _append_neural_summary(lines: List[str], section: dict) -> None:
    lines.extend(["## Neural model", ""])
    meta = section.get("model_meta") or {}
    metrics = (section.get("metrics") or {}).get("metrics_by_split") or {}
    lines.append(f"- Model class: {_value(meta.get('model_class'))}")
    lines.append(f"- Device used: {_value(meta.get('device_used'))}")
    lines.append(f"- Epochs completed: {_value(meta.get('epochs_completed'))}")
    lines.append(f"- Best epoch: {_value(meta.get('best_epoch'))}")
    lines.append(f"- Stopped early: {_value(meta.get('stopped_early'))}")
    lines.append(f"- Train rows: {_value(meta.get('train_rows'))}")
    lines.append(f"- Val rows: {_value(meta.get('val_rows'))}")
    lines.append(f"- Calib rows: {_value(meta.get('calib_rows'))}")
    lines.append(f"- Test rows: {_value(meta.get('test_rows'))}")
    lines.append(f"- Warnings: {_format_list(meta.get('warnings'))}")
    lines.append(
        "- Note: Current neural model is gene-level and not the final branch-site BABAPPA architecture."
    )
    _append_metrics_table(lines, metrics)


def _append_neural_calibration_summary(lines: List[str], section: dict) -> None:
    lines.extend(["## Neural calibration", ""])
    calibration = section.get("calibration") or {}
    metrics = (section.get("metrics") or {}).get("metrics_by_split_calibrated") or {}
    lines.append(
        f"- Calibration method: {_value(calibration.get('calibration_method'))}"
    )
    lines.append(f"- Temperature: {_value(calibration.get('temperature'))}")
    lines.append(
        f"- Selected threshold: {_value(calibration.get('selected_threshold'))}"
    )
    lines.append(f"- Target FDR: {_value(calibration.get('target_fdr'))}")
    lines.append(f"- Calibration warnings: {_format_list(calibration.get('warnings'))}")
    _append_metrics_table(lines, metrics)


def _append_stratified_calibration_summary(lines: List[str], section: dict) -> None:
    lines.extend(["## Stratified calibration", ""])
    calibration = section.get("calibration") or {}
    metrics_by_group = (section.get("metrics") or {}).get("metrics_by_group") or {}
    lines.append(f"- Group column: {_value(calibration.get('group_column'))}")
    lines.append(
        f"- Probability column: {_value(calibration.get('probability_column'))}"
    )
    lines.append(f"- Target FDR: {_value(calibration.get('target_fdr'))}")
    lines.append(
        f"- Groups: {_format_list(sorted((calibration.get('groups') or {}).keys()))}"
    )
    lines.append(f"- Warnings: {_format_list(calibration.get('warnings'))}")
    lines.extend(
        [
            "",
            "| Group | n | AUROC | Precision | Recall | F1 |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for group, metrics in sorted(metrics_by_group.items()):
        if not isinstance(metrics, dict):
            continue
        lines.append(
            "| {group} | {n} | {auroc} | {precision} | {recall} | {f1} |".format(
                group=group,
                n=_value(metrics.get("n")),
                auroc=_format_float(metrics.get("auroc")),
                precision=_format_float(metrics.get("precision")),
                recall=_format_float(metrics.get("recall")),
                f1=_format_float(metrics.get("f1")),
            )
        )
    lines.append("")


def _append_threshold_policy_summary(lines: List[str], section: dict) -> None:
    lines.extend(["## Threshold policy", ""])
    profiles_json = section.get("profiles") or {}
    profiles = profiles_json.get("profiles") or {}
    lines.append(f"- Model name: {_value(profiles_json.get('model_name'))}")
    lines.append(f"- Probability used: {_value(profiles_json.get('probability_used'))}")
    lines.append(f"- Selection split: {_value(profiles_json.get('selection_split'))}")
    lines.append(f"- Target FDR: {_value(profiles_json.get('target_fdr'))}")
    lines.append(f"- Warnings: {_format_list(profiles_json.get('warnings'))}")
    lines.extend(
        [
            "",
            "| Profile | Selected threshold | Selection FDR | Selection precision | Selection recall |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for profile_name in [
        "default_0_5",
        "strict_fdr",
        "max_f1",
        "max_mcc",
        "balanced_youden",
        "high_precision",
        "high_recall",
    ]:
        profile = profiles.get(profile_name)
        if not isinstance(profile, dict):
            continue
        metrics = profile.get("selection_metrics") or {}
        lines.append(
            "| {profile} | {threshold} | {fdr} | {precision} | {recall} |".format(
                profile=profile_name,
                threshold=_format_float(profile.get("selected_threshold")),
                fdr=_format_float(metrics.get("empirical_fdr")),
                precision=_format_float(metrics.get("precision")),
                recall=_format_float(metrics.get("recall")),
            )
        )
    lines.append("")


def _append_stratified_eval_summary(lines: List[str], section: dict) -> None:
    lines.extend(["## Stratified evaluation", ""])
    payload = section.get("stratified_eval") or {}
    findings = payload.get("key_findings") or {}
    lines.append(f"- Model name: {_value(payload.get('model_name'))}")
    lines.append(f"- Probability column: {_value(payload.get('probability_column'))}")
    lines.append(
        f"- Profiles evaluated: {_format_list([p.get('profile') for p in payload.get('profiles_evaluated', []) if isinstance(p, dict)])}"
    )
    lines.append(f"- Warnings: {_format_list(payload.get('warnings'))}")
    lines.append(
        f"- Best saturation tier by AUROC: {_format_mapping(findings.get('best_saturation_tier_by_auroc'))}"
    )
    lines.append(
        f"- Worst saturation tier by AUROC: {_format_mapping(findings.get('worst_saturation_tier_by_auroc'))}"
    )
    lines.append(
        f"- Best method by AUROC: {_format_mapping(findings.get('best_method_by_auroc'))}"
    )
    lines.append(
        f"- Worst method by AUROC: {_format_mapping(findings.get('worst_method_by_auroc'))}"
    )
    lines.append("")


def _append_neural_diagnostics_summary(lines: List[str], section: dict) -> None:
    lines.extend(["## Neural diagnostics", ""])
    payload = section.get("diagnostics") or {}
    metadata = payload.get("metadata_summary") or {}
    history = payload.get("history_summary") or {}
    lines.append(f"- Model name: {_value(payload.get('model_name'))}")
    lines.append(f"- Architecture: {_value(metadata.get('architecture'))}")
    lines.append(f"- Training preset: {_value(metadata.get('training_preset'))}")
    lines.append(f"- Group weighting: {_value(metadata.get('group_weighting'))}")
    lines.append(f"- Sampler: {_value(metadata.get('sampler'))}")
    lines.append(f"- Epochs completed: {_value(history.get('epochs_completed'))}")
    lines.append(f"- Best epoch: {_value(history.get('best_epoch'))}")
    lines.append(f"- Warnings: {_format_list(payload.get('warnings'))}")
    lines.append("")


def _append_ablation_comparison_summary(lines: List[str], section: dict) -> None:
    lines.extend(["## Ablation comparison", ""])
    payload = section.get("comparison") or {}
    recommendation = payload.get("recommendation") or {}
    lines.append(
        f"- Models compared: {_format_list([m.get('model_name') for m in payload.get('models', []) if isinstance(m, dict)])}"
    )
    lines.append(f"- Recommended model: {_value(recommendation.get('best_model'))}")
    lines.append(f"- Recommendation: {_value(recommendation.get('text'))}")
    lines.append(f"- Warnings: {_format_list(payload.get('warnings'))}")
    lines.append("")


def _append_label_signal_audit_summary(lines: List[str], section: dict) -> None:
    lines.extend(["## Label-signal audit", ""])
    payload = section.get("audit") or {}
    lines.append(f"- Rows: {_value(payload.get('n_rows'))}")
    lines.append(f"- Numeric features: {_value(payload.get('n_numeric_features'))}")
    top = payload.get("top_features_by_auroc_distance") or []
    if top:
        lines.append(
            "- Top AUROC-distance feature: {feature} (AUROC {auroc})".format(
                feature=_value(top[0].get("feature")),
                auroc=_format_float(top[0].get("auroc")),
            )
        )
    lines.append(f"- Interpretation: {_value(payload.get('interpretation'))}")
    lines.append(f"- Warnings: {_format_list(payload.get('warnings'))}")
    lines.append("")


def _append_leakage_audit_summary(lines: List[str], section: dict) -> None:
    lines.extend(["## Leakage audit", ""])
    payload = section.get("audit") or {}
    lines.append(f"- Status: {_value(payload.get('status'))}")
    lines.append(
        "- Strict leakage columns: {value}".format(
            value=_format_mapping(payload.get("strict_leakage_columns_present"))
        )
    )
    lines.append(
        f"- Recommended exclusions: {_format_list(payload.get('recommended_excluded_columns'))}"
    )
    lines.append(f"- Warnings: {_format_list(payload.get('warnings'))}")
    lines.append("")


def _append_stability_benchmark_summary(lines: List[str], section: dict) -> None:
    lines.extend(["## Stability benchmark", ""])
    payload = section.get("benchmark") or {}
    aggregate = payload.get("aggregate_summary") or {}
    lines.append(
        f"- Best validation preset: {_value(aggregate.get('best_preset_by_mean_val_auroc'))}"
    )
    lines.append(
        f"- Best test preset: {_value(aggregate.get('best_preset_by_mean_test_auroc'))}"
    )
    lines.append(
        f"- Instability warnings: {_format_list(aggregate.get('instability_warnings'))}"
    )
    lines.append(f"- Warnings: {_format_list(payload.get('warnings'))}")
    lines.append("")


def _append_site_label_summary(lines: List[str], section: dict) -> None:
    lines.extend(["## Site-label overview", ""])
    payload = section.get("summary") or {}
    lines.append(f"- Site records: {_value(payload.get('n_site_records'))}")
    lines.append(f"- Positive sites: {_value(payload.get('n_positive_sites'))}")
    lines.append(
        f"- Positive-site fraction: {_value(payload.get('positive_site_fraction'))}"
    )
    lines.append(f"- Warnings: {_format_list(payload.get('warnings'))}")
    lines.append(
        "- Note: Oracle site labels are supervised targets only, not predictive features."
    )
    lines.append("")


def _append_site_dataset_summary(lines: List[str], section: dict) -> None:
    lines.extend(["## Site dataset overview", ""])
    payload = section.get("index") or {}
    lines.append(f"- Site rows: {_value(payload.get('n_site_rows'))}")
    lines.append(f"- Positive sites: {_value(payload.get('n_positive_sites'))}")
    lines.append(f"- Negative sites: {_value(payload.get('n_negative_sites'))}")
    lines.append(
        f"- Saturation tier counts: {_format_mapping(payload.get('saturation_tier_counts'))}"
    )
    lines.append(f"- Method counts: {_format_mapping(payload.get('method_counts'))}")
    lines.append(f"- Warnings: {_format_list(payload.get('warnings'))}")
    lines.append("")


def _append_site_leakage_summary(lines: List[str], section: dict) -> None:
    lines.extend(["## Site leakage overview", ""])
    payload = section.get("audit") or {}
    lines.append(f"- Status: {_value(payload.get('status'))}")
    lines.append(
        f"- Forbidden columns: {_format_list(payload.get('forbidden_columns_present'))}"
    )
    lines.append(
        f"- Near-perfect predictors: {_format_list([item.get('column') for item in payload.get('near_perfect_univariate_columns', []) if isinstance(item, dict)])}"
    )
    lines.append(f"- Warnings: {_format_list(payload.get('warnings'))}")
    lines.append("")


def _append_site_baseline_summary(lines: List[str], section: dict) -> None:
    lines.extend(["## Site baseline overview", ""])
    meta = section.get("model_meta") or {}
    metrics = (section.get("metrics") or {}).get("metrics_by_split") or {}
    lines.append(f"- Feature columns: {_format_list(meta.get('feature_columns'))}")
    lines.append(f"- Train rows: {_value(meta.get('train_rows'))}")
    lines.append(f"- Warnings: {_format_list(meta.get('warnings'))}")
    _append_metrics_table(lines, metrics)


def _append_site_neural_summary(lines: List[str], section: dict) -> None:
    lines.extend(["## Site neural overview", ""])
    meta = section.get("model_meta") or {}
    metrics = (section.get("metrics") or {}).get("metrics_by_split") or {}
    lines.append(f"- Features: {_value(meta.get('n_features'))}")
    lines.append(f"- Best epoch: {_value(meta.get('best_epoch'))}")
    lines.append(f"- Monitor metric: {_value(meta.get('monitor_metric'))}")
    lines.append(f"- Note: {_value(meta.get('note'))}")
    _append_metrics_table(lines, metrics)


def _append_site_calibration_summary(lines: List[str], section: dict) -> None:
    lines.extend(["## Site calibration overview", ""])
    payload = section.get("calibration") or {}
    lines.append(f"- Method: {_value(payload.get('calibration_method'))}")
    lines.append(f"- Temperature: {_value(payload.get('temperature'))}")
    lines.append(f"- Selected threshold: {_value(payload.get('selected_threshold'))}")
    lines.append(f"- Warnings: {_format_list(payload.get('warnings'))}")
    lines.append("")


def _append_site_threshold_policy_summary(lines: List[str], section: dict) -> None:
    lines.extend(["## Site threshold policy overview", ""])
    payload = section.get("profiles") or {}
    lines.append(f"- Selection split: {_value(payload.get('selection_split'))}")
    lines.append(f"- Target FDR: {_value(payload.get('target_fdr'))}")
    lines.append(f"- Profiles: {_format_list(sorted((payload.get('profiles') or {}).keys()))}")
    lines.append(f"- Warnings: {_format_list(payload.get('warnings'))}")
    lines.append("")


def _append_site_stratified_eval_summary(lines: List[str], section: dict) -> None:
    lines.extend(["## Site stratified evaluation overview", ""])
    payload = section.get("eval") or {}
    lines.append(f"- Profiles: {_format_list(payload.get('profiles_evaluated'))}")
    lines.append(f"- Warnings: {_format_list(payload.get('warnings'))}")
    lines.append("")


def _append_site_aggregation_summary(lines: List[str], section: dict) -> None:
    lines.extend(["## Site-to-gene aggregation overview", ""])
    payload = section.get("metrics") or {}
    default = (payload.get("gene_level_metrics_default") or {}).get("all", {})
    lines.append(f"- Family-method rows: {_value(payload.get('n_family_method_rows'))}")
    lines.append(f"- Default score: {_value(payload.get('default_score'))}")
    lines.append(f"- Default AUROC: {_format_float(default.get('auroc'))}")
    lines.append(f"- Interpretation: {_value(payload.get('interpretation'))}")
    lines.append("")


def _append_generic_site_json_summary(lines: List[str], title: str, section: dict) -> None:
    lines.extend([f"## {title}", ""])
    payload = section.get("payload") or {}
    lines.append(f"- Directory: `{section.get('directory')}`")
    if payload.get("recommendation"):
        lines.append(f"- Recommendation: {_value(payload.get('recommendation'))}")
    if payload.get("warnings") is not None:
        lines.append(f"- Warnings: {_format_list(payload.get('warnings'))}")
    if payload.get("observed"):
        lines.append(f"- Observed: {_value(payload.get('observed'))}")
    if payload.get("aggregate_summary") is not None:
        lines.append("- Aggregate summary: available")
    lines.append("")


def _append_generated_files(lines: List[str], generated_files: dict) -> None:
    lines.extend(["## Files generated", ""])
    for label, path in generated_files.items():
        lines.append(f"- `{label}`: `{path}`")
    lines.append("")


def _append_limitations(lines: List[str]) -> None:
    lines.extend(
        [
            "## Limitations",
            "",
            "- This report may describe the current lightweight BABAPPA implementation.",
            "- The current baseline is not the final branch-site deep-learning model.",
            "- Current neural model is gene-level, not branch-site.",
            "- The internal identity/codon_dropout alignment methods are scaffolds, not replacements for external biological aligners.",
            "- Current alignment methods are still identity/codon_dropout scaffolds unless external aligners are later used.",
            "- Calibration is only as reliable as the calibration split.",
            "- The simulator is not yet the final saturation-aware codon-likelihood simulator described in the manuscript.",
            "- Threshold-policy profiles should be selected according to the scientific use case and calibration quality.",
            "- Stratified evaluation currently reflects simulator-level saturation labels and scaffold alignment methods.",
            "- Multi-saturation panels are benchmark substrates, not final biological validation datasets.",
            "- Site-level oracle learning is supervised method development, not real-data inference.",
            "",
        ]
    )


def _append_metrics_table(lines: List[str], metrics_by_split: dict) -> None:
    if not metrics_by_split:
        lines.extend(["", "No metrics were available.", ""])
        return
    lines.extend(
        [
            "",
            "| Split | n | Accuracy | AUROC | Precision | Recall |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for split in ["train", "val", "calib", "test", "all"]:
        metrics = metrics_by_split.get(split)
        if not isinstance(metrics, dict):
            continue
        lines.append(
            "| {split} | {n} | {accuracy} | {auroc} | {precision} | {recall} |".format(
                split=split,
                n=_value(metrics.get("n")),
                accuracy=_format_float(metrics.get("accuracy")),
                auroc=_format_float(metrics.get("auroc")),
                precision=_format_float(metrics.get("precision")),
                recall=_format_float(metrics.get("recall")),
            )
        )
    lines.append("")


def _append_tsv_preview(lines: List[str], title: str, summary: dict) -> None:
    lines.append("")
    lines.append(f"### {title}")
    lines.append("")
    if not summary.get("exists"):
        lines.extend(["TSV file was not available.", ""])
        return
    lines.append(f"- Rows: {_value(summary.get('n_rows'))}")
    lines.append(f"- Columns: {_format_list(summary.get('fieldnames'))}")
    lines.append("")


def _supplied_inputs(config: ReportConfig) -> Dict[str, str]:
    labels = [
        "sim_dir",
        "sim_audit_dir",
        "align_dir",
        "tensor_dir",
        "dataset_dir",
        "baseline_dir",
        "calibration_dir",
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
    ]
    return {
        label: str(getattr(config, label))
        for label in labels
        if getattr(config, label) is not None
    }


def _format_float(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def _value(value: Any) -> str:
    if value is None:
        return "NA"
    return str(value)


def _format_list(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, list):
        return ", ".join(str(item) for item in value) if value else "none"
    return str(value)


def _format_mapping(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, dict):
        if not value:
            return "none"
        return ", ".join(f"{key}={val}" for key, val in sorted(value.items()))
    return str(value)


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
