"""Branch-site calibration and threshold-policy wrappers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from babappa import __version__
from babappa.calibration.baseline import (
    brier_score,
    expected_calibration_error,
    fit_temperature_grid,
    select_threshold_by_fdr,
    temperature_scale_probs,
)
from babappa.calibration.threshold_policy import ThresholdPolicyConfig, build_threshold_policy
from babappa.datasets.index import read_tsv, write_tsv
from babappa.site.baseline import _compute_binary_metrics

BRANCH_CALIBRATION_VERSION = __version__
CALIBRATED_FIELDNAMES = [
    "branch_site_id",
    "family_id",
    "method",
    "saturation_tier",
    "split",
    "branch_id",
    "site_index_zero",
    "y_branch_site",
    "y_site",
    "gene_label",
    "prob_positive_raw",
    "prob_positive_calibrated",
    "pred_label_calibrated",
    "correct_calibrated",
]


@dataclass(frozen=True)
class BranchSiteCalibrationConfig:
    model_dir: str
    outdir: str
    target_fdr: float = 0.10
    calibration_method: str = "temperature"
    threshold_grid_size: int = 501
    min_threshold: float = 0.0
    max_threshold: float = 1.0
    n_bins: int = 20

    def __post_init__(self) -> None:
        model_path = Path(self.model_dir)
        if not model_path.exists():
            raise ValueError(f"model_dir does not exist: {model_path}")
        if not (model_path / "branch_site_neural_predictions.tsv").exists():
            raise ValueError(f"model_dir is missing branch_site_neural_predictions.tsv: {model_path}")
        if self.calibration_method not in {"none", "temperature"}:
            raise ValueError("calibration_method must be none or temperature")
        if not 0 <= self.target_fdr <= 1:
            raise ValueError("target_fdr must be between 0 and 1")
        Path(self.outdir).mkdir(parents=True, exist_ok=True)


def calibrate_branch_site_model(config: BranchSiteCalibrationConfig) -> dict:
    """Calibrate branch-site neural probabilities."""
    rows = read_tsv(Path(config.model_dir) / "branch_site_neural_predictions.tsv")
    if not rows:
        raise ValueError("branch_site_neural_predictions.tsv contains no rows")
    y = np.array([int(float(row["y_branch_site"])) for row in rows], dtype=np.int32)
    probs = np.array([float(row["prob_positive"]) for row in rows], dtype=np.float64)
    calib_mask = np.array([row.get("split") == "calib" for row in rows])
    warnings: List[str] = []
    if config.calibration_method == "none":
        temperature = 1.0
        calibrated = probs.copy()
    else:
        fit = fit_temperature_grid(y[calib_mask], probs[calib_mask])
        temperature = float(fit["temperature"])
        warnings.extend(fit.get("warnings", []))
        calibrated = temperature_scale_probs(probs, temperature)
    threshold_selection = select_threshold_by_fdr(
        y_true=y[calib_mask],
        probs=calibrated[calib_mask],
        target_fdr=config.target_fdr,
        min_threshold=config.min_threshold,
        max_threshold=config.max_threshold,
        threshold_grid_size=config.threshold_grid_size,
    )
    warnings.extend(threshold_selection.get("warnings", []))
    threshold = float(threshold_selection["selected_threshold"])
    pred = (calibrated >= threshold).astype(np.int32)
    output_rows = []
    for index, row in enumerate(rows):
        label = int(y[index])
        predicted = int(pred[index])
        output_rows.append(
            {
                "branch_site_id": row.get("branch_site_id", ""),
                "family_id": row.get("family_id", ""),
                "method": row.get("method", ""),
                "saturation_tier": row.get("saturation_tier", "unknown") or "unknown",
                "split": row.get("split", ""),
                "branch_id": row.get("branch_id", ""),
                "site_index_zero": row.get("site_index_zero", ""),
                "y_branch_site": label,
                "y_site": row.get("y_site", ""),
                "gene_label": row.get("gene_label", ""),
                "prob_positive_raw": float(probs[index]),
                "prob_positive_calibrated": float(calibrated[index]),
                "pred_label_calibrated": predicted,
                "correct_calibrated": int(predicted == label),
            }
        )
    outdir = Path(config.outdir)
    calibration_path = outdir / "branch_site_calibration.json"
    predictions_path = outdir / "branch_site_calibrated_predictions.tsv"
    metrics_path = outdir / "branch_site_calibrated_metrics.json"
    markdown_path = outdir / "branch_site_calibration.md"
    metrics = {
        "branch_site_calibration_version": BRANCH_CALIBRATION_VERSION,
        "metrics_by_split_raw": _metrics_by_field(rows, y, probs, 0.5, "split", True),
        "metrics_by_split_calibrated": _metrics_by_field(rows, y, calibrated, threshold, "split", True),
        "metrics_by_method_calibrated": _metrics_by_field(rows, y, calibrated, threshold, "method", True),
        "selected_threshold": threshold,
        "temperature": temperature,
    }
    payload = {
        "branch_site_calibration_version": BRANCH_CALIBRATION_VERSION,
        "source_model_dir": str(Path(config.model_dir)),
        "calibration_method": config.calibration_method,
        "temperature": temperature,
        "target_fdr": config.target_fdr,
        "selected_threshold": threshold,
        "threshold_selection": threshold_selection,
        "warnings": sorted(set(warnings)),
        "calibration_split_size": int(calib_mask.sum()),
        "calibration_split_positive_count": int(y[calib_mask].sum()),
        "raw_calibration_metrics": _calibration_metrics(y[calib_mask], probs[calib_mask]),
        "calibrated_calibration_metrics": _calibration_metrics(y[calib_mask], calibrated[calib_mask]),
        "generated_files": {
            "calibration": str(calibration_path),
            "predictions": str(predictions_path),
            "metrics": str(metrics_path),
            "markdown": str(markdown_path),
        },
        "note": "Branch-site calibration for research-alpha simulation-supervised classifier.",
    }
    _write_json(calibration_path, payload)
    write_tsv(predictions_path, output_rows, CALIBRATED_FIELDNAMES)
    _write_json(metrics_path, metrics)
    markdown_path.write_text(_render_calibration_markdown(payload), encoding="utf-8")
    return {
        "status": "ok",
        "outdir": str(outdir),
        "calibration": str(calibration_path),
        "predictions": str(predictions_path),
        "metrics": str(metrics_path),
        "markdown": str(markdown_path),
        "temperature": temperature,
        "selected_threshold": threshold,
        "warnings": payload["warnings"],
    }


@dataclass(frozen=True)
class BranchSiteThresholdPolicyConfig:
    predictions_tsv: str
    outdir: str
    probability_column: str = "prob_positive"
    calibrated_probability_column: Optional[str] = None
    label_column: str = "y_branch_site"
    split_column: str = "split"
    selection_split: str = "calib"
    target_fdr: float = 0.10
    precision_floor: float = 0.80
    recall_floor: float = 0.80
    threshold_grid_size: int = 501
    min_threshold: float = 0.0
    max_threshold: float = 1.0
    model_name: str = "branch_site_model"

    def __post_init__(self) -> None:
        Path(self.outdir).mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class BranchAggregationThresholdPolicyConfig:
    outdir: str
    aggregation_dir: Optional[str] = None
    predictions_tsv: Optional[str] = None
    score_column: str = "max_branch_probability"
    label_column: str = "gene_label"
    split_column: str = "split"
    selection_split: str = "calib"
    target_fdr: float = 0.10
    precision_floor: float = 0.80
    recall_floor: float = 0.80
    threshold_grid_size: int = 501
    min_threshold: float = 0.0
    max_threshold: float = 1.0
    model_name: str = "branch_to_gene"

    def __post_init__(self) -> None:
        if self.aggregation_dir is None and self.predictions_tsv is None:
            raise ValueError("aggregation_dir or predictions_tsv must be supplied")
        Path(self.outdir).mkdir(parents=True, exist_ok=True)


def build_branch_site_threshold_policy(config: BranchSiteThresholdPolicyConfig) -> dict:
    generic = build_threshold_policy(
        ThresholdPolicyConfig(
            predictions_tsv=config.predictions_tsv,
            outdir=config.outdir,
            probability_column=config.probability_column,
            calibrated_probability_column=config.calibrated_probability_column,
            label_column=config.label_column,
            split_column=config.split_column,
            selection_split=config.selection_split,
            target_fdr=config.target_fdr,
            precision_floor=config.precision_floor,
            recall_floor=config.recall_floor,
            threshold_grid_size=config.threshold_grid_size,
            min_threshold=config.min_threshold,
            max_threshold=config.max_threshold,
            model_name=config.model_name,
        )
    )
    return _rename_policy_outputs(Path(config.outdir), "branch_site", generic["warnings"], "Branch-site threshold-policy profiles.")


def build_branch_aggregation_threshold_policy(config: BranchAggregationThresholdPolicyConfig) -> dict:
    predictions = Path(config.predictions_tsv) if config.predictions_tsv else Path(config.aggregation_dir or "") / "branch_to_gene_predictions.tsv"
    if not predictions.exists():
        raise ValueError(f"branch aggregation predictions do not exist: {predictions}")
    generic = build_threshold_policy(
        ThresholdPolicyConfig(
            predictions_tsv=str(predictions),
            outdir=config.outdir,
            probability_column=config.score_column,
            label_column=config.label_column,
            split_column=config.split_column,
            selection_split=config.selection_split,
            target_fdr=config.target_fdr,
            precision_floor=config.precision_floor,
            recall_floor=config.recall_floor,
            threshold_grid_size=config.threshold_grid_size,
            min_threshold=config.min_threshold,
            max_threshold=config.max_threshold,
            model_name=config.model_name,
        )
    )
    return _rename_policy_outputs(Path(config.outdir), "branch_aggregation", generic["warnings"], "Branch aggregation threshold-policy profiles.")


def validate_branch_site_calibration_dir(calibration_dir: str | Path) -> dict:
    return _validate_dir(Path(calibration_dir), "branch_site_calibration.json", "branch_site_calibrated_predictions.tsv", "branch_site_calibrated_metrics.json", "branch_site_calibration.md")


def validate_branch_site_threshold_policy_dir(policy_dir: str | Path) -> dict:
    return _validate_policy_dir(Path(policy_dir), "branch_site")


def validate_branch_aggregation_threshold_policy_dir(policy_dir: str | Path) -> dict:
    return _validate_policy_dir(Path(policy_dir), "branch_aggregation")


def _rename_policy_outputs(outdir: Path, prefix: str, warnings: List[str], note: str) -> dict:
    mapping = {
        "threshold_profiles.json": f"{prefix}_threshold_profiles.json",
        "threshold_profiles.tsv": f"{prefix}_threshold_profiles.tsv",
        "threshold_profile_metrics.tsv": f"{prefix}_threshold_profile_metrics.tsv",
        "threshold_policy_curve.tsv": f"{prefix}_threshold_policy_curve.tsv",
        "threshold_policy.md": f"{prefix}_threshold_policy.md",
    }
    for source_name, dest_name in mapping.items():
        source = outdir / source_name
        dest = outdir / dest_name
        if source.exists():
            dest.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    json_path = outdir / f"{prefix}_threshold_profiles.json"
    if json_path.exists():
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        payload["note"] = note
        payload["generated_files"] = {
            "profiles_json": str(json_path),
            "profiles_tsv": str(outdir / f"{prefix}_threshold_profiles.tsv"),
            "profile_metrics_tsv": str(outdir / f"{prefix}_threshold_profile_metrics.tsv"),
            "curve_tsv": str(outdir / f"{prefix}_threshold_policy_curve.tsv"),
            "markdown": str(outdir / f"{prefix}_threshold_policy.md"),
        }
        _write_json(json_path, payload)
    return {
        "status": "ok",
        "outdir": str(outdir),
        "profiles_json": str(json_path),
        "profiles_tsv": str(outdir / f"{prefix}_threshold_profiles.tsv"),
        "profile_metrics_tsv": str(outdir / f"{prefix}_threshold_profile_metrics.tsv"),
        "curve_tsv": str(outdir / f"{prefix}_threshold_policy_curve.tsv"),
        "markdown": str(outdir / f"{prefix}_threshold_policy.md"),
        "warnings": warnings,
    }


def _validate_policy_dir(path: Path, prefix: str) -> dict:
    failures: List[str] = []
    warnings: List[str] = []
    for suffix in ["threshold_profiles.json", "threshold_profiles.tsv", "threshold_profile_metrics.tsv", "threshold_policy_curve.tsv", "threshold_policy.md"]:
        target = path / f"{prefix}_{suffix}"
        if not target.exists():
            failures.append(f"missing_file:{target}")
        elif target.suffix == ".md" and not target.read_text(encoding="utf-8").strip():
            failures.append(f"empty_file:{target}")
    return {"status": "fail" if failures else "ok", "n_fail": len(failures), "n_warning": len(warnings), "failures": failures, "warnings": warnings}


def _validate_dir(path: Path, json_name: str, predictions_name: str, metrics_name: str, markdown_name: str) -> dict:
    failures: List[str] = []
    warnings: List[str] = []
    rows = []
    for name in [json_name, predictions_name, metrics_name, markdown_name]:
        target = path / name
        if not target.exists():
            failures.append(f"missing_file:{target}")
        elif target.suffix == ".md" and not target.read_text(encoding="utf-8").strip():
            failures.append(f"empty_file:{target}")
    predictions = path / predictions_name
    if predictions.exists():
        rows = read_tsv(predictions)
    if not rows:
        failures.append("no_predictions")
    return {"status": "fail" if failures else "ok", "n_predictions": len(rows), "n_fail": len(failures), "n_warning": len(warnings), "failures": failures, "warnings": warnings}


def _calibration_metrics(y: np.ndarray, prob: np.ndarray) -> dict:
    return {"n": int(y.size), "positives": int(y.sum()) if y.size else 0, "brier": brier_score(y, prob), "ece": expected_calibration_error(y, prob)}


def _metrics_by_field(rows: List[dict], y: np.ndarray, prob: np.ndarray, threshold: float, field: str, include_all: bool) -> Dict[str, dict]:
    metrics = {}
    for value in sorted({row.get(field, "") for row in rows}):
        mask = np.array([row.get(field, "") == value for row in rows])
        metrics[value or "unknown"] = _compute_binary_metrics(y[mask], prob[mask], threshold)
    if include_all:
        metrics["all"] = _compute_binary_metrics(y, prob, threshold)
    return metrics


def _render_calibration_markdown(payload: dict) -> str:
    return "\n".join([
        "# Branch-site calibration",
        "",
        f"- Method: {payload.get('calibration_method')}",
        f"- Temperature: {payload.get('temperature')}",
        f"- Selected threshold: {payload.get('selected_threshold')}",
        "",
        "This calibration is simulation-supervised and research-alpha.",
        "",
    ])


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
