"""Threshold-policy profiling and operating-point selection."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from babappa import __version__
from babappa.datasets.index import read_tsv, write_tsv

THRESHOLD_POLICY_VERSION = __version__
VALID_SPLITS = ["train", "val", "calib", "test"]
VALID_SELECTION_SPLITS = set(VALID_SPLITS + ["all"])
PROFILE_NAMES = [
    "default_0_5",
    "strict_fdr",
    "max_f1",
    "max_mcc",
    "balanced_youden",
    "high_precision",
    "high_recall",
]
CURVE_FIELDNAMES = [
    "model_name",
    "profile_source",
    "split",
    "threshold",
    "tp",
    "fp",
    "tn",
    "fn",
    "called_positive",
    "called_negative",
    "empirical_fdr",
    "precision",
    "recall",
    "specificity",
    "f1",
    "accuracy",
    "mcc",
]
PROFILE_FIELDNAMES = [
    "profile",
    "selected_threshold",
    "selection_split",
    "selection_empirical_fdr",
    "selection_precision",
    "selection_recall",
    "selection_specificity",
    "selection_f1",
    "selection_mcc",
    "warning",
]
PROFILE_METRIC_FIELDNAMES = [
    "profile",
    "split",
    "threshold",
    "tp",
    "fp",
    "tn",
    "fn",
    "called_positive",
    "called_negative",
    "empirical_fdr",
    "precision",
    "recall",
    "specificity",
    "f1",
    "accuracy",
    "mcc",
]


@dataclass(frozen=True)
class ThresholdPolicyConfig:
    """Configuration for threshold-policy profiling."""

    predictions_tsv: str
    outdir: str
    probability_column: str = "prob_positive"
    calibrated_probability_column: Optional[str] = None
    label_column: str = "gene_label"
    split_column: str = "split"
    selection_split: str = "calib"
    target_fdr: float = 0.10
    precision_floor: float = 0.80
    recall_floor: float = 0.80
    threshold_grid_size: int = 501
    min_threshold: float = 0.0
    max_threshold: float = 1.0
    model_name: str = "model"
    warn_degenerate_thresholds: bool = True
    degenerate_call_fraction: float = 0.98
    min_non_degenerate_threshold: Optional[float] = None

    def __post_init__(self) -> None:
        predictions_path = Path(self.predictions_tsv)
        if not predictions_path.exists():
            raise ValueError(f"predictions_tsv does not exist: {predictions_path}")
        if self.selection_split not in VALID_SELECTION_SPLITS:
            allowed = ", ".join(sorted(VALID_SELECTION_SPLITS))
            raise ValueError(f"selection_split must be one of: {allowed}")
        if not 0 <= self.target_fdr <= 1:
            raise ValueError("target_fdr must be between 0 and 1")
        if not 0 <= self.precision_floor <= 1:
            raise ValueError("precision_floor must be between 0 and 1")
        if not 0 <= self.recall_floor <= 1:
            raise ValueError("recall_floor must be between 0 and 1")
        if self.threshold_grid_size < 2:
            raise ValueError("threshold_grid_size must be >= 2")
        if self.min_threshold < 0:
            raise ValueError("min_threshold must be >= 0")
        if self.max_threshold > 1:
            raise ValueError("max_threshold must be <= 1")
        if self.min_threshold >= self.max_threshold:
            raise ValueError("min_threshold must be < max_threshold")
        if not 0.5 < self.degenerate_call_fraction <= 1:
            raise ValueError("degenerate_call_fraction must be > 0.5 and <= 1")
        if self.min_non_degenerate_threshold is not None and not (
            self.min_threshold <= self.min_non_degenerate_threshold <= self.max_threshold
        ):
            raise ValueError(
                "min_non_degenerate_threshold must be within threshold bounds"
            )
        Path(self.outdir).mkdir(parents=True, exist_ok=True)


def threshold_metrics(y_true: np.ndarray, prob: np.ndarray, threshold: float) -> dict:
    """Compute threshold-dependent binary classification metrics."""
    y_true = np.asarray(y_true, dtype=np.int32)
    prob = np.asarray(prob, dtype=np.float64)
    y_pred = (prob >= threshold).astype(np.int32)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    called_positive = tp + fp
    called_negative = tn + fn
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)
    accuracy = None if y_true.size == 0 else _safe_div(tp + tn, int(y_true.size))
    f1 = None
    if precision is not None and recall is not None and (precision + recall) > 0:
        f1 = 2.0 * precision * recall / (precision + recall)

    mcc = None
    denominator = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    if denominator > 0:
        denom_sqrt = math.sqrt(float(denominator))
        if denom_sqrt > 0:
            mcc = ((tp * tn) - (fp * fn)) / denom_sqrt

    return {
        "threshold": float(threshold),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "called_positive": called_positive,
        "called_negative": called_negative,
        "empirical_fdr": fp / max(1, called_positive),
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "accuracy": accuracy,
        "mcc": mcc,
    }


def build_threshold_policy(config: ThresholdPolicyConfig) -> dict:
    """Build threshold-policy profiles and split-wise threshold curves."""
    outdir = Path(config.outdir)
    rows = read_tsv(Path(config.predictions_tsv))
    if not rows:
        raise ValueError("predictions_tsv contains no rows")

    warnings: List[str] = []
    probability_used = _choose_probability_column(config, rows, warnings)
    y_true = _numeric_array(rows, config.label_column, dtype=np.int32)
    probs = _numeric_array(rows, probability_used, dtype=np.float64)
    if np.any((probs < 0) | (probs > 1)):
        raise ValueError(f"{probability_used} contains values outside [0, 1]")

    thresholds = np.linspace(
        config.min_threshold, config.max_threshold, config.threshold_grid_size
    )
    split_masks = _split_masks(rows, config.split_column)
    curve_by_split = _threshold_curve_for_splits(
        config.model_name, split_masks, y_true, probs, thresholds
    )
    selection_metrics = curve_by_split[config.selection_split]
    profiles, profile_warnings = _select_profiles(config, selection_metrics)
    warnings.extend(profile_warnings)
    profile_metrics = _evaluate_profiles(profiles, split_masks, y_true, probs)
    profile_rows = _profile_rows(profiles, config.selection_split)

    profiles_json = outdir / "threshold_profiles.json"
    profiles_tsv = outdir / "threshold_profiles.tsv"
    profile_metrics_tsv = outdir / "threshold_profile_metrics.tsv"
    curve_tsv = outdir / "threshold_policy_curve.tsv"
    markdown = outdir / "threshold_policy.md"

    curve_rows = []
    for split in VALID_SPLITS + ["all"]:
        for metric in curve_by_split[split]:
            curve_rows.append(
                {
                    "model_name": config.model_name,
                    "profile_source": "grid",
                    "split": split,
                    **_metric_tsv_values(metric),
                }
            )

    metric_rows = []
    for profile_name in PROFILE_NAMES:
        for split in VALID_SPLITS + ["all"]:
            metric_rows.append(
                {
                    "profile": profile_name,
                    "split": split,
                    **_metric_tsv_values(profile_metrics[profile_name][split]),
                }
            )

    sorted_warnings = sorted(set(warnings))
    payload = {
        "threshold_policy_version": THRESHOLD_POLICY_VERSION,
        "model_name": config.model_name,
        "predictions_tsv": str(Path(config.predictions_tsv)),
        "probability_used": probability_used,
        "selection_split": config.selection_split,
        "target_fdr": config.target_fdr,
        "precision_floor": config.precision_floor,
        "recall_floor": config.recall_floor,
        "warn_degenerate_thresholds": config.warn_degenerate_thresholds,
        "degenerate_call_fraction": config.degenerate_call_fraction,
        "min_non_degenerate_threshold": config.min_non_degenerate_threshold,
        "profiles": profiles,
        "warnings": sorted_warnings,
        "generated_files": {
            "profiles_json": str(profiles_json),
            "profiles_tsv": str(profiles_tsv),
            "profile_metrics_tsv": str(profile_metrics_tsv),
            "curve_tsv": str(curve_tsv),
            "markdown": str(markdown),
        },
        "note": (
            "Threshold-policy profiles for gene-level BABAPPA predictions; "
            "not final branch-site operating-point selection."
        ),
    }

    _write_json(profiles_json, payload)
    write_tsv(profiles_tsv, profile_rows, PROFILE_FIELDNAMES)
    write_tsv(profile_metrics_tsv, metric_rows, PROFILE_METRIC_FIELDNAMES)
    write_tsv(curve_tsv, curve_rows, CURVE_FIELDNAMES)
    markdown.write_text(
        _render_markdown(payload, profile_metrics, sorted_warnings),
        encoding="utf-8",
    )

    return {
        "status": "ok",
        "outdir": str(outdir),
        "profiles_json": str(profiles_json),
        "profiles_tsv": str(profiles_tsv),
        "profile_metrics_tsv": str(profile_metrics_tsv),
        "curve_tsv": str(curve_tsv),
        "markdown": str(markdown),
        "warnings": sorted_warnings,
    }


def _choose_probability_column(
    config: ThresholdPolicyConfig, rows: List[dict], warnings: List[str]
) -> str:
    columns = set(rows[0].keys())
    if (
        config.calibrated_probability_column
        and config.calibrated_probability_column in columns
    ):
        return config.calibrated_probability_column
    if config.calibrated_probability_column:
        warnings.append("calibrated_probability_column_missing_using_probability_column")
    if config.probability_column not in columns:
        raise ValueError(
            f"probability column not found: {config.probability_column}"
        )
    return config.probability_column


def _numeric_array(rows: List[dict], column: str, dtype: Any) -> np.ndarray:
    values = []
    for row in rows:
        if column not in row:
            raise ValueError(f"required column missing: {column}")
        try:
            values.append(float(row[column]))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{column} is not numeric: {row.get(column)}") from exc
    return np.asarray(values, dtype=dtype)


def _split_masks(rows: List[dict], split_column: str) -> Dict[str, np.ndarray]:
    if split_column not in rows[0]:
        raise ValueError(f"required split column missing: {split_column}")
    masks = {
        split: np.asarray([row.get(split_column) == split for row in rows], dtype=bool)
        for split in VALID_SPLITS
    }
    masks["all"] = np.ones(len(rows), dtype=bool)
    return masks


def _threshold_curve_for_splits(
    model_name: str,
    split_masks: Dict[str, np.ndarray],
    y_true: np.ndarray,
    probs: np.ndarray,
    thresholds: np.ndarray,
) -> Dict[str, List[dict]]:
    del model_name
    return {
        split: [
            threshold_metrics(y_true[mask], probs[mask], float(threshold))
            for threshold in thresholds
        ]
        for split, mask in split_masks.items()
    }


def _select_profiles(
    config: ThresholdPolicyConfig, selection_metrics: List[dict]
) -> tuple[Dict[str, dict], List[str]]:
    warnings: List[str] = []
    profiles: Dict[str, dict] = {}

    profiles["default_0_5"] = _profile_from_metric(
        "default_0_5",
        _nearest_threshold(selection_metrics, 0.5),
        "Fixed threshold 0.5.",
    )

    strict = [
        metric
        for metric in selection_metrics
        if metric["called_positive"] > 0
        and metric["empirical_fdr"] <= config.target_fdr
    ]
    if strict:
        metric = sorted(
            strict,
            key=lambda row: (
                -_value_or_floor(row["recall"]),
                row["empirical_fdr"],
                row["threshold"],
            ),
        )[0]
        profiles["strict_fdr"] = _profile_from_metric(
            "strict_fdr",
            metric,
            "Conservative discovery profile constrained by empirical FDR.",
        )
    else:
        warning = "no_threshold_met_target_fdr"
        warnings.append(warning)
        profiles["strict_fdr"] = _profile_from_metric(
            "strict_fdr",
            _nearest_threshold(selection_metrics, 0.5),
            "No threshold met target FDR; defaulted to 0.5.",
            warning,
        )

    max_f1_candidates = [m for m in selection_metrics if m["f1"] is not None]
    profiles["max_f1"] = _best_or_default(
        "max_f1",
        max_f1_candidates,
        selection_metrics,
        warnings,
        "no_threshold_with_defined_f1",
        key=lambda row: (-row["f1"], -_value_or_floor(row["mcc"]), row["threshold"]),
        description="Balanced classification profile maximizing F1.",
        config=config,
    )

    max_mcc_candidates = [m for m in selection_metrics if m["mcc"] is not None]
    profiles["max_mcc"] = _best_or_default(
        "max_mcc",
        max_mcc_candidates,
        selection_metrics,
        warnings,
        "no_threshold_with_defined_mcc",
        key=lambda row: (-row["mcc"], -_value_or_floor(row["f1"]), row["threshold"]),
        description="Class-imbalance-aware profile maximizing MCC.",
        config=config,
    )

    youden_candidates = [
        m
        for m in selection_metrics
        if m["recall"] is not None and m["specificity"] is not None
    ]
    profiles["balanced_youden"] = _best_or_default(
        "balanced_youden",
        youden_candidates,
        selection_metrics,
        warnings,
        "no_threshold_with_defined_youden",
        key=lambda row: (
            -((row["recall"] or 0.0) + (row["specificity"] or 0.0) - 1.0),
            -_value_or_floor(row["mcc"]),
            row["threshold"],
        ),
        description="Sensitivity/specificity balance profile maximizing Youden index.",
        config=config,
    )

    high_precision = [
        m
        for m in selection_metrics
        if m["called_positive"] > 0
        and m["precision"] is not None
        and m["precision"] >= config.precision_floor
    ]
    if high_precision:
        metric = sorted(
            high_precision,
            key=lambda row: (
                -_value_or_floor(row["recall"]),
                -_value_or_floor(row["precision"]),
                row["threshold"],
            ),
        )[0]
        profiles["high_precision"] = _profile_from_metric(
            "high_precision",
            metric,
            "Low false-positive follow-up profile constrained by precision floor.",
        )
    else:
        warning = "no_threshold_met_precision_floor"
        warnings.append(warning)
        profiles["high_precision"] = _profile_from_metric(
            "high_precision",
            _nearest_threshold(selection_metrics, 0.5),
            "No threshold met precision floor; defaulted to 0.5.",
            warning,
        )

    high_recall = [
        m
        for m in selection_metrics
        if m["recall"] is not None and m["recall"] >= config.recall_floor
    ]
    if high_recall:
        metric = sorted(
            high_recall,
            key=lambda row: (
                row["empirical_fdr"],
                -_value_or_floor(row["specificity"]),
                -row["threshold"],
            ),
        )[0]
        profiles["high_recall"] = _profile_from_metric(
            "high_recall",
            metric,
            "Screening profile constrained by recall floor.",
        )
    else:
        warning = "no_threshold_met_recall_floor"
        warnings.append(warning)
        profiles["high_recall"] = _profile_from_metric(
            "high_recall",
            _nearest_threshold(selection_metrics, 0.5),
            "No threshold met recall floor; defaulted to 0.5.",
            warning,
        )

    if config.warn_degenerate_thresholds:
        for profile_name, profile in profiles.items():
            profile_warnings = _profile_degeneracy_warnings(
                config, profile["selection_metrics"]
            )
            if profile_warnings:
                profile.setdefault("warnings", []).extend(profile_warnings)
                profile["warnings"] = sorted(set(profile["warnings"]))
                profile["warning"] = ";".join(profile["warnings"])
                warnings.extend(
                    f"{profile_name}:{warning}" for warning in profile_warnings
                )

    return profiles, warnings


def _best_or_default(
    name: str,
    candidates: List[dict],
    selection_metrics: List[dict],
    warnings: List[str],
    warning: str,
    key: Any,
    description: str,
    config: ThresholdPolicyConfig,
) -> dict:
    if candidates:
        ranked_candidates = _prefer_non_degenerate_candidates(candidates, config)
        return _profile_from_metric(
            name, sorted(ranked_candidates, key=key)[0], description
        )
    warnings.append(warning)
    return _profile_from_metric(
        name,
        _nearest_threshold(selection_metrics, 0.5),
        f"{description} No valid candidate; defaulted to 0.5.",
        warning,
    )


def _profile_from_metric(
    name: str, metric: dict, description: str, warning: Optional[str] = None
) -> dict:
    warnings = [warning] if warning else []
    return {
        "profile": name,
        "selected_threshold": float(metric["threshold"]),
        "selection_metrics": metric,
        "description": description,
        "warning": warning or "",
        "warnings": warnings,
    }


def _prefer_non_degenerate_candidates(
    candidates: List[dict], config: ThresholdPolicyConfig
) -> List[dict]:
    non_degenerate = []
    for candidate in candidates:
        fraction = _called_positive_fraction(candidate)
        if fraction is None:
            continue
        if (
            (1.0 - config.degenerate_call_fraction)
            < fraction
            < config.degenerate_call_fraction
        ):
            if (
                config.min_non_degenerate_threshold is None
                or candidate["threshold"] >= config.min_non_degenerate_threshold
            ):
                non_degenerate.append(candidate)
    return non_degenerate or candidates


def _profile_degeneracy_warnings(
    config: ThresholdPolicyConfig, metric: dict
) -> List[str]:
    warnings = []
    threshold = float(metric.get("threshold", 0.0))
    if np.isclose(threshold, config.min_threshold) or np.isclose(
        threshold, config.max_threshold
    ):
        warnings.append("selected_boundary_threshold")
    fraction = _called_positive_fraction(metric)
    if fraction is None:
        return warnings
    if fraction >= config.degenerate_call_fraction:
        warnings.append("selected_all_or_nearly_all_positive")
    if fraction <= (1.0 - config.degenerate_call_fraction):
        warnings.append("selected_all_or_nearly_all_negative")
    return warnings


def _called_positive_fraction(metric: dict) -> Optional[float]:
    called_positive = int(metric.get("called_positive") or 0)
    called_negative = int(metric.get("called_negative") or 0)
    total = called_positive + called_negative
    if total == 0:
        return None
    return called_positive / total


def _nearest_threshold(metrics: List[dict], threshold: float) -> dict:
    if not metrics:
        return threshold_metrics(np.asarray([], dtype=np.int32), np.asarray([]), threshold)
    return min(metrics, key=lambda row: abs(row["threshold"] - threshold))


def _evaluate_profiles(
    profiles: Dict[str, dict],
    split_masks: Dict[str, np.ndarray],
    y_true: np.ndarray,
    probs: np.ndarray,
) -> Dict[str, Dict[str, dict]]:
    evaluated = {}
    for profile_name, profile in profiles.items():
        threshold = float(profile["selected_threshold"])
        evaluated[profile_name] = {
            split: threshold_metrics(y_true[mask], probs[mask], threshold)
            for split, mask in split_masks.items()
        }
    return evaluated


def _profile_rows(profiles: Dict[str, dict], selection_split: str) -> List[dict]:
    rows = []
    for profile_name in PROFILE_NAMES:
        profile = profiles[profile_name]
        metrics = profile["selection_metrics"]
        rows.append(
            {
                "profile": profile_name,
                "selected_threshold": _format_value(profile["selected_threshold"]),
                "selection_split": selection_split,
                "selection_empirical_fdr": _format_value(metrics["empirical_fdr"]),
                "selection_precision": _format_value(metrics["precision"]),
                "selection_recall": _format_value(metrics["recall"]),
                "selection_specificity": _format_value(metrics["specificity"]),
                "selection_f1": _format_value(metrics["f1"]),
                "selection_mcc": _format_value(metrics["mcc"]),
                "warning": profile.get("warning") or "",
            }
        )
    return rows


def _metric_tsv_values(metric: dict) -> dict:
    return {
        key: _format_value(metric.get(key))
        for key in [
            "threshold",
            "tp",
            "fp",
            "tn",
            "fn",
            "called_positive",
            "called_negative",
            "empirical_fdr",
            "precision",
            "recall",
            "specificity",
            "f1",
            "accuracy",
            "mcc",
        ]
    }


def _render_markdown(
    payload: dict,
    profile_metrics: Dict[str, Dict[str, dict]],
    warnings: List[str],
) -> str:
    lines = [
        "# Threshold policy report",
        "",
        "## Input",
        "",
        f"- Model name: {payload['model_name']}",
        f"- Predictions TSV: `{payload['predictions_tsv']}`",
        "",
        "## Probability source",
        "",
        f"- Probability column used: `{payload['probability_used']}`",
        "",
        "## Selection split",
        "",
        f"- Selection split: {payload['selection_split']}",
        f"- Target FDR: {payload['target_fdr']}",
        f"- Precision floor: {payload['precision_floor']}",
        f"- Recall floor: {payload['recall_floor']}",
        "",
        "## Profiles",
        "",
        "- strict_fdr is for conservative discovery.",
        "- max_f1 is for balanced classification.",
        "- max_mcc is useful under class imbalance.",
        "- balanced_youden balances sensitivity and specificity.",
        "- high_precision is for low false-positive follow-up.",
        "- high_recall is for screening.",
        "",
        "| Profile | Threshold | Selection FDR | Selection precision | Selection recall | Warning |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for profile_name in PROFILE_NAMES:
        profile = payload["profiles"][profile_name]
        metric = profile["selection_metrics"]
        lines.append(
            "| {profile} | {threshold} | {fdr} | {precision} | {recall} | {warning} |".format(
                profile=profile_name,
                threshold=_format_markdown_float(profile["selected_threshold"]),
                fdr=_format_markdown_float(metric.get("empirical_fdr")),
                precision=_format_markdown_float(metric.get("precision")),
                recall=_format_markdown_float(metric.get("recall")),
                warning=profile.get("warning") or "",
            )
        )

    lines.extend(
        [
            "",
            "## Profile metrics by split",
            "",
            "| Profile | Split | Threshold | F1 | MCC | Precision | Recall | Specificity |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for profile_name in PROFILE_NAMES:
        for split in VALID_SPLITS + ["all"]:
            metric = profile_metrics[profile_name][split]
            lines.append(
                "| {profile} | {split} | {threshold} | {f1} | {mcc} | {precision} | {recall} | {specificity} |".format(
                    profile=profile_name,
                    split=split,
                    threshold=_format_markdown_float(metric.get("threshold")),
                    f1=_format_markdown_float(metric.get("f1")),
                    mcc=_format_markdown_float(metric.get("mcc")),
                    precision=_format_markdown_float(metric.get("precision")),
                    recall=_format_markdown_float(metric.get("recall")),
                    specificity=_format_markdown_float(metric.get("specificity")),
                )
            )

    lines.extend(["", "## Warnings", ""])
    if warnings:
        lines.extend(f"- {warning}" for warning in warnings)
    else:
        lines.append("- none")

    lines.extend(["", "## Recommended interpretation", ""])
    lines.extend(_recommended_interpretation(payload, profile_metrics, warnings))
    return "\n".join(lines).rstrip() + "\n"


def _recommended_interpretation(
    payload: dict,
    profile_metrics: Dict[str, Dict[str, dict]],
    warnings: List[str],
) -> List[str]:
    max_f1_all = profile_metrics["max_f1"]["all"].get("f1")
    max_mcc_all = profile_metrics["max_mcc"]["all"].get("mcc")
    strict_recall = profile_metrics["strict_fdr"][payload["selection_split"]].get(
        "recall"
    )
    lines = []
    if "probability_collapse" in warnings:
        lines.append("- Probability collapse was detected; do not rely on thresholds yet.")
    if strict_recall is not None and strict_recall < 0.25:
        lines.append(
            "- Strict discovery mode is very conservative and may have low recall at this calibration size."
        )
    if (
        (max_f1_all is not None and max_f1_all > 0.5)
        or (max_mcc_all is not None and max_mcc_all > 0.2)
    ) and (strict_recall is not None and strict_recall < 0.5):
        lines.append(
            "- The model may be useful for ranking or screening, but strict discovery needs larger calibration or stronger score separation."
        )
    if not lines:
        lines.append(
            "- Compare max_f1, max_mcc, and strict_fdr profiles against the intended scientific use case before scaling."
        )
    return lines


def _value_or_floor(value: Optional[float], floor: float = -1.0) -> float:
    if value is None:
        return floor
    return float(value)


def _safe_div(numerator: float, denominator: float) -> Optional[float]:
    if denominator == 0:
        return None
    return numerator / denominator


def _format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.12g}"
    return str(value)


def _format_markdown_float(value: Any) -> str:
    if value is None or value == "":
        return "NA"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
