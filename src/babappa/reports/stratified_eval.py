"""Saturation- and method-stratified prediction evaluation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from babappa import __version__
from babappa.datasets.index import read_tsv, write_tsv
from babappa.models.baseline import compute_binary_metrics

STRATIFIED_EVAL_VERSION = __version__
VALID_SPLITS = ["train", "val", "calib", "test"]
SATURATION_TIERS = ["low", "moderate", "high", "extreme", "unknown"]
DEFAULT_METHODS = ["identity", "codon_dropout"]
GROUP_TYPES = [
    "split",
    "saturation_tier",
    "method",
    "split_x_saturation",
    "split_x_method",
    "saturation_x_method",
    "split_x_saturation_x_method",
]
PROFILE_ORDER = [
    "default_0_5",
    "strict_fdr",
    "max_f1",
    "max_mcc",
    "balanced_youden",
    "high_precision",
    "high_recall",
]
STRATIFIED_FIELDNAMES = [
    "model_name",
    "profile",
    "threshold",
    "group_type",
    "split",
    "saturation_tier",
    "method",
    "n",
    "positives",
    "negatives",
    "accuracy",
    "precision",
    "recall",
    "specificity",
    "f1",
    "mcc",
    "auroc",
]


@dataclass(frozen=True)
class StratifiedEvalConfig:
    """Configuration for stratified BABAPPA prediction evaluation."""

    predictions_tsv: str
    outdir: str
    model_name: str = "model"
    probability_column: str = "prob_positive"
    label_column: str = "gene_label"
    split_column: str = "split"
    method_column: str = "method"
    saturation_column: str = "saturation_tier"
    threshold: float = 0.5
    threshold_policy_dir: Optional[str] = None

    def __post_init__(self) -> None:
        predictions_path = Path(self.predictions_tsv)
        if not predictions_path.exists():
            raise ValueError(f"predictions_tsv does not exist: {predictions_path}")
        if not 0 <= self.threshold <= 1:
            raise ValueError("threshold must be between 0 and 1")
        if self.threshold_policy_dir is not None:
            policy_path = Path(self.threshold_policy_dir)
            if not policy_path.exists():
                raise ValueError(f"threshold_policy_dir does not exist: {policy_path}")
            if not policy_path.is_dir():
                raise ValueError(f"threshold_policy_dir is not a directory: {policy_path}")
        Path(self.outdir).mkdir(parents=True, exist_ok=True)


def load_json_if_exists(path: Path) -> Optional[dict]:
    """Load JSON object if it exists."""
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON file is not an object: {path}")
    return payload


def safe_float(value: Any, default: float = 0.0) -> float:
    """Convert a value to float with a fallback."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    """Convert a value to int with a fallback."""
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def stratified_evaluate_predictions(config: StratifiedEvalConfig) -> dict:
    """Compute stratified metrics from a prediction TSV."""
    outdir = Path(config.outdir)
    rows = read_tsv(Path(config.predictions_tsv))
    if not rows:
        raise ValueError("predictions_tsv contains no rows")

    warnings: List[str] = []
    _ensure_column(rows, config.probability_column)
    _ensure_column(rows, config.label_column)
    _fill_missing_column(rows, config.split_column, "unknown", warnings)
    _fill_missing_column(rows, config.method_column, "unknown", warnings)
    _fill_missing_column(rows, config.saturation_column, "unknown", warnings)

    y_true = np.asarray(
        [safe_int(row.get(config.label_column)) for row in rows], dtype=np.int32
    )
    probs = np.asarray(
        [safe_float(row.get(config.probability_column)) for row in rows],
        dtype=np.float64,
    )
    if np.any((probs < 0) | (probs > 1)):
        raise ValueError(f"{config.probability_column} contains values outside [0, 1]")

    profiles, profile_warnings = _load_profiles(config)
    warnings.extend(profile_warnings)
    split_values = _split_values(rows, config.split_column)
    saturation_values = _saturation_values(rows, config.saturation_column)
    method_values = _method_values(rows, config.method_column)

    metric_rows: List[dict] = []
    for profile_name, threshold in profiles:
        metric_rows.extend(
            _evaluate_profile(
                rows=rows,
                y_true=y_true,
                probs=probs,
                config=config,
                profile=profile_name,
                threshold=threshold,
                split_values=split_values,
                saturation_values=saturation_values,
                method_values=method_values,
            )
        )

    key_findings = _key_findings(metric_rows, profiles[0][0] if profiles else "")
    json_path = outdir / "stratified_eval.json"
    tsv_path = outdir / "stratified_metrics.tsv"
    markdown_path = outdir / "stratified_eval.md"
    sorted_warnings = sorted(set(warnings))

    payload = {
        "stratified_eval_version": STRATIFIED_EVAL_VERSION,
        "model_name": config.model_name,
        "predictions_tsv": str(Path(config.predictions_tsv)),
        "probability_column": config.probability_column,
        "label_column": config.label_column,
        "threshold": config.threshold,
        "threshold_policy_dir": config.threshold_policy_dir,
        "profiles_evaluated": [
            {"profile": profile, "threshold": threshold}
            for profile, threshold in profiles
        ],
        "warnings": sorted_warnings,
        "generated_files": {
            "json": str(json_path),
            "tsv": str(tsv_path),
            "markdown": str(markdown_path),
        },
        "key_findings": key_findings,
    }
    _write_json(json_path, payload)
    write_tsv(tsv_path, metric_rows, STRATIFIED_FIELDNAMES)
    markdown_path.write_text(_render_markdown(payload, metric_rows), encoding="utf-8")

    return {
        "status": "ok",
        "outdir": str(outdir),
        "json": str(json_path),
        "tsv": str(tsv_path),
        "markdown": str(markdown_path),
        "warnings": sorted_warnings,
    }


def _ensure_column(rows: List[dict], column: str) -> None:
    if column not in rows[0]:
        raise ValueError(f"required column missing: {column}")


def _fill_missing_column(
    rows: List[dict], column: str, default: str, warnings: List[str]
) -> None:
    if column in rows[0]:
        return
    warnings.append(f"missing_column_{column}_using_{default}")
    for row in rows:
        row[column] = default


def _load_profiles(config: StratifiedEvalConfig) -> tuple[List[tuple[str, float]], List[str]]:
    warnings: List[str] = []
    if config.threshold_policy_dir is None:
        return [("fixed_threshold", float(config.threshold))], warnings

    profiles_path = Path(config.threshold_policy_dir) / "threshold_profiles.json"
    try:
        payload = load_json_if_exists(profiles_path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        warnings.append(f"could_not_load_threshold_policy:{profiles_path}:{exc}")
        return [("fixed_threshold", float(config.threshold))], warnings
    if payload is None:
        warnings.append("threshold_policy_profiles_json_missing_using_fixed_threshold")
        return [("fixed_threshold", float(config.threshold))], warnings

    profile_payload = payload.get("profiles") or {}
    ordered_names = [
        name for name in PROFILE_ORDER if name in profile_payload
    ] + sorted(name for name in profile_payload if name not in PROFILE_ORDER)
    profiles = []
    for profile_name in ordered_names:
        profile = profile_payload.get(profile_name)
        if not isinstance(profile, dict):
            continue
        threshold = profile.get("selected_threshold")
        if threshold is None:
            continue
        profiles.append((str(profile_name), safe_float(threshold, config.threshold)))
    if not profiles:
        warnings.append("threshold_policy_has_no_profiles_using_fixed_threshold")
        profiles = [("fixed_threshold", float(config.threshold))]
    return profiles, warnings


def _split_values(rows: List[dict], split_column: str) -> List[str]:
    observed = sorted(
        {
            str(row.get(split_column) or "unknown")
            for row in rows
            if str(row.get(split_column) or "unknown") not in VALID_SPLITS
        }
    )
    return VALID_SPLITS + observed + ["all"]


def _saturation_values(rows: List[dict], saturation_column: str) -> List[str]:
    observed = sorted(
        {
            str(row.get(saturation_column) or "unknown")
            for row in rows
            if str(row.get(saturation_column) or "unknown") not in SATURATION_TIERS
        }
    )
    return SATURATION_TIERS + observed + ["all"]


def _method_values(rows: List[dict], method_column: str) -> List[str]:
    observed = sorted(
        {
            str(row.get(method_column) or "unknown")
            for row in rows
            if str(row.get(method_column) or "unknown") not in DEFAULT_METHODS
        }
    )
    return DEFAULT_METHODS + observed + ["all"]


def _evaluate_profile(
    rows: List[dict],
    y_true: np.ndarray,
    probs: np.ndarray,
    config: StratifiedEvalConfig,
    profile: str,
    threshold: float,
    split_values: List[str],
    saturation_values: List[str],
    method_values: List[str],
) -> List[dict]:
    metric_rows = []
    saturation_non_all = [tier for tier in saturation_values if tier != "all"]
    method_non_all = [method for method in method_values if method != "all"]

    for split in split_values:
        metric_rows.append(
            _metric_row(
                config,
                profile,
                threshold,
                "split",
                rows,
                y_true,
                probs,
                split=split,
                saturation_tier="all",
                method="all",
            )
        )
    for tier in saturation_values:
        metric_rows.append(
            _metric_row(
                config,
                profile,
                threshold,
                "saturation_tier",
                rows,
                y_true,
                probs,
                split="all",
                saturation_tier=tier,
                method="all",
            )
        )
    for method in method_values:
        metric_rows.append(
            _metric_row(
                config,
                profile,
                threshold,
                "method",
                rows,
                y_true,
                probs,
                split="all",
                saturation_tier="all",
                method=method,
            )
        )
    for split in [value for value in split_values if value != "all"]:
        for tier in saturation_non_all:
            metric_rows.append(
                _metric_row(
                    config,
                    profile,
                    threshold,
                    "split_x_saturation",
                    rows,
                    y_true,
                    probs,
                    split=split,
                    saturation_tier=tier,
                    method="all",
                )
            )
        for method in method_non_all:
            metric_rows.append(
                _metric_row(
                    config,
                    profile,
                    threshold,
                    "split_x_method",
                    rows,
                    y_true,
                    probs,
                    split=split,
                    saturation_tier="all",
                    method=method,
                )
            )
    for tier in saturation_non_all:
        for method in method_non_all:
            metric_rows.append(
                _metric_row(
                    config,
                    profile,
                    threshold,
                    "saturation_x_method",
                    rows,
                    y_true,
                    probs,
                    split="all",
                    saturation_tier=tier,
                    method=method,
                )
            )
    for split in [value for value in split_values if value != "all"]:
        for tier in saturation_non_all:
            for method in method_non_all:
                metric_rows.append(
                    _metric_row(
                        config,
                        profile,
                        threshold,
                        "split_x_saturation_x_method",
                        rows,
                        y_true,
                        probs,
                        split=split,
                        saturation_tier=tier,
                        method=method,
                    )
                )
    return metric_rows


def _metric_row(
    config: StratifiedEvalConfig,
    profile: str,
    threshold: float,
    group_type: str,
    rows: List[dict],
    y_true: np.ndarray,
    probs: np.ndarray,
    split: str,
    saturation_tier: str,
    method: str,
) -> dict:
    mask = _group_mask(rows, config, split, saturation_tier, method)
    metrics = compute_binary_metrics(y_true[mask], probs[mask], threshold)
    return {
        "model_name": config.model_name,
        "profile": profile,
        "threshold": _format_value(float(threshold)),
        "group_type": group_type,
        "split": split,
        "saturation_tier": saturation_tier,
        "method": method,
        **{key: _format_value(metrics.get(key)) for key in [
            "n",
            "positives",
            "negatives",
            "accuracy",
            "precision",
            "recall",
            "specificity",
            "f1",
            "mcc",
            "auroc",
        ]},
    }


def _group_mask(
    rows: List[dict],
    config: StratifiedEvalConfig,
    split: str,
    saturation_tier: str,
    method: str,
) -> np.ndarray:
    keep = []
    for row in rows:
        split_ok = split == "all" or str(row.get(config.split_column) or "unknown") == split
        saturation_ok = (
            saturation_tier == "all"
            or str(row.get(config.saturation_column) or "unknown") == saturation_tier
        )
        method_ok = method == "all" or str(row.get(config.method_column) or "unknown") == method
        keep.append(split_ok and saturation_ok and method_ok)
    return np.asarray(keep, dtype=bool)


def _key_findings(metric_rows: List[dict], primary_profile: str) -> dict:
    saturation_rows = [
        row
        for row in metric_rows
        if row["profile"] == primary_profile
        and row["group_type"] == "saturation_tier"
        and row["saturation_tier"] != "all"
        and _has_metric(row, "auroc")
    ]
    method_rows = [
        row
        for row in metric_rows
        if row["profile"] == primary_profile
        and row["group_type"] == "method"
        and row["method"] != "all"
        and _has_metric(row, "auroc")
    ]
    low_moderate = [
        safe_float(row["auroc"])
        for row in saturation_rows
        if row["saturation_tier"] in {"low", "moderate"} and _has_metric(row, "auroc")
    ]
    high_extreme = [
        safe_float(row["auroc"])
        for row in saturation_rows
        if row["saturation_tier"] in {"high", "extreme"} and _has_metric(row, "auroc")
    ]

    findings = {
        "primary_profile": primary_profile,
        "best_saturation_tier_by_auroc": _best_row(
            saturation_rows, "saturation_tier", reverse=True
        ),
        "worst_saturation_tier_by_auroc": _best_row(
            saturation_rows, "saturation_tier", reverse=False
        ),
        "best_method_by_auroc": _best_row(method_rows, "method", reverse=True),
        "worst_method_by_auroc": _best_row(method_rows, "method", reverse=False),
        "low_moderate_auroc_mean": _mean_or_none(low_moderate),
        "high_extreme_auroc_mean": _mean_or_none(high_extreme),
        "performance_drops_from_low_moderate_to_high_extreme": None,
    }
    if (
        findings["low_moderate_auroc_mean"] is not None
        and findings["high_extreme_auroc_mean"] is not None
    ):
        findings["performance_drops_from_low_moderate_to_high_extreme"] = (
            findings["high_extreme_auroc_mean"]
            < findings["low_moderate_auroc_mean"]
        )
    return findings


def _best_row(rows: List[dict], label_column: str, reverse: bool) -> Optional[dict]:
    if not rows:
        return None
    selected = sorted(rows, key=lambda row: safe_float(row["auroc"]), reverse=reverse)[0]
    return {
        label_column: selected[label_column],
        "auroc": safe_float(selected["auroc"]),
        "n": safe_int(selected["n"]),
    }


def _has_metric(row: dict, key: str) -> bool:
    return row.get(key) not in (None, "")


def _mean_or_none(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return float(np.mean(values))


def _render_markdown(payload: dict, metric_rows: List[dict]) -> str:
    primary_profile = payload["key_findings"].get("primary_profile") or ""
    lines = [
        "# Stratified evaluation report",
        "",
        "## Input",
        "",
        f"- Model name: {payload['model_name']}",
        f"- Predictions TSV: `{payload['predictions_tsv']}`",
        f"- Probability column: `{payload['probability_column']}`",
        f"- Label column: `{payload['label_column']}`",
        "",
        "## Profiles evaluated",
        "",
    ]
    for profile in payload["profiles_evaluated"]:
        lines.append(
            f"- {profile['profile']}: threshold {_format_markdown_float(profile['threshold'])}"
        )
    lines.append("")
    _append_group_table(lines, "Split-level metrics", metric_rows, primary_profile, "split")
    _append_group_table(
        lines,
        "Saturation-tier metrics",
        metric_rows,
        primary_profile,
        "saturation_tier",
    )
    _append_group_table(lines, "Method-level metrics", metric_rows, primary_profile, "method")
    _append_group_table(
        lines,
        "Saturation x method metrics",
        metric_rows,
        primary_profile,
        "saturation_x_method",
    )
    lines.extend(["## Key findings", ""])
    for key, value in payload["key_findings"].items():
        lines.append(f"- `{key}`: {_format_jsonish(value)}")
    lines.extend(["", "## Warnings", ""])
    warnings = payload.get("warnings") or []
    if warnings:
        lines.extend(f"- {warning}" for warning in warnings)
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Interpretation caveats",
            "",
            "- Current model is gene-level, not branch-site.",
            "- Current internal alignment methods are scaffolds.",
            "- Saturation tiers come from the current simulator labels and are not yet final observed saturation diagnostics.",
            "- Small group sizes can make AUROC unstable.",
            "",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def _append_group_table(
    lines: List[str],
    title: str,
    metric_rows: List[dict],
    profile: str,
    group_type: str,
) -> None:
    lines.extend([f"## {title}", ""])
    rows = [
        row
        for row in metric_rows
        if row["profile"] == profile and row["group_type"] == group_type
    ]
    if not rows:
        lines.extend(["No rows available.", ""])
        return
    lines.extend(
        [
            "| Split | Saturation tier | Method | n | AUROC | F1 | MCC | Recall | Specificity |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        if row["n"] in ("", "0"):
            continue
        lines.append(
            "| {split} | {tier} | {method} | {n} | {auroc} | {f1} | {mcc} | {recall} | {specificity} |".format(
                split=row["split"],
                tier=row["saturation_tier"],
                method=row["method"],
                n=row["n"],
                auroc=_format_markdown_float(row["auroc"]),
                f1=_format_markdown_float(row["f1"]),
                mcc=_format_markdown_float(row["mcc"]),
                recall=_format_markdown_float(row["recall"]),
                specificity=_format_markdown_float(row["specificity"]),
            )
        )
    lines.append("")


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


def _format_jsonish(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return str(value)


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
