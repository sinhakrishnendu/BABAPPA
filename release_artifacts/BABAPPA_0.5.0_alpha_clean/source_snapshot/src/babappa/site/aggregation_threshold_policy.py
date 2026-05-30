"""Threshold-policy profiling for site-to-gene aggregation scores."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from babappa.calibration.threshold_policy import ThresholdPolicyConfig, build_threshold_policy


@dataclass(frozen=True)
class AggregationThresholdPolicyConfig:
    """Configuration for aggregation-level threshold policies."""

    outdir: str
    aggregation_dir: str | None = None
    predictions_tsv: str | None = None
    score_column: str = "max_site_probability"
    label_column: str = "gene_label"
    split_column: str = "split"
    selection_split: str = "calib"
    target_fdr: float = 0.10
    precision_floor: float = 0.80
    recall_floor: float = 0.80
    threshold_grid_size: int = 501
    min_threshold: float = 0.0
    max_threshold: float = 1.0
    model_name: str = "site_to_gene"

    def __post_init__(self) -> None:
        if self.aggregation_dir is None and self.predictions_tsv is None:
            raise ValueError("aggregation_dir or predictions_tsv must be supplied")
        if self.aggregation_dir is not None:
            path = Path(self.aggregation_dir)
            if not path.exists():
                raise ValueError(f"aggregation_dir does not exist: {path}")
            if self.predictions_tsv is None and not (path / "site_to_gene_predictions.tsv").exists():
                raise ValueError(f"aggregation_dir is missing site_to_gene_predictions.tsv: {path}")
        if self.predictions_tsv is not None and not Path(self.predictions_tsv).exists():
            raise ValueError(f"predictions_tsv does not exist: {self.predictions_tsv}")
        Path(self.outdir).mkdir(parents=True, exist_ok=True)


def build_aggregation_threshold_policy(config: AggregationThresholdPolicyConfig) -> dict:
    """Build aggregation threshold profiles using the generic threshold-policy engine."""
    predictions = (
        Path(config.predictions_tsv)
        if config.predictions_tsv is not None
        else Path(config.aggregation_dir or "") / "site_to_gene_predictions.tsv"
    )
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
    outdir = Path(config.outdir)
    mapping = {
        "threshold_profiles.json": "aggregation_threshold_profiles.json",
        "threshold_profiles.tsv": "aggregation_threshold_profiles.tsv",
        "threshold_profile_metrics.tsv": "aggregation_threshold_profile_metrics.tsv",
        "threshold_policy_curve.tsv": "aggregation_threshold_policy_curve.tsv",
        "threshold_policy.md": "aggregation_threshold_policy.md",
    }
    for source_name, dest_name in mapping.items():
        source = outdir / source_name
        dest = outdir / dest_name
        if source.exists():
            dest.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    json_path = outdir / "aggregation_threshold_profiles.json"
    if json_path.exists():
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        payload["note"] = "Aggregation-level threshold profiles for site-to-gene support scores."
        payload["score_column"] = config.score_column
        payload["generated_files"] = {
            "profiles_json": str(json_path),
            "profiles_tsv": str(outdir / "aggregation_threshold_profiles.tsv"),
            "profile_metrics_tsv": str(outdir / "aggregation_threshold_profile_metrics.tsv"),
            "curve_tsv": str(outdir / "aggregation_threshold_policy_curve.tsv"),
            "markdown": str(outdir / "aggregation_threshold_policy.md"),
        }
        json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "status": "ok",
        "outdir": str(outdir),
        "profiles_json": str(json_path),
        "profiles_tsv": str(outdir / "aggregation_threshold_profiles.tsv"),
        "profile_metrics_tsv": str(outdir / "aggregation_threshold_profile_metrics.tsv"),
        "curve_tsv": str(outdir / "aggregation_threshold_policy_curve.tsv"),
        "markdown": str(outdir / "aggregation_threshold_policy.md"),
        "warnings": generic["warnings"],
    }
