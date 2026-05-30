"""Site-level threshold-policy wrapper."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from babappa.calibration.threshold_policy import ThresholdPolicyConfig, build_threshold_policy


@dataclass(frozen=True)
class SiteThresholdPolicyConfig:
    """Configuration for site-level threshold-policy profiling."""

    predictions_tsv: str
    outdir: str
    probability_column: str = "prob_positive"
    label_column: str = "y_site"
    split_column: str = "split"
    selection_split: str = "calib"
    target_fdr: float = 0.10
    precision_floor: float = 0.80
    recall_floor: float = 0.80
    threshold_grid_size: int = 501
    min_threshold: float = 0.0
    max_threshold: float = 1.0
    model_name: str = "site_model"
    calibrated_probability_column: Optional[str] = None

    def __post_init__(self) -> None:
        Path(self.outdir).mkdir(parents=True, exist_ok=True)


def build_site_threshold_policy(config: SiteThresholdPolicyConfig) -> dict:
    """Build site threshold-policy profiles with site-specific artifact names."""
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
    outdir = Path(config.outdir)
    mapping = {
        "threshold_profiles.json": "site_threshold_profiles.json",
        "threshold_profiles.tsv": "site_threshold_profiles.tsv",
        "threshold_profile_metrics.tsv": "site_threshold_profile_metrics.tsv",
        "threshold_policy_curve.tsv": "site_threshold_policy_curve.tsv",
        "threshold_policy.md": "site_threshold_policy.md",
    }
    for source_name, dest_name in mapping.items():
        source = outdir / source_name
        dest = outdir / dest_name
        if source.exists():
            dest.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    json_path = outdir / "site_threshold_profiles.json"
    if json_path.exists():
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        payload["note"] = "Site-level threshold-policy profiles for oracle-supervised predictions."
        payload["generated_files"] = {
            "profiles_json": str(json_path),
            "profiles_tsv": str(outdir / "site_threshold_profiles.tsv"),
            "profile_metrics_tsv": str(outdir / "site_threshold_profile_metrics.tsv"),
            "curve_tsv": str(outdir / "site_threshold_policy_curve.tsv"),
            "markdown": str(outdir / "site_threshold_policy.md"),
        }
        json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "status": "ok",
        "outdir": str(outdir),
        "profiles_json": str(json_path),
        "profiles_tsv": str(outdir / "site_threshold_profiles.tsv"),
        "profile_metrics_tsv": str(outdir / "site_threshold_profile_metrics.tsv"),
        "curve_tsv": str(outdir / "site_threshold_policy_curve.tsv"),
        "markdown": str(outdir / "site_threshold_policy.md"),
        "warnings": generic["warnings"],
    }
