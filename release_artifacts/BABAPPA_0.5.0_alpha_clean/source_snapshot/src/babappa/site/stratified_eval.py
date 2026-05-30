"""Stratified evaluation for site-level predictions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from babappa import __version__
from babappa.datasets.index import read_tsv, write_tsv
from babappa.site.baseline import _compute_binary_metrics

SITE_STRATIFIED_EVAL_VERSION = __version__
FIELDNAMES = [
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
class SiteStratifiedEvalConfig:
    """Configuration for site stratified evaluation."""

    predictions_tsv: str
    outdir: str
    probability_column: str = "prob_positive"
    label_column: str = "y_site"
    threshold: float = 0.5
    threshold_policy_dir: Optional[str] = None

    def __post_init__(self) -> None:
        if not Path(self.predictions_tsv).exists():
            raise ValueError(f"predictions_tsv does not exist: {self.predictions_tsv}")
        if not 0 <= self.threshold <= 1:
            raise ValueError("threshold must be between 0 and 1")
        if self.threshold_policy_dir is not None and not Path(self.threshold_policy_dir).exists():
            raise ValueError(f"threshold_policy_dir does not exist: {self.threshold_policy_dir}")
        Path(self.outdir).mkdir(parents=True, exist_ok=True)


def site_stratified_evaluate(config: SiteStratifiedEvalConfig) -> dict:
    """Evaluate site predictions across split/saturation/method groups."""
    rows = read_tsv(Path(config.predictions_tsv))
    if not rows:
        raise ValueError("predictions_tsv contains no rows")
    profiles = _profiles(config)
    y = np.array([int(float(row[config.label_column])) for row in rows], dtype=np.int32)
    prob = np.array([float(row[config.probability_column]) for row in rows], dtype=np.float64)
    output_rows = []
    for profile, threshold in profiles.items():
        output_rows.extend(_profile_rows(rows, y, prob, profile, threshold))
    outdir = Path(config.outdir)
    json_path = outdir / "site_stratified_eval.json"
    tsv_path = outdir / "site_stratified_metrics.tsv"
    markdown_path = outdir / "site_stratified_eval.md"
    payload = {
        "site_stratified_eval_version": SITE_STRATIFIED_EVAL_VERSION,
        "predictions_tsv": str(Path(config.predictions_tsv)),
        "probability_column": config.probability_column,
        "label_column": config.label_column,
        "profiles_evaluated": [
            {"profile": name, "threshold": threshold}
            for name, threshold in profiles.items()
        ],
        "warnings": [],
        "generated_files": {
            "json": str(json_path),
            "tsv": str(tsv_path),
            "markdown": str(markdown_path),
        },
    }
    _write_json(json_path, payload)
    write_tsv(tsv_path, output_rows, FIELDNAMES)
    markdown_path.write_text(_render_markdown(payload, output_rows), encoding="utf-8")
    return {
        "status": "ok",
        "outdir": str(outdir),
        "json": str(json_path),
        "tsv": str(tsv_path),
        "markdown": str(markdown_path),
        "warnings": [],
    }


def _profiles(config: SiteStratifiedEvalConfig) -> Dict[str, float]:
    if config.threshold_policy_dir is None:
        return {"fixed_threshold": config.threshold}
    path = Path(config.threshold_policy_dir) / "site_threshold_profiles.json"
    if not path.exists():
        return {"fixed_threshold": config.threshold}
    payload = json.loads(path.read_text(encoding="utf-8"))
    profiles = {}
    for name, profile in (payload.get("profiles") or {}).items():
        if isinstance(profile, dict) and profile.get("selected_threshold") is not None:
            profiles[name] = float(profile["selected_threshold"])
    return profiles or {"fixed_threshold": config.threshold}


def _profile_rows(rows: List[dict], y: np.ndarray, prob: np.ndarray, profile: str, threshold: float) -> List[dict]:
    output = []
    group_specs = [
        ("split", ["split"]),
        ("saturation_tier", ["saturation_tier"]),
        ("method", ["method"]),
        ("split_x_saturation", ["split", "saturation_tier"]),
        ("split_x_method", ["split", "method"]),
        ("saturation_x_method", ["saturation_tier", "method"]),
    ]
    for group_type, fields in group_specs:
        keys = sorted({tuple(row.get(field, "unknown") for field in fields) for row in rows})
        for key in keys:
            mask = np.array([tuple(row.get(field, "unknown") for field in fields) == key for row in rows])
            metrics = _compute_binary_metrics(y[mask], prob[mask], threshold)
            row = {
                "profile": profile,
                "threshold": threshold,
                "group_type": group_type,
                "split": "",
                "saturation_tier": "",
                "method": "",
                **metrics,
            }
            for field, value in zip(fields, key):
                row[field] = value
            output.append(row)
    return output


def _render_markdown(payload: dict, rows: List[dict]) -> str:
    return "\n".join(
        [
            "# Site stratified evaluation",
            "",
            f"- Predictions: `{payload.get('predictions_tsv')}`",
            f"- Profiles evaluated: {payload.get('profiles_evaluated')}",
            f"- Metric rows: {len(rows)}",
            "",
            "## Interpretation caveats",
            "",
            "- Site-level evaluation is oracle-supervised simulation development.",
            "- Small groups can make AUROC unstable.",
            "",
        ]
    )


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
