"""Aggregate site-level probabilities into gene/family-level support scores."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np

from babappa import __version__
from babappa.datasets.index import read_tsv, write_tsv
from babappa.site.baseline import _compute_binary_metrics

SITE_AGGREGATION_VERSION = __version__
PREDICTION_FIELDNAMES = [
    "family_id",
    "method",
    "split",
    "saturation_tier",
    "gene_label",
    "n_sites",
    "max_site_probability",
    "mean_site_probability",
    "top5_mean_site_probability",
    "top10_mean_site_probability",
    "count_sites_prob_ge_0_5",
    "count_sites_prob_ge_0_8",
    "fraction_sites_prob_ge_0_5",
    "fraction_sites_prob_ge_0_8",
]
SCORE_COLUMNS = [
    "max_site_probability",
    "mean_site_probability",
    "top5_mean_site_probability",
    "top10_mean_site_probability",
    "fraction_sites_prob_ge_0_5",
]


@dataclass(frozen=True)
class SiteAggregationConfig:
    """Configuration for site-to-gene aggregation."""

    predictions_tsv: str
    gene_dataset_dir: str
    outdir: str
    probability_column: str = "prob_positive"
    label_column: str = "y_site"
    gene_label_column: str = "gene_label"
    split_column: str = "split"
    method_column: str = "method"
    saturation_column: str = "saturation_tier"

    def __post_init__(self) -> None:
        if not Path(self.predictions_tsv).exists():
            raise ValueError(f"predictions_tsv does not exist: {self.predictions_tsv}")
        dataset_dir = Path(self.gene_dataset_dir)
        if not dataset_dir.exists():
            raise ValueError(f"gene_dataset_dir does not exist: {dataset_dir}")
        if not (dataset_dir / "splits.tsv").exists():
            raise ValueError(f"gene_dataset_dir is missing splits.tsv: {dataset_dir}")
        Path(self.outdir).mkdir(parents=True, exist_ok=True)


def aggregate_site_predictions(config: SiteAggregationConfig) -> dict:
    """Aggregate site probabilities to family/method rows and evaluate gene labels."""
    site_rows = read_tsv(Path(config.predictions_tsv))
    gene_lookup = _gene_lookup(Path(config.gene_dataset_dir), config)
    grouped: Dict[tuple, List[dict]] = defaultdict(list)
    for row in site_rows:
        grouped[(row.get("family_id", ""), row.get(config.method_column, ""))].append(row)
    output_rows = []
    for (family_id, method), rows in sorted(grouped.items()):
        probs = np.array([float(row[config.probability_column]) for row in rows], dtype=np.float64)
        split = rows[0].get(config.split_column, "")
        tier = rows[0].get(config.saturation_column, "unknown") or "unknown"
        lookup = gene_lookup.get((family_id, method), {})
        gene_label = lookup.get(config.gene_label_column, "")
        if gene_label == "":
            gene_label = lookup.get("gene_label", "")
        if lookup:
            split = lookup.get("split", split)
            tier = lookup.get("saturation_tier", tier)
        output_rows.append(
            {
                "family_id": family_id,
                "method": method,
                "split": split,
                "saturation_tier": tier,
                "gene_label": gene_label,
                "n_sites": int(probs.size),
                "max_site_probability": float(probs.max()) if probs.size else 0.0,
                "mean_site_probability": float(probs.mean()) if probs.size else 0.0,
                "top5_mean_site_probability": _topk_mean(probs, 5),
                "top10_mean_site_probability": _topk_mean(probs, 10),
                "count_sites_prob_ge_0_5": int((probs >= 0.5).sum()),
                "count_sites_prob_ge_0_8": int((probs >= 0.8).sum()),
                "fraction_sites_prob_ge_0_5": float((probs >= 0.5).mean()) if probs.size else 0.0,
                "fraction_sites_prob_ge_0_8": float((probs >= 0.8).mean()) if probs.size else 0.0,
            }
        )
    outdir = Path(config.outdir)
    predictions_path = outdir / "site_to_gene_predictions.tsv"
    metrics_path = outdir / "site_to_gene_metrics.json"
    markdown_path = outdir / "site_to_gene_aggregation.md"
    write_tsv(predictions_path, output_rows, PREDICTION_FIELDNAMES)
    metrics = _metrics(output_rows)
    payload = {
        "site_aggregation_version": SITE_AGGREGATION_VERSION,
        "predictions_tsv": str(Path(config.predictions_tsv)),
        "gene_dataset_dir": str(Path(config.gene_dataset_dir)),
        "n_family_method_rows": len(output_rows),
        "score_columns": SCORE_COLUMNS,
        **metrics,
        "generated_files": {
            "predictions": str(predictions_path),
            "metrics": str(metrics_path),
            "markdown": str(markdown_path),
        },
        "interpretation": (
            "Site-to-gene aggregation is the first proper route from local site evidence "
            "to gene-level support; direct whole-gene classifiers should not replace it."
        ),
    }
    _write_json(metrics_path, payload)
    markdown_path.write_text(_render_markdown(payload), encoding="utf-8")
    return {
        "status": "ok",
        "outdir": str(outdir),
        "predictions": str(predictions_path),
        "metrics": str(metrics_path),
        "markdown": str(markdown_path),
        "n_family_method_rows": len(output_rows),
    }


def _gene_lookup(dataset_dir: Path, config: SiteAggregationConfig) -> Dict[tuple, dict]:
    rows = read_tsv(dataset_dir / "splits.tsv")
    return {
        (row.get("family_id", ""), row.get(config.method_column, "")): row
        for row in rows
    }


def _topk_mean(probs: np.ndarray, k: int) -> float:
    if probs.size == 0:
        return 0.0
    return float(np.sort(probs)[-min(k, probs.size):].mean())


def _metrics(rows: List[dict]) -> dict:
    labeled = [row for row in rows if str(row.get("gene_label", "")) in {"0", "1", "0.0", "1.0"}]
    y = np.array([int(float(row["gene_label"])) for row in labeled], dtype=np.int32)
    by_score = {}
    for score in SCORE_COLUMNS:
        values = np.array([float(row[score]) for row in labeled], dtype=np.float64)
        by_score[score] = {
            "all": _compute_binary_metrics(y, values, threshold=0.5),
            "by_split": _metrics_by_field(labeled, y, values, "split"),
            "by_saturation_tier": _metrics_by_field(labeled, y, values, "saturation_tier"),
            "by_method": _metrics_by_field(labeled, y, values, "method"),
        }
    return {
        "default_score": "max_site_probability",
        "gene_level_metrics_by_score": by_score,
        "gene_level_metrics_default": by_score.get("max_site_probability", {}),
    }


def _metrics_by_field(rows: List[dict], y: np.ndarray, score: np.ndarray, field: str) -> Dict[str, dict]:
    result = {}
    for value in sorted({row.get(field, "") for row in rows}):
        mask = np.array([row.get(field, "") == value for row in rows])
        result[value or "unknown"] = _compute_binary_metrics(y[mask], score[mask], threshold=0.5)
    return result


def _render_markdown(payload: dict) -> str:
    default = payload.get("gene_level_metrics_default", {}).get("all", {})
    return "\n".join(
        [
            "# Site-to-gene aggregation",
            "",
            f"- Family-method rows: {payload.get('n_family_method_rows')}",
            f"- Default score: {payload.get('default_score')}",
            f"- Default AUROC: {default.get('auroc')}",
            "",
            "## Interpretation",
            "",
            "This is the first proper route from site evidence to gene-level support.",
            "Gene-level support should be derived from site evidence, not direct whole-gene classification.",
            "",
        ]
    )


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
