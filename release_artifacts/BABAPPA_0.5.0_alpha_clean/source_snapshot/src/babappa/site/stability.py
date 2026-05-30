"""Repeated-seed stability benchmark for site-level neural models."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from babappa import __version__
from babappa.datasets.index import write_tsv
from babappa.site.aggregate import SiteAggregationConfig, aggregate_site_predictions
from babappa.site.neural_audit import validate_site_neural_dir
from babappa.site.neural_train import SiteNeuralTrainConfig, train_site_neural_model
from babappa.training.neural_env import safe_import_torch

SITE_STABILITY_VERSION = __version__
FIELDNAMES = [
    "seed",
    "level",
    "split",
    "saturation_tier",
    "method",
    "metric_scope",
    "accuracy",
    "auroc",
    "f1",
    "mcc",
    "precision",
    "recall",
    "specificity",
    "warning",
]
METRIC_NAMES = ["accuracy", "auroc", "f1", "mcc", "precision", "recall", "specificity"]


@dataclass(frozen=True)
class SiteStabilityConfig:
    """Configuration for site-level neural stability benchmarking."""

    site_dataset_dir: str
    outdir: str
    seeds: List[int] = None  # type: ignore[assignment]
    device: str = "cpu"
    epochs: int = 5
    batch_size: int = 256
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    hidden_dim: int = 64
    dropout: float = 0.1
    positive_class_weight: str = "auto"
    monitor_metric: str = "val_auroc"
    max_train_items: Optional[int] = 4096
    max_val_items: Optional[int] = 1024
    max_calib_items: Optional[int] = 1024
    max_test_items: Optional[int] = 1024
    run_training: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "seeds", self.seeds or [42, 43, 44])
        dataset = Path(self.site_dataset_dir)
        if not dataset.exists():
            raise ValueError(f"site_dataset_dir does not exist: {dataset}")
        if not self.seeds:
            raise ValueError("seeds must be non-empty")
        if self.epochs < 1:
            raise ValueError("epochs must be >= 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")
        if self.monitor_metric not in {"val_loss", "val_auroc"}:
            raise ValueError("monitor_metric must be one of: val_loss, val_auroc")
        Path(self.outdir).mkdir(parents=True, exist_ok=True)


def run_site_stability_benchmark(config: SiteStabilityConfig) -> dict:
    """Run repeated-seed site neural stability benchmark."""
    outdir = Path(config.outdir)
    warnings: List[str] = []
    rows: List[dict] = []
    model_summaries = {}
    torch, error = safe_import_torch()
    if torch is None and config.run_training:
        warnings.append(f"torch_unavailable:{error!r}")

    gene_dataset_dir = _infer_gene_dataset_dir(Path(config.site_dataset_dir), warnings)
    if not config.run_training:
        warnings.append("run_training_false")

    for seed in config.seeds:
        if torch is None or not config.run_training:
            continue
        model_dir = outdir / "models" / f"seed_{seed}"
        summary = train_site_neural_model(
            SiteNeuralTrainConfig(
                site_dataset_dir=config.site_dataset_dir,
                outdir=str(model_dir),
                seed=seed,
                device=config.device,
                epochs=config.epochs,
                batch_size=config.batch_size,
                learning_rate=config.learning_rate,
                weight_decay=config.weight_decay,
                hidden_dim=config.hidden_dim,
                dropout=config.dropout,
                positive_class_weight=config.positive_class_weight,
                monitor_metric=config.monitor_metric,
                max_train_items=config.max_train_items,
                max_val_items=config.max_val_items,
                max_calib_items=config.max_calib_items,
                max_test_items=config.max_test_items,
            )
        )
        validation = validate_site_neural_dir(model_dir)
        metrics = _load_json(model_dir / "site_neural_metrics.json")
        model_summaries[str(seed)] = {
            "model_dir": str(model_dir),
            "validation": validation,
            "best_epoch": summary.get("best_epoch"),
            "metrics_by_split": metrics.get("metrics_by_split", {}),
        }
        _append_metric_rows(rows, seed, "site", metrics)
        if gene_dataset_dir is not None:
            agg_dir = outdir / "aggregation" / f"seed_{seed}"
            agg = aggregate_site_predictions(
                SiteAggregationConfig(
                    predictions_tsv=str(model_dir / "site_neural_predictions.tsv"),
                    gene_dataset_dir=str(gene_dataset_dir),
                    outdir=str(agg_dir),
                )
            )
            agg_metrics = _load_json(Path(agg["metrics"]))
            _append_aggregation_rows(rows, seed, agg_metrics)

    aggregate_summary = _aggregate_summary(rows)
    payload = {
        "site_stability_version": SITE_STABILITY_VERSION,
        "site_dataset_dir": str(Path(config.site_dataset_dir)),
        "config": asdict(config),
        "seeds": config.seeds,
        "models": model_summaries,
        "aggregate_summary": aggregate_summary,
        "best_seed_by_val_auroc": _best_seed(rows, "val"),
        "mean_std_by_metric": aggregate_summary,
        "warnings": sorted(set(warnings)),
        "generated_files": {
            "json": str(outdir / "site_stability_benchmark.json"),
            "tsv": str(outdir / "site_stability_results.tsv"),
            "markdown": str(outdir / "site_stability_benchmark.md"),
        },
    }
    _write_json(outdir / "site_stability_benchmark.json", payload)
    write_tsv(outdir / "site_stability_results.tsv", rows, FIELDNAMES)
    (outdir / "site_stability_benchmark.md").write_text(_render_markdown(payload), encoding="utf-8")
    return {
        "status": "ok",
        "outdir": str(outdir),
        "json": str(outdir / "site_stability_benchmark.json"),
        "tsv": str(outdir / "site_stability_results.tsv"),
        "markdown": str(outdir / "site_stability_benchmark.md"),
        "warnings": payload["warnings"],
    }


def _append_metric_rows(rows: List[dict], seed: int, level: str, metrics: dict) -> None:
    for split, values in metrics.get("metrics_by_split", {}).items():
        rows.append(_metric_row(seed, level, split, "all", "all", "split", values, ""))
    for tier, values in metrics.get("metrics_by_saturation_tier", {}).items():
        rows.append(_metric_row(seed, level, "all", tier, "all", "saturation_tier", values, ""))
    for method, values in metrics.get("metrics_by_method", {}).items():
        rows.append(_metric_row(seed, level, "all", "all", method, "method", values, ""))


def _append_aggregation_rows(rows: List[dict], seed: int, metrics: dict) -> None:
    default = metrics.get("gene_level_metrics_default", {})
    by_split = default.get("by_split", {})
    for split, values in by_split.items():
        rows.append(_metric_row(seed, "gene_aggregation", split, "all", "all", "split", values, ""))
    all_values = default.get("all")
    if all_values:
        rows.append(_metric_row(seed, "gene_aggregation", "all", "all", "all", "split", all_values, ""))


def _metric_row(seed: int, level: str, split: str, tier: str, method: str, scope: str, values: dict, warning: str) -> dict:
    return {
        "seed": seed,
        "level": level,
        "split": split,
        "saturation_tier": tier,
        "method": method,
        "metric_scope": scope,
        **{metric: values.get(metric) for metric in METRIC_NAMES},
        "warning": warning,
    }


def _aggregate_summary(rows: List[dict]) -> dict:
    summary = {}
    groups = sorted({(row["level"], row["metric_scope"], row["split"], row["saturation_tier"], row["method"]) for row in rows})
    for group in groups:
        level, scope, split, tier, method = group
        subset = [row for row in rows if (row["level"], row["metric_scope"], row["split"], row["saturation_tier"], row["method"]) == group]
        key = "::".join(group)
        summary[key] = {}
        for metric in METRIC_NAMES:
            values = np.array([float(row[metric]) for row in subset if row.get(metric) not in (None, "")], dtype=np.float64)
            summary[key][metric] = {
                "mean": float(values.mean()) if values.size else None,
                "std": float(values.std(ddof=0)) if values.size else None,
                "n": int(values.size),
            }
    return summary


def _best_seed(rows: List[dict], split: str) -> Optional[int]:
    candidates = [
        row for row in rows
        if row["level"] == "site" and row["metric_scope"] == "split" and row["split"] == split and row.get("auroc") is not None
    ]
    if not candidates:
        return None
    return int(max(candidates, key=lambda row: float(row["auroc"]))["seed"])


def _infer_gene_dataset_dir(site_dataset_dir: Path, warnings: List[str]) -> Optional[Path]:
    index_path = site_dataset_dir / "site_dataset_index.json"
    if not index_path.exists():
        warnings.append("could_not_infer_gene_dataset_dir")
        return None
    dataset = _load_json(index_path).get("dataset_dir")
    if not dataset:
        warnings.append("could_not_infer_gene_dataset_dir")
        return None
    path = Path(dataset)
    if not path.exists():
        path = site_dataset_dir.parent / dataset
    if not path.exists():
        warnings.append("gene_dataset_dir_not_found_for_aggregation")
        return None
    if not (path / "splits.tsv").exists():
        warnings.append("gene_dataset_dir_missing_splits_for_aggregation")
        return None
    return path


def _render_markdown(payload: dict) -> str:
    summary = payload.get("aggregate_summary", {})
    all_site = summary.get("site::split::test::all::all") or summary.get("site::split::all::all::all") or {}
    test_auroc = (all_site.get("auroc") or {}).get("mean")
    test_std = (all_site.get("auroc") or {}).get("std")
    stable = test_auroc is not None and test_std is not None and test_std <= 0.03
    interpretation = (
        "The model is stable under the tested seeds." if stable else
        "Inspect seed-to-seed AUROC variance before release claims."
    )
    return "\n".join(
        [
            "# Site-neural stability benchmark",
            "",
            "## Configuration",
            "",
            f"- Seeds: {payload.get('seeds')}",
            "",
            "## Site-level stability",
            "",
            f"- Mean test AUROC: {test_auroc}",
            f"- Test AUROC std: {test_std}",
            "",
            "## Saturation-tier stability",
            "",
            "See `site_stability_results.tsv` for tier-wise stability.",
            "",
            "## Gene-aggregation stability",
            "",
            "If gene aggregation is perfect across seeds, treat it as oracle upper-bound behavior and run decoy/null controls before publication claims.",
            "",
            "## Warnings",
            "",
            *[f"- {warning}" for warning in (payload.get("warnings") or ["none"])],
            "",
            "## Interpretation",
            "",
            interpretation,
            "",
        ]
    )


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
