"""Compare site-level baseline and neural model outputs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from babappa import __version__
from babappa.datasets.index import write_tsv

SITE_MODEL_COMPARISON_VERSION = __version__
METRICS = ["auroc", "f1", "mcc", "precision", "recall", "specificity", "accuracy"]
TSV_FIELDNAMES = [
    "scope",
    "group",
    "metric",
    "baseline",
    "neural",
    "neural_minus_baseline",
    "better_model",
]


@dataclass(frozen=True)
class SiteModelCompareConfig:
    """Configuration for site-level model comparison."""

    outdir: str
    site_baseline_dir: str
    site_neural_dir: str
    site_stratified_eval_dir: Optional[str] = None
    site_aggregation_dir: Optional[str] = None
    title: str = "BABAPPA site model comparison"

    def __post_init__(self) -> None:
        baseline = Path(self.site_baseline_dir)
        neural = Path(self.site_neural_dir)
        if not (baseline / "site_baseline_metrics.json").exists():
            raise ValueError(f"missing baseline metrics: {baseline}")
        if not (neural / "site_neural_metrics.json").exists():
            raise ValueError(f"missing neural metrics: {neural}")
        if self.site_stratified_eval_dir is not None and not Path(self.site_stratified_eval_dir).exists():
            raise ValueError(f"site_stratified_eval_dir does not exist: {self.site_stratified_eval_dir}")
        if self.site_aggregation_dir is not None and not Path(self.site_aggregation_dir).exists():
            raise ValueError(f"site_aggregation_dir does not exist: {self.site_aggregation_dir}")
        Path(self.outdir).mkdir(parents=True, exist_ok=True)


def compare_site_models(config: SiteModelCompareConfig) -> dict:
    """Compare site baseline and neural metrics."""
    outdir = Path(config.outdir)
    baseline = _load_json(Path(config.site_baseline_dir) / "site_baseline_metrics.json")
    neural = _load_json(Path(config.site_neural_dir) / "site_neural_metrics.json")
    aggregation = (
        _load_json(Path(config.site_aggregation_dir) / "site_to_gene_metrics.json")
        if config.site_aggregation_dir is not None
        else None
    )
    rows: List[dict] = []
    comparisons = {
        "split": _compare_scope(
            baseline.get("metrics_by_split", {}),
            neural.get("metrics_by_split", {}),
            rows,
            "split",
        ),
        "saturation_tier": _compare_scope(
            baseline.get("metrics_by_saturation_tier", {}),
            neural.get("metrics_by_saturation_tier", {}),
            rows,
            "saturation_tier",
        ),
        "method": _compare_scope(
            baseline.get("metrics_by_method", {}),
            neural.get("metrics_by_method", {}),
            rows,
            "method",
        ),
    }
    recommendation = _recommend(comparisons)
    payload = {
        "site_model_comparison_version": SITE_MODEL_COMPARISON_VERSION,
        "title": config.title,
        "inputs": {
            "site_baseline_dir": config.site_baseline_dir,
            "site_neural_dir": config.site_neural_dir,
            "site_stratified_eval_dir": config.site_stratified_eval_dir,
            "site_aggregation_dir": config.site_aggregation_dir,
        },
        "comparison": comparisons,
        "site_to_gene_aggregation": _aggregation_summary(aggregation),
        "recommendation": recommendation,
        "generated_files": {
            "json": str(outdir / "site_model_comparison.json"),
            "tsv": str(outdir / "site_model_comparison.tsv"),
            "markdown": str(outdir / "site_model_comparison.md"),
        },
    }
    _write_json(outdir / "site_model_comparison.json", payload)
    write_tsv(outdir / "site_model_comparison.tsv", rows, TSV_FIELDNAMES)
    (outdir / "site_model_comparison.md").write_text(_render_markdown(payload), encoding="utf-8")
    return {
        "status": "ok",
        "outdir": str(outdir),
        "json": str(outdir / "site_model_comparison.json"),
        "tsv": str(outdir / "site_model_comparison.tsv"),
        "markdown": str(outdir / "site_model_comparison.md"),
        "recommendation": recommendation,
    }


def _compare_scope(baseline: Dict[str, dict], neural: Dict[str, dict], rows: List[dict], scope: str) -> dict:
    result = {}
    for group in sorted(set(baseline) | set(neural)):
        group_result = {}
        for metric in METRICS:
            b = baseline.get(group, {}).get(metric)
            n = neural.get(group, {}).get(metric)
            delta = None if b is None or n is None else float(n) - float(b)
            better = ""
            if delta is not None:
                better = "neural" if delta > 0 else "baseline" if delta < 0 else "tie"
            rows.append(
                {
                    "scope": scope,
                    "group": group,
                    "metric": metric,
                    "baseline": b,
                    "neural": n,
                    "neural_minus_baseline": delta,
                    "better_model": better,
                }
            )
            group_result[metric] = {
                "baseline": b,
                "neural": n,
                "neural_minus_baseline": delta,
                "better_model": better,
            }
        result[group] = group_result
    return result


def _recommend(comparisons: dict) -> str:
    split_all = comparisons.get("split", {}).get("all", {})
    auroc_delta = (split_all.get("auroc") or {}).get("neural_minus_baseline")
    precision_delta = (split_all.get("precision") or {}).get("neural_minus_baseline")
    if auroc_delta is not None and auroc_delta > 0.02 and (precision_delta is None or precision_delta > 0):
        return "Use the neural site scorer as the primary scorer, retaining the NumPy baseline as a transparent reference."
    return "Keep the NumPy baseline as a transparent reference and use the neural scorer after stability and calibration checks."


def _aggregation_summary(payload: Optional[dict]) -> dict:
    if not payload:
        return {}
    default = payload.get("gene_level_metrics_default", {}).get("all", {})
    return {
        "default_score": payload.get("default_score"),
        "all_auroc": default.get("auroc"),
        "all_f1": default.get("f1"),
        "n_family_method_rows": payload.get("n_family_method_rows"),
    }


def _render_markdown(payload: dict) -> str:
    split_all = payload.get("comparison", {}).get("split", {}).get("all", {})
    agg = payload.get("site_to_gene_aggregation", {})
    return "\n".join(
        [
            f"# {payload.get('title')}",
            "",
            "## Baseline vs neural by split",
            "",
            f"- All-split AUROC delta: {(split_all.get('auroc') or {}).get('neural_minus_baseline')}",
            f"- All-split F1 delta: {(split_all.get('f1') or {}).get('neural_minus_baseline')}",
            "",
            "## Baseline vs neural by saturation tier",
            "",
            "See `site_model_comparison.tsv` for tier-wise deltas.",
            "",
            "## Baseline vs neural by method",
            "",
            "See `site_model_comparison.tsv` for method-wise deltas.",
            "",
            "## Site-to-gene aggregation",
            "",
            f"- Default score: {agg.get('default_score')}",
            f"- All AUROC: {agg.get('all_auroc')}",
            "",
            "## Interpretation",
            "",
            "Site-level model comparison is meaningful because both models use the same oracle-supervised site dataset and leakage policy.",
            "",
            "## Recommendation",
            "",
            payload.get("recommendation", ""),
            "",
        ]
    )


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
