"""Branch/site aggregation for branch-conditioned BABAPPA predictions."""

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

BRANCH_AGGREGATION_VERSION = __version__
SITE_TO_BRANCH_FIELDNAMES = [
    "family_id",
    "method",
    "split",
    "saturation_tier",
    "branch_id",
    "branch_label",
    "gene_label",
    "n_sites",
    "max_branch_site_probability",
    "mean_branch_site_probability",
    "top5_mean_branch_site_probability",
    "count_sites_prob_ge_0_5",
    "fraction_sites_prob_ge_0_5",
]
BRANCH_TO_GENE_FIELDNAMES = [
    "family_id",
    "method",
    "split",
    "saturation_tier",
    "gene_label",
    "n_branches",
    "max_branch_probability",
    "mean_branch_probability",
    "count_branches_prob_ge_0_5",
]


@dataclass(frozen=True)
class BranchAggregationConfig:
    """Configuration for branch-site aggregation."""

    predictions_tsv: str
    outdir: str
    probability_column: str = "prob_positive"
    label_column: str = "y_branch_site"
    gene_label_column: str = "gene_label"

    def __post_init__(self) -> None:
        if not Path(self.predictions_tsv).exists():
            raise ValueError(f"predictions_tsv does not exist: {self.predictions_tsv}")
        Path(self.outdir).mkdir(parents=True, exist_ok=True)


def aggregate_branch_sites(config: BranchAggregationConfig) -> dict:
    """Aggregate branch-site probabilities to site-to-branch and branch-to-gene tables."""
    rows = read_tsv(Path(config.predictions_tsv))
    if not rows:
        raise ValueError("predictions_tsv contains no rows")
    site_to_branch = _site_to_branch(rows, config)
    branch_to_gene = _branch_to_gene(site_to_branch)
    metrics = _metrics(site_to_branch, branch_to_gene)
    outdir = Path(config.outdir)
    site_to_branch_path = outdir / "branch_site_to_branch_predictions.tsv"
    branch_to_gene_path = outdir / "branch_to_gene_predictions.tsv"
    metrics_path = outdir / "branch_aggregation_metrics.json"
    markdown_path = outdir / "branch_aggregation_report.md"
    write_tsv(site_to_branch_path, site_to_branch, SITE_TO_BRANCH_FIELDNAMES)
    write_tsv(branch_to_gene_path, branch_to_gene, BRANCH_TO_GENE_FIELDNAMES)
    payload = {
        "branch_aggregation_version": BRANCH_AGGREGATION_VERSION,
        "predictions_tsv": str(Path(config.predictions_tsv)),
        "n_site_to_branch_rows": len(site_to_branch),
        "n_branch_to_gene_rows": len(branch_to_gene),
        "default_branch_score": "max_branch_site_probability",
        "default_gene_score": "max_branch_probability",
        **metrics,
        "generated_files": {
            "site_to_branch": str(site_to_branch_path),
            "branch_to_gene": str(branch_to_gene_path),
            "metrics": str(metrics_path),
            "markdown": str(markdown_path),
        },
        "interpretation": "Branch-site scores are first aggregated to branch support, then to gene support.",
    }
    _write_json(metrics_path, payload)
    markdown_path.write_text(_render_markdown(payload), encoding="utf-8")
    return {
        "status": "ok",
        "outdir": str(outdir),
        "site_to_branch": str(site_to_branch_path),
        "branch_to_gene": str(branch_to_gene_path),
        "metrics": str(metrics_path),
        "markdown": str(markdown_path),
        "n_site_to_branch_rows": len(site_to_branch),
        "n_branch_to_gene_rows": len(branch_to_gene),
    }


def validate_branch_aggregation_dir(aggregation_dir: str | Path) -> dict:
    """Validate branch aggregation artifacts."""
    path = Path(aggregation_dir)
    failures: List[str] = []
    warnings: List[str] = []
    branch_rows = _read_tsv(path / "branch_site_to_branch_predictions.tsv", failures)
    gene_rows = _read_tsv(path / "branch_to_gene_predictions.tsv", failures)
    _load_json(path / "branch_aggregation_metrics.json", failures)
    markdown = path / "branch_aggregation_report.md"
    if not markdown.exists():
        failures.append(f"missing_file:{markdown}")
    elif not markdown.read_text(encoding="utf-8").strip():
        failures.append("empty_markdown")
    for row in branch_rows:
        for column in ["max_branch_site_probability", "mean_branch_site_probability", "top5_mean_branch_site_probability", "fraction_sites_prob_ge_0_5"]:
            _check_probability(row, column, failures)
    for row in gene_rows:
        for column in ["max_branch_probability", "mean_branch_probability"]:
            _check_probability(row, column, failures)
    if not branch_rows:
        failures.append("no_site_to_branch_rows")
    if not gene_rows:
        failures.append("no_branch_to_gene_rows")
    return {
        "status": "fail" if failures else "ok",
        "n_rows": len(branch_rows),
        "n_gene_rows": len(gene_rows),
        "n_fail": len(failures),
        "n_warning": len(warnings),
        "failures": failures,
        "warnings": warnings,
    }


def _site_to_branch(rows: List[dict], config: BranchAggregationConfig) -> List[dict]:
    grouped: Dict[tuple, List[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row.get("family_id", ""), row.get("method", ""), row.get("branch_id", ""))].append(row)
    output = []
    for (family_id, method, branch_id), group in sorted(grouped.items()):
        probs = np.array([float(row.get(config.probability_column, 0.0)) for row in group], dtype=np.float64)
        labels = [row.get(config.label_column, "") for row in group]
        branch_label = 1 if any(str(label) in {"1", "1.0"} for label in labels) else 0
        gene_label = _first_nonempty(row.get(config.gene_label_column, "") for row in group)
        split = _first_nonempty(row.get("split", "") for row in group)
        tier = _first_nonempty(row.get("saturation_tier", "") for row in group) or "unknown"
        output.append(
            {
                "family_id": family_id,
                "method": method,
                "split": split,
                "saturation_tier": tier,
                "branch_id": branch_id,
                "branch_label": branch_label,
                "gene_label": gene_label,
                "n_sites": int(probs.size),
                "max_branch_site_probability": float(probs.max()) if probs.size else 0.0,
                "mean_branch_site_probability": float(probs.mean()) if probs.size else 0.0,
                "top5_mean_branch_site_probability": _topk_mean(probs, 5),
                "count_sites_prob_ge_0_5": int((probs >= 0.5).sum()),
                "fraction_sites_prob_ge_0_5": float((probs >= 0.5).mean()) if probs.size else 0.0,
            }
        )
    return output


def _branch_to_gene(rows: List[dict]) -> List[dict]:
    grouped: Dict[tuple, List[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row.get("family_id", ""), row.get("method", ""))].append(row)
    output = []
    for (family_id, method), group in sorted(grouped.items()):
        probs = np.array([float(row.get("max_branch_site_probability", 0.0)) for row in group], dtype=np.float64)
        output.append(
            {
                "family_id": family_id,
                "method": method,
                "split": _first_nonempty(row.get("split", "") for row in group),
                "saturation_tier": _first_nonempty(row.get("saturation_tier", "") for row in group) or "unknown",
                "gene_label": _first_nonempty(row.get("gene_label", "") for row in group),
                "n_branches": len(group),
                "max_branch_probability": float(probs.max()) if probs.size else 0.0,
                "mean_branch_probability": float(probs.mean()) if probs.size else 0.0,
                "count_branches_prob_ge_0_5": int((probs >= 0.5).sum()),
            }
        )
    return output


def _metrics(branch_rows: List[dict], gene_rows: List[dict]) -> dict:
    branch_labeled = [row for row in branch_rows if str(row.get("branch_label", "")) in {"0", "1", "0.0", "1.0"}]
    gene_labeled = [row for row in gene_rows if str(row.get("gene_label", "")) in {"0", "1", "0.0", "1.0"}]
    branch_y = np.array([int(float(row["branch_label"])) for row in branch_labeled], dtype=np.int32)
    branch_score = np.array([float(row["max_branch_site_probability"]) for row in branch_labeled], dtype=np.float64)
    gene_y = np.array([int(float(row["gene_label"])) for row in gene_labeled], dtype=np.int32)
    gene_score = np.array([float(row["max_branch_probability"]) for row in gene_labeled], dtype=np.float64)
    return {
        "branch_level_metrics_default": {
            "all": _compute_binary_metrics(branch_y, branch_score, threshold=0.5),
            "by_split": _metrics_by_field(branch_labeled, branch_y, branch_score, "split"),
            "by_method": _metrics_by_field(branch_labeled, branch_y, branch_score, "method"),
        },
        "gene_level_metrics_default": {
            "all": _compute_binary_metrics(gene_y, gene_score, threshold=0.5),
            "by_split": _metrics_by_field(gene_labeled, gene_y, gene_score, "split"),
            "by_method": _metrics_by_field(gene_labeled, gene_y, gene_score, "method"),
        },
    }


def _metrics_by_field(rows: List[dict], y: np.ndarray, score: np.ndarray, field: str) -> Dict[str, dict]:
    result = {}
    for value in sorted({row.get(field, "") for row in rows}):
        mask = np.array([row.get(field, "") == value for row in rows])
        result[value or "unknown"] = _compute_binary_metrics(y[mask], score[mask], threshold=0.5)
    return result


def _topk_mean(probs: np.ndarray, k: int) -> float:
    if probs.size == 0:
        return 0.0
    return float(np.sort(probs)[-min(k, probs.size):].mean())


def _first_nonempty(values) -> str:
    for value in values:
        if value not in ("", None):
            return str(value)
    return ""


def _check_probability(row: dict, column: str, failures: List[str]) -> None:
    try:
        value = float(row.get(column, "nan"))
    except ValueError:
        failures.append(f"invalid_probability:{column}:{row.get('family_id')}:{row.get('branch_id', '')}")
        return
    if not 0 <= value <= 1:
        failures.append(f"probability_out_of_range:{column}:{value}")


def _read_tsv(path: Path, failures: List[str]) -> List[dict]:
    if not path.exists():
        failures.append(f"missing_file:{path}")
        return []
    try:
        return read_tsv(path)
    except OSError as exc:
        failures.append(f"could_not_read_tsv:{path}:{exc}")
        return []


def _load_json(path: Path, failures: List[str]) -> dict:
    if not path.exists():
        failures.append(f"missing_file:{path}")
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        failures.append(f"could_not_parse_json:{path}:{exc}")
        return {}
    return payload if isinstance(payload, dict) else {}


def _render_markdown(payload: dict) -> str:
    branch = payload.get("branch_level_metrics_default", {}).get("all", {})
    gene = payload.get("gene_level_metrics_default", {}).get("all", {})
    return "\n".join(
        [
            "# Branch/site aggregation",
            "",
            f"- Site-to-branch rows: {payload.get('n_site_to_branch_rows')}",
            f"- Branch-to-gene rows: {payload.get('n_branch_to_gene_rows')}",
            f"- Branch-level AUROC: {branch.get('auroc')}",
            f"- Gene-level AUROC: {gene.get('auroc')}",
            "",
            "This is branch-conditioned simulation-supervised aggregation, not empirical branch-site inference.",
            "",
        ]
    )


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
