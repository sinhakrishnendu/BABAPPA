"""Construct site-level feature datasets from oracle site labels."""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from babappa import __version__
from babappa.datasets.index import read_tsv, write_tsv
from babappa.training.neural_data import resolve_tensor_file

SITE_DATASET_VERSION = __version__
SITE_FEATURE_FIELDNAMES = [
    "site_id",
    "family_id",
    "original_family_id",
    "source_dataset",
    "method",
    "saturation_tier",
    "split",
    "tensor_file",
    "labels_file",
    "site_index_zero",
    "site_index_one",
    "aligned_site_index_zero",
    "aligned_site_index_one",
    "original_site_index_zero",
    "original_site_index_one",
    "mapping_status",
    "mapping_confidence",
    "mappable_site",
    "y_site",
    "site_relative_position",
    "n_taxa",
    "n_codons",
    "codon_id_mean",
    "codon_id_std",
    "codon_id_min",
    "codon_id_max",
    "codon_id_range",
    "codon_id_unique_count",
    "gap_fraction",
    "non_gap_fraction",
    "taxon_codon_variability",
    "foreground_taxon_present",
    "foreground_taxon_index",
    "foreground_codon_id",
    "background_codon_id_mean",
    "foreground_background_abs_delta",
    "foreground_gap",
]
SITE_SPLIT_FIELDNAMES = [
    "site_id",
    "family_id",
    "method",
    "saturation_tier",
    "split",
    "site_index_zero",
    "aligned_site_index_zero",
    "original_site_index_zero",
    "mapping_status",
    "mappable_site",
    "y_site",
]


@dataclass(frozen=True)
class SiteDatasetConfig:
    """Configuration for building a site-level BABAPPA dataset."""

    dataset_dir: str
    oracle_labels_tsv: str
    outdir: str
    include_foreground_context: bool = True
    max_sites_per_family_method: Optional[int] = None
    negative_downsample_ratio: Optional[float] = None
    seed: int = 42
    require_mappable_sites: bool = True

    def __post_init__(self) -> None:
        dataset_path = Path(self.dataset_dir)
        labels_path = Path(self.oracle_labels_tsv)
        out_path = Path(self.outdir)
        if not dataset_path.exists():
            raise ValueError(f"dataset_dir does not exist: {dataset_path}")
        if not labels_path.exists():
            raise ValueError(f"oracle_labels_tsv does not exist: {labels_path}")
        if self.max_sites_per_family_method is not None and self.max_sites_per_family_method <= 0:
            raise ValueError("max_sites_per_family_method must be positive when supplied")
        if self.negative_downsample_ratio is not None and self.negative_downsample_ratio <= 0:
            raise ValueError("negative_downsample_ratio must be > 0 when supplied")
        out_path.mkdir(parents=True, exist_ok=True)


def build_site_dataset(config: SiteDatasetConfig) -> dict:
    """Build a site-level feature table and split table."""
    dataset_dir = Path(config.dataset_dir)
    outdir = Path(config.outdir)
    warnings: List[str] = []
    oracle_rows = read_tsv(Path(config.oracle_labels_tsv))
    oracle_rows = _filter_mappable_rows(oracle_rows, config, warnings)
    oracle_rows = _limit_sites_per_family_method(oracle_rows, config.max_sites_per_family_method)
    oracle_rows, downsample_stats = _downsample_negatives(oracle_rows, config)
    tensor_cache: Dict[str, Tuple[np.ndarray, List[str]]] = {}
    feature_rows: List[dict] = []
    split_rows: List[dict] = []

    for row in oracle_rows:
        try:
            tensor, taxa_order = _load_tensor_cached(
                row.get("tensor_file", ""), dataset_dir, tensor_cache
            )
        except (OSError, ValueError, FileNotFoundError) as exc:
            warnings.append(f"tensor_load_failed:{row.get('tensor_file', '')}:{exc}")
            continue
        site_index = _site_index_for_tensor(row)
        if site_index < 0 or site_index >= tensor.shape[1]:
            warnings.append(
                f"site_index_out_of_bounds:{row.get('family_id', '')}:{site_index}"
            )
            continue
        site_id = _site_id(row.get("family_id", ""), row.get("method", ""), site_index)
        features = _extract_site_features(row, tensor, taxa_order, site_index, config, warnings)
        feature_row = {
            "site_id": site_id,
            "family_id": row.get("family_id", ""),
            "original_family_id": row.get("original_family_id", ""),
            "source_dataset": row.get("source_dataset", ""),
            "method": row.get("method", ""),
            "saturation_tier": row.get("saturation_tier", "unknown") or "unknown",
            "split": row.get("split", ""),
            "tensor_file": row.get("tensor_file", ""),
            "labels_file": row.get("labels_file", ""),
            "site_index_zero": site_index,
            "site_index_one": site_index + 1,
            "aligned_site_index_zero": row.get("aligned_site_index_zero", ""),
            "aligned_site_index_one": row.get("aligned_site_index_one", ""),
            "original_site_index_zero": row.get("original_site_index_zero", ""),
            "original_site_index_one": row.get("original_site_index_one", ""),
            "mapping_status": row.get("mapping_status", ""),
            "mapping_confidence": row.get("mapping_confidence", ""),
            "mappable_site": row.get("mappable_site", ""),
            "y_site": _safe_binary(row.get("y_site")),
        }
        feature_row.update(features)
        feature_rows.append(feature_row)
        split_rows.append(
            {
                "site_id": site_id,
                "family_id": feature_row["family_id"],
                "method": feature_row["method"],
                "saturation_tier": feature_row["saturation_tier"],
                "split": feature_row["split"],
                "site_index_zero": site_index,
                "aligned_site_index_zero": feature_row["aligned_site_index_zero"],
                "original_site_index_zero": feature_row["original_site_index_zero"],
                "mapping_status": feature_row["mapping_status"],
                "mappable_site": feature_row["mappable_site"],
                "y_site": feature_row["y_site"],
            }
        )

    feature_rows.sort(key=lambda row: (row["family_id"], row["method"], int(row["site_index_zero"])))
    split_rows.sort(key=lambda row: row["site_id"])
    features_path = outdir / "site_features.tsv"
    splits_path = outdir / "site_splits.tsv"
    index_path = outdir / "site_dataset_index.json"
    markdown_path = outdir / "site_dataset.md"
    write_tsv(features_path, feature_rows, SITE_FEATURE_FIELDNAMES)
    write_tsv(splits_path, split_rows, SITE_SPLIT_FIELDNAMES)

    n_positive = sum(int(row["y_site"]) for row in feature_rows)
    n_negative = len(feature_rows) - n_positive
    payload = {
        "site_dataset_version": SITE_DATASET_VERSION,
        "dataset_dir": str(dataset_dir),
        "oracle_labels_tsv": str(Path(config.oracle_labels_tsv)),
        "n_site_rows": len(feature_rows),
        "n_positive_sites": n_positive,
        "n_negative_sites": n_negative,
        "positive_fraction": None if not feature_rows else n_positive / len(feature_rows),
        "split_counts": dict(sorted(Counter(row["split"] for row in feature_rows).items())),
        "positive_counts_by_split": dict(
            sorted(Counter(row["split"] for row in feature_rows if int(row["y_site"]) == 1).items())
        ),
        "saturation_tier_counts": dict(
            sorted(Counter(row["saturation_tier"] for row in feature_rows).items())
        ),
        "method_counts": dict(sorted(Counter(row["method"] for row in feature_rows).items())),
        "negative_downsample_ratio": config.negative_downsample_ratio,
        "require_mappable_sites": config.require_mappable_sites,
        "downsampling_stats": downsample_stats,
        "files": {
            "features": str(features_path),
            "splits": str(splits_path),
            "index": str(index_path),
            "markdown": str(markdown_path),
        },
        "warnings": sorted(set(warnings)),
        "note": "Site-level oracle labels are supervised targets; truth fields are excluded from predictive features.",
    }
    _write_json(index_path, payload)
    markdown_path.write_text(_render_markdown(payload), encoding="utf-8")
    return {
        "status": "ok",
        "outdir": str(outdir),
        "features": str(features_path),
        "splits": str(splits_path),
        "index": str(index_path),
        "markdown": str(markdown_path),
        "n_site_rows": len(feature_rows),
        "n_positive_sites": n_positive,
        "n_negative_sites": n_negative,
        "warnings": payload["warnings"],
    }


def _filter_mappable_rows(
    rows: List[dict], config: SiteDatasetConfig, warnings: List[str]
) -> List[dict]:
    if not config.require_mappable_sites:
        return rows
    if not rows or "mappable_site" not in rows[0]:
        return rows
    if not any(row.get("mappable_site") not in ("", None) for row in rows):
        return rows
    kept = [row for row in rows if str(row.get("mappable_site", "1")).strip() in {"1", "1.0", "true", "True"}]
    dropped = len(rows) - len(kept)
    if dropped:
        warnings.append(f"unmappable_site_rows_dropped:{dropped}")
    return kept


def _site_index_for_tensor(row: dict) -> int:
    if row.get("aligned_site_index_zero") not in ("", None):
        return _safe_int(row.get("aligned_site_index_zero"), default=-1)
    return _safe_int(row.get("site_index_zero"), default=-1)


def _limit_sites_per_family_method(rows: List[dict], limit: Optional[int]) -> List[dict]:
    if limit is None:
        return rows
    grouped: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row.get("family_id", ""), row.get("method", ""))].append(row)
    limited: List[dict] = []
    for key in sorted(grouped):
        group_rows = sorted(grouped[key], key=lambda row: int(float(row.get("site_index_zero", 0))))
        positives = [row for row in group_rows if str(row.get("y_site")) == "1"]
        negatives = [row for row in group_rows if str(row.get("y_site")) != "1"]
        selected = positives[:limit]
        remaining = max(0, limit - len(selected))
        selected.extend(negatives[:remaining])
        limited.extend(selected)
    return limited


def _downsample_negatives(rows: List[dict], config: SiteDatasetConfig) -> Tuple[List[dict], dict]:
    if config.negative_downsample_ratio is None:
        return rows, {}
    rng = random.Random(config.seed)
    grouped: Dict[Tuple[str, str, str], List[dict]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                row.get("split", ""),
                row.get("saturation_tier", "unknown") or "unknown",
                row.get("method", ""),
            )
        ].append(row)
    selected: List[dict] = []
    stats = {}
    for key in sorted(grouped):
        group_rows = grouped[key]
        positives = [row for row in group_rows if str(row.get("y_site")) == "1"]
        negatives = [row for row in group_rows if str(row.get("y_site")) != "1"]
        if positives:
            max_negatives = int(round(config.negative_downsample_ratio * len(positives)))
            max_negatives = max(max_negatives, 1)
            sampled_negatives = list(negatives)
            rng.shuffle(sampled_negatives)
            sampled_negatives = sampled_negatives[:max_negatives]
        else:
            sampled_negatives = negatives
        selected.extend(positives)
        selected.extend(sampled_negatives)
        stats["::".join(key)] = {
            "positives": len(positives),
            "negatives_before": len(negatives),
            "negatives_after": len(sampled_negatives),
        }
    return selected, stats


def _load_tensor_cached(
    tensor_file: str, dataset_dir: Path, cache: Dict[str, Tuple[np.ndarray, List[str]]]
) -> Tuple[np.ndarray, List[str]]:
    if tensor_file in cache:
        return cache[tensor_file]
    tensor_path = resolve_tensor_file(tensor_file, dataset_dir)
    with np.load(tensor_path, allow_pickle=False) as shard:
        if "X" not in shard.files:
            raise ValueError(f"tensor shard missing X array: {tensor_path}")
        tensor = shard["X"]
        taxa_order = [str(value) for value in shard["taxa_order"].tolist()] if "taxa_order" in shard.files else []
    if tensor.ndim != 3:
        raise ValueError(f"X array is not 3-dimensional: {tensor_path}")
    cache[tensor_file] = (tensor, taxa_order)
    return tensor, taxa_order


def _extract_site_features(
    row: dict,
    tensor: np.ndarray,
    taxa_order: List[str],
    site_index: int,
    config: SiteDatasetConfig,
    warnings: List[str],
) -> dict:
    n_taxa, n_codons, n_channels = tensor.shape
    site_values = tensor[:, site_index, :]
    codon_ids = site_values[:, 0].astype(np.float64)
    gaps = site_values[:, 1].astype(np.float64) if n_channels > 1 else np.zeros(n_taxa)
    foreground_taxon = row.get("foreground_taxon", "")
    foreground_present = 1 if foreground_taxon else 0
    foreground_index = -1
    if config.include_foreground_context and foreground_taxon and taxa_order:
        try:
            foreground_index = taxa_order.index(foreground_taxon)
        except ValueError:
            warnings.append(f"foreground_taxon_unresolved:{foreground_taxon}")
    elif config.include_foreground_context and foreground_taxon:
        warnings.append(f"foreground_taxon_unresolved:{foreground_taxon}")

    if foreground_index >= 0:
        foreground_codon_id = float(codon_ids[foreground_index])
        foreground_gap = float(gaps[foreground_index])
        background_mask = np.ones(n_taxa, dtype=bool)
        background_mask[foreground_index] = False
        background_values = codon_ids[background_mask]
        background_mean = float(background_values.mean()) if background_values.size else float(codon_ids.mean())
        delta = abs(foreground_codon_id - background_mean)
    else:
        foreground_codon_id = -1.0
        foreground_gap = 0.0
        background_mean = float(codon_ids.mean()) if codon_ids.size else 0.0
        delta = 0.0

    codon_min = float(codon_ids.min()) if codon_ids.size else 0.0
    codon_max = float(codon_ids.max()) if codon_ids.size else 0.0
    return {
        "site_relative_position": 0.0 if n_codons <= 1 else site_index / (n_codons - 1),
        "n_taxa": n_taxa,
        "n_codons": n_codons,
        "codon_id_mean": float(codon_ids.mean()) if codon_ids.size else 0.0,
        "codon_id_std": float(codon_ids.std()) if codon_ids.size else 0.0,
        "codon_id_min": codon_min,
        "codon_id_max": codon_max,
        "codon_id_range": codon_max - codon_min,
        "codon_id_unique_count": int(np.unique(codon_ids.astype(np.int64)).size),
        "gap_fraction": float(gaps.mean()) if gaps.size else 0.0,
        "non_gap_fraction": float(1.0 - gaps.mean()) if gaps.size else 1.0,
        "taxon_codon_variability": float(codon_ids.std()) if codon_ids.size else 0.0,
        "foreground_taxon_present": foreground_present,
        "foreground_taxon_index": foreground_index,
        "foreground_codon_id": foreground_codon_id,
        "background_codon_id_mean": background_mean,
        "foreground_background_abs_delta": float(delta),
        "foreground_gap": foreground_gap,
    }


def _site_id(family_id: str, method: str, site_index: int) -> str:
    return f"{family_id}::{method}::site_{site_index:06d}"


def _safe_int(value: object, default: int = 0) -> int:
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return default


def _safe_binary(value: object) -> int:
    return 1 if str(value).strip() in {"1", "1.0", "true", "True"} else 0


def _render_markdown(payload: dict) -> str:
    lines = [
        "# BABAPPA site-level dataset",
        "",
        "## Inputs",
        "",
        f"- Dataset directory: `{payload.get('dataset_dir')}`",
        f"- Oracle labels TSV: `{payload.get('oracle_labels_tsv')}`",
        "",
        "## Site counts",
        "",
        f"- Site rows: {payload.get('n_site_rows')}",
        f"- Positive sites: {payload.get('n_positive_sites')}",
        f"- Negative sites: {payload.get('n_negative_sites')}",
        f"- Positive fraction: {payload.get('positive_fraction')}",
        "",
        "## Class balance",
        "",
        f"- Positive counts by split: {payload.get('positive_counts_by_split')}",
        "",
        "## Splits",
        "",
        f"- Split counts: {payload.get('split_counts')}",
        "",
        "## Saturation tiers",
        "",
        f"- Saturation tier counts: {payload.get('saturation_tier_counts')}",
        "",
        "## Methods",
        "",
        f"- Method counts: {payload.get('method_counts')}",
        "",
        "## Warnings",
        "",
    ]
    warnings = payload.get("warnings") or []
    lines.extend([f"- {warning}" for warning in warnings] if warnings else ["- none"])
    lines.extend(
        [
            "",
            "## Leakage note",
            "",
            "The target `y_site` is derived from oracle selected-site labels and is never used as an input feature.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
