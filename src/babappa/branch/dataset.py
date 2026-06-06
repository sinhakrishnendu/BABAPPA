"""Branch-conditioned site feature datasets for BABAPPA."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from babappa import __version__
from babappa.datasets.index import read_tsv, write_tsv
from babappa.training.neural_data import resolve_tensor_file

BRANCH_SITE_DATASET_VERSION = __version__
BRANCH_FEATURE_FIELDNAMES = [
    "branch_site_id",
    "family_id",
    "original_family_id",
    "source_dataset",
    "method",
    "saturation_tier",
    "split",
    "tensor_file",
    "labels_file",
    "branch_id",
    "foreground_taxon",
    "site_index_zero",
    "site_index_one",
    "aligned_site_index_zero",
    "aligned_site_index_one",
    "original_site_index_zero",
    "original_site_index_one",
    "mapping_status",
    "mapping_confidence",
    "mappable_site",
    "y_branch_site",
    "y_site",
    "gene_label",
    "foreground_branch_present",
    "branch_label_source",
    "site_relative_position",
    "site_centered_position",
    "site_terminal_distance",
    "n_taxa",
    "n_codons",
    "log_n_taxa",
    "log_n_codons",
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
    "branch_query_id_numeric",
    "foreground_codon_id",
    "foreground_gap",
    "branch_codon_id",
    "branch_gap",
    "background_mean_codon_id",
    "foreground_background_codon_delta",
    "branch_background_codon_delta",
]
BRANCH_SPLIT_FIELDNAMES = [
    "branch_site_id",
    "family_id",
    "method",
    "saturation_tier",
    "split",
    "branch_id",
    "site_index_zero",
    "aligned_site_index_zero",
    "original_site_index_zero",
    "mapping_status",
    "mappable_site",
    "y_branch_site",
]
FORBIDDEN_FEATURE_COLUMNS = {
    "y_branch_site",
    "y_site",
    "gene_label",
    "selected_sites",
    "n_selected_sites",
    "truth_label",
    "positive_sites",
    "oracle_selected_sites",
    "site_labels",
    "branch_labels",
    "branch_label_source",
    "labels_file",
}
METADATA_COLUMNS = {
    "branch_site_id",
    "family_id",
    "original_family_id",
    "source_dataset",
    "method",
    "saturation_tier",
    "split",
    "tensor_file",
    "labels_file",
    "branch_id",
    "foreground_taxon",
    "site_index_one",
    "aligned_site_index_one",
    "original_site_index_one",
    "mapping_status",
    "mapping_confidence",
    "mappable_site",
    "branch_label_source",
}
LEAKAGE_TOKENS = ("selected", "truth", "positive", "label")
SENSITIVE_CONTEXT_COLUMNS = [
    "foreground_branch_present",
    "foreground_taxon_present",
    "foreground_taxon_index",
    "branch_query_id_numeric",
    "foreground_codon_id",
    "foreground_gap",
    "foreground_background_codon_delta",
]


@dataclass(frozen=True)
class BranchSiteDatasetConfig:
    """Configuration for building branch-conditioned site features."""

    dataset_dir: str
    branch_site_labels_tsv: str
    outdir: str
    negative_downsample_ratio: Optional[float] = None
    seed: int = 42
    require_mappable_sites: bool = True
    max_input_rows: Optional[int] = None
    max_output_rows: Optional[int] = None
    max_rows_per_split: Optional[int] = None
    max_negatives_per_positive: Optional[float] = None
    streaming: bool = True

    def __post_init__(self) -> None:
        dataset_path = Path(self.dataset_dir)
        label_path = Path(self.branch_site_labels_tsv)
        if not dataset_path.exists():
            raise ValueError(f"dataset_dir does not exist: {dataset_path}")
        if not label_path.exists():
            raise ValueError(f"branch_site_labels_tsv does not exist: {label_path}")
        if self.negative_downsample_ratio is not None and self.negative_downsample_ratio <= 0:
            raise ValueError("negative_downsample_ratio must be > 0 when supplied")
        if self.max_negatives_per_positive is not None and self.max_negatives_per_positive <= 0:
            raise ValueError("max_negatives_per_positive must be > 0 when supplied")
        for name in ("max_input_rows", "max_output_rows", "max_rows_per_split"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be > 0 when supplied")
        Path(self.outdir).mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class BranchSiteDatasetMergeConfig:
    """Configuration for merging branch-site feature datasets."""

    dataset_dirs: Sequence[str] | str
    outdir: str

    def __post_init__(self) -> None:
        dirs = _parse_dataset_dirs(self.dataset_dirs)
        if not dirs:
            raise ValueError("dataset_dirs must contain at least one branch-site dataset directory")
        for directory in dirs:
            path = Path(directory)
            for filename in ("branch_site_features.tsv", "branch_site_splits.tsv"):
                if not (path / filename).exists():
                    raise ValueError(f"dataset_dir is missing {filename}: {path}")
        object.__setattr__(self, "dataset_dirs", dirs)
        Path(self.outdir).mkdir(parents=True, exist_ok=True)


def build_branch_site_dataset(config: BranchSiteDatasetConfig) -> dict:
    """Build branch-conditioned site feature and split tables."""
    if config.streaming:
        return _build_branch_site_dataset_streaming(config)
    return _build_branch_site_dataset_in_memory(config)


def merge_branch_site_datasets(config: BranchSiteDatasetMergeConfig) -> dict:
    """Merge multiple branch-site feature datasets into one trainable directory."""

    outdir = Path(config.outdir)
    features_path = outdir / "branch_site_features.tsv"
    splits_path = outdir / "branch_site_splits.tsv"
    index_path = outdir / "branch_site_dataset_index.json"
    markdown_path = outdir / "branch_site_dataset.md"
    seen_ids = set()
    rows_written = 0
    positives_written = 0
    split_counts: Counter[str] = Counter()
    method_counts: Counter[str] = Counter()
    tier_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    warnings: List[str] = []

    with features_path.open("w", encoding="utf-8", newline="") as feature_handle, splits_path.open(
        "w", encoding="utf-8", newline=""
    ) as split_handle:
        feature_writer = csv.DictWriter(feature_handle, fieldnames=BRANCH_FEATURE_FIELDNAMES, delimiter="\t", lineterminator="\n")
        split_writer = csv.DictWriter(split_handle, fieldnames=BRANCH_SPLIT_FIELDNAMES, delimiter="\t", lineterminator="\n")
        feature_writer.writeheader()
        split_writer.writeheader()
        for dataset_dir in config.dataset_dirs:
            path = Path(dataset_dir)
            feature_rows = read_tsv(path / "branch_site_features.tsv")
            split_rows = read_tsv(path / "branch_site_splits.tsv")
            split_by_id = {row.get("branch_site_id", ""): row for row in split_rows}
            for row in feature_rows:
                branch_site_id = row.get("branch_site_id", "")
                if not branch_site_id:
                    warnings.append(f"missing_branch_site_id:{path}")
                    continue
                if branch_site_id in seen_ids:
                    warnings.append(f"duplicate_branch_site_id_skipped:{branch_site_id}")
                    continue
                seen_ids.add(branch_site_id)
                feature_writer.writerow(_project_row(row, BRANCH_FEATURE_FIELDNAMES))
                split_row = split_by_id.get(branch_site_id)
                if split_row is None:
                    warnings.append(f"missing_split_row:{branch_site_id}")
                    split_row = {key: row.get(key, "") for key in BRANCH_SPLIT_FIELDNAMES}
                split_writer.writerow(_project_row(split_row, BRANCH_SPLIT_FIELDNAMES))
                rows_written += 1
                positives_written += 1 if str(row.get("y_branch_site")) == "1" else 0
                split_counts[row.get("split", "")] += 1
                method_counts[row.get("method", "")] += 1
                tier_counts[row.get("saturation_tier", "")] += 1
                source_counts[str(path)] += 1

    payload = {
        "branch_site_dataset_version": BRANCH_SITE_DATASET_VERSION,
        "merge_kind": "branch_site_feature_table_merge",
        "source_dataset_dirs": [str(path) for path in config.dataset_dirs],
        "dataset_dir": "",
        "n_branch_site_rows": rows_written,
        "n_positive_branch_sites": positives_written,
        "n_negative_branch_sites": rows_written - positives_written,
        "positive_fraction": None if rows_written == 0 else positives_written / rows_written,
        "split_counts": dict(sorted(split_counts.items())),
        "saturation_tier_counts": dict(sorted(tier_counts.items())),
        "method_counts": dict(sorted(method_counts.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "feature_columns": _branch_site_feature_columns_from_fieldnames(),
        "forbidden_as_features": sorted(FORBIDDEN_FEATURE_COLUMNS),
        "sensitive_context_columns": [
            column for column in SENSITIVE_CONTEXT_COLUMNS if column in _branch_site_feature_columns_from_fieldnames()
        ],
        "files": {
            "features": str(features_path),
            "splits": str(splits_path),
            "index": str(index_path),
            "markdown": str(markdown_path),
        },
        "warnings": sorted(set(warnings)),
        "note": "Merged feature table for storage-safe variable-length retraining. Raw tensors are not required after features are written.",
    }
    _write_json(index_path, payload)
    markdown_path.write_text(_render_markdown(payload), encoding="utf-8")
    return {
        "status": "ok",
        "outdir": str(outdir),
        "features": str(features_path),
        "splits": str(splits_path),
        "index": str(index_path),
        "n_branch_site_rows": rows_written,
        "n_positive_branch_sites": positives_written,
        "n_negative_branch_sites": rows_written - positives_written,
        "warnings": payload["warnings"],
    }


def _build_branch_site_dataset_in_memory(config: BranchSiteDatasetConfig) -> dict:
    """Legacy branch-site dataset builder used only when streaming is disabled."""
    dataset_dir = Path(config.dataset_dir)
    outdir = Path(config.outdir)
    warnings: List[str] = []
    rows = read_tsv(Path(config.branch_site_labels_tsv))
    if config.max_input_rows is not None and len(rows) > config.max_input_rows:
        rows = rows[: config.max_input_rows]
        warnings.append(f"max_input_rows_reached:{config.max_input_rows}")
    rows = _filter_mappable_rows(rows, config, warnings)
    rows, downsampling_stats = _downsample_negatives(rows, config)
    if config.max_output_rows is not None and len(rows) > config.max_output_rows:
        positives = [row for row in rows if str(row.get("y_branch_site")) == "1"]
        negatives = [row for row in rows if str(row.get("y_branch_site")) != "1"]
        remaining = max(0, config.max_output_rows - len(positives))
        rows = positives + negatives[:remaining]
        warnings.append(f"max_output_rows_reached:{config.max_output_rows}")
    if config.max_rows_per_split is not None:
        limited = []
        split_counts: Counter[str] = Counter()
        for row in rows:
            split = row.get("split", "")
            if split_counts[split] >= config.max_rows_per_split:
                continue
            limited.append(row)
            split_counts[split] += 1
        if len(limited) < len(rows):
            warnings.append(f"max_rows_per_split_reached:{config.max_rows_per_split}")
        rows = limited
    tensor_cache: Dict[str, Tuple[np.ndarray, List[str]]] = {}
    site_feature_cache: Dict[Tuple[str, int], dict] = {}
    feature_rows: List[dict] = []
    split_rows: List[dict] = []

    for row in rows:
        built = _build_feature_and_split_row(
            row,
            dataset_dir,
            tensor_cache,
            warnings,
            site_feature_cache=site_feature_cache,
        )
        if built is None:
            continue
        feature_row, split_row = built
        feature_rows.append(feature_row)
        split_rows.append(split_row)

    feature_rows.sort(key=lambda row: (row["family_id"], row["method"], row["branch_id"], int(row["site_index_zero"])))
    split_rows.sort(key=lambda row: row["branch_site_id"])
    features_path = outdir / "branch_site_features.tsv"
    splits_path = outdir / "branch_site_splits.tsv"
    index_path = outdir / "branch_site_dataset_index.json"
    markdown_path = outdir / "branch_site_dataset.md"
    write_tsv(features_path, feature_rows, BRANCH_FEATURE_FIELDNAMES)
    write_tsv(splits_path, split_rows, BRANCH_SPLIT_FIELDNAMES)
    n_positive = sum(int(row["y_branch_site"]) for row in feature_rows)
    feature_columns = get_branch_site_feature_columns(feature_rows)
    payload = {
        "branch_site_dataset_version": BRANCH_SITE_DATASET_VERSION,
        "dataset_dir": str(dataset_dir),
        "branch_site_labels_tsv": str(Path(config.branch_site_labels_tsv)),
        "n_branch_site_rows": len(feature_rows),
        "n_positive_branch_sites": n_positive,
        "n_negative_branch_sites": len(feature_rows) - n_positive,
        "positive_fraction": None if not feature_rows else n_positive / len(feature_rows),
        "split_counts": dict(sorted(Counter(row["split"] for row in feature_rows).items())),
        "saturation_tier_counts": dict(sorted(Counter(row["saturation_tier"] for row in feature_rows).items())),
        "method_counts": dict(sorted(Counter(row["method"] for row in feature_rows).items())),
        "branch_counts": dict(sorted(Counter(row["branch_id"] for row in feature_rows).items())),
        "feature_columns": feature_columns,
        "forbidden_as_features": sorted(FORBIDDEN_FEATURE_COLUMNS),
        "sensitive_context_columns": [column for column in SENSITIVE_CONTEXT_COLUMNS if column in feature_columns],
        "negative_downsample_ratio": config.negative_downsample_ratio,
        "max_negatives_per_positive": config.max_negatives_per_positive,
        "require_mappable_sites": config.require_mappable_sites,
        "streaming": False,
        "max_input_rows": config.max_input_rows,
        "max_output_rows": config.max_output_rows,
        "max_rows_per_split": config.max_rows_per_split,
        "downsampling_stats": downsampling_stats,
        "files": {
            "features": str(features_path),
            "splits": str(splits_path),
            "index": str(index_path),
            "markdown": str(markdown_path),
        },
        "warnings": sorted(set(warnings)),
        "note": "Branch-site supervised targets are labels only; site/gene labels are diagnostic metadata, not features.",
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
        "n_branch_site_rows": len(feature_rows),
        "n_positive_branch_sites": n_positive,
        "n_negative_branch_sites": len(feature_rows) - n_positive,
        "warnings": payload["warnings"],
    }


def _build_branch_site_dataset_streaming(config: BranchSiteDatasetConfig) -> dict:
    """Build a branch-site dataset without materializing the label table."""
    dataset_dir = Path(config.dataset_dir)
    outdir = Path(config.outdir)
    labels_path = Path(config.branch_site_labels_tsv)
    warnings: List[str] = []
    scan = _scan_branch_label_counts(labels_path, config)
    warnings.extend(scan["warnings"])
    positive_total = int(scan["positive_total"])
    if config.max_output_rows is not None and positive_total > config.max_output_rows:
        raise ValueError(
            "max_output_rows is smaller than the number of positive branch-site rows "
            f"({config.max_output_rows} < {positive_total}); positives are always retained"
        )
    if config.max_rows_per_split is not None:
        for split, count in scan["positive_counts_by_split"].items():
            if count > config.max_rows_per_split:
                raise ValueError(
                    "max_rows_per_split is smaller than positive rows in split "
                    f"{split!r} ({config.max_rows_per_split} < {count}); positives are always retained"
                )
    negative_targets = _negative_targets(scan, config)
    features_path = outdir / "branch_site_features.tsv"
    splits_path = outdir / "branch_site_splits.tsv"
    index_path = outdir / "branch_site_dataset_index.json"
    markdown_path = outdir / "branch_site_dataset.md"
    tensor_cache: Dict[str, Tuple[np.ndarray, List[str]]] = {}
    site_feature_cache: Dict[Tuple[str, int], dict] = {}
    rows_written = 0
    positives_written = 0
    negatives_written = 0
    skipped_negative_rows = 0
    tensor_failed_rows = 0
    split_counts: Counter[str] = Counter()
    method_counts: Counter[str] = Counter()
    tier_counts: Counter[str] = Counter()
    branch_counts: Counter[str] = Counter()
    branch_label_source_counts: Counter[str] = Counter()
    foreground_branch_counts: Counter[str] = Counter()
    negative_seen_by_key: Counter[Tuple[str, str, str]] = Counter()
    negative_written_by_key: Counter[Tuple[str, str, str]] = Counter()
    positive_remaining_total = positive_total
    positive_remaining_by_split: Counter[str] = Counter(scan["positive_counts_by_split"])

    with features_path.open("w", encoding="utf-8", newline="") as feature_handle, splits_path.open(
        "w", encoding="utf-8", newline=""
    ) as split_handle:
        feature_writer = csv.DictWriter(feature_handle, fieldnames=BRANCH_FEATURE_FIELDNAMES, delimiter="\t", lineterminator="\n")
        split_writer = csv.DictWriter(split_handle, fieldnames=BRANCH_SPLIT_FIELDNAMES, delimiter="\t", lineterminator="\n")
        feature_writer.writeheader()
        split_writer.writeheader()
        for row in _iter_label_rows(labels_path, config.max_input_rows):
            if not _row_passes_mappability(row, config, bool(scan["mappable_values_present"])):
                continue
            split = row.get("split", "")
            is_positive = str(row.get("y_branch_site")) == "1"
            if is_positive:
                positive_remaining_total = max(0, positive_remaining_total - 1)
                positive_remaining_by_split[split] = max(0, positive_remaining_by_split[split] - 1)
            else:
                key = _sampling_key(row)
                negative_seen_by_key[key] += 1
                if not _should_keep_streaming_negative(
                    row=row,
                    key=key,
                    scan=scan,
                    negative_targets=negative_targets,
                    negative_seen_index=negative_seen_by_key[key],
                    negative_written_by_key=negative_written_by_key,
                    rows_written=rows_written,
                    split_counts=split_counts,
                    positive_remaining_total=positive_remaining_total,
                    positive_remaining_by_split=positive_remaining_by_split,
                    config=config,
                ):
                    skipped_negative_rows += 1
                    continue
            if is_positive and _positive_would_exceed_caps(
                split=split,
                rows_written=rows_written,
                split_counts=split_counts,
                config=config,
                warnings=warnings,
            ):
                continue
            built = _build_feature_and_split_row(
                row,
                dataset_dir,
                tensor_cache,
                warnings,
                cache_limit=8,
                site_feature_cache=site_feature_cache,
                site_feature_cache_limit=4096,
            )
            if built is None:
                tensor_failed_rows += 1
                continue
            feature_row, split_row = built
            feature_writer.writerow(_project_row(feature_row, BRANCH_FEATURE_FIELDNAMES))
            split_writer.writerow(_project_row(split_row, BRANCH_SPLIT_FIELDNAMES))
            rows_written += 1
            split_counts[feature_row["split"]] += 1
            method_counts[feature_row["method"]] += 1
            tier_counts[feature_row["saturation_tier"]] += 1
            branch_counts[feature_row["branch_id"]] += 1
            branch_label_source_counts[feature_row["branch_label_source"]] += 1
            foreground_branch_counts[str(feature_row["foreground_branch_present"])] += 1
            if is_positive:
                positives_written += 1
            else:
                negatives_written += 1
                negative_written_by_key[_sampling_key(row)] += 1

    feature_columns = _branch_site_feature_columns_from_fieldnames()
    downsampling_stats = _downsampling_stats_from_scan(scan, negative_targets, negative_written_by_key)
    payload = {
        "branch_site_dataset_version": BRANCH_SITE_DATASET_VERSION,
        "dataset_dir": str(dataset_dir),
        "branch_site_labels_tsv": str(labels_path),
        "total_input_rows": scan["total_input_rows"],
        "eligible_input_rows": scan["eligible_input_rows"],
        "mappability_filtered_rows": scan["mappability_filtered_rows"],
        "n_branch_site_rows": rows_written,
        "rows_written": rows_written,
        "n_positive_branch_sites": positives_written,
        "positives_written": positives_written,
        "n_negative_branch_sites": negatives_written,
        "negatives_written": negatives_written,
        "positive_fraction": None if rows_written == 0 else positives_written / rows_written,
        "skipped_negative_rows": skipped_negative_rows,
        "tensor_failed_rows": tensor_failed_rows,
        "split_counts": dict(sorted(split_counts.items())),
        "saturation_tier_counts": dict(sorted(tier_counts.items())),
        "method_counts": dict(sorted(method_counts.items())),
        "branch_counts": dict(sorted(branch_counts.items())),
        "branch_label_source_counts": dict(sorted(branch_label_source_counts.items())),
        "foreground_branch_present_counts": dict(sorted(foreground_branch_counts.items())),
        "feature_columns": feature_columns,
        "forbidden_as_features": sorted(FORBIDDEN_FEATURE_COLUMNS),
        "sensitive_context_columns": [column for column in SENSITIVE_CONTEXT_COLUMNS if column in feature_columns],
        "negative_downsample_ratio": config.negative_downsample_ratio,
        "max_negatives_per_positive": config.max_negatives_per_positive,
        "require_mappable_sites": config.require_mappable_sites,
        "streaming": True,
        "max_input_rows": config.max_input_rows,
        "max_output_rows": config.max_output_rows,
        "max_rows_per_split": config.max_rows_per_split,
        "downsampling_stats": downsampling_stats,
        "files": {
            "features": str(features_path),
            "splits": str(splits_path),
            "index": str(index_path),
            "markdown": str(markdown_path),
        },
        "warnings": sorted(set(warnings)),
        "note": (
            "Streaming branch-site dataset builder. Positives are retained unless an explicit hard cap "
            "makes that impossible; negatives are sampled before tensor feature extraction."
        ),
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
        "n_branch_site_rows": rows_written,
        "n_positive_branch_sites": positives_written,
        "n_negative_branch_sites": negatives_written,
        "total_input_rows": scan["total_input_rows"],
        "warnings": payload["warnings"],
    }


def validate_branch_site_dataset_dir(branch_site_dataset_dir: str | Path) -> dict:
    """Validate branch-site dataset artifacts."""
    path = Path(branch_site_dataset_dir)
    failures: List[str] = []
    warnings: List[str] = []
    index_path = path / "branch_site_dataset_index.json"
    features_path = path / "branch_site_features.tsv"
    splits_path = path / "branch_site_splits.tsv"
    payload = _load_json(index_path, failures)
    rows = _read_rows(features_path, failures)
    split_rows = _read_rows(splits_path, failures)
    _check_required(rows, set(BRANCH_SPLIT_FIELDNAMES) | {"tensor_file", "gene_label", "y_site"}, features_path, failures)
    _check_required(split_rows, set(BRANCH_SPLIT_FIELDNAMES), splits_path, failures)
    seen = set()
    n_positive = 0
    family_splits: Dict[str, set] = {}
    for row in rows:
        branch_site_id = row.get("branch_site_id", "")
        if branch_site_id in seen:
            failures.append(f"duplicate_branch_site_id:{branch_site_id}")
        seen.add(branch_site_id)
        if row.get("y_branch_site") not in {"0", "1", 0, 1}:
            failures.append(f"invalid_y_branch_site:{branch_site_id}:{row.get('y_branch_site')}")
        n_positive += 1 if str(row.get("y_branch_site")) == "1" else 0
        family_splits.setdefault(row.get("family_id", ""), set()).add(row.get("split", ""))
    for family_id, splits in family_splits.items():
        if family_id and len(splits) > 1:
            failures.append(f"family_id_multiple_splits:{family_id}:{sorted(splits)}")
    feature_columns = payload.get("feature_columns", []) if isinstance(payload, dict) else []
    forbidden_features = sorted(set(feature_columns) & FORBIDDEN_FEATURE_COLUMNS)
    if forbidden_features:
        failures.append("forbidden_branch_feature_columns:" + ",".join(forbidden_features))
    if not rows:
        failures.append("no_branch_site_rows")
    elif n_positive == 0:
        warnings.append("no_positive_branch_sites")
    if payload and payload.get("n_branch_site_rows") != len(rows):
        warnings.append("index_branch_site_row_count_mismatch")
    if rows and payload.get("dataset_dir"):
        dataset_dir = Path(str(payload.get("dataset_dir")))
        for row in rows[:25]:
            try:
                resolve_tensor_file(row.get("tensor_file", ""), dataset_dir)
            except FileNotFoundError:
                failures.append(f"tensor_file_unresolved:{row.get('tensor_file', '')}")
            except Exception as exc:  # pragma: no cover - defensive validator path
                warnings.append(f"tensor_file_resolution_warning:{row.get('tensor_file', '')}:{exc}")
    return {
        "status": "fail" if failures else "ok",
        "n_rows": len(rows),
        "n_positive_branch_sites": n_positive,
        "n_fail": len(failures),
        "n_warning": len(warnings),
        "failures": failures,
        "warnings": warnings,
    }


def get_branch_site_feature_columns(rows: List[dict]) -> List[str]:
    """Return numeric non-leaking branch-site feature columns."""
    if not rows:
        return []
    selected = []
    for column in rows[0].keys():
        if _exclude_feature_column(column):
            continue
        if all(_to_float_or_none(row.get(column)) is not None for row in rows):
            selected.append(column)
    return selected


def _branch_site_feature_columns_from_fieldnames() -> List[str]:
    return [column for column in BRANCH_FEATURE_FIELDNAMES if not _exclude_feature_column(column)]


def _iter_label_rows(labels_path: Path, max_input_rows: Optional[int] = None):
    with labels_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for index, row in enumerate(reader, start=1):
            if max_input_rows is not None and index > max_input_rows:
                break
            yield row


def _parse_dataset_dirs(value: Sequence[str] | str) -> List[str]:
    if isinstance(value, str):
        parts = value.replace("\n", ",").split(",")
    else:
        parts = []
        for item in value:
            parts.extend(str(item).replace("\n", ",").split(","))
    return [part.strip() for part in parts if part.strip()]


def _scan_branch_label_counts(labels_path: Path, config: BranchSiteDatasetConfig) -> dict:
    all_counts = _empty_scan_counts()
    mappable_counts = _empty_scan_counts()
    total_input_rows = 0
    mappable_values_present = False
    max_input_rows_reached = False
    with labels_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for index, row in enumerate(reader, start=1):
            if config.max_input_rows is not None and index > config.max_input_rows:
                max_input_rows_reached = True
                break
            total_input_rows += 1
            if row.get("mappable_site") not in ("", None):
                mappable_values_present = True
            _increment_scan_counts(all_counts, row)
            if _row_is_true_mappable(row):
                _increment_scan_counts(mappable_counts, row)
    selected = mappable_counts if config.require_mappable_sites and mappable_values_present else all_counts
    warnings = []
    if max_input_rows_reached:
        warnings.append(f"max_input_rows_reached:{config.max_input_rows}")
    return {
        **selected,
        "total_input_rows": total_input_rows,
        "eligible_input_rows": selected["rows"],
        "mappability_filtered_rows": total_input_rows - selected["rows"],
        "mappable_values_present": mappable_values_present,
        "warnings": warnings,
    }


def _empty_scan_counts() -> dict:
    return {
        "rows": 0,
        "positive_total": 0,
        "negative_total": 0,
        "positive_counts_by_key": Counter(),
        "negative_counts_by_key": Counter(),
        "positive_counts_by_split": Counter(),
        "negative_counts_by_split": Counter(),
    }


def _increment_scan_counts(counts: dict, row: dict) -> None:
    key = _sampling_key(row)
    split = row.get("split", "")
    counts["rows"] += 1
    if str(row.get("y_branch_site")) == "1":
        counts["positive_total"] += 1
        counts["positive_counts_by_key"][key] += 1
        counts["positive_counts_by_split"][split] += 1
    else:
        counts["negative_total"] += 1
        counts["negative_counts_by_key"][key] += 1
        counts["negative_counts_by_split"][split] += 1


def _negative_targets(scan: dict, config: BranchSiteDatasetConfig) -> Counter[Tuple[str, str, str]]:
    ratio = config.max_negatives_per_positive
    if ratio is None:
        ratio = config.negative_downsample_ratio
    targets: Counter[Tuple[str, str, str]] = Counter()
    for key, negative_count in scan["negative_counts_by_key"].items():
        if ratio is None:
            target = negative_count
        else:
            positive_count = scan["positive_counts_by_key"].get(key, 0)
            target = int(round(float(ratio) * positive_count))
            if positive_count > 0 and target == 0:
                target = 1
            target = min(target, negative_count)
        targets[key] = max(0, int(target))
    return targets


def _should_keep_streaming_negative(
    row: dict,
    key: Tuple[str, str, str],
    scan: dict,
    negative_targets: Counter[Tuple[str, str, str]],
    negative_seen_index: int,
    negative_written_by_key: Counter[Tuple[str, str, str]],
    rows_written: int,
    split_counts: Counter[str],
    positive_remaining_total: int,
    positive_remaining_by_split: Counter[str],
    config: BranchSiteDatasetConfig,
) -> bool:
    target = negative_targets.get(key, 0)
    if target <= 0 or negative_written_by_key[key] >= target:
        return False
    split = row.get("split", "")
    if config.max_output_rows is not None and rows_written + positive_remaining_total >= config.max_output_rows:
        return False
    if config.max_rows_per_split is not None and split_counts[split] + positive_remaining_by_split[split] >= config.max_rows_per_split:
        return False
    negative_count = scan["negative_counts_by_key"].get(key, 0)
    if negative_count <= target:
        return True
    return _deterministic_negative_accept(row, negative_seen_index, negative_count, target, config.seed)


def _deterministic_negative_accept(row: dict, seen_index: int, negative_count: int, target: int, seed: int) -> bool:
    token = "|".join(
        [
            str(seed),
            row.get("family_id", ""),
            row.get("method", ""),
            row.get("branch_id", ""),
            row.get("site_index_zero", ""),
            row.get("aligned_site_index_zero", ""),
            str(seen_index),
        ]
    )
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    value = int.from_bytes(digest, byteorder="big", signed=False) / float(2**64 - 1)
    return value < (target / negative_count)


def _positive_would_exceed_caps(
    split: str,
    rows_written: int,
    split_counts: Counter[str],
    config: BranchSiteDatasetConfig,
    warnings: List[str],
) -> bool:
    if config.max_output_rows is not None and rows_written >= config.max_output_rows:
        warnings.append(f"positive_row_skipped_by_max_output_rows:{config.max_output_rows}")
        return True
    if config.max_rows_per_split is not None and split_counts[split] >= config.max_rows_per_split:
        warnings.append(f"positive_row_skipped_by_max_rows_per_split:{split}:{config.max_rows_per_split}")
        return True
    return False


def _downsampling_stats_from_scan(
    scan: dict,
    negative_targets: Counter[Tuple[str, str, str]],
    negative_written_by_key: Counter[Tuple[str, str, str]],
) -> dict:
    stats = {}
    keys = sorted(set(scan["positive_counts_by_key"]) | set(scan["negative_counts_by_key"]))
    for key in keys:
        stats["::".join(key)] = {
            "positives": int(scan["positive_counts_by_key"].get(key, 0)),
            "negatives_before": int(scan["negative_counts_by_key"].get(key, 0)),
            "negative_target": int(negative_targets.get(key, 0)),
            "negatives_after": int(negative_written_by_key.get(key, 0)),
        }
    return stats


def _sampling_key(row: dict) -> Tuple[str, str, str]:
    return (row.get("split", ""), row.get("saturation_tier", "unknown") or "unknown", row.get("method", ""))


def _row_passes_mappability(row: dict, config: BranchSiteDatasetConfig, mappable_values_present: bool) -> bool:
    if not config.require_mappable_sites:
        return True
    if not mappable_values_present:
        return True
    return _row_is_true_mappable(row)


def _row_is_true_mappable(row: dict) -> bool:
    return str(row.get("mappable_site", "")).strip() in {"1", "1.0", "true", "True"}


def _project_row(row: dict, fieldnames: List[str]) -> dict:
    return {field: row.get(field, "") for field in fieldnames}


def _build_feature_and_split_row(
    row: dict,
    dataset_dir: Path,
    tensor_cache: Dict[str, Tuple[np.ndarray, List[str]]],
    warnings: List[str],
    cache_limit: Optional[int] = None,
    site_feature_cache: Optional[Dict[Tuple[str, int], dict]] = None,
    site_feature_cache_limit: Optional[int] = None,
) -> Optional[Tuple[dict, dict]]:
    try:
        tensor, taxa_order = _load_tensor_cached(row.get("tensor_file", ""), dataset_dir, tensor_cache, cache_limit=cache_limit)
    except (OSError, ValueError, FileNotFoundError) as exc:
        warnings.append(f"tensor_load_failed:{row.get('tensor_file', '')}:{exc}")
        return None
    site_index = _site_index_for_tensor(row)
    if site_index < 0 or site_index >= tensor.shape[1]:
        warnings.append(f"site_index_out_of_bounds:{row.get('family_id', '')}:{site_index}")
        return None
    branch_site_id = _branch_site_id(row.get("family_id", ""), row.get("method", ""), row.get("branch_id", ""), site_index)
    extracted = _extract_branch_features(
        row,
        tensor,
        taxa_order,
        site_index,
        warnings,
        site_feature_cache=site_feature_cache,
        site_feature_cache_limit=site_feature_cache_limit,
    )
    feature_row = {
        "branch_site_id": branch_site_id,
        "family_id": row.get("family_id", ""),
        "original_family_id": row.get("original_family_id", ""),
        "source_dataset": row.get("source_dataset", ""),
        "method": row.get("method", ""),
        "saturation_tier": row.get("saturation_tier", "unknown") or "unknown",
        "split": row.get("split", ""),
        "tensor_file": row.get("tensor_file", ""),
        "labels_file": row.get("labels_file", ""),
        "branch_id": row.get("branch_id", ""),
        "foreground_taxon": row.get("foreground_taxon", ""),
        "site_index_zero": site_index,
        "site_index_one": site_index + 1,
        "aligned_site_index_zero": row.get("aligned_site_index_zero", ""),
        "aligned_site_index_one": _one_based(row.get("aligned_site_index_zero", "")),
        "original_site_index_zero": row.get("original_site_index_zero", ""),
        "original_site_index_one": _one_based(row.get("original_site_index_zero", "")),
        "mapping_status": row.get("mapping_status", ""),
        "mapping_confidence": row.get("mapping_confidence", ""),
        "mappable_site": row.get("mappable_site", ""),
        "y_branch_site": _safe_binary(row.get("y_branch_site")),
        "y_site": _safe_binary(row.get("y_site")),
        "gene_label": _safe_binary(row.get("gene_label")),
        "foreground_branch_present": _safe_binary(row.get("foreground_branch_present")),
        "branch_label_source": row.get("branch_label_source", ""),
    }
    feature_row.update(extracted)
    split_row = {
        "branch_site_id": branch_site_id,
        "family_id": feature_row["family_id"],
        "method": feature_row["method"],
        "saturation_tier": feature_row["saturation_tier"],
        "split": feature_row["split"],
        "branch_id": feature_row["branch_id"],
        "site_index_zero": site_index,
        "aligned_site_index_zero": feature_row["aligned_site_index_zero"],
        "original_site_index_zero": feature_row["original_site_index_zero"],
        "mapping_status": feature_row["mapping_status"],
        "mappable_site": feature_row["mappable_site"],
        "y_branch_site": feature_row["y_branch_site"],
    }
    return feature_row, split_row


def _filter_mappable_rows(rows: List[dict], config: BranchSiteDatasetConfig, warnings: List[str]) -> List[dict]:
    if not config.require_mappable_sites:
        return rows
    if not rows or "mappable_site" not in rows[0]:
        return rows
    if not any(row.get("mappable_site") not in ("", None) for row in rows):
        return rows
    kept = [row for row in rows if str(row.get("mappable_site", "1")).strip() in {"1", "1.0", "true", "True"}]
    dropped = len(rows) - len(kept)
    if dropped:
        warnings.append(f"unmappable_branch_site_rows_dropped:{dropped}")
    return kept


def _downsample_negatives(rows: List[dict], config: BranchSiteDatasetConfig) -> Tuple[List[dict], dict]:
    ratio = config.max_negatives_per_positive
    if ratio is None:
        ratio = config.negative_downsample_ratio
    if ratio is None:
        return rows, {}
    rng = random.Random(config.seed)
    grouped: Dict[Tuple[str, str, str], List[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row.get("split", ""), row.get("saturation_tier", "unknown") or "unknown", row.get("method", ""))].append(row)
    selected: List[dict] = []
    stats = {}
    for key in sorted(grouped):
        group = grouped[key]
        positives = [row for row in group if str(row.get("y_branch_site")) == "1"]
        negatives = [row for row in group if str(row.get("y_branch_site")) != "1"]
        if positives:
            max_negatives = max(1, int(round(ratio * len(positives))))
            rng.shuffle(negatives)
            negatives = negatives[:max_negatives]
        selected.extend(positives)
        selected.extend(negatives)
        stats["::".join(key)] = {
            "positives": len(positives),
            "negatives_before": len(group) - len(positives),
            "negatives_after": len(negatives),
        }
    return selected, stats


def _load_tensor_cached(
    tensor_file: str,
    dataset_dir: Path,
    cache: Dict[str, Tuple[np.ndarray, List[str]]],
    cache_limit: Optional[int] = None,
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
    if cache_limit is not None:
        while len(cache) > cache_limit:
            cache.pop(next(iter(cache)))
    return tensor, taxa_order


def _site_index_for_tensor(row: dict) -> int:
    if row.get("aligned_site_index_zero") not in ("", None):
        return _safe_int(row.get("aligned_site_index_zero"), default=-1)
    return _safe_int(row.get("site_index_zero"), default=-1)


def _extract_branch_features(
    row: dict,
    tensor: np.ndarray,
    taxa_order: List[str],
    site_index: int,
    warnings: List[str],
    site_feature_cache: Optional[Dict[Tuple[str, int], dict]] = None,
    site_feature_cache_limit: Optional[int] = None,
) -> dict:
    n_taxa, n_codons, n_channels = tensor.shape
    tensor_key = row.get("tensor_file", "")
    cache_key = (tensor_key, site_index)
    cached_site = site_feature_cache.get(cache_key) if site_feature_cache is not None else None
    if cached_site is None:
        site_values = tensor[:, site_index, :]
        codon_ids = site_values[:, 0].astype(np.float64)
        gaps = site_values[:, 1].astype(np.float64) if n_channels > 1 else np.zeros(n_taxa)
        codon_min = float(codon_ids.min()) if codon_ids.size else 0.0
        codon_max = float(codon_ids.max()) if codon_ids.size else 0.0
        codon_mean = float(codon_ids.mean()) if codon_ids.size else 0.0
        codon_std = float(codon_ids.std()) if codon_ids.size else 0.0
        gap_mean = float(gaps.mean()) if gaps.size else 0.0
        site_relative = 0.0 if n_codons <= 1 else site_index / (n_codons - 1)
        cached_site = {
            "codon_ids": codon_ids,
            "gaps": gaps,
            "base_features": {
                "site_relative_position": site_relative,
                "site_centered_position": site_relative - 0.5,
                "site_terminal_distance": min(site_relative, 1.0 - site_relative),
                "n_taxa": n_taxa,
                "n_codons": n_codons,
                "log_n_taxa": math.log1p(n_taxa),
                "log_n_codons": math.log1p(n_codons),
                "codon_id_mean": codon_mean,
                "codon_id_std": codon_std,
                "codon_id_min": codon_min,
                "codon_id_max": codon_max,
                "codon_id_range": codon_max - codon_min,
                "codon_id_unique_count": int(np.unique(codon_ids.astype(np.int64)).size),
                "gap_fraction": gap_mean,
                "non_gap_fraction": float(1.0 - gap_mean),
                "taxon_codon_variability": codon_std,
            },
        }
        if site_feature_cache is not None:
            site_feature_cache[cache_key] = cached_site
            if site_feature_cache_limit is not None:
                while len(site_feature_cache) > site_feature_cache_limit:
                    site_feature_cache.pop(next(iter(site_feature_cache)))
    codon_ids = cached_site["codon_ids"]
    gaps = cached_site["gaps"]
    foreground_taxon = row.get("foreground_taxon", "")
    branch_id = row.get("branch_id", "")
    foreground_index = _taxon_index(foreground_taxon, taxa_order)
    branch_index = _taxon_index(branch_id, taxa_order)
    if foreground_taxon and foreground_index < 0:
        warnings.append(f"foreground_taxon_unresolved:{foreground_taxon}")
    if branch_id and branch_index < 0:
        warnings.append(f"branch_id_unresolved:{branch_id}")
    background_indices = np.ones(n_taxa, dtype=bool)
    if branch_index >= 0:
        background_indices[branch_index] = False
    background = codon_ids[background_indices]
    background_mean = float(background.mean()) if background.size else float(codon_ids.mean())
    foreground_codon = float(codon_ids[foreground_index]) if foreground_index >= 0 else -1.0
    branch_codon = float(codon_ids[branch_index]) if branch_index >= 0 else -1.0
    foreground_gap = float(gaps[foreground_index]) if foreground_index >= 0 else 0.0
    branch_gap = float(gaps[branch_index]) if branch_index >= 0 else 0.0
    features = dict(cached_site["base_features"])
    features.update({
        "foreground_taxon_present": 1 if foreground_taxon else 0,
        "foreground_taxon_index": foreground_index,
        "branch_query_id_numeric": branch_index,
        "foreground_codon_id": foreground_codon,
        "foreground_gap": foreground_gap,
        "branch_codon_id": branch_codon,
        "branch_gap": branch_gap,
        "background_mean_codon_id": background_mean,
        "foreground_background_codon_delta": abs(foreground_codon - background_mean) if foreground_index >= 0 else 0.0,
        "branch_background_codon_delta": abs(branch_codon - background_mean) if branch_index >= 0 else 0.0,
    })
    return features


def _exclude_feature_column(column: str) -> bool:
    if column in METADATA_COLUMNS or column in FORBIDDEN_FEATURE_COLUMNS:
        return True
    lowered = column.lower()
    if column in {"foreground_taxon_present", "foreground_taxon_index", "foreground_branch_present"}:
        return False
    if any(token in lowered for token in LEAKAGE_TOKENS):
        return True
    return False


def _taxon_index(taxon: str, taxa_order: List[str]) -> int:
    if not taxon or not taxa_order:
        return -1
    try:
        return taxa_order.index(taxon)
    except ValueError:
        return -1


def _branch_site_id(family_id: str, method: str, branch_id: str, site_index: int) -> str:
    return f"{family_id}::{method}::{branch_id}::site_{site_index:06d}"


def _one_based(value: object) -> str:
    if value in ("", None):
        return ""
    return str(_safe_int(value) + 1)


def _safe_int(value: object, default: int = 0) -> int:
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return default


def _safe_binary(value: object) -> int:
    return 1 if str(value).strip() in {"1", "1.0", "true", "True"} else 0


def _to_float_or_none(value: object) -> Optional[float]:
    try:
        if value in ("", None):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _read_rows(path: Path, failures: List[str]) -> List[dict]:
    if not path.exists():
        failures.append(f"missing_file:{path}")
        return []
    try:
        return read_tsv(path)
    except OSError as exc:
        failures.append(f"could_not_read_tsv:{path}:{exc}")
        return []


def _check_required(rows: List[dict], required: set, path: Path, failures: List[str]) -> None:
    if not path.exists():
        return
    if rows:
        missing = sorted(required - set(rows[0].keys()))
    else:
        import csv
        with path.open("r", encoding="utf-8", newline="") as handle:
            missing = sorted(required - set(csv.DictReader(handle, delimiter="\t").fieldnames or []))
    if missing:
        failures.append(f"missing_required_columns:{path}:{','.join(missing)}")


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
    if not isinstance(payload, dict):
        failures.append(f"json_not_object:{path}")
        return {}
    return payload


def _render_markdown(payload: dict) -> str:
    return "\n".join(
        [
            "# BABAPPA branch-site dataset",
            "",
            f"- Branch-site rows: {payload.get('n_branch_site_rows')}",
            f"- Positive branch-sites: {payload.get('n_positive_branch_sites')}",
            f"- Positive fraction: {payload.get('positive_fraction')}",
            f"- Feature columns: {', '.join(payload.get('feature_columns', []))}",
            "",
            "## Leakage boundary",
            "",
            "`y_branch_site` is the supervised target. `y_site` and `gene_label` remain diagnostic metadata and are excluded from features.",
            "Foreground context is retained as biologically sensitive context and must be interpreted cautiously.",
            "",
        ]
    )


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
