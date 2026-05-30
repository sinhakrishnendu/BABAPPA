"""Merge multiple BABAPPA dataset indexes into one trainable dataset."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from babappa import __version__
from babappa.datasets.index import SPLIT_NAMES, read_tsv, write_tsv

MERGED_DATASET_VERSION = __version__
MERGED_FEATURE_PREFIX = [
    "family_id",
    "original_family_id",
    "source_dataset",
]
MERGED_SPLIT_FIELDNAMES = [
    "family_id",
    "original_family_id",
    "source_dataset",
    "method",
    "tensor_file",
    "gene_label",
    "saturation_tier",
    "split",
]


@dataclass(frozen=True)
class DatasetMergeConfig:
    """Configuration for merging BABAPPA dataset indexes."""

    dataset_dirs: List[str]
    outdir: str
    names: Optional[List[str]] = None
    seed: int = 42
    resplit: bool = True
    train_fraction: float = 0.8
    val_fraction: float = 0.1
    calib_fraction: float = 0.05
    test_fraction: float = 0.05
    split_by_family: bool = True

    def __post_init__(self) -> None:
        if not self.dataset_dirs:
            raise ValueError("dataset_dirs must be non-empty")
        for directory in self.dataset_dirs:
            dataset_dir = Path(directory)
            if not dataset_dir.exists():
                raise ValueError(f"dataset_dir does not exist: {dataset_dir}")
            for filename in ("dataset_index.json", "features.tsv", "splits.tsv"):
                if not (dataset_dir / filename).exists():
                    raise ValueError(f"dataset_dir is missing {filename}: {dataset_dir}")
        if self.names is not None:
            if len(self.names) != len(self.dataset_dirs):
                raise ValueError("names must have the same length as dataset_dirs")
            if len(set(self.names)) != len(self.names):
                raise ValueError("names must be unique")
            object.__setattr__(self, "names", [name for name in self.names])
        else:
            derived_names = _derive_source_names(self.dataset_dirs)
            object.__setattr__(self, "names", derived_names)
        if self.resplit:
            fractions = [
                self.train_fraction,
                self.val_fraction,
                self.calib_fraction,
                self.test_fraction,
            ]
            if any(fraction <= 0 for fraction in fractions):
                raise ValueError("split fractions must be positive")
            if abs(sum(fractions) - 1.0) > 1e-6:
                raise ValueError("split fractions must sum to 1.0")
        Path(self.outdir).mkdir(parents=True, exist_ok=True)


def _derive_source_names(dataset_dirs: List[str]) -> List[str]:
    names: List[str] = []
    for index, directory in enumerate(dataset_dirs):
        path = Path(directory)
        name = path.name or f"dataset_{index + 1}"
        if name == "dataset" and path.parent.name:
            name = path.parent.name
        names.append(name)
    if len(set(names)) != len(names):
        return [f"dataset_{index + 1}" for index in range(len(dataset_dirs))]
    return names


def merge_dataset_indexes(config: DatasetMergeConfig) -> dict:
    """Merge feature/split tables and write a unified dataset index."""
    outdir = Path(config.outdir)
    merged_features: List[dict] = []
    merged_splits: List[dict] = []
    source_indexes = []

    for source_name, dataset_dir_string in zip(config.names or [], config.dataset_dirs):
        dataset_dir = Path(dataset_dir_string)
        dataset_index = _read_json(dataset_dir / "dataset_index.json")
        source_indexes.append(dataset_index)
        tensor_dir = Path(str(dataset_index.get("tensor_dir", "")))
        feature_rows = read_tsv(dataset_dir / "features.tsv")
        split_rows = read_tsv(dataset_dir / "splits.tsv")
        split_lookup = {
            (row.get("family_id"), row.get("method"), row.get("tensor_file")): row
            for row in split_rows
        }
        for feature_row in feature_rows:
            original_family_id = feature_row.get("family_id", "")
            merged_family_id = f"{source_name}::{original_family_id}"
            original_tensor_file = feature_row.get("tensor_file", "")
            resolved_tensor_file = _resolvable_tensor_path(
                tensor_dir, original_tensor_file, outdir
            )
            split_row = split_lookup.get(
                (
                    original_family_id,
                    feature_row.get("method", ""),
                    original_tensor_file,
                ),
                {},
            )
            merged_feature = dict(feature_row)
            merged_feature["family_id"] = merged_family_id
            merged_feature["original_family_id"] = original_family_id
            merged_feature["source_dataset"] = source_name
            merged_feature["tensor_file"] = resolved_tensor_file
            for path_field in ("tensor_meta_file", "labels_file"):
                if path_field in merged_feature:
                    merged_feature[path_field] = _resolvable_tensor_path(
                        tensor_dir, merged_feature[path_field], outdir
                    )
            merged_features.append(merged_feature)
            merged_splits.append(
                {
                    "family_id": merged_family_id,
                    "original_family_id": original_family_id,
                    "source_dataset": source_name,
                    "method": feature_row.get("method", ""),
                    "tensor_file": resolved_tensor_file,
                    "gene_label": feature_row.get("gene_label", split_row.get("gene_label", "")),
                    "saturation_tier": feature_row.get(
                        "saturation_tier", split_row.get("saturation_tier", "")
                    ),
                    "split": split_row.get("split", ""),
                }
            )

    if config.resplit:
        split_assignments = _assign_merged_splits(merged_splits, config)
        for row in merged_splits:
            key = _split_key(row, config.split_by_family)
            row["split"] = split_assignments[key]
    else:
        for row in merged_splits:
            if row["split"] not in SPLIT_NAMES:
                raise ValueError(f"source split row has invalid split: {row['split']}")

    merged_features.sort(key=lambda row: (row["family_id"], row["method"], row["tensor_file"]))
    merged_splits.sort(key=lambda row: (row["family_id"], row["method"], row["tensor_file"]))
    feature_fieldnames = _merged_feature_fieldnames(merged_features)
    features_path = outdir / "features.tsv"
    splits_path = outdir / "splits.tsv"
    index_path = outdir / "dataset_index.json"
    write_tsv(features_path, merged_features, feature_fieldnames)
    write_tsv(splits_path, merged_splits, MERGED_SPLIT_FIELDNAMES)

    methods = sorted({row["method"] for row in merged_splits if row.get("method")})
    family_ids = sorted({row["family_id"] for row in merged_splits if row.get("family_id")})
    saturation_tier_counts = _saturation_tier_counts(merged_splits)
    index_payload = {
        "dataset_index_version": MERGED_DATASET_VERSION,
        "merged_dataset_version": MERGED_DATASET_VERSION,
        "source_dataset_dirs": [str(Path(directory)) for directory in config.dataset_dirs],
        "source_dataset_names": list(config.names or []),
        "source_dataset_indexes": source_indexes,
        "n_rows": len(merged_splits),
        "n_families": len(family_ids),
        "methods": methods,
        "seed": config.seed,
        "resplit": config.resplit,
        "split_by_family": config.split_by_family,
        "fractions": {
            "train": config.train_fraction,
            "val": config.val_fraction,
            "calib": config.calib_fraction,
            "test": config.test_fraction,
        },
        "split_counts_rows": _count_splits(merged_splits),
        "split_counts_families": _count_family_splits(merged_splits),
        "positive_counts_by_split": _count_positive_by_split(merged_splits),
        "saturation_tier_counts": saturation_tier_counts,
        "files": {
            "features": str(features_path),
            "splits": str(splits_path),
        },
    }
    _write_json(index_path, index_payload)

    return {
        "status": "ok",
        "outdir": str(outdir),
        "features": str(features_path),
        "splits": str(splits_path),
        "index": str(index_path),
        "n_rows": len(merged_splits),
        "n_families": len(family_ids),
        "saturation_tier_counts": saturation_tier_counts,
    }


def _resolvable_tensor_path(tensor_dir: Path, relative_path: str, outdir: Path) -> str:
    path = tensor_dir / relative_path
    try:
        return str(path.relative_to(outdir.parent))
    except ValueError:
        pass
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def _merged_feature_fieldnames(rows: List[dict]) -> List[str]:
    fieldnames = list(MERGED_FEATURE_PREFIX)
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    return fieldnames


def _assign_merged_splits(rows: List[dict], config: DatasetMergeConfig) -> Dict[str, str]:
    if config.split_by_family:
        units = sorted({row["family_id"] for row in rows})
    else:
        units = sorted(_split_key(row, False) for row in rows)
    shuffled = list(units)
    random.Random(config.seed).shuffle(shuffled)
    split_counts = _split_counts(len(shuffled), config)
    assignments: Dict[str, str] = {}
    cursor = 0
    for split_name in SPLIT_NAMES:
        count = split_counts[split_name]
        for unit in shuffled[cursor:cursor + count]:
            assignments[unit] = split_name
        cursor += count
    return assignments


def _split_counts(n_units: int, config: DatasetMergeConfig) -> Dict[str, int]:
    fractions = {
        "train": config.train_fraction,
        "val": config.val_fraction,
        "calib": config.calib_fraction,
        "test": config.test_fraction,
    }
    if n_units >= len(SPLIT_NAMES):
        counts = {name: 1 for name in SPLIT_NAMES}
        assignable_units = n_units - len(SPLIT_NAMES)
    else:
        counts = {name: 0 for name in SPLIT_NAMES}
        assignable_units = n_units
    raw_counts = {name: fractions[name] * assignable_units for name in SPLIT_NAMES}
    for name in SPLIT_NAMES:
        counts[name] += int(raw_counts[name])
    remaining = n_units - sum(counts.values())
    remainders = sorted(
        SPLIT_NAMES,
        key=lambda name: (raw_counts[name] - int(raw_counts[name]), fractions[name]),
        reverse=True,
    )
    for index in range(remaining):
        counts[remainders[index % len(remainders)]] += 1
    return counts


def _split_key(row: dict, split_by_family: bool) -> str:
    if split_by_family:
        return row["family_id"]
    return f"{row['family_id']}::{row['method']}::{row['tensor_file']}"


def _count_splits(rows: List[dict]) -> Dict[str, int]:
    counts = {split_name: 0 for split_name in SPLIT_NAMES}
    for row in rows:
        counts[row["split"]] = counts.get(row["split"], 0) + 1
    return counts


def _count_family_splits(rows: List[dict]) -> Dict[str, int]:
    families_by_split = {split_name: set() for split_name in SPLIT_NAMES}
    for row in rows:
        families_by_split.setdefault(row["split"], set()).add(row["family_id"])
    return {split: len(families) for split, families in families_by_split.items()}


def _count_positive_by_split(rows: List[dict]) -> Dict[str, int]:
    positives = {split_name: 0 for split_name in SPLIT_NAMES}
    seen = set()
    for row in rows:
        key = (row["split"], row["family_id"])
        if key in seen:
            continue
        seen.add(key)
        if str(row.get("gene_label")) == "1":
            positives[row["split"]] = positives.get(row["split"], 0) + 1
    return positives


def _saturation_tier_counts(rows: List[dict]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    seen = set()
    for row in rows:
        family_id = row.get("family_id", "")
        if family_id in seen:
            continue
        seen.add(family_id)
        tier = row.get("saturation_tier") or "unknown"
        counts[tier] = counts.get(tier, 0) + 1
    return dict(sorted(counts.items()))


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON file is not an object: {path}")
    return payload


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
