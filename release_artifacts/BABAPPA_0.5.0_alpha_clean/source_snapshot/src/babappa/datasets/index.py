"""Dataset indexing, feature extraction, and deterministic splits."""

from __future__ import annotations

import csv
import json
import os
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from babappa import __version__

DATASET_INDEX_VERSION = __version__
SPLIT_NAMES = ["train", "val", "calib", "test"]
FEATURE_FIELDNAMES = [
    "family_id",
    "method",
    "tensor_file",
    "tensor_meta_file",
    "labels_file",
    "gene_label",
    "saturation_tier",
    "foreground_taxon",
    "n_selected_sites",
    "n_taxa",
    "n_codons",
    "n_channels",
    "gap_codon_count",
    "gap_codon_fraction",
    "codon_id_mean",
    "codon_id_std",
    "codon_id_min",
    "codon_id_max",
    "codon_id_nonzero_fraction",
    "unique_codon_id_count",
    "unique_codon_id_fraction",
    "mean_taxon_codon_id_std",
    "mean_site_codon_id_std",
]
SPLIT_FIELDNAMES = [
    "family_id",
    "method",
    "tensor_file",
    "gene_label",
    "saturation_tier",
    "split",
]


@dataclass(frozen=True)
class DatasetIndexConfig:
    """Configuration for building a BABAPPA dataset index."""

    tensor_dir: str
    outdir: str
    methods: Optional[List[str]] = None
    seed: int = 42
    train_fraction: float = 0.8
    val_fraction: float = 0.1
    calib_fraction: float = 0.05
    test_fraction: float = 0.05
    split_by_family: bool = True
    workers: int = 1

    def __post_init__(self) -> None:
        tensor_path = Path(self.tensor_dir)
        out_path = Path(self.outdir)
        if not tensor_path.exists():
            raise ValueError(f"tensor_dir does not exist: {tensor_path}")
        manifest_path = tensor_path / "tensor_manifest.json"
        if not manifest_path.exists():
            raise ValueError(f"tensor_dir is missing tensor_manifest.json: {tensor_path}")
        audit_path = tensor_path / "tensor_audit.tsv"
        if not audit_path.exists():
            raise ValueError(f"tensor_dir is missing tensor_audit.tsv: {tensor_path}")

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
        if self.workers < 1:
            raise ValueError("workers must be >= 1")

        tensor_manifest = _read_json(manifest_path)
        available_methods = tensor_manifest.get("methods")
        if not isinstance(available_methods, list) or not available_methods:
            raise ValueError("tensor manifest does not contain non-empty methods")

        if self.methods is None or not self.methods:
            resolved_methods = list(available_methods)
        else:
            unknown_methods = sorted(set(self.methods) - set(available_methods))
            if unknown_methods:
                unknown = ", ".join(unknown_methods)
                allowed = ", ".join(str(method) for method in available_methods)
                raise ValueError(
                    f"unknown dataset method(s): {unknown}; available: {allowed}"
                )
            resolved_methods = list(self.methods)

        object.__setattr__(self, "methods", resolved_methods)
        out_path.mkdir(parents=True, exist_ok=True)


def read_tsv(path: Path) -> List[Dict[str, str]]:
    """Read a TSV file as a list of dictionaries."""
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: List[dict], fieldnames: List[str]) -> None:
    """Write rows to TSV using a stable column order."""
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def extract_tensor_features(
    tensor_file: Path, meta_json: Path, labels_json: Path
) -> dict:
    """Extract simple non-leaking gene-level features from one tensor shard."""
    with np.load(tensor_file, allow_pickle=False) as shard:
        if "X" not in shard.files:
            raise ValueError(f"tensor file missing X array: {tensor_file}")
        tensor = shard["X"]

    if tensor.ndim != 3:
        raise ValueError(f"X array is not 3-dimensional: {tensor_file}")

    meta = _read_json(meta_json)
    labels = _read_json(labels_json)
    codon_ids = tensor[:, :, 0].astype(np.float64)
    nonzero_mask = codon_ids != 0
    unique_ids = np.unique(codon_ids.astype(np.int64))
    total_positions = codon_ids.size
    n_taxa, n_codons, n_channels = tensor.shape

    if n_channels > 1:
        gap_codon_count = int(tensor[:, :, 1].sum())
        gap_codon_fraction = (
            0.0 if total_positions == 0 else gap_codon_count / total_positions
        )
    else:
        gap_codon_count = int(meta.get("gap_codon_count", 0))
        gap_codon_fraction = float(meta.get("gap_codon_fraction", 0.0))

    unique_count = int(unique_ids.size)
    return {
        "family_id": str(meta.get("family_id", labels.get("family_id", ""))),
        "method": str(meta.get("method", "")),
        "tensor_file": str(tensor_file),
        "tensor_meta_file": str(meta_json),
        "labels_file": str(labels_json),
        "gene_label": labels.get("gene_label"),
        "saturation_tier": labels.get("saturation_tier"),
        "foreground_taxon": labels.get("foreground_taxon") or "",
        "n_selected_sites": labels.get("n_selected_sites", 0),
        "n_taxa": n_taxa,
        "n_codons": n_codons,
        "n_channels": n_channels,
        "gap_codon_count": gap_codon_count,
        "gap_codon_fraction": gap_codon_fraction,
        "codon_id_mean": float(codon_ids.mean()),
        "codon_id_std": float(codon_ids.std()),
        "codon_id_min": int(codon_ids.min()),
        "codon_id_max": int(codon_ids.max()),
        "codon_id_nonzero_fraction": float(nonzero_mask.sum() / total_positions),
        "unique_codon_id_count": unique_count,
        "unique_codon_id_fraction": float(unique_count / total_positions),
        "mean_taxon_codon_id_std": float(np.std(codon_ids, axis=1).mean()),
        "mean_site_codon_id_std": float(np.std(codon_ids, axis=0).mean()),
    }


def build_dataset_index(config: DatasetIndexConfig) -> dict:
    """Build feature table, splits, and dataset index metadata."""
    tensor_dir = Path(config.tensor_dir)
    outdir = Path(config.outdir)
    tensor_manifest = _read_json(tensor_dir / "tensor_manifest.json")
    audit_rows = read_tsv(tensor_dir / "tensor_audit.tsv")
    selected_methods = set(config.methods or tensor_manifest["methods"])
    feature_rows: List[dict] = []
    feature_tasks: List[tuple[str, str, str, str]] = []

    for audit_row in audit_rows:
        if audit_row.get("status") != "ok":
            continue
        if audit_row.get("method") not in selected_methods:
            continue
        tensor_file = tensor_dir / audit_row["tensor_file"]
        meta_json = tensor_file.with_name(
            tensor_file.name.replace(".tensor.npz", ".tensor_meta.json")
        )
        family_id = audit_row["family_id"]
        labels_json = tensor_dir / "families" / family_id / f"{family_id}.labels.json"
        feature_tasks.append(
            (
                str(tensor_file),
                str(meta_json),
                str(labels_json),
                str(tensor_dir),
            )
        )

    effective_workers = min(config.workers, len(feature_tasks) or 1, os.cpu_count() or config.workers)
    if effective_workers <= 1 or len(feature_tasks) <= 1:
        feature_rows = [_extract_tensor_features_task(task) for task in feature_tasks]
    else:
        with ProcessPoolExecutor(max_workers=effective_workers) as executor:
            futures = {
                executor.submit(_extract_tensor_features_task, task): task
                for task in feature_tasks
            }
            feature_rows = [future.result() for future in as_completed(futures)]

    feature_rows.sort(key=lambda row: (row["family_id"], row["method"]))
    split_assignments = _assign_splits(feature_rows, config)
    split_rows = []
    for row in feature_rows:
        split = split_assignments[_split_key(row, config.split_by_family)]
        split_rows.append(
            {
                "family_id": row["family_id"],
                "method": row["method"],
                "tensor_file": row["tensor_file"],
                "gene_label": row["gene_label"],
                "saturation_tier": row["saturation_tier"],
                "split": split,
            }
        )

    features_path = outdir / "features.tsv"
    splits_path = outdir / "splits.tsv"
    index_path = outdir / "dataset_index.json"
    write_tsv(features_path, feature_rows, FEATURE_FIELDNAMES)
    write_tsv(splits_path, split_rows, SPLIT_FIELDNAMES)

    family_ids = sorted({row["family_id"] for row in feature_rows})
    index_payload = {
        "dataset_index_version": DATASET_INDEX_VERSION,
        "tensor_dir": str(tensor_dir),
        "n_rows": len(feature_rows),
        "n_families": len(family_ids),
        "methods": list(config.methods or []),
        "seed": config.seed,
        "split_by_family": config.split_by_family,
        "workers": effective_workers,
        "requested_workers": config.workers,
        "fractions": {
            "train": config.train_fraction,
            "val": config.val_fraction,
            "calib": config.calib_fraction,
            "test": config.test_fraction,
        },
        "split_counts_rows": _count_splits(split_rows, key="split"),
        "split_counts_families": _count_family_splits(split_rows),
        "positive_counts_by_split": _count_positive_by_split(split_rows),
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
        "n_rows": len(feature_rows),
        "n_families": len(family_ids),
        "methods": list(config.methods or []),
        "workers": effective_workers,
        "requested_workers": config.workers,
    }


def _assign_splits(feature_rows: List[dict], config: DatasetIndexConfig) -> Dict[str, str]:
    if config.split_by_family:
        units = sorted({row["family_id"] for row in feature_rows})
    else:
        units = [
            f"{row['family_id']}::{row['method']}::{row['tensor_file']}"
            for row in feature_rows
        ]

    shuffled = list(units)
    rng = random.Random(config.seed)
    rng.shuffle(shuffled)
    split_counts = _split_counts(len(shuffled), config)
    assignments: Dict[str, str] = {}
    cursor = 0
    for split_name in SPLIT_NAMES:
        count = split_counts[split_name]
        for unit in shuffled[cursor:cursor + count]:
            assignments[unit] = split_name
        cursor += count
    return assignments


def _extract_tensor_features_task(payload: tuple[str, str, str, str]) -> dict:
    tensor_file = Path(payload[0])
    meta_json = Path(payload[1])
    labels_json = Path(payload[2])
    tensor_dir = Path(payload[3])
    features = extract_tensor_features(tensor_file, meta_json, labels_json)
    features["tensor_file"] = str(tensor_file.relative_to(tensor_dir))
    features["tensor_meta_file"] = str(meta_json.relative_to(tensor_dir))
    features["labels_file"] = str(labels_json.relative_to(tensor_dir))
    return features


def _split_counts(n_units: int, config: DatasetIndexConfig) -> Dict[str, int]:
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
        key=lambda name: (raw_counts[name] - counts[name], fractions[name]),
        reverse=True,
    )
    for index in range(remaining):
        counts[remainders[index % len(remainders)]] += 1
    return counts


def _split_key(row: dict, split_by_family: bool) -> str:
    if split_by_family:
        return row["family_id"]
    return f"{row['family_id']}::{row['method']}::{row['tensor_file']}"


def _count_splits(rows: List[dict], key: str) -> Dict[str, int]:
    counts = {split_name: 0 for split_name in SPLIT_NAMES}
    for row in rows:
        counts[row[key]] = counts.get(row[key], 0) + 1
    return counts


def _count_family_splits(split_rows: List[dict]) -> Dict[str, int]:
    families_by_split = {split_name: set() for split_name in SPLIT_NAMES}
    for row in split_rows:
        families_by_split[row["split"]].add(row["family_id"])
    return {split: len(families) for split, families in families_by_split.items()}


def _count_positive_by_split(split_rows: List[dict]) -> Dict[str, int]:
    positives = {split_name: 0 for split_name in SPLIT_NAMES}
    seen_families = set()
    for row in split_rows:
        key = (row["split"], row["family_id"])
        if key in seen_families:
            continue
        seen_families.add(key)
        if str(row["gene_label"]) == "1":
            positives[row["split"]] = positives.get(row["split"], 0) + 1
    return positives


def _read_json(path: Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON file is not an object: {path}")
    return payload


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
