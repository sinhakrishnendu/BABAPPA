"""Deterministic resplitting for existing BABAPPA dataset indexes."""

from __future__ import annotations

import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from babappa import __version__
from babappa.datasets.index import read_tsv, write_tsv

RESPLIT_DATASET_VERSION = __version__
SPLIT_NAMES = ["train", "val", "calib", "test"]


@dataclass(frozen=True)
class ResplitDatasetConfig:
    """Configuration for deterministic dataset resplitting."""

    dataset_dir: str
    outdir: str
    seed: int
    train_fraction: float = 0.8
    val_fraction: float = 0.1
    calib_fraction: float = 0.05
    test_fraction: float = 0.05
    split_by_family: bool = True

    def __post_init__(self) -> None:
        dataset_path = Path(self.dataset_dir)
        if not dataset_path.exists():
            raise ValueError(f"dataset_dir does not exist: {dataset_path}")
        for required in ["dataset_index.json", "features.tsv", "splits.tsv"]:
            if not (dataset_path / required).exists():
                raise ValueError(f"dataset_dir is missing {required}: {dataset_path}")
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


def resplit_dataset(config: ResplitDatasetConfig) -> dict:
    """Copy features.tsv and rewrite splits.tsv with deterministic splits."""
    source_dir = Path(config.dataset_dir)
    outdir = Path(config.outdir)
    feature_rows = read_tsv(source_dir / "features.tsv")
    split_rows = read_tsv(source_dir / "splits.tsv")
    if not split_rows:
        raise ValueError("splits.tsv has no rows")

    shutil.copyfile(source_dir / "features.tsv", outdir / "features.tsv")
    assignments = _assign_splits(split_rows, config)
    fieldnames = list(split_rows[0].keys())
    if "split" not in fieldnames:
        fieldnames.append("split")
    rewritten_rows = []
    for row in split_rows:
        rewritten = dict(row)
        rewritten["split"] = assignments[_split_key(row, config.split_by_family)]
        rewritten_rows.append(rewritten)
    write_tsv(outdir / "splits.tsv", rewritten_rows, fieldnames)

    payload = _dataset_index_payload(config, source_dir, feature_rows, rewritten_rows)
    _write_json(outdir / "dataset_index.json", payload)
    return {
        "status": "ok",
        "outdir": str(outdir),
        "features": str(outdir / "features.tsv"),
        "splits": str(outdir / "splits.tsv"),
        "index": str(outdir / "dataset_index.json"),
        "n_rows": payload["n_rows"],
        "n_families": payload["n_families"],
        "split_counts_rows": payload["split_counts_rows"],
    }


def _assign_splits(rows: List[dict], config: ResplitDatasetConfig) -> Dict[str, str]:
    keys = sorted({_split_key(row, config.split_by_family) for row in rows})
    rng = random.Random(config.seed)
    rng.shuffle(keys)
    n = len(keys)
    train_end = int(round(n * config.train_fraction))
    val_end = train_end + int(round(n * config.val_fraction))
    calib_end = val_end + int(round(n * config.calib_fraction))
    boundaries = {
        "train": set(keys[:train_end]),
        "val": set(keys[train_end:val_end]),
        "calib": set(keys[val_end:calib_end]),
        "test": set(keys[calib_end:]),
    }
    assignments = {}
    for split, split_keys in boundaries.items():
        for key in split_keys:
            assignments[key] = split
    for key in keys:
        assignments.setdefault(key, "test")
    return assignments


def _split_key(row: dict, split_by_family: bool) -> str:
    if split_by_family:
        return row.get("family_id", "")
    return "\t".join([row.get("family_id", ""), row.get("method", ""), row.get("tensor_file", "")])


def _dataset_index_payload(
    config: ResplitDatasetConfig,
    source_dir: Path,
    feature_rows: List[dict],
    split_rows: List[dict],
) -> dict:
    family_ids = sorted({row.get("family_id", "") for row in split_rows})
    methods = sorted({row.get("method", "") for row in split_rows if row.get("method")})
    return {
        "dataset_index_version": RESPLIT_DATASET_VERSION,
        "resplit_dataset_version": RESPLIT_DATASET_VERSION,
        "source_dataset_dir": str(source_dir),
        "n_rows": len(split_rows),
        "n_families": len(family_ids),
        "methods": methods,
        "seed": config.seed,
        "split_by_family": config.split_by_family,
        "fractions": {
            "train": config.train_fraction,
            "val": config.val_fraction,
            "calib": config.calib_fraction,
            "test": config.test_fraction,
        },
        "split_counts_rows": _split_counts_rows(split_rows),
        "split_counts_families": _split_counts_families(split_rows),
        "positive_counts_by_split": _positive_counts_by_split(split_rows),
        "saturation_tier_counts": _saturation_tier_counts(split_rows or feature_rows),
        "files": {
            "features": "features.tsv",
            "splits": "splits.tsv",
            "index": "dataset_index.json",
        },
    }


def _split_counts_rows(rows: List[dict]) -> dict:
    return {split: sum(1 for row in rows if row.get("split") == split) for split in SPLIT_NAMES}


def _split_counts_families(rows: List[dict]) -> dict:
    counts = {}
    for split in SPLIT_NAMES:
        counts[split] = len({row.get("family_id", "") for row in rows if row.get("split") == split})
    return counts


def _positive_counts_by_split(rows: List[dict]) -> dict:
    counts = {}
    for split in SPLIT_NAMES:
        counts[split] = sum(
            1
            for row in rows
            if row.get("split") == split and str(row.get("gene_label")) in {"1", "1.0"}
        )
    return counts


def _saturation_tier_counts(rows: List[dict]) -> dict:
    counts: Dict[str, int] = {}
    for row in rows:
        tier = row.get("saturation_tier") or "unknown"
        counts[tier] = counts.get(tier, 0) + 1
    return dict(sorted(counts.items()))


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
