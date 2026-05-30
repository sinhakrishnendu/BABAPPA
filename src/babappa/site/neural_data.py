"""Feature-table data loading for site-level neural classifiers."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from babappa.datasets.index import read_tsv
from babappa.site.baseline import get_site_feature_columns
from babappa.training.neural_env import safe_import_torch

VALID_SITE_SPLITS = {"train", "val", "calib", "test", "all"}


@dataclass(frozen=True)
class SiteNeuralDatasetConfig:
    """Configuration for loading site-level feature rows."""

    site_dataset_dir: str
    split: Optional[str] = None
    max_items: Optional[int] = None
    seed: int = 42

    def __post_init__(self) -> None:
        dataset_dir = Path(self.site_dataset_dir)
        if not dataset_dir.exists():
            raise ValueError(f"site_dataset_dir does not exist: {dataset_dir}")
        for filename in ("site_features.tsv", "site_splits.tsv", "site_dataset_index.json"):
            if not (dataset_dir / filename).exists():
                raise ValueError(f"site_dataset_dir is missing {filename}: {dataset_dir}")
        if self.split is not None and self.split not in VALID_SITE_SPLITS:
            allowed = ", ".join(sorted(VALID_SITE_SPLITS))
            raise ValueError(f"split must be one of: {allowed}")
        if self.max_items is not None and self.max_items <= 0:
            raise ValueError("max_items must be positive when supplied")


def load_site_feature_arrays(
    config: SiteNeuralDatasetConfig,
    feature_columns: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, List[dict], List[str]]:
    """Load site-level numeric features, labels, metadata, and feature names."""
    dataset_dir = Path(config.site_dataset_dir)
    rows = read_tsv(dataset_dir / "site_features.tsv")
    if config.split and config.split != "all":
        rows = [row for row in rows if row.get("split") == config.split]
    if config.max_items is not None and len(rows) > config.max_items:
        rows = _sample_rows(rows, config.max_items, config.seed)
    rows.sort(key=lambda row: (row.get("family_id", ""), row.get("method", ""), int(float(row.get("site_index_zero", 0)))))
    selected_columns = list(feature_columns or get_site_feature_columns(rows))
    X = np.zeros((len(rows), len(selected_columns)), dtype=np.float32)
    y = np.zeros(len(rows), dtype=np.float32)
    metadata: List[dict] = []
    for row_index, row in enumerate(rows):
        y[row_index] = 1.0 if str(row.get("y_site")).strip() in {"1", "1.0"} else 0.0
        for column_index, column in enumerate(selected_columns):
            X[row_index, column_index] = _safe_float(row.get(column))
        metadata.append(
            {
                "site_id": row.get("site_id", ""),
                "family_id": row.get("family_id", ""),
                "method": row.get("method", ""),
                "saturation_tier": row.get("saturation_tier", "unknown") or "unknown",
                "split": row.get("split", ""),
                "site_index_zero": row.get("site_index_zero", ""),
            }
        )
    return X, y, metadata, selected_columns


def _sample_rows(rows: List[dict], max_items: int, seed: int) -> List[dict]:
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    return shuffled[:max_items]


def _safe_float(value: object) -> float:
    try:
        if value in ("", None):
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


class SiteFeatureTorchDataset:
    """Torch dataset wrapper around site-level feature arrays."""

    def __init__(
        self,
        config: SiteNeuralDatasetConfig,
        feature_columns: Optional[List[str]] = None,
        feature_mean: Optional[np.ndarray] = None,
        feature_std: Optional[np.ndarray] = None,
    ):
        torch, error = safe_import_torch()
        if torch is None:
            raise RuntimeError(
                "PyTorch is not available. Install torch or use an environment containing torch."
            ) from error
        self.torch = torch
        X, y, metadata, columns = load_site_feature_arrays(config, feature_columns)
        if feature_mean is not None and feature_std is not None and X.size:
            X = (X - feature_mean.astype(np.float32)) / feature_std.astype(np.float32)
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)
        self.metadata = metadata
        self.feature_columns = columns

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, index: int) -> dict:
        return {
            "x": self.torch.as_tensor(self.X[index], dtype=self.torch.float32),
            "y": self.torch.tensor(float(self.y[index]), dtype=self.torch.float32),
            "metadata": self.metadata[index],
        }


def collate_site_feature_batch(items: List[dict]) -> dict:
    """Collate site feature items into a torch batch."""
    if not items:
        raise ValueError("cannot collate empty site batch")
    torch, error = safe_import_torch()
    if torch is None:
        raise RuntimeError(
            "PyTorch is not available. Install torch or use an environment containing torch."
        ) from error
    return {
        "x": torch.stack([item["x"] for item in items], dim=0),
        "y": torch.stack([item["y"] for item in items], dim=0),
        "metadata": [item["metadata"] for item in items],
    }
