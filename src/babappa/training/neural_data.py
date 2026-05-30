"""Neural data-loading infrastructure for BABAPPA tensor shards."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

from babappa.datasets.index import read_tsv
from babappa.training.neural_env import safe_import_torch

VALID_NEURAL_SPLITS = {"train", "val", "calib", "test", "all"}
SATURATION_TIER_TO_ID = {
    "unknown": 0,
    "low": 1,
    "moderate": 2,
    "high": 3,
    "extreme": 4,
}
REQUIRED_SPLIT_COLUMNS = [
    "family_id",
    "method",
    "split",
    "tensor_file",
    "gene_label",
]


def saturation_tier_to_id(value: str) -> int:
    """Map saturation tier labels to stable small integer IDs."""
    tier = (value or "unknown").strip().lower()
    return SATURATION_TIER_TO_ID.get(tier, SATURATION_TIER_TO_ID["unknown"])


@dataclass(frozen=True)
class NeuralDatasetConfig:
    """Configuration for inspecting or loading BABAPPA tensor dataset rows."""

    dataset_dir: str
    split: str = "train"
    methods: Optional[List[str]] = None
    max_items: Optional[int] = None
    require_torch: bool = False

    def __post_init__(self) -> None:
        dataset_path = Path(self.dataset_dir)
        if not dataset_path.exists():
            raise ValueError(f"dataset_dir does not exist: {dataset_path}")
        if not (dataset_path / "dataset_index.json").exists():
            raise ValueError(f"dataset_dir is missing dataset_index.json: {dataset_path}")
        if not (dataset_path / "splits.tsv").exists():
            raise ValueError(f"dataset_dir is missing splits.tsv: {dataset_path}")
        if self.split not in VALID_NEURAL_SPLITS:
            allowed = ", ".join(sorted(VALID_NEURAL_SPLITS))
            raise ValueError(f"split must be one of: {allowed}")
        if self.max_items is not None and self.max_items <= 0:
            raise ValueError("max_items must be positive when provided")
        if self.methods is not None:
            resolved_methods = [method for method in self.methods if method]
            object.__setattr__(self, "methods", resolved_methods or None)


def resolve_tensor_file(path_string: str, dataset_dir: Path) -> Path:
    """Resolve tensor paths stored in dataset splits."""
    raw_path = Path(path_string)
    candidates = []
    if raw_path.is_absolute():
        candidates.append(raw_path)
    else:
        candidates.extend(
            [
                Path.cwd() / raw_path,
                dataset_dir / raw_path,
                dataset_dir.parent / raw_path,
            ]
        )
        tensor_dir = _tensor_dir_from_index(dataset_dir)
        if tensor_dir is not None:
            candidates.append(tensor_dir / raw_path)

    for candidate in candidates:
        if candidate.exists():
            return candidate

    checked = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(
        f"could not resolve tensor_file {path_string!r}; checked: {checked}"
    )


def load_neural_rows(config: NeuralDatasetConfig) -> List[dict]:
    """Load and deterministically filter neural dataset split rows."""
    dataset_dir = Path(config.dataset_dir)
    rows = read_tsv(dataset_dir / "splits.tsv")
    filtered = []
    selected_methods = set(config.methods or [])
    for row in rows:
        _validate_split_row(row)
        if config.split != "all" and row["split"] != config.split:
            continue
        if selected_methods and row["method"] not in selected_methods:
            continue
        resolved_row = dict(row)
        if not resolved_row.get("saturation_tier"):
            resolved_row["saturation_tier"] = "unknown"
        filtered.append(resolved_row)

    filtered.sort(key=lambda row: (row["family_id"], row["method"], row["tensor_file"]))
    if config.max_items is not None:
        filtered = filtered[: config.max_items]
    return filtered


def load_tensor_and_label(row: dict, dataset_dir: Path) -> tuple[np.ndarray, int, dict]:
    """Load one tensor shard and its gene-level label from a split row."""
    tensor_file = resolve_tensor_file(row["tensor_file"], dataset_dir)
    with np.load(tensor_file, allow_pickle=False) as shard:
        if "X" not in shard.files:
            raise ValueError(f"tensor shard missing X array: {tensor_file}")
        tensor = shard["X"]
    if tensor.ndim != 3:
        raise ValueError(f"X array is not 3-dimensional: {tensor_file}")
    try:
        label = int(float(row["gene_label"]))
    except ValueError as exc:
        raise ValueError(f"gene_label is not numeric: {row.get('gene_label')}") from exc
    if label not in {0, 1}:
        raise ValueError(f"gene_label must be 0 or 1: {label}")

    metadata = {
        "family_id": row["family_id"],
        "method": row["method"],
        "split": row["split"],
        "tensor_file": row["tensor_file"],
        "resolved_tensor_file": str(tensor_file),
        "saturation_tier": row.get("saturation_tier", "unknown") or "unknown",
    }
    return tensor, label, metadata


def inspect_neural_dataset(config: NeuralDatasetConfig) -> dict:
    """Inspect neural dataset rows and one example tensor without requiring torch."""
    dataset_dir = Path(config.dataset_dir)
    rows = load_neural_rows(config)
    warnings = []
    class_counts = {"0": 0, "1": 0}
    for row in rows:
        label = str(int(float(row["gene_label"])))
        class_counts[label] = class_counts.get(label, 0) + 1

    example_shape = None
    example_dtype = None
    example_tensor_file = None
    if rows:
        tensor, _label, metadata = load_tensor_and_label(rows[0], dataset_dir)
        example_shape = list(tensor.shape)
        example_dtype = str(tensor.dtype)
        example_tensor_file = metadata["tensor_file"]
    else:
        warnings.append("no_rows_after_filtering")

    return {
        "status": "ok",
        "dataset_dir": str(dataset_dir),
        "split": config.split,
        "methods": list(config.methods or []),
        "n_rows": len(rows),
        "n_families": len({row["family_id"] for row in rows}),
        "class_counts": class_counts,
        "methods_present": sorted({row["method"] for row in rows}),
        "example_shape": example_shape,
        "example_dtype": example_dtype,
        "example_tensor_file": example_tensor_file,
        "warnings": warnings,
    }


class BabappaTensorDataset:
    """Lazy torch dataset for BABAPPA tensor shards."""

    def __init__(self, config: NeuralDatasetConfig):
        torch, error = safe_import_torch()
        if torch is None:
            message = (
                "PyTorch is not available. Install torch or use an environment "
                "containing torch."
            )
            if config.require_torch:
                raise RuntimeError(message) from None
            raise RuntimeError(f"{message} Import error: {error}") from None
        self.torch = torch
        self.config = config
        self.dataset_dir = Path(config.dataset_dir)
        self.rows = load_neural_rows(config)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        tensor, label, metadata = load_tensor_and_label(row, self.dataset_dir)
        return {
            "X": self.torch.as_tensor(tensor, dtype=self.torch.long),
            "y": self.torch.tensor(label, dtype=self.torch.float32),
            "family_id": metadata["family_id"],
            "method": metadata["method"],
            "split": metadata["split"],
            "tensor_file": metadata["tensor_file"],
            "saturation_tier": metadata["saturation_tier"],
            "saturation_id": self.torch.tensor(
                saturation_tier_to_id(metadata["saturation_tier"]),
                dtype=self.torch.long,
            ),
        }


def collate_babappa_batch(items: List[dict]) -> dict:
    """Stack BABAPPA torch dataset items into a batch."""
    if not items:
        raise ValueError("cannot collate an empty batch")
    torch, error = safe_import_torch()
    if torch is None:
        raise RuntimeError(
            "PyTorch is not available. Install torch or use an environment containing torch."
        ) from None

    first_shape = tuple(items[0]["X"].shape)
    for item in items:
        if tuple(item["X"].shape) != first_shape:
            raise ValueError("cannot collate tensors with different shapes")

    return {
        "X": torch.stack([item["X"] for item in items], dim=0),
        "y": torch.stack([item["y"] for item in items], dim=0),
        "family_id": [item["family_id"] for item in items],
        "method": [item["method"] for item in items],
        "split": [item["split"] for item in items],
        "tensor_file": [item["tensor_file"] for item in items],
        "saturation_tier": [item["saturation_tier"] for item in items],
        "saturation_id": torch.stack(
            [
                item["saturation_id"]
                if hasattr(item["saturation_id"], "shape")
                else torch.tensor(item["saturation_id"], dtype=torch.long)
                for item in items
            ],
            dim=0,
        ),
    }


def make_smoke_batch(config: NeuralDatasetConfig, batch_size: int = 4) -> dict:
    """Create and summarize the first small neural-data batch."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    torch, _error = safe_import_torch()
    if torch is None:
        raise RuntimeError(
            "PyTorch is not available. Install torch or use an environment containing torch."
        ) from None
    dataset_config = NeuralDatasetConfig(
        dataset_dir=config.dataset_dir,
        split=config.split,
        methods=config.methods,
        max_items=config.max_items,
        require_torch=True,
    )
    dataset = BabappaTensorDataset(dataset_config)
    if len(dataset) == 0:
        raise ValueError("no neural dataset rows available for smoke batch")
    items = [dataset[index] for index in range(min(batch_size, len(dataset)))]
    batch = collate_babappa_batch(items)
    return {
        "batch_size": int(batch["X"].shape[0]),
        "X_shape": list(batch["X"].shape),
        "y_shape": list(batch["y"].shape),
        "X_dtype": str(batch["X"].dtype),
        "y_dtype": str(batch["y"].dtype),
        "family_ids": batch["family_id"],
        "methods": batch["method"],
        "saturation_tiers": batch["saturation_tier"],
        "saturation_ids": [int(value) for value in batch["saturation_id"].tolist()],
        "split": config.split,
    }


def _validate_split_row(row: dict) -> None:
    missing = [column for column in REQUIRED_SPLIT_COLUMNS if column not in row]
    if missing:
        raise ValueError(f"splits.tsv row missing columns: {', '.join(missing)}")


def _tensor_dir_from_index(dataset_dir: Path) -> Optional[Path]:
    index_path = dataset_dir / "dataset_index.json"
    try:
        with index_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    tensor_dir_value = payload.get("tensor_dir")
    if not tensor_dir_value:
        return None
    tensor_dir = Path(str(tensor_dir_value))
    if tensor_dir.is_absolute():
        return tensor_dir
    candidates = [Path.cwd() / tensor_dir, dataset_dir.parent / tensor_dir]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return tensor_dir
