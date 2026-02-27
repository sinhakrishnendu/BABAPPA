from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .neutral import NeutralSpec
from .schemas import validate_energy_payload
from .system_info import now_utc_iso


@dataclass
class EnergyModel:
    feature_names: tuple[str, ...]
    mean: np.ndarray
    covariance: np.ndarray
    precision: np.ndarray
    training_samples: int
    training_alignments: int
    max_training_length: int
    ridge: float
    seed: int | None
    objective: float
    created_at_utc: str
    neutral_spec: NeutralSpec | None = None
    training_mode: str = "alignment"
    energy_family: str = "QuadraticEnergy"
    energy_capacity: int = 0
    optimizer: str = "closed_form_score_matching"
    schema_version: int = 3

    @classmethod
    def create(
        cls,
        *,
        feature_names: tuple[str, ...],
        mean: np.ndarray,
        covariance: np.ndarray,
        precision: np.ndarray,
        training_samples: int,
        training_alignments: int,
        max_training_length: int,
        ridge: float,
        seed: int | None,
        objective: float,
        neutral_spec: NeutralSpec | None = None,
        training_mode: str = "alignment",
        energy_family: str = "QuadraticEnergy",
        optimizer: str = "closed_form_score_matching",
    ) -> "EnergyModel":
        capacity = int(mean.shape[0])
        return cls(
            feature_names=feature_names,
            mean=mean,
            covariance=covariance,
            precision=precision,
            training_samples=training_samples,
            training_alignments=training_alignments,
            max_training_length=max_training_length,
            ridge=ridge,
            seed=seed,
            objective=objective,
            created_at_utc=now_utc_iso(),
            neutral_spec=neutral_spec,
            training_mode=training_mode,
            energy_family=energy_family,
            energy_capacity=capacity,
            optimizer=optimizer,
        )

    def energy(self, features: np.ndarray) -> np.ndarray:
        centered = features - self.mean
        values = 0.5 * np.einsum("...i,ij,...j->...", centered, self.precision, centered)
        return values.astype(float, copy=False)

    def gradient(self, features: np.ndarray) -> np.ndarray:
        centered = features - self.mean
        return centered @ self.precision.T

    def laplacian(self, features: np.ndarray) -> np.ndarray:
        trace = float(np.trace(self.precision))
        return np.full(features.shape[0], trace, dtype=float)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "feature_names": list(self.feature_names),
            "mean": self.mean.tolist(),
            "covariance": self.covariance.tolist(),
            "precision": self.precision.tolist(),
            "training_samples": self.training_samples,
            "training_alignments": self.training_alignments,
            "max_training_length": self.max_training_length,
            "ridge": self.ridge,
            "seed": self.seed,
            "objective": self.objective,
            "created_at_utc": self.created_at_utc,
            "neutral_spec": (None if self.neutral_spec is None else self.neutral_spec.to_dict()),
            "training_mode": self.training_mode,
            "energy_family": self.energy_family,
            "energy_capacity": self.energy_capacity,
            "optimizer": self.optimizer,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "EnergyModel":
        schema_version = int(payload.get("schema_version", 1))
        if schema_version not in {1, 2, 3}:
            raise ValueError(f"Unsupported model schema version: {schema_version}")

        neutral_spec_payload = payload.get("neutral_spec")
        neutral_spec = (
            None
            if neutral_spec_payload is None
            else NeutralSpec.from_dict(dict(neutral_spec_payload))
        )

        return cls(
            schema_version=schema_version,
            feature_names=tuple(str(x) for x in payload["feature_names"]),  # type: ignore[index]
            mean=np.array(payload["mean"], dtype=float),  # type: ignore[index]
            covariance=np.array(payload["covariance"], dtype=float),  # type: ignore[index]
            precision=np.array(payload["precision"], dtype=float),  # type: ignore[index]
            training_samples=int(payload["training_samples"]),  # type: ignore[index]
            training_alignments=int(payload["training_alignments"]),  # type: ignore[index]
            max_training_length=int(payload["max_training_length"]),  # type: ignore[index]
            ridge=float(payload["ridge"]),  # type: ignore[index]
            seed=(None if payload.get("seed") is None else int(payload["seed"])),
            objective=float(payload["objective"]),  # type: ignore[index]
            created_at_utc=str(payload["created_at_utc"]),  # type: ignore[index]
            neutral_spec=neutral_spec,
            training_mode=str(payload.get("training_mode", "alignment")),
            energy_family=str(payload.get("energy_family", "QuadraticEnergy")),
            energy_capacity=int(payload.get("energy_capacity", len(payload["mean"]))),  # type: ignore[index]
            optimizer=str(payload.get("optimizer", "closed_form_score_matching")),
        )


def save_model(model: EnergyModel, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(model.to_dict(), handle, indent=2, sort_keys=True)
        handle.write("\n")


def load_model(path: str | Path) -> EnergyModel:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Model file does not contain a JSON object: {path}")
    validate_energy_payload(payload)
    return EnergyModel.from_dict(payload)
