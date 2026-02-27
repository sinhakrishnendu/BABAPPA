from __future__ import annotations

import json
from pathlib import Path
from typing import Any


SCHEMAS_DIR = Path(__file__).resolve().parents[2] / "schemas"


def load_schema(schema_filename: str) -> dict[str, Any]:
    path = SCHEMAS_DIR / schema_filename
    if not path.exists():
        raise FileNotFoundError(f"Schema not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Schema {path} must be a JSON object.")
    return payload


def _ensure_type(payload: Any, expected: type, label: str) -> None:
    if not isinstance(payload, expected):
        raise ValueError(f"{label} must be {expected.__name__}.")


def _require_keys(payload: dict[str, Any], keys: list[str], label: str) -> None:
    missing = [k for k in keys if k not in payload]
    if missing:
        raise ValueError(f"{label} missing required keys: {', '.join(missing)}")


def validate_neutral_model_payload(payload: dict[str, Any]) -> None:
    _ensure_type(payload, dict, "neutral model payload")
    _require_keys(
        payload,
        ["schema_version", "model_family", "genetic_code_table", "kappa", "omega"],
        "neutral model payload",
    )
    if int(payload["schema_version"]) != 1:
        raise ValueError("neutral_model schema_version must be 1.")
    if str(payload["model_family"]).upper() != "GY94":
        raise ValueError("neutral model family must be GY94.")
    if float(payload["omega"]) != 1.0:
        raise ValueError("neutral model omega must be exactly 1.0.")
    if float(payload["kappa"]) <= 0:
        raise ValueError("neutral model kappa must be > 0.")
    if not payload.get("tree_newick") and not payload.get("tree_file"):
        raise ValueError("neutral model must include tree_newick or tree_file.")
    codon_freq = payload.get("codon_frequencies")
    if codon_freq is None and not payload.get("codon_frequencies_method"):
        raise ValueError(
            "neutral model must include codon_frequencies or codon_frequencies_method."
        )
    if codon_freq is not None and not isinstance(codon_freq, dict):
        raise ValueError("codon_frequencies must be a JSON object.")


def validate_phi_spec_payload(payload: dict[str, Any]) -> None:
    _ensure_type(payload, dict, "phi spec payload")
    _require_keys(payload, ["schema_version", "name", "feature_names"], "phi spec payload")
    if int(payload["schema_version"]) != 1:
        raise ValueError("phi schema_version must be 1.")
    if not isinstance(payload["feature_names"], list) or not payload["feature_names"]:
        raise ValueError("phi feature_names must be a non-empty list.")


def validate_energy_payload(payload: dict[str, Any]) -> None:
    _ensure_type(payload, dict, "energy payload")
    schema_version = int(payload.get("schema_version", 1))
    if schema_version >= 3:
        _require_keys(
            payload,
            [
                "schema_version",
                "energy_family",
                "feature_names",
                "mean",
                "covariance",
                "precision",
                "training_samples",
            ],
            "energy payload",
        )
        if str(payload["energy_family"]) != "QuadraticEnergy":
            raise ValueError("energy_family must be QuadraticEnergy.")
    else:
        _require_keys(
            payload,
            ["schema_version", "feature_names", "mean", "covariance", "precision", "training_samples"],
            "energy payload",
        )
    if int(payload["training_samples"]) <= 0:
        raise ValueError("training_samples must be > 0.")


def validate_manifest_payload(payload: dict[str, Any], manifest_kind: str) -> None:
    _ensure_type(payload, dict, "manifest payload")
    if int(payload.get("schema_version", -1)) != 1:
        raise ValueError("manifest schema_version must be 1.")
    _require_keys(
        payload,
        ["command", "tool_version", "system", "seed", "seed_policy"],
        "manifest payload",
    )
    kind = manifest_kind.lower()
    if kind == "training":
        _require_keys(payload, ["m0_hash", "phi_hash", "model_hash"], "training manifest")
    elif kind == "analysis":
        _require_keys(
            payload,
            ["m0_hash", "phi_hash", "model_hash", "tail", "N"],
            "analysis manifest",
        )
    elif kind == "benchmark":
        _require_keys(payload, ["grid", "pack_dir"], "benchmark manifest")
    else:
        raise ValueError(f"Unknown manifest kind: {manifest_kind}")


def validate_dataset_payload(payload: dict[str, Any]) -> None:
    _ensure_type(payload, dict, "dataset payload")
    _require_keys(payload, ["schema_version", "tree_path", "genes"], "dataset payload")
    if int(payload["schema_version"]) != 1:
        raise ValueError("dataset schema_version must be 1.")
    genes = payload["genes"]
    if not isinstance(genes, list) or not genes:
        raise ValueError("dataset genes must be a non-empty list.")
    for idx, gene in enumerate(genes, start=1):
        if not isinstance(gene, dict):
            raise ValueError(f"dataset genes[{idx}] must be an object.")
        _require_keys(gene, ["gene_id", "alignment_path"], f"dataset genes[{idx}]")
