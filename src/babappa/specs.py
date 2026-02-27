from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .hash_utils import sha256_json
from .neutral import NeutralSpec
from .representation import FEATURE_NAMES
from .schemas import validate_neutral_model_payload, validate_phi_spec_payload


def default_phi_spec() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "name": "phi_default_v1",
        "description": "Deterministic codon-column feature map shipped with BABAPPA.",
        "feature_names": list(FEATURE_NAMES),
    }


def load_phi_spec(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        payload = default_phi_spec()
        validate_phi_spec_payload(payload)
        return payload
    p = Path(path)
    with p.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"phi spec must be a JSON object: {p}")
    validate_phi_spec_payload(payload)
    return payload


def phi_hash(phi_spec: dict[str, Any]) -> str:
    return sha256_json(phi_spec)


def load_neutral_model_spec(path: str | Path) -> tuple[NeutralSpec, dict[str, Any]]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"neutral model file must be JSON object: {p}")
    validate_neutral_model_payload(payload)

    tree_newick = payload.get("tree_newick")
    if not tree_newick:
        tree_file = payload.get("tree_file")
        if not tree_file:
            raise ValueError("neutral model must include tree_newick or tree_file.")
        tree_path = Path(tree_file)
        if not tree_path.is_absolute():
            tree_path = (p.parent / tree_path).resolve()
        tree_newick = tree_path.read_text(encoding="utf-8").strip()

    codon_frequencies = payload.get("codon_frequencies")
    if codon_frequencies is not None:
        codon_frequencies = {str(k): float(v) for k, v in dict(codon_frequencies).items()}

    spec = NeutralSpec(
        tree_newick=str(tree_newick),
        kappa=float(payload["kappa"]),
        omega=float(payload["omega"]),
        codon_frequencies=codon_frequencies,
        simulator="gy94",
    )
    return spec, payload


def m0_hash(neutral_payload: dict[str, Any]) -> str:
    return sha256_json(neutral_payload)
