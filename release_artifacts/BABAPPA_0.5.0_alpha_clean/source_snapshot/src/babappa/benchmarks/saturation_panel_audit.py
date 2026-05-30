"""Validation utilities for saturation panel outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Union


def validate_saturation_panel_dir(panel_dir: Union[str, Path]) -> dict:
    """Validate a BABAPPA saturation panel directory."""
    path = Path(panel_dir)
    failures: List[str] = []
    warnings: List[str] = []
    panel_json = path / "saturation_panel.json"
    panel_markdown = path / "saturation_panel.md"

    payload = _read_panel(panel_json, failures)
    _validate_markdown(panel_markdown, failures)
    if isinstance(payload, dict):
        tier_outputs = payload.get("tier_outputs")
        if not isinstance(tier_outputs, dict) or not tier_outputs:
            failures.append("saturation_panel.json missing non-empty tier_outputs")
        else:
            for tier, outputs in tier_outputs.items():
                _validate_tier(str(tier), outputs, failures, warnings)

    return {
        "status": "fail" if failures else "ok",
        "n_fail": len(failures),
        "n_warning": len(warnings),
        "failures": failures,
        "warnings": warnings,
    }


def _read_panel(path: Path, failures: List[str]) -> object:
    if not path.exists():
        failures.append(f"missing saturation_panel.json: {path}")
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        failures.append(f"unreadable saturation_panel.json: {exc}")
        return None
    if not isinstance(payload, dict):
        failures.append("saturation_panel.json is not a JSON object")
        return None
    return payload


def _validate_markdown(path: Path, failures: List[str]) -> None:
    if not path.exists():
        failures.append(f"missing saturation_panel.md: {path}")
        return
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        failures.append(f"unreadable saturation_panel.md: {exc}")
        return
    if not text.strip():
        failures.append("saturation_panel.md is empty")


def _validate_tier(
    tier: str, outputs: object, failures: List[str], warnings: List[str]
) -> None:
    if not isinstance(outputs, dict):
        failures.append(f"{tier}: tier output is not an object")
        return
    _require_file(outputs.get("sim_dir"), "manifest.json", tier, failures)
    _require_file(outputs.get("sim_audit_dir"), "dataset_summary.json", tier, failures)
    _require_file(outputs.get("align_dir"), "alignment_manifest.json", tier, failures)
    if outputs.get("tensor_dir") is not None:
        _require_file(outputs.get("tensor_dir"), "tensor_manifest.json", tier, failures)
    else:
        warnings.append(f"{tier}: tensor outputs were not built")
    if outputs.get("dataset_dir") is not None:
        _require_file(outputs.get("dataset_dir"), "dataset_index.json", tier, failures)
    else:
        warnings.append(f"{tier}: dataset outputs were not indexed")


def _require_file(
    directory: object, filename: str, tier: str, failures: List[str]
) -> None:
    if not directory:
        failures.append(f"{tier}: missing directory for {filename}")
        return
    path = Path(str(directory)) / filename
    if not path.exists():
        failures.append(f"{tier}: missing {filename}: {path}")
