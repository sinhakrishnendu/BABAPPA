"""Validation for large-run planning artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List


def validate_large_run_plan_dir(plan_dir: str | Path) -> dict:
    """Validate that a large-run plan contains templates only."""
    path = Path(plan_dir)
    failures: List[str] = []
    warnings: List[str] = []
    required = {
        "commands": path / "large_run_commands.sh",
        "reference": path / "large_run_commands_commented_reference.sh",
        "monitor": path / "monitor_commands.sh",
        "external": path / "external_aligner_run_commands.sh",
        "expected": path / "expected_outputs.json",
        "markdown": path / "large_run_plan.md",
    }
    for label, file_path in required.items():
        if not file_path.exists():
            failures.append(f"missing_{label}:{file_path}")
    commands_text = required["commands"].read_text(encoding="utf-8") if required["commands"].exists() else ""
    if "USER-RUN ONLY" not in commands_text:
        failures.append("commands_missing_user_run_only_marker")
    if not commands_text.startswith("#!/usr/bin/env bash\nset -euo pipefail"):
        failures.append("commands_missing_executable_header")
    if "babappa make-saturation-panel" not in commands_text:
        failures.append("commands_missing_executable_babappa_steps")
    external_text = required["external"].read_text(encoding="utf-8") if required["external"].exists() else ""
    if "USER-RUN ONLY" not in external_text:
        failures.append("external_commands_missing_user_run_only_marker")
    if required["external"].exists() and "babappa build-site-map" not in external_text:
        failures.append("external_commands_missing_site_map_step")
    reference_text = required["reference"].read_text(encoding="utf-8") if required["reference"].exists() else ""
    if "USER-RUN ONLY" not in reference_text:
        failures.append("reference_missing_user_run_only_marker")
    for line in reference_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("babappa ") or stripped.startswith("conda "):
            failures.append(f"uncommented_reference_command:{stripped}")
    payload = _load_json(required["expected"], failures)
    if payload.get("planner_executed_commands") not in ([], None):
        failures.append("planner_executed_commands_not_empty")
    if (path / "planner_execution.log").exists():
        failures.append("planner_execution_log_present")
    if required["markdown"].exists() and not required["markdown"].read_text(encoding="utf-8").strip():
        failures.append("empty_markdown")
    return {
        "status": "fail" if failures else "ok",
        "n_fail": len(failures),
        "n_warning": len(warnings),
        "failures": failures,
        "warnings": warnings,
    }


def _load_json(path: Path, failures: List[str]) -> dict:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        failures.append(f"could_not_parse_json:{path}:{exc}")
        return {}
    if not isinstance(payload, dict):
        failures.append(f"json_not_object:{path}")
        return {}
    return payload
