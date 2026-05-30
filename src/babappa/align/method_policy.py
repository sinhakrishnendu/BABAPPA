"""Method-level quarantine policy for external aligner workflows."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from babappa import __version__
from babappa.datasets.index import write_tsv

METHOD_POLICY_VERSION = __version__
METHOD_POLICY_FIELDNAMES = [
    "method",
    "attempted_families",
    "successful_families",
    "failed_families",
    "failure_fraction",
    "site_map_unique_fraction",
    "site_map_conflict_fraction",
    "site_map_frame_error_fraction",
    "recommendation",
    "reason",
]


@dataclass(frozen=True)
class MethodPolicyConfig:
    """Configuration for method-level external aligner policy."""

    align_dir: str
    site_map_dir: str
    outdir: str
    max_frame_error_fraction: float = 0.0
    max_conflict_fraction: float = 0.03
    max_method_failure_fraction: float = 0.01

    def __post_init__(self) -> None:
        if not Path(self.align_dir).exists():
            raise ValueError(f"align_dir does not exist: {self.align_dir}")
        if not Path(self.align_dir, "alignment_manifest.json").exists():
            raise ValueError(f"align_dir is missing alignment_manifest.json: {self.align_dir}")
        if not Path(self.site_map_dir).exists():
            raise ValueError(f"site_map_dir does not exist: {self.site_map_dir}")
        if not Path(self.site_map_dir, "site_map_manifest.json").exists():
            raise ValueError(f"site_map_dir is missing site_map_manifest.json: {self.site_map_dir}")
        for name, value in [
            ("max_frame_error_fraction", self.max_frame_error_fraction),
            ("max_conflict_fraction", self.max_conflict_fraction),
            ("max_method_failure_fraction", self.max_method_failure_fraction),
        ]:
            if value < 0 or value > 1:
                raise ValueError(f"{name} must be between 0 and 1")
        Path(self.outdir).mkdir(parents=True, exist_ok=True)


def build_method_policy(config: MethodPolicyConfig) -> dict:
    """Build method-level usability/quarantine policy artifacts."""
    align_dir = Path(config.align_dir)
    site_map_dir = Path(config.site_map_dir)
    outdir = Path(config.outdir)
    align_manifest = _read_json(align_dir / "alignment_manifest.json")
    site_map_manifest = _read_json(site_map_dir / "site_map_manifest.json")
    site_map_summary = _site_map_summary(site_map_manifest)
    method_status = _alignment_method_status(align_manifest)
    methods = sorted(set(method_status) | set(site_map_summary))
    rows = [
        _policy_row(method, method_status.get(method, {}), site_map_summary.get(method, {}), config)
        for method in methods
    ]
    json_path = outdir / "method_policy.json"
    tsv_path = outdir / "method_policy.tsv"
    md_path = outdir / "method_policy.md"
    payload = {
        "method_policy_version": METHOD_POLICY_VERSION,
        "align_dir": str(align_dir),
        "site_map_dir": str(site_map_dir),
        "thresholds": {
            "max_frame_error_fraction": config.max_frame_error_fraction,
            "max_conflict_fraction": config.max_conflict_fraction,
            "max_method_failure_fraction": config.max_method_failure_fraction,
        },
        "methods": rows,
        "usable_methods": [row["method"] for row in rows if row["recommendation"] in {"usable", "caution"}],
        "quarantined_methods": [row["method"] for row in rows if row["recommendation"] == "quarantine"],
        "generated_files": {
            "json": str(json_path),
            "tsv": str(tsv_path),
            "markdown": str(md_path),
        },
    }
    _write_json(json_path, payload)
    write_tsv(tsv_path, rows, METHOD_POLICY_FIELDNAMES)
    md_path.write_text(_render_markdown(payload), encoding="utf-8")
    return {
        "status": "ok",
        "outdir": str(outdir),
        "json": str(json_path),
        "tsv": str(tsv_path),
        "markdown": str(md_path),
        "usable_methods": payload["usable_methods"],
        "quarantined_methods": payload["quarantined_methods"],
        "n_methods": len(rows),
    }


def validate_method_policy_dir(policy_dir: str | Path) -> dict:
    """Validate method-policy artifacts."""
    path = Path(policy_dir)
    failures: List[str] = []
    warnings: List[str] = []
    json_path = path / "method_policy.json"
    tsv_path = path / "method_policy.tsv"
    md_path = path / "method_policy.md"
    payload = _load_json(json_path, failures)
    rows = _load_tsv(tsv_path, failures)
    if not md_path.exists():
        failures.append(f"missing_file:{md_path}")
    elif not md_path.read_text(encoding="utf-8").strip():
        failures.append(f"empty_markdown:{md_path}")
    if payload and not isinstance(payload.get("methods"), list):
        failures.append("json_methods_not_list")
    if rows:
        missing = sorted(set(METHOD_POLICY_FIELDNAMES) - set(rows[0]))
        if missing:
            failures.append(f"tsv_missing_columns:{','.join(missing)}")
        valid = {"usable", "caution", "quarantine"}
        for row in rows:
            if row.get("recommendation") not in valid:
                failures.append(f"invalid_recommendation:{row.get('method')}:{row.get('recommendation')}")
            for column in [
                "failure_fraction",
                "site_map_unique_fraction",
                "site_map_conflict_fraction",
                "site_map_frame_error_fraction",
            ]:
                value = _safe_float(row.get(column), default=0.0)
                if value < 0 or value > 1:
                    failures.append(f"fraction_out_of_range:{row.get('method')}:{column}:{value}")
    elif tsv_path.exists():
        warnings.append("method_policy_tsv_has_no_rows")
    return {
        "status": "fail" if failures else "ok",
        "n_fail": len(failures),
        "n_warning": len(warnings),
        "failures": failures,
        "warnings": warnings,
    }


def _alignment_method_status(manifest: dict) -> Dict[str, dict]:
    statuses = manifest.get("method_status")
    if isinstance(statuses, dict) and statuses:
        return {str(method): dict(status) for method, status in statuses.items() if method}
    n_families = int(manifest.get("n_families") or len(manifest.get("family_ids") or []))
    methods = set(manifest.get("methods_requested") or []) | set(manifest.get("methods") or [])
    skipped = manifest.get("methods_skipped") if isinstance(manifest.get("methods_skipped"), dict) else {}
    result = {}
    for method in methods:
        method = str(method)
        failed = 0
        if method in skipped:
            reason = str(skipped[method])
            if reason.startswith("family_method_failures:"):
                failed = _safe_int(reason.split(":", 1)[1], 0)
            elif reason == "executable_unavailable":
                failed = n_families
        attempted = n_families if method in (manifest.get("methods") or []) or method in skipped else 0
        successful = max(0, attempted - failed)
        result[method] = {
            "method": method,
            "attempted_families": attempted,
            "successful_families": successful,
            "failed_families": failed,
            "failure_fraction": 0.0 if attempted <= 0 else failed / attempted,
            "failure_reasons": [str(skipped[method])] if method in skipped else [],
        }
    return result


def _site_map_summary(manifest: dict) -> Dict[str, dict]:
    rows = manifest.get("method_summary") or []
    result = {}
    if isinstance(rows, list):
        for row in rows:
            if isinstance(row, dict) and row.get("method"):
                result[str(row["method"])] = row
    return result


def _policy_row(method: str, align_status: dict, site_status: dict, config: MethodPolicyConfig) -> dict:
    attempted = _safe_int(align_status.get("attempted_families"), 0)
    successful = _safe_int(align_status.get("successful_families"), 0)
    failed = _safe_int(align_status.get("failed_families"), max(0, attempted - successful))
    failure_fraction = _safe_float(align_status.get("failure_fraction"), 0.0 if attempted <= 0 else failed / attempted)
    unique_fraction = _safe_float(site_status.get("unique_fraction"), 0.0)
    conflict_fraction = _safe_float(site_status.get("conflict_fraction"), 0.0)
    frame_error_fraction = _safe_float(site_status.get("frame_error_fraction"), 0.0)
    recommendation = "usable"
    reasons: List[str] = []
    if attempted == 0 and successful == 0:
        recommendation = "quarantine"
        reasons.append("method_not_attempted_or_unavailable")
    if failure_fraction > config.max_method_failure_fraction:
        recommendation = "quarantine"
        reasons.append(f"failure_fraction>{config.max_method_failure_fraction:g}")
    elif failed > 0:
        reasons.append("low_failure_fraction_accepted")
    if frame_error_fraction > config.max_frame_error_fraction:
        recommendation = "quarantine"
        reasons.append(f"frame_error_fraction>{config.max_frame_error_fraction:g}")
    if conflict_fraction > config.max_conflict_fraction:
        recommendation = "quarantine"
        reasons.append(f"conflict_fraction>{config.max_conflict_fraction:g}")
    elif conflict_fraction > 0:
        if recommendation != "quarantine":
            recommendation = "caution"
        reasons.append("nonzero_conflict_fraction")
    if not reasons:
        reasons.append("passes_policy_thresholds")
    return {
        "method": method,
        "attempted_families": attempted,
        "successful_families": successful,
        "failed_families": failed,
        "failure_fraction": _format_fraction(failure_fraction),
        "site_map_unique_fraction": _format_fraction(unique_fraction),
        "site_map_conflict_fraction": _format_fraction(conflict_fraction),
        "site_map_frame_error_fraction": _format_fraction(frame_error_fraction),
        "recommendation": recommendation,
        "reason": ";".join(reasons),
    }


def _render_markdown(payload: dict) -> str:
    lines = [
        "# BABAPPA aligner method policy",
        "",
        "## Inputs",
        "",
        f"- Alignment directory: `{payload.get('align_dir')}`",
        f"- Site-map directory: `{payload.get('site_map_dir')}`",
        "",
        "## Usable methods",
        "",
    ]
    usable = payload.get("usable_methods") or []
    lines.extend([f"- {method}" for method in usable] if usable else ["- none"])
    lines.extend(["", "## Quarantined methods", ""])
    quarantined = payload.get("quarantined_methods") or []
    lines.extend([f"- {method}" for method in quarantined] if quarantined else ["- none"])
    lines.extend(["", "## Per-method details", ""])
    for row in payload.get("methods") or []:
        lines.append(
            "- {method}: recommendation={recommendation}, failures={failed_families}/{attempted_families}, "
            "frame_error={site_map_frame_error_fraction}, conflict={site_map_conflict_fraction}, reason={reason}".format(**row)
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "Methods marked usable or caution can be tensorized by generated workflows; methods marked quarantine must be excluded from mapped oracle-label training.",
        "",
    ])
    return "\n".join(lines)


def _load_json(path: Path, failures: List[str]) -> Optional[dict]:
    if not path.exists():
        failures.append(f"missing_file:{path}")
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        failures.append(f"could_not_parse_json:{path}:{exc}")
        return None
    if not isinstance(payload, dict):
        failures.append(f"json_not_object:{path}")
        return None
    return payload


def _load_tsv(path: Path, failures: List[str]) -> List[dict]:
    if not path.exists():
        failures.append(f"missing_file:{path}")
        return []
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle, delimiter="\t"))
    except OSError as exc:
        failures.append(f"could_not_read_tsv:{path}:{exc}")
        return []


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON file is not an object: {path}")
    return payload


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _safe_float(value: object, default: float) -> float:
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return default


def _safe_int(value: object, default: int) -> int:
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return default


def _format_fraction(value: float) -> str:
    return f"{float(value):.12g}"
