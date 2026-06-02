"""Simulation-matched empirical calibration and scoring planners."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from babappa import __version__

FORBIDDEN_EMPIRICAL_INPUT_COLUMNS = [
    "branch_site_truth.tsv",
    "selected_sites",
    "truth.json",
    "branch_truth.json",
    "oracle",
    "y_branch_site",
    "y_site",
    "gene_label",
]


@dataclass(frozen=True)
class SimulationMatchedCalibrationPlanConfig:
    """Configuration for a simulation-matched empirical calibration plan."""

    empirical_validation_dir: str
    deployable_model_package: str
    outdir: str


@dataclass(frozen=True)
class EmpiricalScoringPlanConfig:
    """Configuration for an empirical scoring plan scaffold."""

    cds_fasta: str
    tree: str
    foreground: str
    deployable_model_package: str
    outdir: str
    methods: Sequence[str] | str
    device: str = "auto"
    allow_diagnostic_out_of_domain: bool = False


def plan_simulation_matched_calibration(config: SimulationMatchedCalibrationPlanConfig) -> Dict[str, Any]:
    """Create a planner for empirical-QC-matched null simulation calibration."""

    outdir = Path(config.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    package_dir = Path(config.deployable_model_package)
    empirical_dir = Path(config.empirical_validation_dir)
    manifest = _read_optional_json(package_dir / "model_manifest.json")
    empirical_features = _collect_empirical_qc_features(empirical_dir)
    proposed = _propose_simulation_parameters(empirical_features)
    missing = [] if empirical_dir.exists() else [f"missing_empirical_validation_dir:{empirical_dir}"]
    payload = {
        "version": __version__,
        "status": "planned_with_missing_empirical_qc" if missing else "planned",
        "empirical_validation_dir": str(empirical_dir),
        "deployable_model_package": str(package_dir),
        "source_package": manifest.get("package_name", "unknown"),
        "empirical_qc_features": empirical_features,
        "proposed_simulation_parameters": proposed,
        "missing_inputs": missing,
        "heavy_jobs_executed": False,
        "claim_boundary": "Planner only; not final empirical branch-site inference.",
        "estimated_runtime": _estimate_runtime(proposed),
        "estimated_disk": _estimate_disk(proposed),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(outdir / "simulation_matched_calibration_plan.json", payload)
    (outdir / "simulation_matched_calibration_plan.md").write_text(
        _render_simulation_matched_plan_md(payload),
        encoding="utf-8",
    )
    (outdir / "proposed_null_simulation_commands.sh").write_text(
        _render_null_commands(payload),
        encoding="utf-8",
    )
    (outdir / "proposed_null_simulation_commands.sh").chmod(0o755)
    (outdir / "proposed_alt_simulation_commands.sh").write_text(
        _render_alt_commands(payload),
        encoding="utf-8",
    )
    (outdir / "proposed_alt_simulation_commands.sh").chmod(0o755)
    _write_json(outdir / "expected_outputs.json", _expected_calibration_outputs(outdir))
    return {
        "status": payload["status"],
        "outdir": str(outdir),
        "json": str(outdir / "simulation_matched_calibration_plan.json"),
        "markdown": str(outdir / "simulation_matched_calibration_plan.md"),
        "commands": str(outdir / "proposed_null_simulation_commands.sh"),
        "alt_commands": str(outdir / "proposed_alt_simulation_commands.sh"),
        "heavy_jobs_executed": False,
        "missing_inputs": missing,
    }


def plan_empirical_scoring(config: EmpiricalScoringPlanConfig) -> Dict[str, Any]:
    """Create a non-executing empirical scoring scaffold."""

    outdir = Path(config.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    methods = _parse_csv(config.methods)
    forbidden = _check_forbidden_empirical_inputs([Path(config.cds_fasta), Path(config.tree)])
    if forbidden:
        payload = {
            "version": __version__,
            "status": "blocked",
            "forbidden_inputs": forbidden,
            "claim_boundary": "Truth-derived empirical inputs are blocked.",
        }
        _write_json(outdir / "empirical_scoring_plan.json", payload)
        (outdir / "empirical_scoring_plan.md").write_text(_render_blocked_scoring_md(payload), encoding="utf-8")
        raise ValueError("truth-derived empirical input blocked: " + ",".join(forbidden))

    manifest = _read_optional_json(Path(config.deployable_model_package) / "model_manifest.json")
    payload = {
        "version": __version__,
        "status": "planned",
        "cds_fasta": config.cds_fasta,
        "tree": config.tree,
        "foreground": config.foreground,
        "deployable_model_package": config.deployable_model_package,
        "outdir_name": str(outdir),
        "package_name": manifest.get("package_name", "unknown"),
        "methods": methods,
        "device": config.device,
        "current_stopping_point": "full_empirical_bridge_script_generated_research_alpha_empirical_feature_extraction",
        "allow_diagnostic_out_of_domain": config.allow_diagnostic_out_of_domain,
        "truth_derived_inputs_blocked": True,
        "forbidden_empirical_input_columns": FORBIDDEN_EMPIRICAL_INPUT_COLUMNS,
        "heavy_jobs_executed": False,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(outdir / "empirical_scoring_plan.json", payload)
    (outdir / "run_empirical_scoring.sh").write_text(_render_run_empirical_scoring(payload), encoding="utf-8")
    (outdir / "validate_empirical_scoring.sh").write_text(_render_validate_empirical_scoring(payload), encoding="utf-8")
    (outdir / "summarize_empirical_scoring.sh").write_text(_render_summarize_empirical_scoring(payload), encoding="utf-8")
    for filename in ["run_empirical_scoring.sh", "validate_empirical_scoring.sh", "summarize_empirical_scoring.sh"]:
        (outdir / filename).chmod(0o755)
    (outdir / "empirical_scoring_plan.md").write_text(_render_empirical_scoring_md(payload), encoding="utf-8")
    return {
        "status": "planned",
        "outdir": str(outdir),
        "run_script": str(outdir / "run_empirical_scoring.sh"),
        "validate_script": str(outdir / "validate_empirical_scoring.sh"),
        "summarize_script": str(outdir / "summarize_empirical_scoring.sh"),
        "markdown": str(outdir / "empirical_scoring_plan.md"),
        "current_stopping_point": payload["current_stopping_point"],
    }


def _collect_empirical_qc_features(path: Path) -> Dict[str, Any]:
    features: Dict[str, Any] = {}
    if not path.exists():
        return features
    for candidate in sorted(path.glob("*.json")):
        data = _read_optional_json(candidate)
        _merge_interesting_fields(features, data)
    for candidate in sorted(path.glob("*.tsv")):
        rows = _read_tsv_rows(candidate)
        if rows:
            _merge_interesting_fields(features, rows[0])
    return features


def _merge_interesting_fields(features: Dict[str, Any], data: Dict[str, Any]) -> None:
    interesting = {
        "n_taxa",
        "n_codons",
        "gc_content",
        "mean_pairwise_p_distance",
        "saturation_proxy",
        "gap_fraction",
        "foreground_branch",
        "foreground_taxon",
        "tree_shape_summary",
        "alignment_disagreement",
        "ambiguous_base_fraction",
    }
    for key, value in _walk_dict(data):
        if key in interesting and key not in features:
            features[key] = value


def _walk_dict(data: Dict[str, Any]) -> Iterable[tuple[str, Any]]:
    for key, value in data.items():
        if isinstance(value, dict):
            yield from _walk_dict(value)
        else:
            yield str(key), value


def _propose_simulation_parameters(features: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "n_taxa": _coerce_int(features.get("n_taxa"), 8),
        "n_codons": _coerce_int(features.get("n_codons"), 180),
        "gc_content": _coerce_float(features.get("gc_content"), None),
        "mean_pairwise_p_distance": _coerce_float(features.get("mean_pairwise_p_distance"), None),
        "saturation_proxy": features.get("saturation_proxy", "estimate_from_pairwise_distance"),
        "gap_fraction": _coerce_float(features.get("gap_fraction"), None),
        "ambiguous_base_fraction": _coerce_float(features.get("ambiguous_base_fraction"), None),
        "foreground": features.get("foreground_taxon") or features.get("foreground_branch") or "user_supplied_foreground",
        "tree_shape_summary": features.get("tree_shape_summary", "pending_empirical_qc"),
        "alignment_disagreement": features.get("alignment_disagreement", "pending_alignment_ensemble_qc"),
        "recommended_tier": _recommended_tier(features),
        "null_replicates_initial": 1000,
        "alt_replicates_initial": 250,
        "notes": "Initial null calibration scaffold; manual execution only.",
    }


def _recommended_tier(features: Dict[str, Any]) -> str:
    p_distance = _coerce_float(features.get("mean_pairwise_p_distance"), None)
    if p_distance is None:
        return "match_after_qc"
    if p_distance < 0.05:
        return "low"
    if p_distance < 0.12:
        return "moderate"
    if p_distance < 0.25:
        return "high"
    return "extreme"


def _render_simulation_matched_plan_md(payload: Dict[str, Any]) -> str:
    lines = [
        "# Simulation-matched empirical calibration plan",
        "",
        f"- status: `{payload['status']}`",
        f"- deployable model package: `{payload['deployable_model_package']}`",
        "- heavy jobs executed: `False`",
        "",
        "## Scientific boundary",
        "",
        "This is a planning scaffold only. It does not authorize final empirical branch-site inference.",
        "",
        "## Proposed simulation parameters",
        "",
    ]
    for key, value in payload["proposed_simulation_parameters"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend([
        "",
        "## Estimated resources",
        "",
        f"- runtime: `{payload.get('estimated_runtime')}`",
        f"- disk: `{payload.get('estimated_disk')}`",
    ])
    if payload["missing_inputs"]:
        lines.extend(["", "## Missing inputs", ""])
        lines.extend(f"- {item}" for item in payload["missing_inputs"])
    lines.append("")
    return "\n".join(lines)


def _render_null_commands(payload: Dict[str, Any]) -> str:
    params = payload["proposed_simulation_parameters"]
    return "\n".join([
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "echo 'MANUAL EXECUTION SCRIPT - REVIEW BEFORE RUNNING'",
        "echo 'This scaffold proposes simulation-matched null calibration; it does not run automatically.'",
        "",
        "# Proposed heavy command placeholder, intentionally commented:",
        "# babappa simulate \\",
        f"#   --outdir sim_empirical_matched_null_{params['recommended_tier']} \\",
        "#   --n-families 1000 \\",
        f"#   --n-taxa {params['n_taxa']} \\",
        f"#   --n-codons {params['n_codons']} \\",
        f"#   --saturation-tier {params['recommended_tier']}",
        "",
    ])


def _render_alt_commands(payload: Dict[str, Any]) -> str:
    params = payload["proposed_simulation_parameters"]
    return "\n".join([
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "echo 'MANUAL EXECUTION SCRIPT - REVIEW BEFORE RUNNING'",
        "echo 'Optional matched alternative simulations for calibration stress testing; not run automatically.'",
        "",
        "# Proposed heavy command placeholder, intentionally commented:",
        "# babappa simulate \\",
        f"#   --outdir sim_empirical_matched_alt_{params['recommended_tier']} \\",
        "#   --n-families 250 \\",
        f"#   --n-taxa {params['n_taxa']} \\",
        f"#   --n-codons {params['n_codons']} \\",
        f"#   --saturation-tier {params['recommended_tier']} \\",
        "#   --positive-rate 0.5",
        "",
    ])


def _expected_calibration_outputs(outdir: Path) -> Dict[str, Any]:
    return {
        "expected_outputs": [
            str(outdir / "simulation_matched_calibration_plan.json"),
            str(outdir / "simulation_matched_calibration_plan.md"),
            str(outdir / "proposed_null_simulation_commands.sh"),
            str(outdir / "proposed_alt_simulation_commands.sh"),
        ],
        "future_user_run_outputs": [
            "empirical_matched_null_simulations/",
            "empirical_matched_calibration/",
            "empirical_applicability_report.json",
        ],
    }


def _estimate_runtime(params: Dict[str, Any]) -> str:
    n_taxa = _coerce_int(params.get("n_taxa"), 8)
    n_codons = _coerce_int(params.get("n_codons"), 180)
    reps = _coerce_int(params.get("null_replicates_initial"), 1000)
    score = n_taxa * n_codons * reps
    if score < 2_000_000:
        return "minutes_to_low_hours"
    if score < 20_000_000:
        return "hours"
    return "multi_hour_to_day_scale"


def _estimate_disk(params: Dict[str, Any]) -> str:
    n_taxa = _coerce_int(params.get("n_taxa"), 8)
    n_codons = _coerce_int(params.get("n_codons"), 180)
    reps = _coerce_int(params.get("null_replicates_initial"), 1000)
    approx_mb = max(10, int(n_taxa * n_codons * reps / 10_000))
    return f"roughly_{approx_mb}_MB_plus_alignment_outputs"


def _check_forbidden_empirical_inputs(paths: List[Path]) -> List[str]:
    found: List[str] = []
    lowered_terms = [term.lower() for term in FORBIDDEN_EMPIRICAL_INPUT_COLUMNS]
    for path in paths:
        text_path = str(path).lower()
        if any(term in text_path for term in lowered_terms):
            found.append(f"path:{path}")
            continue
        if path.exists() and path.is_file():
            try:
                head = path.read_text(encoding="utf-8", errors="ignore")[:4096].lower()
            except OSError:
                continue
            for term in lowered_terms:
                if term in {"oracle", "selected_sites"}:
                    continue
                if term in head:
                    found.append(f"content:{path}:{term}")
                    break
    return found


def _render_run_empirical_scoring(payload: Dict[str, Any]) -> str:
    outdir = str(Path(payload["outdir_name"]))
    input_dir = f"{outdir}/empirical_input"
    alignment_dir = f"{outdir}/empirical_alignment"
    feature_dir = f"{outdir}/empirical_features"
    audit_dir = f"{outdir}/empirical_feature_audit"
    applicability_dir = f"{outdir}/empirical_applicability"
    scoring_dir = f"{outdir}/empirical_scores"
    calibration_dir = f"{outdir}/simulation_matched_calibration_plan"
    report_dir = f"{outdir}/empirical_report"
    guard = []
    if not payload.get("allow_diagnostic_out_of_domain"):
        guard = [
            "python - <<'PY'",
            "import json",
            f"p='{applicability_dir}/empirical_applicability.json'",
            "s=json.load(open(p)).get('applicability_status')",
            "raise SystemExit('Applicability is out_of_domain; rerun with --allow-diagnostic-out-of-domain for diagnostic-only scoring.') if s == 'out_of_domain' else 0",
            "PY",
        ]
    return "\n".join([
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "echo 'MANUAL EXECUTION SCRIPT - REVIEW BEFORE RUNNING'",
        f"babappa validate-deployable-model-package --package-dir {payload['deployable_model_package']}",
        f"babappa validate-empirical-input --cds-fasta {payload['cds_fasta']} --tree {payload['tree']} --foreground {payload['foreground']} --outdir {input_dir}",
        f"babappa run-empirical-alignment-ensemble --cds-fasta {payload['cds_fasta']} --tree {payload['tree']} --foreground {payload['foreground']} --outdir {alignment_dir} --methods {','.join(payload['methods'])} --require-babappalign false --threads 4",
        f"babappa extract-empirical-branch-site-features --empirical-validation-dir {input_dir} --alignment-dir {alignment_dir} --deployable-model-package {payload['deployable_model_package']} --outdir {feature_dir} --foreground {payload['foreground']}",
        f"babappa audit-empirical-features --features {feature_dir}/empirical_branch_site_features.tsv --deployable-model-package {payload['deployable_model_package']} --outdir {audit_dir}",
        f"babappa empirical-applicability --empirical-validation-dir {input_dir} --empirical-feature-dir {feature_dir} --deployable-model-package {payload['deployable_model_package']} --outdir {applicability_dir}",
        *guard,
        f"babappa score-empirical-branch-sites --features {feature_dir}/empirical_branch_site_features.tsv --deployable-model-package {payload['deployable_model_package']} --applicability-dir {applicability_dir} --outdir {scoring_dir} --device {payload['device']}",
        f"babappa plan-simulation-matched-calibration --empirical-validation-dir {input_dir} --deployable-model-package {payload['deployable_model_package']} --outdir {calibration_dir}",
        f"babappa make-empirical-branch-site-report --outdir {report_dir} --empirical-validation-dir {input_dir} --alignment-dir {alignment_dir} --feature-dir {feature_dir} --feature-audit-dir {audit_dir} --applicability-dir {applicability_dir} --scoring-dir {scoring_dir} --simulation-matched-calibration-plan {calibration_dir} --deployable-model-package {payload['deployable_model_package']}",
        "",
    ])


def _render_validate_empirical_scoring(payload: Dict[str, Any]) -> str:
    return "\n".join([
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"test -f {payload['cds_fasta']}",
        f"test -f {payload['tree']}",
        f"babappa validate-deployable-model-package --package-dir {payload['deployable_model_package']}",
        "echo 'Empirical scoring plan inputs are present; truth-derived inputs remain blocked.'",
        "",
    ])


def _render_summarize_empirical_scoring(payload: Dict[str, Any]) -> str:
    return "\n".join([
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "echo 'Empirical scoring summary scaffold only.'",
        f"echo 'Current stopping point: {payload['current_stopping_point']}'",
        "",
    ])


def _render_empirical_scoring_md(payload: Dict[str, Any]) -> str:
    return "\n".join([
        "# Empirical scoring plan",
        "",
        f"- status: `{payload['status']}`",
        f"- CDS FASTA: `{payload['cds_fasta']}`",
        f"- tree: `{payload['tree']}`",
        f"- foreground: `{payload['foreground']}`",
        f"- methods: `{','.join(payload['methods'])}`",
        f"- device: `{payload['device']}`",
        f"- current stopping point: `{payload['current_stopping_point']}`",
        "",
        "Truth-derived empirical inputs are blocked. This scaffold does not run empirical prediction yet.",
        "",
    ])


def _render_blocked_scoring_md(payload: Dict[str, Any]) -> str:
    lines = ["# Empirical scoring plan blocked", "", "Truth-derived empirical inputs were detected.", ""]
    lines.extend(f"- {item}" for item in payload["forbidden_inputs"])
    lines.append("")
    return "\n".join(lines)


def _read_optional_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _read_tsv_rows(path: Path) -> List[Dict[str, str]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    if not lines:
        return []
    header = lines[0].split("\t")
    rows: List[Dict[str, str]] = []
    for line in lines[1:2]:
        values = line.split("\t")
        rows.append({key: values[index] if index < len(values) else "" for index, key in enumerate(header)})
    return rows


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _parse_csv(value: Sequence[str] | str) -> List[str]:
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    parsed: List[str] = []
    for item in value:
        parsed.extend(part.strip() for part in str(item).split(",") if part.strip())
    return parsed


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _coerce_float(value: Any, default: float | None) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
