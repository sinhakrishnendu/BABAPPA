"""Fast preflight validation for generated Apple Silicon/MPS branch-truth plans."""

from __future__ import annotations

import csv
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from babappa import __version__
from babappa.align.backends import BABAPPALIGN_MODEL_INSTALL_COMMAND, babappalign_model_status
from babappa.align.external import smoke_aligner
from babappa.simulate.audit import read_fasta

MPS_PREFLIGHT_VERSION = __version__
BARE_STAGE_TOKENS = {
    "t_sim",
    "t_align",
    "t_site_map",
    "t_policy",
    "t_tensor",
    "t_dataset",
    "t_branch",
}
REQUIRED_COMMANDS = [
    "babappa",
    "python",
    "conda",
    "mafft",
    "muscle",
    "babappalign",
    "curl",
    "awk",
    "sed",
    "grep",
    "find",
]


@dataclass(frozen=True)
class MPSPlanPreflightConfig:
    """Configuration for fast Mac MPS plan preflight."""

    plan_dir: str
    scale: str
    require_babappalign: bool = True
    require_mps: bool = True
    conda_env: str = "molevo"
    allow_partial_resume: bool = False
    run_align_external_smoke: bool = False

    def __post_init__(self) -> None:
        if self.scale not in {"10k", "100k"}:
            raise ValueError("scale must be 10k or 100k")


@dataclass(frozen=True)
class MPSPlanScriptValidationConfig:
    """Configuration for lightweight script-only Mac MPS plan validation."""

    plan_dir: str
    scale: str | None = None


def preflight_explicit_branch_truth_mps_plan(config: MPSPlanPreflightConfig) -> dict:
    """Run fast preflight checks and write JSON/TSV/Markdown reports."""
    plan_dir = Path(config.plan_dir)
    rows: list[dict] = []
    scripts = _plan_scripts(plan_dir, config.scale)
    run_text = scripts["run"].read_text(encoding="utf-8") if scripts["run"].exists() else ""

    _check_plan_files(rows, scripts)
    _check_shell_syntax(rows, scripts)
    _check_suspicious_tokens(rows, scripts["run"])
    _check_required_commands(rows, config)
    _check_babappa_import(rows)
    _check_mps(rows, config.require_mps)
    _check_run_script_env(rows, run_text)
    _check_babappalign_model(rows, config.require_babappalign)
    _check_babappalign_smoke(rows, config.require_babappalign, plan_dir)
    _check_mafft_smoke(rows, required=True)
    _check_muscle_smoke(rows, required=True)
    _check_optional_align_external_smoke(rows, config)
    _check_mac_script_hardening(rows, run_text)
    _check_lock_hardening(rows, run_text)
    _check_stage_markers(rows, config.allow_partial_resume)
    _check_output_collisions(rows, scripts["run"])
    _check_disk_and_memory(rows, config.scale)
    _check_internal_consistency(rows, scripts, config.scale)

    return _write_reports(plan_dir, rows, report_stem="preflight_report")


def validate_mps_plan_script(config: MPSPlanScriptValidationConfig) -> dict:
    """Validate generated script syntax and internal consistency without environment smokes."""
    plan_dir = Path(config.plan_dir)
    scale = config.scale or _infer_scale(plan_dir)
    rows: list[dict] = []
    scripts = _plan_scripts(plan_dir, scale)
    run_text = scripts["run"].read_text(encoding="utf-8") if scripts["run"].exists() else ""
    _check_plan_files(rows, scripts)
    _check_shell_syntax(rows, scripts)
    _check_suspicious_tokens(rows, scripts["run"])
    _check_run_script_env(rows, run_text)
    _check_mac_script_hardening(rows, run_text)
    _check_lock_hardening(rows, run_text)
    _check_internal_consistency(rows, scripts, scale)
    return _write_reports(plan_dir, rows, report_stem="mps_plan_script_validation")


def _plan_scripts(plan_dir: Path, scale: str) -> dict[str, Path]:
    stem = f"explicit_branch_truth_{scale}_mps"
    return {
        "run": plan_dir / f"run_{stem}.sh",
        "monitor": plan_dir / f"monitor_{stem}.sh",
        "validate": plan_dir / f"validate_{stem}.sh",
        "summarize": plan_dir / f"summarize_{stem}.sh",
    }


def _infer_scale(plan_dir: Path) -> str:
    name = plan_dir.name.lower()
    if "100k" in name:
        return "100k"
    return "10k"


def _add_check(
    rows: list[dict],
    name: str,
    status: str,
    required: bool,
    message: str,
    details: dict | None = None,
) -> None:
    rows.append(
        {
            "name": name,
            "status": status,
            "required": required,
            "message": message,
            "details": details or {},
        }
    )


def _check_plan_files(rows: list[dict], scripts: dict[str, Path]) -> None:
    for label, path in scripts.items():
        _add_check(
            rows,
            f"script_exists:{label}",
            "pass" if path.exists() else "fail",
            True,
            str(path),
        )


def _check_shell_syntax(rows: list[dict], scripts: dict[str, Path]) -> None:
    for label, path in scripts.items():
        if not path.exists():
            continue
        proc = subprocess.run(
            ["bash", "-n", str(path)],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        _add_check(
            rows,
            f"bash_n:{label}",
            "pass" if proc.returncode == 0 else "fail",
            True,
            "bash -n passed" if proc.returncode == 0 else (proc.stderr or proc.stdout).strip(),
            {"path": str(path), "return_code": proc.returncode},
        )


def _check_suspicious_tokens(rows: list[dict], run_script: Path) -> None:
    if not run_script.exists():
        return
    findings = _find_suspicious_bare_tokens(run_script.read_text(encoding="utf-8"))
    _add_check(
        rows,
        "suspicious_bare_tokens",
        "fail" if findings else "pass",
        True,
        "bare generated t_* tokens found" if findings else "no bare generated t_* tokens",
        {"findings": findings},
    )


def _find_suspicious_bare_tokens(text: str) -> list[dict]:
    findings: list[dict] = []
    command_re = re.compile(r"(^|[;&|]\s*)(t_[A-Za-z0-9_]+)\b(?!\s*=)")
    for line_no, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        for match in command_re.finditer(line):
            token = match.group(2)
            findings.append({"line": line_no, "token": token, "text": stripped})
        for token in BARE_STAGE_TOKENS:
            if re.search(rf"(^|[ ;]){re.escape(token)}($|[ ;])", line) and not re.search(
                rf"(^|\s){re.escape(token)}=", line
            ):
                findings.append({"line": line_no, "token": token, "text": stripped})
    unique: list[dict] = []
    seen = set()
    for finding in findings:
        key = (finding["line"], finding["token"], finding["text"])
        if key not in seen:
            unique.append(finding)
            seen.add(key)
    return unique


def _check_required_commands(rows: list[dict], config: MPSPlanPreflightConfig) -> None:
    for command in REQUIRED_COMMANDS:
        required = command != "babappalign" or config.require_babappalign
        path = shutil.which(command)
        if command == "conda" and path is None:
            path = os.environ.get("CONDA_EXE")
        _add_check(
            rows,
            f"command:{command}",
            "pass" if path else ("fail" if required else "warn"),
            required,
            path or f"{command} not found on PATH",
        )


def _check_babappa_import(rows: list[dict]) -> None:
    code = "import babappa; print(babappa.__version__)"
    proc = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    version = proc.stdout.strip()
    ok = proc.returncode == 0 and _version_at_least_042_alpha(version)
    _add_check(
        rows,
        "babappa_import_version",
        "pass" if ok else "fail",
        True,
        version if ok else (proc.stderr.strip() or f"version too old: {version}"),
        {"version": version, "return_code": proc.returncode},
    )


def _version_at_least_042_alpha(version: str) -> bool:
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)", version)
    if not match:
        return False
    return tuple(int(part) for part in match.groups()) >= (0, 4, 2)


def _check_mps(rows: list[dict], required: bool) -> None:
    code = """
import torch
assert torch.backends.mps.is_built()
assert torch.backends.mps.is_available()
x = torch.ones((2, 2), device='mps')
y = (x + 1).cpu()
print(float(y.sum()))
"""
    proc = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    ok = proc.returncode == 0
    _add_check(
        rows,
        "torch_mps_tiny_tensor",
        "pass" if ok else ("fail" if required else "warn"),
        required,
        "MPS tiny tensor operation passed" if ok else (proc.stderr.strip() or "MPS unavailable"),
        {"stdout": proc.stdout.strip(), "return_code": proc.returncode},
    )


def _check_run_script_env(rows: list[dict], run_text: str) -> None:
    _add_check(
        rows,
        "mps_fallback_export",
        "pass" if "export PYTORCH_ENABLE_MPS_FALLBACK=1" in run_text else "fail",
        True,
        "run script exports PYTORCH_ENABLE_MPS_FALLBACK=1",
    )


def _check_babappalign_model(rows: list[dict], required: bool) -> None:
    status = babappalign_model_status()
    present = bool(status["model_present"])
    _add_check(
        rows,
        "babappalign_model_cache",
        "pass" if present else ("fail" if required else "warn"),
        required,
        "BABAPPAScore model present" if present else "babappalign_model_missing",
        {
            **status,
            "action": BABAPPALIGN_MODEL_INSTALL_COMMAND,
        },
    )


def _check_babappalign_smoke(rows: list[dict], required: bool, plan_dir: Path) -> None:
    smoke_dir = plan_dir / ".preflight_babappalign_smoke"
    try:
        summary = smoke_aligner("babappalign", smoke_dir, device="cpu", timeout_seconds=60)
        ok = summary.get("status") == "ok" and _validate_fasta_frame(
            smoke_dir / "tiny.babappalign.codon.fasta", {"taxon_a", "taxon_b"}
        )[0]
        message = "BABAPPAlign tiny smoke passed" if ok else str(summary.get("reason"))
        details = dict(summary)
        if not ok:
            details["frame_validation"] = _validate_fasta_frame(
                smoke_dir / "tiny.babappalign.codon.fasta", {"taxon_a", "taxon_b"}
            )[1]
    except Exception as exc:  # pragma: no cover - defensive reporting
        ok = False
        message = str(exc)
        details = {"exception": type(exc).__name__}
    _add_check(
        rows,
        "babappalign_tiny_smoke",
        "pass" if ok else ("fail" if required else "warn"),
        required,
        message,
        details,
    )


def _check_mafft_smoke(rows: list[dict], required: bool) -> None:
    executable = shutil.which("mafft")
    if executable is None:
        _add_check(rows, "mafft_tiny_smoke", "fail" if required else "warn", required, "mafft not found")
        return
    with tempfile.TemporaryDirectory(prefix="babappa_mafft_smoke_") as tmp:
        input_path = Path(tmp) / "tiny.codon.fasta"
        input_path.write_text(">taxon_a\nATGAAACCC\n>taxon_b\nATGAAAGGG\n", encoding="utf-8")
        proc = subprocess.run(
            [executable, "--auto", str(input_path)],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=60,
        )
        output_path = Path(tmp) / "mafft.out.fasta"
        output_path.write_text(proc.stdout, encoding="utf-8")
        ok, reason = _validate_equal_length_fasta(output_path, {"taxon_a", "taxon_b"})
    _add_check(
        rows,
        "mafft_tiny_smoke",
        "pass" if proc.returncode == 0 and ok else "fail",
        required,
        "MAFFT tiny smoke passed" if proc.returncode == 0 and ok else reason or proc.stderr.strip(),
        {"return_code": proc.returncode},
    )


def _check_muscle_smoke(rows: list[dict], required: bool) -> None:
    executable = shutil.which("muscle") or shutil.which("muscle5")
    if executable is None:
        _add_check(rows, "muscle_tiny_smoke", "fail" if required else "warn", required, "muscle not found")
        return
    with tempfile.TemporaryDirectory(prefix="babappa_muscle_smoke_") as tmp:
        input_path = Path(tmp) / "tiny.codon.fasta"
        output_path = Path(tmp) / "muscle.out.fasta"
        input_path.write_text(">taxon_a\nATGAAACCC\n>taxon_b\nATGAAAGGG\n", encoding="utf-8")
        candidates = [
            [executable, "-align", str(input_path), "-output", str(output_path)],
            [executable, "-in", str(input_path), "-out", str(output_path)],
        ]
        proc = None
        ok = False
        reason = ""
        for command in candidates:
            if output_path.exists():
                output_path.unlink()
            proc = subprocess.run(
                command,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=60,
            )
            ok, reason = _validate_equal_length_fasta(output_path, {"taxon_a", "taxon_b"})
            if proc.returncode == 0 and ok:
                break
    return_code = -1 if proc is None else proc.returncode
    stderr = "" if proc is None else proc.stderr.strip()
    _add_check(
        rows,
        "muscle_tiny_smoke",
        "pass" if return_code == 0 and ok else "fail",
        required,
        "MUSCLE tiny smoke passed" if return_code == 0 and ok else reason or stderr,
        {"return_code": return_code},
    )


def _check_optional_align_external_smoke(rows: list[dict], config: MPSPlanPreflightConfig) -> None:
    if not config.run_align_external_smoke:
        _add_check(
            rows,
            "babappa_align_external_tiny_smoke",
            "skip",
            False,
            "optional tiny align-external smoke not requested",
        )
        return
    _add_check(
        rows,
        "babappa_align_external_tiny_smoke",
        "skip",
        False,
        "optional smoke runner is intentionally not invoked by default preflight",
    )


def _check_mac_script_hardening(rows: list[dict], run_text: str) -> None:
    checks = [
        ("no_nvidia_smi", "nvidia-smi" not in run_text, "run script does not call nvidia-smi"),
        ("no_cuda_visible_devices", "CUDA_VISIBLE_DEVICES" not in run_text, "run script does not set CUDA_VISIBLE_DEVICES"),
        ("no_linux_home_assumption", "/home/rajamosai" not in run_text, "run script does not assume /home/rajamosai"),
        ("mac_conda_sources", "$HOME/miniforge3/etc/profile.d/conda.sh" in run_text, "run script has macOS conda sources"),
        ("conda_nounset_guard", "set +u" in run_text and "conda activate" in run_text, "conda activate is guarded from nounset"),
    ]
    for name, ok, message in checks:
        _add_check(rows, name, "pass" if ok else "fail", True, message)


def _check_lock_hardening(rows: list[dict], run_text: str) -> None:
    checks = [
        ("mkdir_lock", "mkdir \"$lock_dir\"" in run_text, "run script uses mkdir-based lock"),
        ("lock_cleanup", "trap cleanup_lock EXIT" in run_text, "run script cleans lock on exit"),
        ("no_required_flock", "command -v flock" not in run_text, "run script does not require flock"),
        ("stale_lock_help", "rm -rf \"$lock_dir\"" in run_text, "run script explains stale lock removal"),
    ]
    for name, ok, message in checks:
        _add_check(rows, name, "pass" if ok else "fail", True, message)


def _check_stage_markers(rows: list[dict], allow_partial_resume: bool) -> None:
    partials = sorted(Path.cwd().glob(".stage_complete_*.partial"))
    completes = sorted(Path.cwd().glob(".stage_complete_*"))
    complete_count = len([path for path in completes if not path.name.endswith(".partial")])
    _add_check(
        rows,
        "stage_complete_markers",
        "pass",
        False,
        f"{complete_count} completed stage marker(s) detected",
        {"markers": [path.name for path in completes if not path.name.endswith(".partial")]},
    )
    _add_check(
        rows,
        "partial_stage_markers",
        "pass" if not partials or allow_partial_resume else "fail",
        not allow_partial_resume,
        "no partial stage markers" if not partials else "partial stage markers require review",
        {
            "markers": [path.name for path in partials],
            "recommendation": "resume only after validating or remove stale .partial markers intentionally",
        },
    )


def _check_output_collisions(rows: list[dict], run_script: Path) -> None:
    if not run_script.exists():
        return
    stages = _parse_run_stage_dirs(run_script.read_text(encoding="utf-8"))
    unsafe: list[dict] = []
    reusable: list[dict] = []
    for stage in stages:
        outdir = Path(stage["outdir"])
        marker = Path(stage["marker"])
        if marker.exists() and not outdir.exists():
            unsafe.append({**stage, "reason": "completed_marker_without_expected_output"})
            continue
        if not outdir.exists():
            continue
        if marker.exists():
            reusable.append({**stage, "reason": "completed_marker_present"})
            continue
        ok, command, message = _validate_existing_output(stage["marker"], outdir)
        if ok:
            reusable.append({**stage, "reason": "validator_passed", "validator": command})
        else:
            unsafe.append({**stage, "reason": message, "validator": command})
    _add_check(
        rows,
        "output_collisions",
        "fail" if unsafe else "pass",
        True,
        "unsafe existing outputs found" if unsafe else "no unsafe output collisions",
        {"unsafe": unsafe, "reusable": reusable},
    )


def _parse_run_stage_dirs(run_text: str) -> list[dict]:
    stages: list[dict] = []
    for line in run_text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("run_stage_dir "):
            continue
        parts = stripped.split()
        if len(parts) >= 4:
            stages.append({"marker": parts[1], "outdir": parts[2], "command": " ".join(parts[3:])})
    return stages


def _validate_existing_output(marker: str, outdir: Path) -> tuple[bool, str, str]:
    marker_name = Path(marker).name
    command = _validator_for_marker(marker_name, outdir)
    if not command:
        return False, "", "no validator available for existing output"
    proc = subprocess.run(
        command,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=300,
    )
    return proc.returncode == 0, " ".join(command), (proc.stderr or proc.stdout).strip()[:500]


def _validator_for_marker(marker_name: str, outdir: Path) -> list[str]:
    if marker_name.endswith("_simulate"):
        return ["babappa", "validate-sim", "--sim-dir", str(outdir), "--require-branch-truth"]
    if marker_name.endswith("_align"):
        return ["babappa", "validate-align", "--align-dir", str(outdir)]
    if marker_name.endswith("_site_map"):
        return ["babappa", "validate-site-map", "--site-map-dir", str(outdir)]
    if marker_name.endswith("_method_policy"):
        return ["babappa", "validate-aligner-method-policy", "--policy-dir", str(outdir)]
    if marker_name.endswith("_tensors"):
        return ["babappa", "validate-tensors", "--tensor-dir", str(outdir)]
    if marker_name.endswith("_index"):
        return ["babappa", "validate-index", "--index-dir", str(outdir)]
    if marker_name.endswith("_labels"):
        return ["babappa", "validate-branch-site-labels", "--label-dir", str(outdir)]
    if marker_name.endswith("_branch_dataset"):
        return ["babappa", "validate-branch-site-dataset", "--branch-site-dataset-dir", str(outdir)]
    if marker_name.endswith("_leakage"):
        return ["babappa", "validate-branch-site-leakage", "--leakage-dir", str(outdir)]
    if marker_name.endswith("_branch_neural"):
        return ["babappa", "validate-branch-site-neural", "--model-dir", str(outdir)]
    if marker_name.endswith("_calibration"):
        return ["babappa", "validate-branch-site-calibration", "--calibration-dir", str(outdir)]
    if marker_name.endswith("_aggregation"):
        return ["babappa", "validate-branch-aggregation", "--aggregation-dir", str(outdir)]
    if marker_name.endswith("_controls"):
        return ["babappa", "validate-branch-aggregation-controls", "--controls-dir", str(outdir)]
    if marker_name.endswith("_threshold"):
        return ["babappa", "validate-branch-site-threshold-policy", "--policy-dir", str(outdir)]
    if marker_name.endswith("_aggregation_policy"):
        return ["babappa", "validate-branch-aggregation-threshold-policy", "--policy-dir", str(outdir)]
    if marker_name.endswith("_summary"):
        return ["babappa", "validate-branch-site-run-summary", "--summary-dir", str(outdir)]
    if marker_name.endswith("_truth_audit"):
        return ["babappa", "validate-branch-truth-status-audit", "--audit-dir", str(outdir)]
    return []


def _check_disk_and_memory(rows: list[dict], scale: str) -> None:
    usage = shutil.disk_usage(Path.cwd())
    free_gb = usage.free / (1024**3)
    warn_gb, fail_gb = (100.0, 50.0) if scale == "10k" else (1000.0, 500.0)
    warn_gb = _env_float(
        f"BABAPPA_PREFLIGHT_{scale.upper()}_WARN_FREE_GB",
        _env_float("BABAPPA_PREFLIGHT_WARN_FREE_GB", warn_gb),
    )
    fail_gb = _env_float(
        f"BABAPPA_PREFLIGHT_{scale.upper()}_FAIL_FREE_GB",
        _env_float("BABAPPA_PREFLIGHT_FAIL_FREE_GB", fail_gb),
    )
    status = "pass"
    if free_gb < fail_gb:
        status = "fail"
    elif free_gb < warn_gb:
        status = "warn"
    _add_check(
        rows,
        "disk_free",
        status,
        free_gb < fail_gb,
        f"{free_gb:.1f} GB free",
        {
            "warn_below_gb": warn_gb,
            "fail_below_gb": fail_gb,
            "override_env": {
                "warn": f"BABAPPA_PREFLIGHT_{scale.upper()}_WARN_FREE_GB",
                "fail": f"BABAPPA_PREFLIGHT_{scale.upper()}_FAIL_FREE_GB",
                "generic_warn": "BABAPPA_PREFLIGHT_WARN_FREE_GB",
                "generic_fail": "BABAPPA_PREFLIGHT_FAIL_FREE_GB",
            },
        },
    )
    mem_bytes = _memory_bytes()
    mem_gb = mem_bytes / (1024**3) if mem_bytes else 0.0
    mem_status = "pass" if mem_gb >= 32.0 else "warn"
    message = f"{mem_gb:.1f} GB memory detected" if mem_bytes else "memory size unavailable"
    if scale == "100k":
        message += "; 100K should remain tier-by-tier with capped outputs on 36 GB"
    _add_check(rows, "system_memory", mem_status, False, message)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if value >= 0 else default


def _memory_bytes() -> int:
    proc = subprocess.run(
        ["sysctl", "-n", "hw.memsize"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        return int(proc.stdout.strip())
    except ValueError:
        return 0


def _check_internal_consistency(rows: list[dict], scripts: dict[str, Path], scale: str) -> None:
    run_text = scripts["run"].read_text(encoding="utf-8") if scripts["run"].exists() else ""
    validate_text = scripts["validate"].read_text(encoding="utf-8") if scripts["validate"].exists() else ""
    checks = [
        ("methods_default", "--methods identity,mafft,babappalign,muscle" in run_text),
        ("feature_policy", "--feature-policy conservative_branch_site" in run_text),
        ("truth_mode", "--truth-mode explicit" in run_text),
        ("device_mps", "--device mps" in run_text),
        ("batch_size_present", "--batch-size " in run_text),
        ("max_output_rows_present", "--max-output-rows " in run_text),
        ("streaming_capped_dataset", "--streaming --max-output-rows" in run_text),
        ("no_prank_default", "prank" not in run_text.lower()),
        ("no_tcoffee_default", "tcoffee" not in run_text.lower()),
        ("validate_align_dirs", "validate-align --align-dir" in validate_text),
        ("validate_site_map_dirs", "validate-site-map --site-map-dir" in validate_text),
        ("preflight_before_simulation", _preflight_before_simulation(run_text, scale)),
    ]
    for name, ok in checks:
        _add_check(rows, f"internal_consistency:{name}", "pass" if ok else "fail", True, name)


def _preflight_before_simulation(run_text: str, scale: str) -> bool:
    preflight = run_text.find("preflight-explicit-branch-truth-mps-plan")
    simulate = run_text.find("babappa simulate")
    return preflight >= 0 and simulate >= 0 and preflight < simulate


def _validate_equal_length_fasta(path: Path, expected_taxa: set[str]) -> tuple[bool, str]:
    try:
        records = read_fasta(path)
    except ValueError as exc:
        return False, str(exc)
    if set(records) != expected_taxa:
        return False, f"expected taxa {sorted(expected_taxa)}, observed {sorted(records)}"
    lengths = {len(sequence) for sequence in records.values()}
    if len(lengths) != 1:
        return False, "aligned lengths are unequal"
    return True, "ok"


def _validate_fasta_frame(path: Path, expected_taxa: set[str]) -> tuple[bool, str]:
    ok, reason = _validate_equal_length_fasta(path, expected_taxa)
    if not ok:
        return ok, reason
    records = read_fasta(path)
    if any(len(sequence) % 3 != 0 for sequence in records.values()):
        return False, "alignment length is not divisible by 3"
    return True, "ok"


def _write_reports(plan_dir: Path, rows: list[dict], report_stem: str) -> dict:
    plan_dir.mkdir(parents=True, exist_ok=True)
    json_path = plan_dir / f"{report_stem}.json"
    tsv_path = plan_dir / f"{report_stem}.tsv"
    md_path = plan_dir / f"{report_stem}.md"
    status = "pass" if not any(row["status"] == "fail" and row["required"] for row in rows) else "fail"
    payload = {
        "preflight_version": MPS_PREFLIGHT_VERSION,
        "status": status,
        "n_checks": len(rows),
        "n_fail": sum(1 for row in rows if row["status"] == "fail"),
        "n_warn": sum(1 for row in rows if row["status"] == "warn"),
        "checks": rows,
        "reports": {"json": str(json_path), "tsv": str(tsv_path), "markdown": str(md_path)},
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with tsv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["name", "status", "required", "message", "details"])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "name": row["name"],
                    "status": row["status"],
                    "required": row["required"],
                    "message": row["message"],
                    "details": json.dumps(row["details"], sort_keys=True),
                }
            )
    md_lines = [
        "# BABAPPA MPS plan preflight",
        "",
        f"- status: {status}",
        f"- checks: {len(rows)}",
        f"- failures: {payload['n_fail']}",
        f"- warnings: {payload['n_warn']}",
        "",
        "| Check | Status | Required | Message |",
        "|---|---:|---:|---|",
    ]
    for row in rows:
        md_lines.append(
            f"| `{row['name']}` | {row['status']} | {row['required']} | {str(row['message']).replace('|', '/')} |"
        )
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return payload
