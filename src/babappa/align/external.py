"""Unified internal/external alignment runner for BABAPPA."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

from babappa import __version__
from babappa.align.backends import (
    INTERNAL_METHODS,
    babappalign_model_status,
    detect_aligner_backends,
    validate_alignment_methods,
    write_aligner_status,
)
from babappa.align.ensemble import AlignmentConfig, align_simulation_directory, write_fasta
from babappa.simulate.audit import read_fasta

EXTERNAL_ALIGNMENT_VERSION = __version__


@dataclass(frozen=True)
class ExternalAlignmentConfig:
    """Configuration for running an internal/external alignment ensemble."""

    sim_dir: str
    outdir: str
    methods: List[str] = field(default_factory=lambda: ["identity", "mafft", "babappalign", "muscle"])
    seed: int = 42
    require_available: bool = False
    keep_intermediate: bool = False
    timeout_seconds: int = 300
    threads: int = 1
    aligner_subprocess_threads: int = 1
    babappalign_device: str = "cpu"
    babappalign_backend: str = "auto"
    babappalign_workers: int = 0
    max_method_failure_fraction: float = 0.01
    allow_missing_babappalign: bool = False

    def __post_init__(self) -> None:
        sim_path = Path(self.sim_dir)
        out_path = Path(self.outdir)
        if not sim_path.exists():
            raise ValueError(f"sim_dir does not exist: {sim_path}")
        if not (sim_path / "manifest.json").exists():
            raise ValueError(f"sim_dir is missing manifest.json: {sim_path}")
        if not self.methods:
            raise ValueError("methods must be non-empty")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be > 0")
        if self.threads < 1:
            raise ValueError("threads must be >= 1")
        if self.aligner_subprocess_threads < 1:
            raise ValueError("aligner_subprocess_threads must be >= 1")
        if self.babappalign_device not in {"auto", "cpu", "cuda", "mps"}:
            raise ValueError("babappalign_device must be one of: auto, cpu, cuda, mps")
        if self.babappalign_backend not in {"auto", "cli", "embedded"}:
            raise ValueError("babappalign_backend must be one of: auto, cli, embedded")
        if self.babappalign_workers < 0:
            raise ValueError("babappalign_workers must be >= 0")
        if self.max_method_failure_fraction < 0 or self.max_method_failure_fraction > 1:
            raise ValueError("max_method_failure_fraction must be between 0 and 1")
        validate_alignment_methods(list(self.methods), require_available=self.require_available)
        if "babappalign" in self.methods and not self.allow_missing_babappalign:
            status = babappalign_model_status()
            if not status["model_present"]:
                raise ValueError(_babappalign_model_missing_message(status))
        out_path.mkdir(parents=True, exist_ok=True)


def run_alignment_ensemble(config: ExternalAlignmentConfig) -> dict:
    """Run available internal/external alignment methods on a simulation directory."""
    sim_path = Path(config.sim_dir)
    outdir = Path(config.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    families_outdir = outdir / "families"
    families_outdir.mkdir(parents=True, exist_ok=True)

    sim_manifest = _read_json(sim_path / "manifest.json")
    family_ids = sim_manifest.get("family_ids")
    if not isinstance(family_ids, list):
        raise ValueError("simulation manifest does not contain a family_ids list")

    warnings = validate_alignment_methods(
        list(config.methods), require_available=config.require_available
    )
    backends = detect_aligner_backends()
    write_aligner_status(outdir / "aligner_status.json", backends)

    created_files: Dict[str, Dict[str, Dict[str, str]]] = {
        str(family_id): {} for family_id in family_ids
    }
    methods_run: List[str] = []
    methods_skipped: Dict[str, str] = {}
    methods_quarantined: Dict[str, str] = {}
    method_status: Dict[str, dict] = {}
    babappalign_effective_workers: int | None = None
    attempted = 0
    ok_count = 0
    failed_count = 0

    def ensure_method_status(method: str) -> dict:
        return method_status.setdefault(
            method,
            {
                "method": method,
                "attempted_families": 0,
                "successful_families": 0,
                "failed_families": 0,
                "failure_fraction": 0.0,
                "failure_reasons": [],
                "quarantined": False,
                "quarantine_reason": "",
            },
        )

    internal_methods = [method for method in config.methods if method in INTERNAL_METHODS]
    if internal_methods:
        internal_summary = align_simulation_directory(
            AlignmentConfig(
                sim_dir=str(sim_path),
                outdir=str(outdir),
                methods=internal_methods,
                seed=config.seed,
            )
        )
        internal_manifest = _read_json(Path(internal_summary["manifest"]))
        internal_created = internal_manifest.get("created_files", {})
        for method in internal_methods:
            stats = ensure_method_status(method)
            stats["attempted_families"] = len(family_ids)
        for family_id in family_ids:
            family_created = internal_created.get(family_id, {})
            for method in internal_methods:
                if method in family_created:
                    created_files[str(family_id)][method] = dict(family_created[method])
                    attempted += 1
                    ok_count += 1
                    stats = ensure_method_status(method)
                    stats["successful_families"] += 1
        for method in internal_methods:
            stats = ensure_method_status(method)
            stats["failed_families"] = max(
                0, int(stats["attempted_families"]) - int(stats["successful_families"])
            )
            stats["failure_fraction"] = _failure_fraction(stats)
            if stats["successful_families"]:
                methods_run.append(method)

    external_methods = [method for method in config.methods if method not in INTERNAL_METHODS]
    resolved_babappalign_backend = (
        _resolve_babappalign_backend(config.babappalign_backend)
        if "babappalign" in external_methods
        else config.babappalign_backend
    )
    if resolved_babappalign_backend == "embedded":
        warnings = [warning for warning in warnings if warning != "method_unavailable:babappalign"]
    for method in external_methods:
        backend = backends[method]
        babappalign_backend = resolved_babappalign_backend
        stats = ensure_method_status(method)
        if not backend.available and not (method == "babappalign" and babappalign_backend == "embedded"):
            methods_skipped[method] = "executable_unavailable"
            stats["failure_reasons"].append("executable_unavailable")
            warnings.append(f"method_unavailable:{method}")
            if config.require_available:
                raise ValueError(f"requested external aligner unavailable: {method}")
            continue
        pending_family_ids: List[str] = []
        for family_id in family_ids:
            attempted += 1
            stats["attempted_families"] += 1
            existing = _existing_family_method_output(
                sim_path=sim_path,
                outdir=outdir,
                family_id=str(family_id),
                method=method,
            )
            if existing is not None:
                ok_count += 1
                stats["successful_families"] += 1
                created_files[str(family_id)][method] = existing
                continue
            pending_family_ids.append(str(family_id))

        def run_one(pending_family_id: str) -> Tuple[str, dict]:
            result = _run_external_family_method(
                sim_path=sim_path,
                outdir=outdir,
                family_id=pending_family_id,
                method=method,
                executable=backend.executable or method,
                timeout_seconds=config.timeout_seconds,
                threads=config.threads,
                aligner_subprocess_threads=config.aligner_subprocess_threads,
                babappalign_device=config.babappalign_device,
                babappalign_backend=babappalign_backend,
            )
            return pending_family_id, result

        method_workers = config.threads
        if method == "babappalign" and config.babappalign_workers > 0:
            method_workers = config.babappalign_workers
        method_workers = max(1, min(int(method_workers), len(pending_family_ids) or 1))
        if method == "babappalign" and babappalign_backend == "embedded":
            method_workers, cap_warning = _cap_embedded_babappalign_workers(method_workers)
            babappalign_effective_workers = method_workers
            if cap_warning:
                warnings.append(cap_warning)
        if method == "babappalign" and babappalign_backend == "embedded" and method_workers > 1:
            results = []
            task_payloads = [
                (
                    str(sim_path),
                    str(outdir),
                    family_id,
                    backend.executable or method,
                    config.timeout_seconds,
                    config.threads,
                    config.aligner_subprocess_threads,
                    config.babappalign_device,
                    babappalign_backend,
                )
                for family_id in pending_family_ids
            ]
            with ProcessPoolExecutor(max_workers=method_workers) as executor:
                futures = {
                    executor.submit(_run_external_family_method_task, payload): payload[2]
                    for payload in task_payloads
                }
                for future in as_completed(futures):
                    results.append(future.result())
        elif method_workers == 1 or len(pending_family_ids) <= 1:
            results = [run_one(family_id) for family_id in pending_family_ids]
        else:
            results = []
            with ThreadPoolExecutor(max_workers=method_workers) as executor:
                futures = {
                    executor.submit(run_one, family_id): family_id
                    for family_id in pending_family_ids
                }
                for future in as_completed(futures):
                    results.append(future.result())

        for family_id, result in sorted(results, key=lambda item: item[0]):
            if result["status"] == "ok":
                ok_count += 1
                stats["successful_families"] += 1
                created_files[family_id][method] = result["files"]
            else:
                failed_count += 1
                stats["failed_families"] += 1
                stats["failure_reasons"].extend(result["warnings"])
                warnings.extend(result["warnings"])
        stats["failure_fraction"] = _failure_fraction(stats)
        stats["failure_reasons"] = sorted(set(stats["failure_reasons"]))
        if stats["successful_families"] == 0:
            methods_skipped[method] = "all_family_method_failures"
            warnings.append(f"method_not_manifested_due_to_failures:{method}:{stats['failed_families']}")
        elif stats["failure_fraction"] > config.max_method_failure_fraction:
            reason = (
                f"failure_fraction_above_max:{stats['failure_fraction']}"
                f">{config.max_method_failure_fraction}"
            )
            stats["quarantined"] = True
            stats["quarantine_reason"] = reason
            methods_quarantined[method] = reason
            methods_skipped[method] = f"family_method_failures:{stats['failed_families']}"
            warnings.append(f"method_quarantined_due_to_failures:{method}:{reason}")
        else:
            methods_run.append(method)
            if stats["failed_families"]:
                warnings.append(
                    f"method_manifested_with_partial_failures:{method}:"
                    f"{stats['failed_families']}/{stats['attempted_families']}"
                )

    failure_rate_by_method = {
        method: float(stats.get("failure_fraction") or 0.0)
        for method, stats in sorted(method_status.items())
    }
    manifest_path = outdir / "alignment_manifest.json"
    manifest = {
        "alignment_manifest_version": EXTERNAL_ALIGNMENT_VERSION,
        "aligner_scaffold_version": EXTERNAL_ALIGNMENT_VERSION,
        "sim_dir": str(sim_path),
        "n_families": len(family_ids),
        "family_ids": family_ids,
        "methods": methods_run,
        "methods_requested": list(config.methods),
        "methods_run": methods_run,
        "methods_skipped": methods_skipped,
        "methods_quarantined": methods_quarantined,
        "method_status": method_status,
        "failure_rate_by_method": failure_rate_by_method,
        "max_method_failure_fraction": config.max_method_failure_fraction,
        "timeout_seconds": config.timeout_seconds,
        "threads": config.threads,
        "aligner_subprocess_threads": config.aligner_subprocess_threads,
        "babappalign_device": config.babappalign_device,
        "babappalign_backend": resolved_babappalign_backend,
        "babappalign_workers": config.babappalign_workers,
        "babappalign_effective_workers": babappalign_effective_workers,
        "seed": config.seed,
        "dropout_rate": 0.02,
        "n_family_method_attempted": attempted,
        "n_family_method_ok": ok_count,
        "n_family_method_failed": failed_count,
        "warnings": sorted(set(warnings)),
        "created_files": created_files,
        "generated_files": {
            "alignment_manifest": str(manifest_path),
            "aligner_status": str(outdir / "aligner_status.json"),
        },
    }
    _write_json(manifest_path, manifest)
    return {
        "status": "ok",
        "outdir": str(outdir),
        "n_families": len(family_ids),
        "methods_requested": list(config.methods),
        "methods_run": methods_run,
        "methods_skipped": methods_skipped,
        "methods_quarantined": methods_quarantined,
        "failure_rate_by_method": failure_rate_by_method,
        "n_family_method_attempted": attempted,
        "n_family_method_ok": ok_count,
        "n_family_method_failed": failed_count,
        "manifest": str(manifest_path),
        "warnings": manifest["warnings"],
    }


def _failure_fraction(stats: dict) -> float:
    attempted = int(stats.get("attempted_families") or 0)
    failed = int(stats.get("failed_families") or 0)
    return 0.0 if attempted <= 0 else failed / attempted


def _cap_embedded_babappalign_workers(requested: int) -> Tuple[int, str | None]:
    """Keep embedded BABAPPAlign from loading too many large models at once."""
    requested = max(1, int(requested))
    if os.environ.get("BABAPPA_ALLOW_OVERLOAD") == "1":
        return requested, None

    cap = None
    warning_prefix = "babappalign_embedded_worker_cap"
    env_cap = os.environ.get("BABAPPA_BABAPPALIGN_MAX_WORKERS", "").strip()
    if env_cap:
        try:
            cap = max(1, int(env_cap))
        except ValueError:
            cap = 4
            warning_prefix = "babappalign_invalid_max_workers_capped"
    if cap is None:
        memory_gb = _physical_memory_gb()
        cap = max(1, int(memory_gb // 9)) if memory_gb else 4
        cap = min(cap, 4)

    effective = max(1, min(requested, cap))
    if effective < requested:
        return (
            effective,
            f"{warning_prefix}:{requested}->{effective}:"
            "set_BABAPPA_BABAPPALIGN_MAX_WORKERS_or_BABAPPA_ALLOW_OVERLOAD=1_to_override",
        )
    return effective, None


def _physical_memory_gb() -> float | None:
    try:
        proc = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        )
        if proc.returncode == 0:
            return int(proc.stdout.strip()) / (1024**3)
    except Exception:
        pass
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return float(pages * page_size) / (1024**3)
    except Exception:
        return None


def _run_external_family_method(
    sim_path: Path,
    outdir: Path,
    family_id: str,
    method: str,
    executable: str,
    timeout_seconds: int,
    threads: int,
    aligner_subprocess_threads: int,
    babappalign_device: str,
    babappalign_backend: str,
) -> dict:
    source_fasta = sim_path / "families" / family_id / f"{family_id}.fasta"
    family_outdir = outdir / "families" / family_id
    family_outdir.mkdir(parents=True, exist_ok=True)
    codon_fasta = family_outdir / f"{family_id}.{method}.codon.fasta"
    map_tsv = family_outdir / f"{family_id}.{method}.map.tsv"
    qc_json = family_outdir / f"{family_id}.{method}.qc.json"
    warnings: List[str] = []
    started = time.monotonic()
    command: List[str] = [executable]
    return_code: int | None = None
    stdout_preview = ""
    stderr_preview = ""
    output_source = ""
    expected_taxa = set(read_fasta(source_fasta).keys())
    child_env = _alignment_subprocess_env(aligner_subprocess_threads)

    try:
        if method == "mafft":
            command = [executable, "--auto", str(source_fasta)]
            with codon_fasta.open("w", encoding="utf-8") as stdout:
                proc = subprocess.run(
                    command,
                    check=False,
                    stdout=stdout,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=timeout_seconds,
                    env=child_env,
                )
            return_code = proc.returncode
            stderr_preview = _preview(proc.stderr)
            output_source = "stdout_redirect"
            if proc.returncode != 0:
                warnings.append(
                    f"external_alignment_failed:{family_id}:{method}:{proc.stderr.strip()[:200]}"
                )
        elif method == "prank":
            prefix = family_outdir / f"{family_id}.{method}"
            command = [executable, f"-d={source_fasta}", f"-o={prefix}"]
            proc = subprocess.run(
                command,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout_seconds,
                env=child_env,
            )
            return_code = proc.returncode
            stdout_preview = _preview(proc.stdout)
            stderr_preview = _preview(proc.stderr)
            candidate = _locate_prank_output(prefix)
            if proc.returncode == 0 and candidate is not None:
                shutil.copyfile(candidate, codon_fasta)
                output_source = str(candidate)
            else:
                warnings.append(
                    f"external_alignment_failed:{family_id}:{method}:{proc.stderr.strip()[:200]}"
                )
        elif method == "babappalign":
            if babappalign_backend == "embedded":
                command, return_code, stdout_preview, stderr_preview, output_source = _run_babappalign_embedded(
                    source_fasta,
                    family_outdir,
                    codon_fasta,
                    babappalign_device,
                )
            else:
                command, return_code, stdout_preview, stderr_preview, output_source = _run_babappalign(
                    executable,
                    source_fasta,
                    family_outdir,
                    codon_fasta,
                    timeout_seconds,
                    babappalign_device,
                    child_env,
                )
            if return_code != 0 or not output_source:
                reason = (
                    "babappalign_model_missing"
                    if _is_babappalign_model_missing(stdout_preview, stderr_preview)
                    else "no_stdout_or_sidecar_fasta"
                )
                warnings.append(
                    f"external_alignment_failed:{family_id}:{method}:{reason}"
                )
        elif method == "muscle":
            command, return_code, stdout_preview, stderr_preview, output_source = _run_muscle(
                executable, source_fasta, codon_fasta, timeout_seconds, child_env
            )
            if return_code != 0 or not output_source:
                warnings.append(
                    f"external_alignment_failed:{family_id}:{method}:{stderr_preview or stdout_preview}"
                )
        elif method == "tcoffee":
            command = [
                executable,
                str(source_fasta),
                "-outfile",
                str(codon_fasta),
                "-output",
                "fasta",
            ]
            proc = subprocess.run(
                command,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout_seconds,
                env=child_env,
            )
            return_code = proc.returncode
            stdout_preview = _preview(proc.stdout)
            stderr_preview = _preview(proc.stderr)
            if proc.returncode == 0 and codon_fasta.exists() and codon_fasta.stat().st_size > 0:
                output_source = str(codon_fasta)
            else:
                warnings.append(
                    f"external_alignment_failed:{family_id}:{method}:{stderr_preview or stdout_preview}"
                )
        else:
            warnings.append(f"unsupported_external_method:{method}")
    except subprocess.TimeoutExpired:
        warnings.append(f"external_alignment_timeout:{family_id}:{method}")
    except OSError as exc:
        warnings.append(f"external_alignment_os_error:{family_id}:{method}:{exc}")

    runtime_seconds = time.monotonic() - started
    if not codon_fasta.exists() or codon_fasta.stat().st_size == 0:
        _write_json(
            qc_json,
            _qc_payload(
                family_id,
                method,
                source_fasta,
                command,
                return_code,
                runtime_seconds,
                "fail",
                warnings,
                stdout_preview,
                stderr_preview,
                output_source,
            ),
        )
        return {"status": "fail", "warnings": warnings, "files": {}}

    records, qc_warnings = _validate_codon_frame(codon_fasta, expected_taxa=expected_taxa)
    warnings.extend(qc_warnings)
    if qc_warnings:
        warnings.append(f"external_alignment_not_codon_frame_preserving:{family_id}:{method}")
        _write_json(
            qc_json,
            _qc_payload(
                family_id,
                method,
                source_fasta,
                command,
                return_code,
                runtime_seconds,
                "fail",
                warnings,
                stdout_preview,
                stderr_preview,
                output_source,
            ),
        )
        return {"status": "fail", "warnings": warnings, "files": {}}

    _write_alignment_map(records, map_tsv, method)
    _write_json(
        qc_json,
        _qc_payload(
            family_id,
            method,
            source_fasta,
            command,
            return_code,
            runtime_seconds,
            "ok",
            warnings,
            stdout_preview,
            stderr_preview,
            output_source,
        )
        | {
            "n_taxa": len(records),
            "expected_taxa_count": len(expected_taxa),
            "observed_taxa": list(records.keys()),
            "alignment_length_nt": len(next(iter(records.values()))) if records else 0,
            "alignment_length_codons": (
                len(next(iter(records.values()))) // 3 if records else 0
            ),
            "frame_status": "ok",
        },
    )
    return {
        "status": "ok",
        "warnings": warnings,
        "files": {
            "codon_fasta": str(codon_fasta.relative_to(outdir)),
            "map": str(map_tsv.relative_to(outdir)),
            "qc": str(qc_json.relative_to(outdir)),
        },
    }


def _run_external_family_method_task(payload: tuple) -> Tuple[str, dict]:
    (
        sim_path,
        outdir,
        family_id,
        executable,
        timeout_seconds,
        threads,
        aligner_subprocess_threads,
        babappalign_device,
        babappalign_backend,
    ) = payload
    result = _run_external_family_method(
        sim_path=Path(sim_path),
        outdir=Path(outdir),
        family_id=str(family_id),
        method="babappalign",
        executable=str(executable),
        timeout_seconds=int(timeout_seconds),
        threads=int(threads),
        aligner_subprocess_threads=int(aligner_subprocess_threads),
        babappalign_device=str(babappalign_device),
        babappalign_backend=str(babappalign_backend),
    )
    return str(family_id), result


def smoke_aligner(
    method: str,
    outdir: str | Path,
    device: str = "cpu",
    timeout_seconds: int = 60,
) -> dict:
    """Run a tiny aligner smoke, with explicit BABAPPAlign model-cache reporting."""
    method = method.strip()
    validate_alignment_methods([method], require_available=False)
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be > 0")
    if device not in {"cpu", "cuda", "mps", "auto"}:
        raise ValueError("device must be one of: auto, cpu, cuda, mps")

    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)
    report_json = out_path / "aligner_smoke_report.json"
    report_md = out_path / "aligner_smoke_report.md"
    backends = detect_aligner_backends()
    backend = backends[method]

    payload = {
        "status": "fail",
        "method": method,
        "reason": "",
        "executable": backend.executable,
        "command": [],
        "return_code": None,
        "stdout_preview": "",
        "stderr_preview": "",
        "output_source": "",
        "report_json": str(report_json),
        "report_md": str(report_md),
    }

    if method == "identity":
        payload.update({"status": "ok", "reason": "internal_identity_available"})
        _write_smoke_aligner_reports(report_json, report_md, payload)
        return payload

    if method != "babappalign":
        if not backend.available:
            payload["reason"] = "executable_unavailable"
            _write_smoke_aligner_reports(report_json, report_md, payload)
            return payload
        payload["status"] = "skipped"
        payload["reason"] = "smoke_not_implemented_for_method"
        _write_smoke_aligner_reports(report_json, report_md, payload)
        return payload

    model_status = babappalign_model_status()
    payload.update(model_status)
    if not model_status["model_present"]:
        payload["reason"] = "babappalign_model_missing"
        _write_smoke_aligner_reports(report_json, report_md, payload)
        return payload
    if not backend.available:
        payload["reason"] = "executable_unavailable"
        _write_smoke_aligner_reports(report_json, report_md, payload)
        return payload

    input_path = out_path / "tiny.codon.fasta"
    codon_fasta = out_path / "tiny.babappalign.codon.fasta"
    input_path.write_text(
        ">taxon_a\nATGAAACCCGGG\n>taxon_b\nATGAAACCCGGG\n",
        encoding="utf-8",
    )
    command, return_code, stdout_preview, stderr_preview, output_source = _run_babappalign(
        backend.executable or "babappalign",
        input_path,
        out_path,
        codon_fasta,
        timeout_seconds,
        "cpu" if device == "auto" else device,
    )
    reason = ""
    status = "ok"
    if return_code != 0 or not output_source:
        status = "fail"
        reason = (
            "babappalign_model_missing"
            if _is_babappalign_model_missing(stdout_preview, stderr_preview)
            else "no_stdout_or_sidecar_fasta"
        )
    payload.update(
        {
            "status": status,
            "reason": reason or "ok",
            "command": command,
            "return_code": return_code,
            "stdout_preview": stdout_preview,
            "stderr_preview": stderr_preview,
            "output_source": output_source,
        }
    )
    _write_smoke_aligner_reports(report_json, report_md, payload)
    return payload


def _existing_family_method_output(
    sim_path: Path,
    outdir: Path,
    family_id: str,
    method: str,
) -> dict | None:
    """Return already validated family-method files so interrupted runs can resume."""
    source_fasta = sim_path / "families" / family_id / f"{family_id}.fasta"
    family_outdir = outdir / "families" / family_id
    codon_fasta = family_outdir / f"{family_id}.{method}.codon.fasta"
    map_tsv = family_outdir / f"{family_id}.{method}.map.tsv"
    qc_json = family_outdir / f"{family_id}.{method}.qc.json"
    if not (codon_fasta.exists() and map_tsv.exists() and qc_json.exists()):
        return None
    if codon_fasta.stat().st_size == 0:
        return None
    try:
        expected_taxa = set(read_fasta(source_fasta).keys())
    except ValueError:
        return None
    _, warnings = _validate_codon_frame(codon_fasta, expected_taxa=expected_taxa)
    if warnings:
        return None
    return {
        "codon_fasta": str(codon_fasta.relative_to(outdir)),
        "map": str(map_tsv.relative_to(outdir)),
        "qc": str(qc_json.relative_to(outdir)),
    }


def _run_babappalign(
    executable: str,
    source_fasta: Path,
    family_outdir: Path,
    codon_fasta: Path,
    timeout_seconds: int,
    device: str,
    env: dict[str, str] | None = None,
) -> Tuple[List[str], int | None, str, str, str]:
    work_input = family_outdir / f"{source_fasta.stem}.babappalign.input.fasta"
    shutil.copyfile(source_fasta, work_input)
    command = [executable, "--mode", "codon", "--device", device, str(work_input)]
    proc = subprocess.run(
        command,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout_seconds,
        env=env,
    )
    output_source = ""
    if proc.returncode == 0:
        stdout_records = _fasta_records_from_stdout(proc.stdout, family_outdir)
        if stdout_records:
            write_fasta(stdout_records, codon_fasta)
            output_source = "stdout_fasta"
        else:
            candidate = _locate_babappalign_output(work_input)
            if candidate is not None:
                shutil.copyfile(candidate, codon_fasta)
                output_source = str(candidate)
    return command, proc.returncode, _preview(proc.stdout), _preview(proc.stderr), output_source


_BABAPPALIGN_EMBEDDED_STATE: dict = {}


def _resolve_babappalign_backend(requested: str) -> str:
    requested = (requested or "auto").lower()
    if requested == "cli":
        return "cli"
    available = _babappalign_embedded_available()
    if requested == "embedded":
        if not available:
            raise ValueError("babappalign embedded backend requested but babappalign Python API is unavailable")
        return "embedded"
    if requested == "auto":
        return "embedded" if available else "cli"
    raise ValueError("babappalign_backend must be one of: auto, cli, embedded")


def _babappalign_embedded_available() -> bool:
    try:
        import babappalign.babappalign  # noqa: F401
    except Exception:
        return False
    return True


def _embedded_babappalign_state(device_choice: str) -> dict:
    state = _BABAPPALIGN_EMBEDDED_STATE
    if state.get("requested_device") == device_choice and state.get("model") is not None:
        return state
    import torch
    import babappalign.babappalign as ba

    device = ba.resolve_device(device_choice)
    model_path = ba.resolve_model_path()
    model = ba.safe_load_model(str(model_path), device)
    if hasattr(model, "eval"):
        model.eval()
    state.clear()
    state.update(
        {
            "ba": ba,
            "torch": torch,
            "requested_device": device_choice,
            "device": device,
            "model": model,
            "embedding_cache": ba.get_cache_dir("embeddings"),
        }
    )
    return state


def _run_babappalign_embedded(
    source_fasta: Path,
    family_outdir: Path,
    codon_fasta: Path,
    device_choice: str,
) -> Tuple[List[str], int | None, str, str, str]:
    command = ["babappalign-embedded", "--mode", "codon", "--device", device_choice, str(source_fasta)]
    try:
        state = _embedded_babappalign_state(device_choice)
        ba = state["ba"]
        torch = state["torch"]
        device = state["device"]
        model = state["model"]
        emb_cache = state["embedding_cache"]

        ids, raw_seqs = ba.read_fasta(source_fasta)
        cds_map = {}
        seqs = []
        for sid, cds in zip(ids, raw_seqs):
            cds = cds.upper().replace("U", "T")
            ba.validate_cds(cds, sid)
            cds_map[sid] = cds
            seqs.append(ba.translate_cds(cds))

        emb_map = {}
        with torch.inference_mode():
            for sid, seq in zip(ids, seqs):
                emb_path = emb_cache / f"{ba.seq_hash(seq)}.pt"
                if emb_path.exists():
                    try:
                        emb = torch.load(emb_path, map_location="cpu")
                    except Exception:
                        try:
                            emb_path.unlink()
                        except OSError:
                            pass
                        emb = ba.embed_sequence(seq, device)
                        tmp_path = emb_path.with_name(f"{emb_path.name}.{os.getpid()}.tmp")
                        torch.save(emb.cpu(), tmp_path)
                        os.replace(tmp_path, emb_path)
                else:
                    emb = ba.embed_sequence(seq, device)
                    tmp_path = emb_path.with_name(f"{emb_path.name}.{os.getpid()}.tmp")
                    torch.save(emb.cpu(), tmp_path)
                    os.replace(tmp_path, emb_path)
                emb_map[sid] = emb.to(device)
            out_ids, out_seqs = ba.progressive_align(
                ids,
                seqs,
                emb_map,
                model,
                device,
                -2.5 * 3,
                -0.7 * 3,
            )

        codon_aligned = [
            ba.backmap_to_codon_alignment(aln, cds_map[sid])
            for sid, aln in zip(out_ids, out_seqs)
        ]
        ba.write_fasta(out_ids, codon_aligned, codon_fasta)
        stdout_preview = f"embedded_babappalign_device:{device}"
        return command, 0, stdout_preview, "", "embedded_babappalign"
    except Exception as exc:
        return command, 1, "", _preview(f"embedded_babappalign_error:{exc}"), ""


def _run_muscle(
    executable: str,
    source_fasta: Path,
    codon_fasta: Path,
    timeout_seconds: int,
    env: dict[str, str] | None = None,
) -> Tuple[List[str], int | None, str, str, str]:
    candidates = [
        [executable, "-align", str(source_fasta), "-output", str(codon_fasta)],
        [executable, "-in", str(source_fasta), "-out", str(codon_fasta)],
    ]
    if "muscle5" in Path(executable).name.lower():
        candidates = candidates[:1]
    stdout_text = ""
    stderr_text = ""
    last_command = candidates[0]
    last_return_code: int | None = None
    for command in candidates:
        if codon_fasta.exists():
            codon_fasta.unlink()
        proc = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout_seconds,
            env=env,
        )
        last_command = command
        last_return_code = proc.returncode
        stdout_text = proc.stdout or ""
        stderr_text = proc.stderr or ""
        if proc.returncode == 0 and codon_fasta.exists() and codon_fasta.stat().st_size > 0:
            return command, proc.returncode, _preview(stdout_text), _preview(stderr_text), str(codon_fasta)
    return last_command, last_return_code, _preview(stdout_text), _preview(stderr_text), ""


def _alignment_subprocess_env(threads: int) -> dict[str, str]:
    """Keep per-family external aligner children from oversubscribing CPU threads."""
    capped = str(max(1, int(threads)))
    env = os.environ.copy()
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "ACCELERATE_MAX_THREADS",
        "BLIS_NUM_THREADS",
        "TORCH_NUM_THREADS",
    ):
        env[name] = capped
    env.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    return env


def _locate_prank_output(prefix: Path) -> Path | None:
    candidates = [
        prefix.with_suffix(".best.fas"),
        prefix.with_suffix(".best.fasta"),
        prefix.with_suffix(".fas"),
        prefix.with_suffix(".fasta"),
    ]
    candidates.extend(prefix.parent.glob(prefix.name + "*.best.fas"))
    candidates.extend(prefix.parent.glob(prefix.name + "*.fas"))
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def _locate_babappalign_output(input_path: Path) -> Path | None:
    candidates = [
        input_path.with_name(input_path.stem + ".codon.aln.fasta"),
        input_path.with_name(input_path.stem + ".codon.aln.fa"),
    ]
    candidates.extend(sorted(input_path.parent.glob(input_path.stem + "*.codon.aln.fasta")))
    candidates.extend(sorted(input_path.parent.glob("*.codon.aln.fasta")))
    for candidate in candidates:
        if candidate.exists() and candidate.is_file() and candidate.stat().st_size > 0:
            return candidate
    return None


def _fasta_records_from_stdout(stdout: str, family_outdir: Path) -> Dict[str, str]:
    if not stdout or ">" not in stdout:
        return {}
    fasta_text = stdout[stdout.find(">") :]
    candidate = family_outdir / ".babappalign.stdout_candidate.fasta"
    candidate.write_text(fasta_text, encoding="utf-8")
    try:
        return read_fasta(candidate)
    except ValueError:
        return {}


def _validate_codon_frame(
    path: Path, expected_taxa: set[str] | None = None
) -> Tuple[Dict[str, str], List[str]]:
    warnings: List[str] = []
    try:
        records = read_fasta(path)
    except ValueError as exc:
        return {}, [f"unreadable_fasta:{path}:{exc}"]
    lengths = [len(sequence) for sequence in records.values()]
    if not records:
        warnings.append(f"empty_alignment:{path}")
    if any(not sequence for sequence in records.values()):
        warnings.append(f"empty_sequence:{path}")
    if expected_taxa is not None:
        missing = sorted(expected_taxa - set(records))
        if missing:
            warnings.append(f"missing_expected_taxa:{path}:{','.join(missing)}")
    if len(set(lengths)) > 1:
        warnings.append(f"unequal_alignment_lengths:{path}")
    if any(length % 3 != 0 for length in lengths):
        warnings.append(f"alignment_length_not_divisible_by_3:{path}")
    for taxon, sequence in records.items():
        for start in range(0, len(sequence), 3):
            codon = sequence[start : start + 3]
            if "-" in codon and codon != "---":
                warnings.append(f"partial_gap_codon:{path}:{taxon}:{start // 3}")
                break
    return records, warnings


def _write_alignment_map(records: Dict[str, str], path: Path, method: str) -> None:
    n_codons = 0 if not records else len(next(iter(records.values()))) // 3
    with path.open("w", encoding="utf-8") as handle:
        handle.write("alignment_column_0based\thomology_id\tnote\n")
        for codon_index in range(n_codons):
            handle.write(
                f"{codon_index}\tH{codon_index + 1:06d}\texternal_{method}_schema\n"
            )


def _qc_payload(
    family_id: str,
    method: str,
    source_fasta: Path,
    command: List[str],
    return_code: int | None,
    runtime_seconds: float,
    status: str,
    warnings: List[str],
    stdout_preview: str = "",
    stderr_preview: str = "",
    output_source: str = "",
) -> dict:
    return {
        "family_id": family_id,
        "method": method,
        "source_fasta": str(source_fasta),
        "command": command,
        "return_code": return_code,
        "runtime_seconds": runtime_seconds,
        "status": status,
        "warnings": warnings,
        "stdout_preview": stdout_preview,
        "stderr_preview": stderr_preview,
        "output_source": output_source,
        "frame_status": "ok" if status == "ok" else "fail",
    }


def _preview(text: str | None, limit: int = 500) -> str:
    return (text or "").replace("\r", "\\r")[:limit]


def _is_babappalign_model_missing(*texts: str) -> bool:
    joined = "\n".join(text or "" for text in texts).lower()
    return (
        "required babappascore model is missing" in joined
        or "babappascore.pt" in joined
    )


def _babappalign_model_missing_message(status: dict | None = None) -> str:
    status = status or babappalign_model_status()
    return (
        "babappalign_model_missing: required BABAPPAScore model is missing. "
        f"Expected file: {status['model_expected_path']}. "
        f"Install with: {status['install_command']}"
    )


def _write_smoke_aligner_reports(json_path: Path, markdown_path: Path, payload: dict) -> None:
    _write_json(json_path, payload)
    lines = [
        "# BABAPPA aligner smoke report",
        "",
        f"- method: {payload.get('method')}",
        f"- status: {payload.get('status')}",
        f"- reason: {payload.get('reason')}",
        f"- executable: {payload.get('executable') or ''}",
    ]
    if payload.get("method") == "babappalign":
        lines.extend(
            [
                f"- model_expected_path: {payload.get('model_expected_path')}",
                f"- model_present: {payload.get('model_present')}",
                f"- model_size_bytes: {payload.get('model_size_bytes')}",
            ]
        )
        if payload.get("install_command"):
            lines.append(f"- install_command: `{payload.get('install_command')}`")
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON file is not an object: {path}")
    return payload


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
