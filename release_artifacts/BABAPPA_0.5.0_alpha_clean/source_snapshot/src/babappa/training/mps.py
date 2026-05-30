"""Lightweight Apple Silicon / MPS smoke and benchmark helpers."""

from __future__ import annotations

import json
import time
import tracemalloc
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Union

import numpy as np

from babappa import __version__
from babappa.site.baseline import _compute_binary_metrics
from babappa.site.neural_model import SiteMLPClassifier
from babappa.training.neural_env import (
    VALID_DEVICES,
    maybe_clear_device_cache,
    mps_runtime_guidance,
    resolve_torch_device,
    safe_import_torch,
)

MPS_SMOKE_VERSION = __version__
APPLE_SILICON_BENCHMARK_VERSION = __version__


@dataclass(frozen=True)
class MPSTrainingSmokeConfig:
    """Configuration for the tiny MPS training smoke."""

    outdir: str = "mps_smoke"
    dataset_dir: Optional[str] = None
    device: str = "mps"
    batch_size: int = 32
    max_items: int = 512
    seed: int = 42
    threads: int = 8

    def __post_init__(self) -> None:
        if self.device not in VALID_DEVICES:
            allowed = ", ".join(sorted(VALID_DEVICES))
            raise ValueError(f"device must be one of: {allowed}")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.max_items < 2:
            raise ValueError("max_items must be >= 2")
        if self.threads < 1:
            raise ValueError("threads must be >= 1")
        Path(self.outdir).mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class AppleSiliconBenchmarkConfig:
    """Configuration for a lightweight Apple Silicon neural benchmark."""

    outdir: str = "apple_silicon_benchmark"
    device: str = "auto"
    batch_sizes: Union[str, Sequence[int]] = "32,64,128,256"
    max_items: int = 4096
    seed: int = 42
    threads: int = 8
    prefer_mps: bool = False

    def __post_init__(self) -> None:
        if self.device not in VALID_DEVICES:
            allowed = ", ".join(sorted(VALID_DEVICES))
            raise ValueError(f"device must be one of: {allowed}")
        if self.max_items < 2:
            raise ValueError("max_items must be >= 2")
        if self.threads < 1:
            raise ValueError("threads must be >= 1")
        batch_sizes = _parse_batch_sizes(self.batch_sizes)
        if not batch_sizes:
            raise ValueError("batch_sizes must not be empty")
        object.__setattr__(self, "batch_sizes", batch_sizes)
        Path(self.outdir).mkdir(parents=True, exist_ok=True)


def run_mps_training_smoke(config: MPSTrainingSmokeConfig) -> dict:
    """Run or skip a tiny forward/backward/checkpoint smoke for MPS."""
    outdir = Path(config.outdir)
    torch, error = safe_import_torch()
    if torch is None:
        return _write_smoke_reports(
            outdir,
            {
                "mps_smoke_version": MPS_SMOKE_VERSION,
                "status": "skipped",
                "reason": f"torch_unavailable:{error}",
                "device_requested": config.device,
                "device_used": None,
                "warnings": ["PyTorch is not available; MPS smoke skipped."],
            },
        )

    try:
        torch.set_num_threads(config.threads)
    except Exception:
        pass
    try:
        device = resolve_torch_device(torch, config.device)
    except RuntimeError as exc:
        return _write_smoke_reports(
            outdir,
            {
                "mps_smoke_version": MPS_SMOKE_VERSION,
                "status": "skipped",
                "reason": str(exc),
                "device_requested": config.device,
                "device_used": None,
                "warnings": [
                    "MPS is unavailable; set up an Apple Silicon PyTorch build or rerun with --device cpu."
                ],
            },
        )

    if config.device == "auto" and device != "mps":
        return _write_smoke_reports(
            outdir,
            {
                "mps_smoke_version": MPS_SMOKE_VERSION,
                "status": "skipped",
                "reason": "auto_did_not_select_mps",
                "device_requested": config.device,
                "device_used": device,
                "warnings": ["MPS is unavailable or lower priority than CUDA; MPS smoke skipped."],
            },
        )

    rng = np.random.default_rng(config.seed)
    n_items = max(config.batch_size, min(config.max_items, max(2, config.batch_size * 2)))
    n_features = 24
    X = rng.normal(size=(n_items, n_features)).astype(np.float32)
    y = rng.integers(0, 2, size=n_items).astype(np.float32)
    batch_x = torch.as_tensor(X[: config.batch_size], dtype=torch.float32, device=device)
    batch_y = torch.as_tensor(y[: config.batch_size], dtype=torch.float32, device=device)
    checkpoint_path = outdir / "mps_smoke_checkpoint.pt"

    try:
        model = SiteMLPClassifier(input_dim=n_features, hidden_dim=16, dropout=0.0).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        loss_fn = torch.nn.BCEWithLogitsLoss()
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch_x)
        loss = loss_fn(logits, batch_y)
        loss.backward()
        optimizer.step()
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        metrics = _compute_binary_metrics(batch_y.detach().cpu().numpy().astype(np.int32), probs, threshold=0.5)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "input_dim": n_features,
                "device_used": device,
                "mps_smoke_version": MPS_SMOKE_VERSION,
            },
            checkpoint_path,
        )
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        checkpoint_read_ok = isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
    except RuntimeError as exc:
        if device == "mps":
            raise RuntimeError(mps_runtime_guidance(exc)) from exc
        raise
    finally:
        maybe_clear_device_cache(torch, device)

    return _write_smoke_reports(
        outdir,
        {
            "mps_smoke_version": MPS_SMOKE_VERSION,
            "status": "ok",
            "reason": None,
            "dataset_dir": config.dataset_dir,
            "device_requested": config.device,
            "device_used": device,
            "batch_size": config.batch_size,
            "max_items": config.max_items,
            "n_items": n_items,
            "loss": float(loss.detach().cpu().item()),
            "metrics": metrics,
            "checkpoint": str(checkpoint_path),
            "checkpoint_read_ok": checkpoint_read_ok,
            "warnings": [] if device == "mps" else ["Smoke ran on CPU by explicit request, not on MPS."],
        },
    )


def validate_mps_smoke_dir(smoke_dir: str | Path) -> dict:
    """Validate MPS smoke reports."""
    path = Path(smoke_dir)
    report_json = path / "mps_smoke_report.json"
    report_md = path / "mps_smoke_report.md"
    failures: List[str] = []
    warnings: List[str] = []
    if not report_json.exists():
        failures.append(f"missing_file:{report_json}")
        payload = {}
    else:
        try:
            payload = json.loads(report_json.read_text("utf-8"))
        except json.JSONDecodeError as exc:
            failures.append(f"invalid_json:{exc}")
            payload = {}
    if not report_md.exists():
        failures.append(f"missing_file:{report_md}")
    status = payload.get("status")
    if status not in {"ok", "skipped"}:
        failures.append(f"invalid_status:{status}")
    if status == "ok":
        checkpoint = payload.get("checkpoint")
        if not checkpoint or not Path(checkpoint).exists():
            failures.append(f"missing_checkpoint:{checkpoint}")
        if payload.get("checkpoint_read_ok") is not True:
            failures.append("checkpoint_read_failed")
    if status == "skipped":
        warnings.append(str(payload.get("reason", "skipped")))
    return {
        "status": "fail" if failures else status or "fail",
        "smoke_dir": str(path),
        "device_used": payload.get("device_used"),
        "n_fail": len(failures),
        "n_warning": len(warnings),
        "failures": failures,
        "warnings": warnings,
    }


def run_apple_silicon_benchmark(config: AppleSiliconBenchmarkConfig) -> dict:
    """Run a synthetic branch-neural benchmark; it is not a scientific metric."""
    outdir = Path(config.outdir)
    torch, error = safe_import_torch()
    if torch is None:
        return _write_benchmark_reports(
            outdir,
            {
                "apple_silicon_benchmark_version": APPLE_SILICON_BENCHMARK_VERSION,
                "status": "skipped",
                "reason": f"torch_unavailable:{error}",
                "device_requested": config.device,
                "device_used": None,
                "rows": [],
                "recommended_batch_size": None,
                "warnings": ["PyTorch is not available; benchmark skipped."],
            },
        )
    try:
        torch.set_num_threads(config.threads)
    except Exception:
        pass
    try:
        device = resolve_torch_device(torch, config.device, prefer_mps=config.prefer_mps)
    except RuntimeError as exc:
        return _write_benchmark_reports(
            outdir,
            {
                "apple_silicon_benchmark_version": APPLE_SILICON_BENCHMARK_VERSION,
                "status": "skipped",
                "reason": str(exc),
                "device_requested": config.device,
                "device_used": None,
                "rows": [],
                "recommended_batch_size": None,
                "warnings": ["Requested accelerator is unavailable; rerun with --device cpu."],
            },
        )

    warnings = []
    if device != "mps":
        warnings.append("MPS was unavailable or not selected; benchmark ran as a lightweight CPU/CUDA fallback.")
    rows = []
    rng = np.random.default_rng(config.seed)
    n_items = int(config.max_items)
    n_features = 32
    X = rng.normal(size=(n_items, n_features)).astype(np.float32)
    y = rng.integers(0, 2, size=n_items).astype(np.float32)

    for batch_size in config.batch_sizes:
        tracemalloc.start()
        start = time.perf_counter()
        row = {"batch_size": int(batch_size), "status": "ok"}
        try:
            model = SiteMLPClassifier(input_dim=n_features, hidden_dim=32, dropout=0.0).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
            loss_fn = torch.nn.BCEWithLogitsLoss()
            for start_index in range(0, n_items, batch_size):
                batch_x = torch.as_tensor(X[start_index:start_index + batch_size], dtype=torch.float32, device=device)
                batch_y = torch.as_tensor(y[start_index:start_index + batch_size], dtype=torch.float32, device=device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(batch_x)
                loss = loss_fn(logits, batch_y)
                loss.backward()
                optimizer.step()
            elapsed = max(time.perf_counter() - start, 1e-9)
            _current, peak = tracemalloc.get_traced_memory()
            row.update(
                {
                    "rows": n_items,
                    "seconds": elapsed,
                    "throughput_rows_per_sec": n_items / elapsed,
                    "peak_python_memory_bytes": int(peak),
                }
            )
        except RuntimeError as exc:
            row.update({"status": "failed", "error": str(exc), "throughput_rows_per_sec": 0.0})
            if device == "mps":
                row["guidance"] = mps_runtime_guidance(exc)
        finally:
            tracemalloc.stop()
            maybe_clear_device_cache(torch, device)
        rows.append(row)

    recommended = _recommend_batch_size(rows, device)
    return _write_benchmark_reports(
        outdir,
        {
            "apple_silicon_benchmark_version": APPLE_SILICON_BENCHMARK_VERSION,
            "status": "ok",
            "reason": None,
            "device_requested": config.device,
            "device_used": device,
            "batch_sizes": list(config.batch_sizes),
            "max_items": config.max_items,
            "rows": rows,
            "recommended_batch_size": recommended,
            "warnings": warnings,
            "note": "Synthetic throughput helper only; do not use for scientific metrics.",
        },
    )


def _recommend_batch_size(rows: List[dict], device: str) -> Optional[int]:
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    if not ok_rows:
        return None
    if device == "mps":
        conservative = [row for row in ok_rows if int(row["batch_size"]) <= 128]
        if conservative:
            return int(max(conservative, key=lambda row: int(row["batch_size"]))["batch_size"])
    best = max(ok_rows, key=lambda row: float(row.get("throughput_rows_per_sec") or 0.0))
    return int(best["batch_size"])


def _write_smoke_reports(outdir: Path, payload: dict) -> dict:
    json_path = outdir / "mps_smoke_report.json"
    md_path = outdir / "mps_smoke_report.md"
    payload["files"] = {"json": str(json_path), "markdown": str(md_path)}
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(_render_smoke_markdown(payload), encoding="utf-8")
    return {**payload, "outdir": str(outdir), "json": str(json_path), "markdown": str(md_path)}


def _write_benchmark_reports(outdir: Path, payload: dict) -> dict:
    json_path = outdir / "apple_silicon_benchmark_report.json"
    md_path = outdir / "apple_silicon_benchmark_report.md"
    payload["files"] = {"json": str(json_path), "markdown": str(md_path)}
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(_render_benchmark_markdown(payload), encoding="utf-8")
    return {**payload, "outdir": str(outdir), "json": str(json_path), "markdown": str(md_path)}


def _render_smoke_markdown(payload: dict) -> str:
    return "\n".join(
        [
            "# BABAPPA MPS smoke report",
            "",
            f"- Status: {payload.get('status')}",
            f"- Device requested: {payload.get('device_requested')}",
            f"- Device used: {payload.get('device_used')}",
            f"- Batch size: {payload.get('batch_size')}",
            f"- Checkpoint read ok: {payload.get('checkpoint_read_ok')}",
            f"- Warnings: {', '.join(payload.get('warnings') or []) or 'none'}",
            "",
            "This is a lightweight portability smoke, not a scientific benchmark.",
            "",
        ]
    )


def _render_benchmark_markdown(payload: dict) -> str:
    lines = [
        "# BABAPPA Apple Silicon benchmark",
        "",
        f"- Status: {payload.get('status')}",
        f"- Device requested: {payload.get('device_requested')}",
        f"- Device used: {payload.get('device_used')}",
        f"- Recommended batch size: {payload.get('recommended_batch_size')}",
        f"- Warnings: {', '.join(payload.get('warnings') or []) or 'none'}",
        "",
        "| batch_size | status | rows/sec | peak Python memory |",
        "|---:|---|---:|---:|",
    ]
    for row in payload.get("rows") or []:
        lines.append(
            f"| {row.get('batch_size')} | {row.get('status')} | "
            f"{float(row.get('throughput_rows_per_sec') or 0.0):.2f} | "
            f"{int(row.get('peak_python_memory_bytes') or 0)} |"
        )
    lines.extend(["", "Synthetic helper only; do not use for scientific metrics.", ""])
    return "\n".join(lines)


def _parse_batch_sizes(value: Union[str, Sequence[int]]) -> List[int]:
    if isinstance(value, str):
        items = [item.strip() for item in value.split(",") if item.strip()]
        return [int(item) for item in items]
    return [int(item) for item in value]
