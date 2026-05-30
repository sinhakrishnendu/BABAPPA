"""Optional PyTorch environment inspection for BABAPPA neural workflows."""

from __future__ import annotations

import os
import platform
import sys
from types import ModuleType
from typing import Optional, Tuple


VALID_DEVICES = {"auto", "cpu", "cuda", "mps"}
CUDA_ENV_VARS = (
    "CUDA_VISIBLE_DEVICES",
    "CUDA_HOME",
    "CUDA_PATH",
    "NVIDIA_VISIBLE_DEVICES",
)


def safe_import_torch() -> Tuple[Optional[ModuleType], Optional[str]]:
    """Import torch if available without making it a hard dependency."""
    try:
        import torch  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on local environment
        return None, repr(exc)
    return torch, None


def get_torch_environment(prefer_mps: bool = False) -> dict:
    """Inspect PyTorch availability and accelerator support."""
    torch, error = safe_import_torch()
    warnings = []
    platform_system = platform.system()
    platform_machine = platform.machine()
    macos_version = platform.mac_ver()[0] if platform_system == "Darwin" else ""
    mps_fallback_env = os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK")
    mps_high_watermark_env = os.environ.get("PYTORCH_MPS_HIGH_WATERMARK_RATIO")
    if platform_system == "Darwin":
        for name in CUDA_ENV_VARS:
            if os.environ.get(name):
                warnings.append(f"cuda_env_var_set_on_mac:{name}")
    if torch is None:
        warnings.append(f"torch_import_failed:{error}")
        return {
            "platform_system": platform_system,
            "platform_machine": platform_machine,
            "macos_version": macos_version,
            "python_executable": sys.executable,
            "torch_available": False,
            "torch_version": None,
            "cuda_available": False,
            "cuda_device_count": 0,
            "cuda_device_names": [],
            "mps_built": False,
            "mps_available": False,
            "recommended_device": "unavailable",
            "mps_fallback_env": mps_fallback_env,
            "mps_high_watermark_env": mps_high_watermark_env,
            "warnings": warnings,
        }

    cuda_available = bool(torch.cuda.is_available())
    cuda_device_count = int(torch.cuda.device_count()) if cuda_available else 0
    cuda_device_names = []
    if cuda_available:
        for device_index in range(cuda_device_count):
            cuda_device_names.append(str(torch.cuda.get_device_name(device_index)))

    mps_built = is_mps_built(torch)
    try:
        mps_available = is_mps_available(torch)
    except Exception as exc:  # pragma: no cover - backend-specific
        mps_available = False
        warnings.append(f"mps_check_failed:{exc}")

    recommended_device = resolve_torch_device(
        torch,
        "auto",
        prefer_mps=prefer_mps,
    )

    return {
        "platform_system": platform_system,
        "platform_machine": platform_machine,
        "macos_version": macos_version,
        "python_executable": sys.executable,
        "torch_available": True,
        "torch_version": str(getattr(torch, "__version__", "unknown")),
        "cuda_available": cuda_available,
        "cuda_device_count": cuda_device_count,
        "cuda_device_names": cuda_device_names,
        "mps_built": mps_built,
        "mps_available": mps_available,
        "recommended_device": recommended_device,
        "mps_fallback_env": mps_fallback_env,
        "mps_high_watermark_env": mps_high_watermark_env,
        "warnings": warnings,
    }


def resolve_torch_device(torch: ModuleType, requested: str, prefer_mps: bool = False) -> str:
    """Resolve a requested device string to a usable PyTorch device name."""
    if requested not in VALID_DEVICES:
        allowed = ", ".join(sorted(VALID_DEVICES))
        raise ValueError(f"device must be one of: {allowed}")
    cuda_available = bool(torch.cuda.is_available())
    mps_available = is_mps_available(torch)
    if requested == "auto":
        if prefer_mps and mps_available:
            return "mps"
        if cuda_available:
            return "cuda"
        if mps_available:
            return "mps"
        return "cpu"
    if requested == "cuda":
        if not cuda_available:
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
        return "cuda"
    if requested == "mps":
        if not mps_available:
            raise RuntimeError(
                "MPS was requested but torch.backends.mps.is_available() is false. "
                "Use --device cpu, or run on an Apple Silicon Python/PyTorch build with MPS support."
            )
        return "mps"
    return "cpu"


def is_mps_built(torch: ModuleType) -> bool:
    """Return whether the local PyTorch build includes the MPS backend."""
    backends = getattr(torch, "backends", None)
    mps = getattr(backends, "mps", None) if backends is not None else None
    if mps is None:
        return False
    is_built = getattr(mps, "is_built", None)
    if is_built is None:
        return True
    try:
        return bool(is_built())
    except Exception:
        return False


def is_mps_available(torch: ModuleType) -> bool:
    """Return whether the MPS backend is available for execution."""
    backends = getattr(torch, "backends", None)
    mps = getattr(backends, "mps", None) if backends is not None else None
    if mps is None:
        return False
    is_available = getattr(mps, "is_available", None)
    if is_available is None:
        return False
    return bool(is_available())


def torch_device(torch: ModuleType, requested: str, prefer_mps: bool = False):
    """Return a torch.device from the shared BABAPPA resolver."""
    return torch.device(resolve_torch_device(torch, requested, prefer_mps=prefer_mps))


def mps_runtime_guidance(exc: BaseException) -> str:
    """Append a compact Apple Silicon fallback hint to an MPS runtime error."""
    return (
        f"{exc}\n"
        "MPS execution failed. Retry with PYTORCH_ENABLE_MPS_FALLBACK=1, "
        "lower --batch-size, or rerun the neural stage with --device cpu."
    )


def maybe_clear_device_cache(torch: ModuleType, device: str) -> None:
    """Clear accelerator caches without calling CUDA APIs for MPS runs."""
    device_name = str(device)
    if device_name.startswith("cuda") and hasattr(torch, "cuda"):
        torch.cuda.empty_cache()
    elif device_name == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()


def format_torch_environment_text(env: dict) -> str:
    """Format a compact human-readable PyTorch environment summary."""
    cuda_names = env.get("cuda_device_names") or []
    lines = [
        f"platform_system: {env.get('platform_system')}",
        f"platform_machine: {env.get('platform_machine')}",
        f"macos_version: {env.get('macos_version') or 'n/a'}",
        f"python_executable: {env.get('python_executable')}",
        f"torch_available: {env.get('torch_available')}",
        f"torch_version: {env.get('torch_version')}",
        f"cuda_available: {env.get('cuda_available')}",
        f"cuda_device_count: {env.get('cuda_device_count')}",
        f"cuda_device_names: {', '.join(cuda_names) if cuda_names else 'none'}",
        f"mps_built: {env.get('mps_built')}",
        f"mps_available: {env.get('mps_available')}",
        f"recommended_device: {env.get('recommended_device')}",
        f"PYTORCH_ENABLE_MPS_FALLBACK: {env.get('mps_fallback_env') or 'unset'}",
        f"PYTORCH_MPS_HIGH_WATERMARK_RATIO: {env.get('mps_high_watermark_env') or 'unset'}",
    ]
    warnings = env.get("warnings") or []
    if warnings:
        lines.append(f"warnings: {', '.join(str(warning) for warning in warnings)}")
    return "\n".join(lines)
