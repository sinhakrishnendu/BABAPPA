"""Alignment backend detection for BABAPPA."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Dict, List

INTERNAL_METHODS = {"identity", "codon_dropout"}
EXTERNAL_METHODS = {"mafft", "prank", "babappalign", "muscle", "tcoffee"}
SUPPORTED_METHODS = INTERNAL_METHODS | EXTERNAL_METHODS
BABAPPALIGN_MODEL_URL = "https://zenodo.org/record/18053201/files/babappascore.pt"
BABAPPALIGN_MODEL_RELATIVE_PATH = ".cache/babappalign/models/babappascore.pt"
BABAPPALIGN_MODEL_INSTALL_COMMAND = (
    f'mkdir -p "$HOME/.cache/babappalign/models" && '
    f'curl -L "{BABAPPALIGN_MODEL_URL}" '
    f'-o "$HOME/{BABAPPALIGN_MODEL_RELATIVE_PATH}"'
)


@dataclass(frozen=True)
class AlignerBackend:
    """Detected alignment backend metadata."""

    name: str
    executable: str | None
    available: bool
    version: str | None
    kind: str
    notes: List[str]
    wrapper_status: str | None = None
    command_template: str | None = None
    runtime_class: str = "moderate"
    production_default: bool = False
    mapped_oracle_default: bool = False
    default_role: str = "optional"
    model_expected_path: str | None = None
    model_present: bool | None = None
    model_size_bytes: int | None = None
    model_status: str | None = None
    install_command: str | None = None

    def as_dict(self) -> dict:
        return asdict(self)


def detect_aligner_backends() -> Dict[str, AlignerBackend]:
    """Detect internal and optional external alignment backends."""
    backends: Dict[str, AlignerBackend] = {
        "identity": AlignerBackend(
            name="identity",
            executable=None,
            available=True,
            version=None,
            kind="internal",
            notes=["internal_identity_alignment"],
            wrapper_status="active",
            command_template="internal identity",
            runtime_class="fast",
            production_default=True,
            mapped_oracle_default=True,
            default_role="production_default",
        ),
        "codon_dropout": AlignerBackend(
            name="codon_dropout",
            executable=None,
            available=True,
            version=None,
            kind="internal",
            notes=["internal_codon_dropout_alignment"],
            wrapper_status="active_unmappable_noise_control",
            command_template="internal codon_dropout",
            runtime_class="fast",
            production_default=False,
            mapped_oracle_default=False,
            default_role="diagnostic_unmappable_noise_control",
        ),
    }
    backends["mafft"] = _detect_external(
        "mafft",
        ["mafft"],
        [["--version"]],
        runtime_class="fast",
        production_default=True,
        mapped_oracle_default=True,
        default_role="production_default",
    )
    backends["prank"] = _detect_external(
        "prank",
        ["prank"],
        [["--help"], ["-help"]],
        runtime_class="diagnostic",
        production_default=False,
        mapped_oracle_default=False,
        default_role="diagnostic_slow",
    )
    backends["babappalign"] = _with_babappalign_model_status(
        _detect_external(
            "babappalign",
            ["babappalign", "babappalign-cli"],
            [["--version"], ["-h"]],
            runtime_class="moderate",
            production_default=True,
            mapped_oracle_default=True,
            default_role="production_default",
        )
    )
    backends["muscle"] = _detect_external(
        "muscle",
        ["muscle", "muscle5"],
        [["-version"], ["--version"], ["-h"]],
        runtime_class="fast",
        production_default=True,
        mapped_oracle_default=True,
        default_role="production_default",
    )
    backends["tcoffee"] = _detect_external(
        "tcoffee",
        ["t_coffee", "tcoffee"],
        [["-version"], ["--version"], ["-h"]],
        runtime_class="diagnostic",
        production_default=False,
        mapped_oracle_default=False,
        default_role="optional_diagnostic",
    )
    return backends


def babappalign_model_path(home: str | Path | None = None) -> Path:
    """Return the expected BABAPPAScore model cache path."""
    root = Path(home).expanduser() if home is not None else Path.home()
    return root / BABAPPALIGN_MODEL_RELATIVE_PATH


def babappalign_model_status(home: str | Path | None = None) -> dict:
    """Return BABAPPAlign model-cache status without attempting a download."""
    path = babappalign_model_path(home)
    present = False
    size_bytes = None
    try:
        present = path.is_file() and path.stat().st_size > 0
        if present:
            size_bytes = path.stat().st_size
    except OSError:
        present = False
        size_bytes = None
    return {
        "model_expected_path": str(path),
        "model_present": present,
        "model_size_bytes": size_bytes,
        "model_status": "model_present" if present else "model_missing",
        "install_command": None if present else BABAPPALIGN_MODEL_INSTALL_COMMAND,
        "model_url": BABAPPALIGN_MODEL_URL,
    }


def supported_alignment_methods(include_unavailable: bool = False) -> List[str]:
    """Return supported method names, optionally including unavailable externals."""
    backends = detect_aligner_backends()
    return [
        name
        for name in sorted(backends)
        if include_unavailable or backends[name].available
    ]


def validate_alignment_methods(
    methods: List[str], require_available: bool = False
) -> List[str]:
    """Validate method names and optional external availability.

    Returns warnings for unavailable optional methods when ``require_available`` is
    false. Raises ``ValueError`` for unknown methods or required unavailable tools.
    """
    if not methods:
        raise ValueError("methods must be non-empty")
    normalized = [str(method).strip() for method in methods if str(method).strip()]
    unknown = sorted(set(normalized) - SUPPORTED_METHODS)
    if unknown:
        allowed = ", ".join(sorted(SUPPORTED_METHODS))
        raise ValueError(
            f"unknown alignment method(s): {', '.join(unknown)}; allowed: {allowed}"
        )
    backends = detect_aligner_backends()
    warnings: List[str] = []
    unavailable = [
        method
        for method in normalized
        if not backends.get(
            method,
            AlignerBackend(method, None, False, None, "external", []),
        ).available
    ]
    if unavailable and require_available:
        raise ValueError(
            "requested alignment method executable is unavailable: "
            + ", ".join(sorted(unavailable))
        )
    for method in sorted(unavailable):
        warnings.append(f"method_unavailable:{method}")
    return warnings


def _with_babappalign_model_status(backend: AlignerBackend) -> AlignerBackend:
    status = babappalign_model_status()
    notes = list(backend.notes)
    if status["model_status"] not in notes:
        notes.append(status["model_status"])
    return replace(
        backend,
        notes=notes,
        model_expected_path=status["model_expected_path"],
        model_present=status["model_present"],
        model_size_bytes=status["model_size_bytes"],
        model_status=status["model_status"],
        install_command=status["install_command"],
    )


def write_aligner_status(
    path: str | Path, backends: Dict[str, AlignerBackend] | None = None
) -> None:
    """Write detected backend status as JSON."""
    payload = {
        name: backend.as_dict()
        for name, backend in (backends or detect_aligner_backends()).items()
    }
    Path(path).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _detect_external(
    name: str,
    executable_names: List[str],
    version_args: List[List[str]],
    runtime_class: str,
    production_default: bool,
    mapped_oracle_default: bool,
    default_role: str,
) -> AlignerBackend:
    notes: List[str] = []
    executable = None
    for candidate in executable_names:
        resolved = shutil.which(candidate)
        if resolved:
            executable = resolved
            if candidate != name:
                notes.append(f"resolved_executable_name:{candidate}")
            break
    if executable is None:
        return AlignerBackend(
            name=name,
            executable=None,
            available=False,
            version=None,
            kind="external",
            notes=["executable_not_found"],
            wrapper_status=_wrapper_status(name),
            command_template=_command_template(name),
            runtime_class=runtime_class,
            production_default=False,
            mapped_oracle_default=False,
            default_role=default_role,
        )

    version = None
    for args in version_args:
        try:
            proc = subprocess.run(
                [executable, *args],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=5,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            notes.append(f"version_probe_failed:{type(exc).__name__}:{' '.join(args)}")
            continue
        text = (proc.stdout or proc.stderr or "").strip()
        if text:
            version = text.splitlines()[0][:200]
            notes.append(f"version_probe:{' '.join(args)}")
            break
        notes.append(f"version_probe_empty:{' '.join(args)}")
    if version is None:
        notes.append("version_unknown")
    if name == "muscle":
        notes.append(_muscle_syntax_note(executable, version))
    return AlignerBackend(
        name=name,
        executable=executable,
        available=True,
        version=version,
        kind="external",
        notes=notes,
        wrapper_status=_wrapper_status(name),
        command_template=_command_template(name),
        runtime_class=runtime_class,
        production_default=production_default,
        mapped_oracle_default=mapped_oracle_default,
        default_role=default_role,
    )


def _muscle_syntax_note(executable: str, version: str | None) -> str:
    text = f"{Path(executable).name} {version or ''}".lower()
    if "muscle5" in text or "muscle v5" in text or "muscle 5" in text:
        return "muscle_syntax:muscle5"
    return "muscle_syntax:auto_probe"


def _wrapper_status(name: str) -> str:
    if name in {"mafft", "prank", "babappalign", "muscle", "tcoffee"}:
        if name == "prank":
            return "active_diagnostic_slow"
        if name == "tcoffee":
            return "active_optional_diagnostic"
        return "active"
    return "unknown"


def _command_template(name: str) -> str | None:
    if name == "mafft":
        return "mafft --auto <input.fasta>"
    if name == "prank":
        return "prank -d=<input.fasta> -o=<output_prefix>"
    if name == "babappalign":
        return "babappalign --mode codon --device cpu|cuda <input.fasta>"
    if name == "muscle":
        return "muscle -align <input.fasta> -output <output.fasta> OR muscle -in <input.fasta> -out <output.fasta>"
    if name == "tcoffee":
        return "t_coffee <input.fasta> -outfile <output.fasta> -output fasta"
    return None
