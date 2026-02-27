from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .io import Alignment, read_fasta
from .phylo import parse_newick


CODEML_DOCKER_IMAGE = "quay.io/biocontainers/paml:4.10.7--hdfd78af_0"
HYPHY_DOCKER_IMAGE = "stevenweaver/hyphy:2.5.63"


@dataclass
class CommandOutcome:
    returncode: int
    stdout: str
    stderr: str
    runtime_sec: float
    command: str


@dataclass
class AdapterRecord:
    method: str
    status: str
    reason: str
    p_value: float | None
    lrt_stat: float | None
    lnl1: float | None
    lnl0: float | None
    runtime_sec: float
    stderr_tail: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "status": self.status,
            "reason": self.reason,
            "p_value": self.p_value,
            "lrt_stat": self.lrt_stat,
            "lnL1": self.lnl1,
            "lnL0": self.lnl0,
            "runtime_sec": self.runtime_sec,
            "stderr_tail": self.stderr_tail,
        }


def _stderr_tail(stderr: str, n: int = 40) -> str:
    lines = stderr.strip().splitlines()
    if not lines:
        return ""
    return "\n".join(lines[-n:])


def _coerce_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except Exception:
            return None
    return None


def _find_numeric_by_key(payload: Any, key_tokens: tuple[str, ...]) -> float | None:
    if isinstance(payload, dict):
        for key, value in payload.items():
            key_l = str(key).lower()
            if any(token in key_l for token in key_tokens):
                found = _coerce_float(value)
                if found is not None:
                    return found
            found_nested = _find_numeric_by_key(value, key_tokens)
            if found_nested is not None:
                return found_nested
    elif isinstance(payload, list):
        for item in payload:
            found = _find_numeric_by_key(item, key_tokens)
            if found is not None:
                return found
    return None


def _parse_pvalue_from_text(text: str) -> float | None:
    # Avoid matching values in scientific model names by requiring explicit p-value label.
    m = re.search(
        r"p(?:[-_\s]?value)?\s*[:=]\s*([0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)",
        text,
        flags=re.IGNORECASE,
    )
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def _parse_lrt_from_text(text: str) -> float | None:
    m = re.search(
        r"(?:LRT|likelihood\s+ratio(?:\s+test)?)\s*[:=]?\s*([0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)",
        text,
        flags=re.IGNORECASE,
    )
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _run_cmd(cmd: list[str], cwd: Path, timeout_sec: int = 1800) -> CommandOutcome:
    started = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=timeout_sec,
    )
    runtime = time.perf_counter() - started
    return CommandOutcome(
        returncode=proc.returncode,
        stdout=proc.stdout,
        stderr=proc.stderr,
        runtime_sec=float(runtime),
        command=" ".join(cmd),
    )


def _select_backend(
    *,
    local_bins: list[str],
    docker_image: str,
    env_bin_key: str,
    env_backend_key: str,
    env_sif_key: str,
) -> tuple[str, str]:
    def _project_local_bin(name: str) -> str | None:
        candidates = [
            Path.cwd() / ".conda_env" / "bin" / name,
            Path(__file__).resolve().parents[2] / ".conda_env" / "bin" / name,
        ]
        for candidate in candidates:
            if candidate.exists() and os.access(candidate, os.X_OK):
                return str(candidate.resolve())
        return None

    backend = os.environ.get(env_backend_key, "auto").strip().lower()
    explicit_bin = os.environ.get(env_bin_key)
    explicit_sif = os.environ.get(env_sif_key)

    if backend == "auto":
        # Container-first policy for portable/reproducible baseline execution.
        if explicit_bin:
            return ("local", explicit_bin)
        if shutil.which("docker"):
            return ("docker", docker_image)
        if shutil.which("singularity"):
            sif_ref = explicit_sif if explicit_sif else f"docker://{docker_image}"
            return ("singularity", sif_ref)
        if explicit_bin:
            return ("local", explicit_bin)
        for name in local_bins:
            local_path = _project_local_bin(name)
            if local_path:
                return ("local", local_path)
            path = shutil.which(name)
            if path:
                return ("local", path)
        raise RuntimeError(
            "No runnable backend found (docker/singularity/local unavailable). "
            f"Tried bins={local_bins}."
        )

    if backend == "local":
        if explicit_bin:
            return ("local", explicit_bin)
        for name in local_bins:
            local_path = _project_local_bin(name)
            if local_path:
                return ("local", local_path)
            path = shutil.which(name)
            if path:
                return ("local", path)
        raise RuntimeError(f"{env_bin_key} not set and local binary not found.")

    if backend == "docker":
        if shutil.which("docker"):
            return ("docker", docker_image)
        raise RuntimeError("docker requested but docker executable was not found.")

    if backend == "singularity":
        if shutil.which("singularity"):
            sif_ref = explicit_sif if explicit_sif else f"docker://{docker_image}"
            return ("singularity", sif_ref)
        raise RuntimeError("singularity requested but singularity executable was not found.")

    raise RuntimeError(
        "No runnable backend found (docker/singularity/local unavailable). "
        f"Tried bins={local_bins}."
    )


def _label_foreground_paml(tree_newick: str, foreground_taxon: str) -> str:
    pattern = re.compile(rf"(?<![A-Za-z0-9_.-]){re.escape(foreground_taxon)}(?=\s*:)")
    out, n_sub = pattern.subn(f"{foreground_taxon} #1", tree_newick, count=1)
    if n_sub != 1:
        raise ValueError(
            f"Unable to label foreground taxon '{foreground_taxon}' in tree for codeml."
        )
    return out


def _label_foreground_hyphy(tree_newick: str, foreground_taxon: str) -> str:
    pattern = re.compile(rf"(?<![A-Za-z0-9_.-]){re.escape(foreground_taxon)}(?=\s*:)")
    out, n_sub = pattern.subn(f"{foreground_taxon}{{Foreground}}", tree_newick, count=1)
    if n_sub != 1:
        raise ValueError(
            f"Unable to label foreground taxon '{foreground_taxon}' in tree for HyPhy RELAX."
        )
    return out


def _pick_foreground_taxon(tree_newick: str, preferred: str | None) -> str:
    leaves = parse_newick(tree_newick).leaf_names()
    if preferred:
        if preferred not in leaves:
            raise ValueError(
                f"Requested foreground taxon '{preferred}' was not found in tree leaves."
            )
        return preferred
    return leaves[0]


def _check_name_compatibility(alignment: Alignment) -> None:
    for name in alignment.names:
        if any(ch.isspace() for ch in name):
            raise ValueError(
                "Alignment taxon names with whitespace are not supported by codeml adapter."
            )


def _write_phylip(alignment: Alignment, path: Path) -> None:
    _check_name_compatibility(alignment)
    if int(alignment.length) % 3 != 0:
        raise ValueError(
            f"codeml adapter requires codon alignment length divisible by 3, got {alignment.length}"
        )
    width = 60
    with path.open("w", encoding="utf-8") as handle:
        # codeml expects nucleotide site count here and tolerates wrapped sequence blocks.
        handle.write(f"{alignment.n_sequences} {alignment.length}\n")
        for name, seq in zip(alignment.names, alignment.sequences):
            s = str(seq).upper()
            label = name[:30]
            handle.write(f"{label.ljust(30)} {s[:width]}\n")
            for i in range(width, len(s), width):
                handle.write(f"{'':30} {s[i : i + width]}\n")


def _write_codeml_ctl(
    path: Path,
    *,
    seqfile: str,
    treefile: str,
    outfile: str,
    model: int,
    nssites: int,
    fix_omega: int,
    omega: float,
    codon_freq: int = 2,
) -> None:
    lines = [
        f"seqfile = {seqfile}",
        f"treefile = {treefile}",
        f"outfile = {outfile}",
        "noisy = 0",
        "verbose = 0",
        "runmode = 0",
        "seqtype = 1",
        f"CodonFreq = {codon_freq}",
        "clock = 0",
        "aaDist = 0",
        f"model = {model}",
        f"NSsites = {nssites}",
        "icode = 0",
        "Mgene = 0",
        "fix_kappa = 0",
        "kappa = 2.0",
        f"fix_omega = {fix_omega}",
        f"omega = {omega}",
        "fix_alpha = 1",
        "alpha = 0.0",
        "Malpha = 0",
        "ncatG = 8",
        "getSE = 0",
        "RateAncestor = 0",
        "Small_Diff = 1e-6",
        "cleandata = 0",
        "method = 0",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _extract_lnl_from_mlc(path: Path) -> float:
    text = path.read_text(encoding="utf-8", errors="replace")
    matches = re.findall(
        r"lnL\([^)]*\)\s*:\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)",
        text,
    )
    if not matches:
        raise ValueError(f"Could not parse lnL from {path}")
    return float(matches[-1])


def _chi2_sf(lrt: float, df: int) -> float:
    x = max(float(lrt), 0.0)
    if df == 1:
        # Survival function for chi-square(1): erfc(sqrt(x/2))
        return float(math.erfc(math.sqrt(x / 2.0)))
    if df == 2:
        return float(math.exp(-x / 2.0))
    raise ValueError(f"Unsupported df for chi-square survival function: {df}")


def _codeml_command(backend: tuple[str, str], cwd: Path, ctl_filename: str) -> list[str]:
    kind, value = backend
    if kind == "local":
        return [value, ctl_filename]
    if kind == "docker":
        return [
            "docker",
            "run",
            "--rm",
            "-v",
            f"{cwd}:/work",
            "-w",
            "/work",
            value,
            "codeml",
            ctl_filename,
        ]
    if kind == "singularity":
        return ["singularity", "exec", value, "codeml", ctl_filename]
    raise ValueError(f"Unsupported backend: {backend}")


def _hyphy_command(
    backend: tuple[str, str],
    *,
    cwd: Path,
    analysis: str,
    alignment_file: str,
    tree_file: str,
    output_file: str,
    extra_args: list[str] | None = None,
) -> list[str]:
    extra = [] if extra_args is None else list(extra_args)
    kind, value = backend
    args = [
        analysis,
        "--alignment",
        alignment_file,
        "--tree",
        tree_file,
        "--output",
        output_file,
    ] + extra
    if kind == "local":
        return [value] + args
    if kind == "docker":
        return [
            "docker",
            "run",
            "--rm",
            "-v",
            f"{cwd}:/work",
            "-w",
            "/work",
            value,
            "hyphy",
        ] + args
    if kind == "singularity":
        return ["singularity", "exec", value, "hyphy"] + args
    raise ValueError(f"Unsupported backend: {backend}")


def _ok_record(
    *,
    method: str,
    p_value: float,
    lrt_stat: float | None,
    lnl1: float | None,
    lnl0: float | None,
    runtime_sec: float,
    stderr_tail: str,
) -> AdapterRecord:
    return AdapterRecord(
        method=method,
        status="OK",
        reason="ok",
        p_value=float(max(min(p_value, 1.0), 0.0)),
        lrt_stat=None if lrt_stat is None else float(max(lrt_stat, 0.0)),
        lnl1=None if lnl1 is None else float(lnl1),
        lnl0=None if lnl0 is None else float(lnl0),
        runtime_sec=float(runtime_sec),
        stderr_tail=stderr_tail,
    )


def _fail_record(
    *,
    method: str,
    reason: str,
    runtime_sec: float = 0.0,
    stderr_tail: str = "",
) -> AdapterRecord:
    return AdapterRecord(
        method=method,
        status="FAIL",
        reason=reason,
        p_value=None,
        lrt_stat=None,
        lnl1=None,
        lnl0=None,
        runtime_sec=float(runtime_sec),
        stderr_tail=stderr_tail,
    )


def _run_codeml_lrt(
    *,
    method_label: str,
    backend: tuple[str, str],
    workdir: Path,
    alt_ctl: str,
    null_ctl: str,
    alt_mlc: str,
    null_mlc: str,
    df: int,
    timeout_sec: int,
) -> AdapterRecord:
    total_runtime = 0.0
    stderr_chunks: list[str] = []

    for ctl in [null_ctl, alt_ctl]:
        cmd = _codeml_command(backend, workdir, ctl)
        try:
            outcome = _run_cmd(cmd, cwd=workdir, timeout_sec=timeout_sec)
        except Exception as exc:
            return _fail_record(
                method=method_label,
                reason=f"codeml_run_exception:{ctl}:{exc}",
                runtime_sec=total_runtime,
                stderr_tail=_stderr_tail("\n".join(stderr_chunks)),
            )
        total_runtime += outcome.runtime_sec
        if outcome.stderr:
            stderr_chunks.append(outcome.stderr)
        if outcome.returncode != 0:
            return _fail_record(
                method=method_label,
                reason=f"codeml_run_failed:{ctl}:rc={outcome.returncode}",
                runtime_sec=total_runtime,
                stderr_tail=_stderr_tail("\n".join(stderr_chunks)),
            )

    try:
        lnl0 = _extract_lnl_from_mlc(workdir / null_mlc)
        lnl1 = _extract_lnl_from_mlc(workdir / alt_mlc)
    except Exception as exc:
        return _fail_record(
            method=method_label,
            reason=f"codeml_parse_failed:{exc}",
            runtime_sec=total_runtime,
            stderr_tail=_stderr_tail("\n".join(stderr_chunks)),
        )

    lrt = max(0.0, 2.0 * (lnl1 - lnl0))
    p = _chi2_sf(lrt, df=df)
    return _ok_record(
        method=method_label,
        p_value=p,
        lrt_stat=lrt,
        lnl1=lnl1,
        lnl0=lnl0,
        runtime_sec=total_runtime,
        stderr_tail=_stderr_tail("\n".join(stderr_chunks)),
    )


def run_codeml_for_gene(
    *,
    alignment_path: Path,
    tree_path: Path,
    workdir: Path,
    foreground_taxon: str | None,
    run_site_model: bool = True,
    timeout_sec: int = 1800,
) -> list[AdapterRecord]:
    workdir.mkdir(parents=True, exist_ok=True)
    try:
        alignment = read_fasta(alignment_path)
        _write_phylip(alignment, workdir / "alignment.phy")
        tree_text = tree_path.read_text(encoding="utf-8").strip()
        fg_taxon = _pick_foreground_taxon(tree_text, foreground_taxon)
        (workdir / "tree.nwk").write_text(tree_text + "\n", encoding="utf-8")
        (workdir / "tree_fg.nwk").write_text(
            _label_foreground_paml(tree_text, fg_taxon) + "\n",
            encoding="utf-8",
        )
    except Exception as exc:
        return [
            _fail_record(method="codeml_branchsite", reason=f"codeml_input_prep_failed:{exc}"),
            _fail_record(method="codeml_site_m7m8", reason=f"codeml_input_prep_failed:{exc}"),
        ]

    try:
        backend = _select_backend(
            local_bins=["codeml"],
            docker_image=CODEML_DOCKER_IMAGE,
            env_bin_key="BABAPPA_CODEML_BIN",
            env_backend_key="BABAPPA_CODEML_BACKEND",
            env_sif_key="BABAPPA_CODEML_SIF",
        )
    except Exception as exc:
        reason = f"codeml_backend_unavailable:{exc}"
        out = [_fail_record(method="codeml_branchsite", reason=reason)]
        if run_site_model:
            out.append(_fail_record(method="codeml_site_m7m8", reason=reason))
        return out

    _write_codeml_ctl(
        workdir / "branchsite_alt.ctl",
        seqfile="alignment.phy",
        treefile="tree_fg.nwk",
        outfile="branchsite_alt.mlc",
        model=2,
        nssites=2,
        fix_omega=0,
        omega=1.5,
    )
    _write_codeml_ctl(
        workdir / "branchsite_null.ctl",
        seqfile="alignment.phy",
        treefile="tree_fg.nwk",
        outfile="branchsite_null.mlc",
        model=2,
        nssites=2,
        fix_omega=1,
        omega=1.0,
    )
    branchsite = _run_codeml_lrt(
        method_label="codeml_branchsite",
        backend=backend,
        workdir=workdir,
        alt_ctl="branchsite_alt.ctl",
        null_ctl="branchsite_null.ctl",
        alt_mlc="branchsite_alt.mlc",
        null_mlc="branchsite_null.mlc",
        df=1,
        timeout_sec=timeout_sec,
    )

    out = [branchsite]
    if run_site_model:
        _write_codeml_ctl(
            workdir / "m8_alt.ctl",
            seqfile="alignment.phy",
            treefile="tree.nwk",
            outfile="m8_alt.mlc",
            model=0,
            nssites=8,
            fix_omega=0,
            omega=1.5,
        )
        _write_codeml_ctl(
            workdir / "m7_null.ctl",
            seqfile="alignment.phy",
            treefile="tree.nwk",
            outfile="m7_null.mlc",
            model=0,
            nssites=7,
            fix_omega=1,
            omega=1.0,
        )
        site = _run_codeml_lrt(
            method_label="codeml_site_m7m8",
            backend=backend,
            workdir=workdir,
            alt_ctl="m8_alt.ctl",
            null_ctl="m7_null.ctl",
            alt_mlc="m8_alt.mlc",
            null_mlc="m7_null.mlc",
            df=2,
            timeout_sec=timeout_sec,
        )
        out.append(site)
    return out


def _parse_hyphy_output(json_path: Path, stdout: str, stderr: str) -> tuple[float | None, float | None]:
    p: float | None = None
    lrt: float | None = None
    if json_path.exists():
        try:
            payload = _read_json(json_path)
            p = _find_numeric_by_key(payload, ("p-value", "pvalue", "p_value"))
            lrt = _find_numeric_by_key(payload, ("lrt", "likelihood ratio", "test statistic"))
        except Exception:
            pass
    if p is None:
        p = _parse_pvalue_from_text(stdout) or _parse_pvalue_from_text(stderr)
    if lrt is None:
        lrt = _parse_lrt_from_text(stdout) or _parse_lrt_from_text(stderr)
    return p, lrt


def run_hyphy_busted_for_gene(
    *,
    alignment_path: Path,
    tree_path: Path,
    workdir: Path,
    timeout_sec: int = 1800,
) -> AdapterRecord:
    workdir.mkdir(parents=True, exist_ok=True)
    aln_local = workdir / "alignment.fasta"
    tree_local = workdir / "tree.nwk"
    out_json = workdir / "busted.json"
    shutil.copy2(alignment_path, aln_local)
    shutil.copy2(tree_path, tree_local)

    try:
        backend = _select_backend(
            local_bins=["hyphy", "HYPHYMP"],
            docker_image=HYPHY_DOCKER_IMAGE,
            env_bin_key="BABAPPA_HYPHY_BIN",
            env_backend_key="BABAPPA_HYPHY_BACKEND",
            env_sif_key="BABAPPA_HYPHY_SIF",
        )
    except Exception as exc:
        return _fail_record(method="busted", reason=f"hyphy_backend_unavailable:{exc}")

    try:
        cmd = _hyphy_command(
            backend,
            cwd=workdir,
            analysis="busted",
            alignment_file=aln_local.name,
            tree_file=tree_local.name,
            output_file=out_json.name,
        )
        outcome = _run_cmd(cmd, cwd=workdir, timeout_sec=timeout_sec)
    except Exception as exc:
        return _fail_record(method="busted", reason=f"hyphy_busted_run_exception:{exc}")

    if outcome.returncode != 0:
        return _fail_record(
            method="busted",
            reason=f"hyphy_busted_failed:rc={outcome.returncode}",
            runtime_sec=outcome.runtime_sec,
            stderr_tail=_stderr_tail(outcome.stderr),
        )

    p, lrt = _parse_hyphy_output(out_json, outcome.stdout, outcome.stderr)
    if p is None:
        return _fail_record(
            method="busted",
            reason="hyphy_busted_parse_failed:p_value_missing",
            runtime_sec=outcome.runtime_sec,
            stderr_tail=_stderr_tail(outcome.stderr),
        )
    return _ok_record(
        method="busted",
        p_value=p,
        lrt_stat=lrt,
        lnl1=None,
        lnl0=None,
        runtime_sec=outcome.runtime_sec,
        stderr_tail=_stderr_tail(outcome.stderr),
    )


def run_hyphy_relax_for_gene(
    *,
    alignment_path: Path,
    tree_path: Path,
    workdir: Path,
    foreground_taxon: str | None,
    timeout_sec: int = 1800,
) -> AdapterRecord:
    workdir.mkdir(parents=True, exist_ok=True)
    aln_local = workdir / "alignment.fasta"
    tree_local = workdir / "tree_relax.nwk"
    out_json = workdir / "relax.json"
    shutil.copy2(alignment_path, aln_local)

    try:
        tree_text = tree_path.read_text(encoding="utf-8").strip()
        fg = _pick_foreground_taxon(tree_text, foreground_taxon)
        tree_local.write_text(_label_foreground_hyphy(tree_text, fg) + "\n", encoding="utf-8")
    except Exception as exc:
        return _fail_record(method="relax", reason=f"hyphy_relax_tree_prepare_failed:{exc}")

    try:
        backend = _select_backend(
            local_bins=["hyphy", "HYPHYMP"],
            docker_image=HYPHY_DOCKER_IMAGE,
            env_bin_key="BABAPPA_HYPHY_BIN",
            env_backend_key="BABAPPA_HYPHY_BACKEND",
            env_sif_key="BABAPPA_HYPHY_SIF",
        )
    except Exception as exc:
        return _fail_record(method="relax", reason=f"hyphy_backend_unavailable:{exc}")

    try:
        cmd = _hyphy_command(
            backend,
            cwd=workdir,
            analysis="relax",
            alignment_file=aln_local.name,
            tree_file=tree_local.name,
            output_file=out_json.name,
            extra_args=["--test", "Foreground"],
        )
        outcome = _run_cmd(cmd, cwd=workdir, timeout_sec=timeout_sec)
    except Exception as exc:
        return _fail_record(method="relax", reason=f"hyphy_relax_run_exception:{exc}")

    if outcome.returncode != 0:
        return _fail_record(
            method="relax",
            reason=f"hyphy_relax_failed:rc={outcome.returncode}",
            runtime_sec=outcome.runtime_sec,
            stderr_tail=_stderr_tail(outcome.stderr),
        )

    p, lrt = _parse_hyphy_output(out_json, outcome.stdout, outcome.stderr)
    if p is None:
        return _fail_record(
            method="relax",
            reason="hyphy_relax_parse_failed:p_value_missing",
            runtime_sec=outcome.runtime_sec,
            stderr_tail=_stderr_tail(outcome.stderr),
        )
    return _ok_record(
        method="relax",
        p_value=p,
        lrt_stat=lrt,
        lnl1=None,
        lnl0=None,
        runtime_sec=outcome.runtime_sec,
        stderr_tail=_stderr_tail(outcome.stderr),
    )


def run_method_for_gene(
    *,
    method: str,
    alignment_path: Path,
    tree_path: Path,
    workdir: Path,
    foreground_taxon: str | None = None,
    run_site_model: bool = True,
    timeout_sec: int = 1800,
) -> list[AdapterRecord]:
    method_l = method.lower()
    if method_l == "codeml":
        return run_codeml_for_gene(
            alignment_path=alignment_path,
            tree_path=tree_path,
            workdir=workdir,
            foreground_taxon=foreground_taxon,
            run_site_model=run_site_model,
            timeout_sec=timeout_sec,
        )
    if method_l == "busted":
        return [
            run_hyphy_busted_for_gene(
                alignment_path=alignment_path,
                tree_path=tree_path,
                workdir=workdir,
                timeout_sec=timeout_sec,
            )
        ]
    if method_l == "relax":
        return [
            run_hyphy_relax_for_gene(
                alignment_path=alignment_path,
                tree_path=tree_path,
                workdir=workdir,
                foreground_taxon=foreground_taxon,
                timeout_sec=timeout_sec,
            )
        ]
    raise ValueError(f"Unsupported method: {method}")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m babappa.baseline_adapters")
    parser.add_argument("--method", choices=["codeml", "busted", "relax"], required=True)
    parser.add_argument("--alignment", required=True)
    parser.add_argument("--tree", required=True)
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--foreground", default=None)
    parser.add_argument("--no-site-model", action="store_true")
    parser.add_argument("--timeout-sec", type=int, default=1800)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    records = run_method_for_gene(
        method=args.method,
        alignment_path=Path(args.alignment).resolve(),
        tree_path=Path(args.tree).resolve(),
        workdir=Path(args.workdir).resolve(),
        foreground_taxon=args.foreground,
        run_site_model=not bool(args.no_site_model),
        timeout_sec=int(args.timeout_sec),
    )
    payload = [r.to_dict() for r in records]
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
