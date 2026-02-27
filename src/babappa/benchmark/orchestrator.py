from __future__ import annotations

import json
import math
import os
import shutil
import stat
import subprocess
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, as_completed, wait
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .._plotting_env import configure_plotting_env
from ..baseline import load_dataset_json, run_baseline_doctor, run_baseline_for_dataset
from ..benchmark_pack import BenchmarkPackConfig, run_benchmark_pack
from ..doctor import run_doctor
from ..engine import (
    analyze_alignment,
    benjamini_hochberg,
    compute_dispersion,
    train_energy_model_from_neutral_spec,
)
from ..evaluation import simulate_mixture_omega_alignment
from ..hash_utils import sha256_file, sha256_json
from ..io import Alignment, read_fasta
from ..neutral import GY94NeutralSimulator, NeutralSpec
from ..phylo import parse_newick
from ..representation import FEATURE_NAMES, alignment_to_matrix
from ..specs import load_neutral_model_spec
from ..stats import rank_p_value
from ..system_info import get_system_metadata
from ..taxa_normalize import normalize_taxon


_BABAPPA_GENE_SUBPROC_CODE = r"""
import json
import sys

from babappa.benchmark.orchestrator import _run_babappa_gene_worker

payload = json.loads(sys.stdin.read())
task = (
    int(payload["idx"]),
    payload["gene"],
    payload["model_payload"],
    int(payload["calibration_size"]),
    str(payload["tail"]),
    int(payload["seed_gene"]),
)
idx, row, runtime = _run_babappa_gene_worker(task)
print(json.dumps({"idx": idx, "row": row, "runtime": runtime}, allow_nan=True))
"""


@dataclass
class BenchmarkRunConfig:
    track: str
    preset: str
    outdir: Path
    seed: int
    tail: str = "right"
    dataset_json: str | None = None
    include_baselines: bool = True
    include_relax: bool = False
    allow_baseline_fail: bool = False
    allow_qc_fail: bool = False
    baseline_container: str = "auto"
    baseline_timeout_sec: int = 1800
    write_plots: bool = True
    require_real_data: bool = False
    calibration_size_override: int | None = None
    empirical_profile: str | None = None
    mode: str = "pilot"
    min_taxa: int | None = None
    require_full_completeness: bool = False
    foreground_species: str | None = None
    jobs: int = 0
    resume: bool = False


def _is_publication_mode(config: BenchmarkRunConfig) -> bool:
    return str(config.mode).strip().lower() == "publication"


def _combo_symbol(method: str) -> str:
    name = str(method).strip().lower()
    symbols = {
        "babappa": "B",
        "busted": "U",
        "relax": "R",
        "meme": "M",
        "fel": "F",
        "fubar": "Q",
    }
    if name in symbols:
        return symbols[name]
    if not name:
        return "X"
    return name[0].upper()


def _ordered_methods(methods: list[str]) -> list[str]:
    order = {"busted": 0, "relax": 1, "meme": 2, "fel": 3, "fubar": 4}
    uniq = sorted({str(m).strip().lower() for m in methods if str(m).strip()})
    return sorted(uniq, key=lambda m: (order.get(m, 100), m))


def _plot_module() -> Any:
    from . import plots

    return plots


def _get_plt() -> Any:
    configure_plotting_env()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _ensure_dirs(outdir: Path) -> dict[str, Path]:
    dirs = {
        "root": outdir,
        "raw": outdir / "raw",
        "tables": outdir / "tables",
        "figures": outdir / "figures",
        "manifests": outdir / "manifests",
        "logs": outdir / "logs",
        "scripts": outdir / "scripts",
        "report": outdir / "report",
    }
    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)
    return dirs


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_tsv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False)


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        out = float(value)
        if math.isnan(out) or math.isinf(out):
            return None
        return out
    except Exception:
        return None


def _safe_tsv(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return ""
    text = str(value)
    return text.replace("\t", " ").replace("\n", " ")


def _write_json_fsync(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    try:
        dfd = os.open(str(path.parent), os.O_RDONLY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    except Exception:
        # Directory fsync is best effort across platforms/filesystems.
        pass


def _append_progress_row(path: Path, row: dict[str, Any]) -> None:
    header = [
        "idx",
        "gene_id",
        "status",
        "runtime_sec",
        "p_value",
        "I",
        "D_obs",
        "mu_null",
        "sigma_null",
        "timestamp",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    needs_header = (not path.exists()) or (path.stat().st_size == 0)
    with path.open("a", encoding="utf-8", buffering=1) as handle:
        if needs_header:
            handle.write("\t".join(header) + "\n")
        line = "\t".join(_safe_tsv(row.get(col, "")) for col in header)
        handle.write(line + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _write_heartbeat(
    *,
    path: Path,
    completed: int,
    total: int,
    active_workers: int,
    elapsed_sec: float,
    last_gene: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        f"completed={int(completed)}/total={int(total)} "
        f"active_workers={int(active_workers)} "
        f"elapsed_sec={int(max(0, elapsed_sec))} "
        f"last_gene={last_gene}\n"
    )
    with path.open("w", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _load_shards(shards_dir: Path) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    if not shards_dir.exists():
        return out
    for shard in sorted(shards_dir.glob("*.json")):
        try:
            payload = json.loads(shard.read_text(encoding="utf-8"))
            idx = int(payload.get("idx"))
            out[idx] = payload
        except Exception:
            continue
    return out


def _write_rebuild_script(outdir: Path) -> None:
    script = outdir / "scripts" / "rebuild_all.sh"
    repo_root = Path(__file__).resolve().parents[3]
    text = """#!/usr/bin/env bash
set -euo pipefail
PACK_DIR="$(cd "$(dirname "$0")/.." && pwd)"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplconfig_babappa}"
mkdir -p "${MPLCONFIGDIR}"
if python -c "import babappa" >/dev/null 2>&1; then
  python -m babappa.cli benchmark rebuild --outdir "${PACK_DIR}"
elif [ -d "__BABAPPA_REPO_ROOT__/src" ]; then
  export PYTHONPATH="__BABAPPA_REPO_ROOT__/src:${PYTHONPATH:-}"
  python -m babappa.cli benchmark rebuild --outdir "${PACK_DIR}"
else
  echo "Could not import babappa. Install BABAPPA or set PYTHONPATH." >&2
  exit 1
fi
"""
    text = text.replace("__BABAPPA_REPO_ROOT__", str(repo_root))
    script.write_text(text, encoding="utf-8")
    mode = script.stat().st_mode
    script.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _write_checksums(outdir: Path) -> None:
    raw = outdir / "raw"
    rows: list[str] = []
    for path in sorted(raw.rglob("*")):
        if not path.is_file():
            continue
        digest = sha256_file(path)
        rel = path.relative_to(outdir)
        rows.append(f"{digest}  {rel}")
    (outdir / "checksums.txt").write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")


@contextmanager
def _baseline_backend_env(container: str) -> Any:
    mode = str(container).strip().lower()
    if mode == "auto":
        yield
        return
    if mode not in {"docker", "singularity", "local"}:
        raise ValueError(f"Unsupported baseline container mode: {container}")
    keys = ("BABAPPA_CODEML_BACKEND", "BABAPPA_HYPHY_BACKEND")
    old = {k: os.environ.get(k) for k in keys}
    try:
        for k in keys:
            os.environ[k] = mode
        yield
    finally:
        for k in keys:
            prev = old[k]
            if prev is None:
                if k in os.environ:
                    del os.environ[k]
            else:
                os.environ[k] = prev


def _enforce_baseline_doctor(
    *,
    outdir: Path,
    methods: list[str],
    allow_baseline_fail: bool,
    container: str,
    timeout_sec: int,
) -> None:
    wanted = sorted({m.strip().lower() for m in methods if m.strip()})
    if not wanted:
        return
    report = run_baseline_doctor(
        methods=tuple(wanted),
        timeout_sec=int(timeout_sec),
        work_dir=outdir / "raw" / "baseline_doctor",
        pull_images=False,
        container=container,
    )
    (outdir / "logs").mkdir(parents=True, exist_ok=True)
    (outdir / "manifests").mkdir(parents=True, exist_ok=True)
    (outdir / "logs" / "baseline_doctor_report.txt").write_text(report.render() + "\n", encoding="utf-8")
    _write_json(
        outdir / "manifests" / "baseline_doctor_manifest.json",
        {
            "schema_version": 1,
            "methods": wanted,
            "timeout_sec": report.timeout_sec,
            "docker_available": report.docker_available,
            "singularity_available": report.singularity_available,
            "has_failures": report.has_failures,
            "results": [m.to_dict() for m in report.methods],
        },
    )
    if report.has_failures and not allow_baseline_fail:
        raise ValueError(
            "Baseline doctor failed. Run `babappa baseline doctor` and resolve backend issues, "
            "or re-run with --allow-baseline-fail."
        )


def _read_tsv_or_empty(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, sep="\t")
    except Exception:
        return pd.DataFrame()


def _fisher_enrichment(a: int, b: int, c: int, d: int) -> tuple[float, float, float, float]:
    def comb(n: int, k: int) -> int:
        if k < 0 or k > n:
            return 0
        return math.comb(n, k)

    row1 = a + b
    col1 = a + c
    col2 = b + d
    n = row1 + c + d
    denom = comb(n, row1)
    p = 0.0
    upper = min(row1, col1)
    for x in range(a, upper + 1):
        p += (comb(col1, x) * comb(col2, row1 - x)) / max(denom, 1)

    aa, bb, cc, dd = a, b, c, d
    if min(aa, bb, cc, dd) == 0:
        aa += 0.5
        bb += 0.5
        cc += 0.5
        dd += 0.5
    odds = (aa * dd) / max(bb * cc, 1e-12)
    se = math.sqrt(1.0 / aa + 1.0 / bb + 1.0 / cc + 1.0 / dd)
    log_or = math.log(max(odds, 1e-12))
    ci_low = math.exp(log_or - 1.96 * se)
    ci_high = math.exp(log_or + 1.96 * se)
    return float(odds), float(ci_low), float(ci_high), float(min(max(p, 0.0), 1.0))


def _write_alignment_fasta(aln: Alignment, path: Path) -> None:
    lines: list[str] = []
    for name, seq in zip(aln.names, aln.sequences):
        lines.append(f">{name}")
        lines.append(seq)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _mosaic_alignment(a: Alignment, b: Alignment, block_nt: int = 90) -> Alignment:
    if a.names != b.names:
        raise ValueError("Mosaic source alignments must have identical taxon ordering.")
    if a.length != b.length:
        raise ValueError("Mosaic source alignments must have identical length.")
    out_seqs: list[str] = []
    for s1, s2 in zip(a.sequences, b.sequences):
        chunks: list[str] = []
        for start in range(0, a.length, block_nt):
            end = min(start + block_nt, a.length)
            use_first = ((start // block_nt) % 2) == 0
            chunks.append(s1[start:end] if use_first else s2[start:end])
        out_seqs.append("".join(chunks))
    return Alignment(names=a.names, sequences=tuple(out_seqs))


def _balanced_tree_newick(taxa_n: int, branch_length: float = 0.1) -> str:
    labels = [f"T{i+1}" for i in range(taxa_n)]

    def _build(names: list[str], is_root: bool = False) -> str:
        if len(names) == 1:
            return f"{names[0]}:{branch_length}"
        mid = len(names) // 2
        left = _build(names[:mid], is_root=False)
        right = _build(names[mid:], is_root=False)
        if is_root:
            return f"({left},{right})"
        return f"({left},{right}):{branch_length}"

    return _build(labels, is_root=True) + ";"


def _render_dependence_plot(dep_df: pd.DataFrame, out_pdf: Path) -> None:
    plots = _plot_module()
    if dep_df.empty:
        plots.plot_report_page(
            ["Dependence simulation not available."],
            out_pdf,
            "Dependence Sensitivity",
        )
        return
    summary = dep_df.groupby("scenario", as_index=False).agg(
        size_hat=("p", lambda x: float(np.mean(np.asarray(x) <= 0.05)))
    )
    plt = _get_plt()
    fig = plt.figure(figsize=(7, 4))
    ax = fig.add_subplot(111)
    ax.bar(summary["scenario"], summary["size_hat"])
    ax.axhline(0.05, color="r", linestyle="--", linewidth=1)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Empirical size @ 0.05")
    ax.set_title("Dependence/Misspecification Sensitivity")
    fig.tight_layout()
    fig.savefig(out_pdf)
    plt.close(fig)


def _simulation_preset_params(preset: str) -> dict[str, Any]:
    fast = os.environ.get("BABAPPA_BENCHMARK_FAST", "0") == "1"
    p = preset.lower()
    if p == "simulation_null_paper":
        if fast:
            return {
                "L_grid": [300],
                "taxa_grid": [8],
                "N_grid": [99],
                "n_null_genes": 4,
                "n_alt_genes": 0,
                "run_null": True,
                "run_power": False,
                "include_baselines": False,
                "include_relax": False,
                "training_replicates": 20,
                "training_length_nt": 600,
                "kappa": 2.0,
            }
        return {
            "L_grid": [300, 600, 1200],
            "taxa_grid": [8, 16],
            "N_grid": [999, 4999],
            "n_null_genes": 2000,
            "n_alt_genes": 0,
            "run_null": True,
            "run_power": False,
            "include_baselines": False,
            "include_relax": False,
            "training_replicates": 1000,
            "training_length_nt": 1800,
            "kappa": 2.0,
        }
    if p == "simulation_power_paper":
        if fast:
            return {
                "L_grid": [600],
                "taxa_grid": [8],
                "N_grid": [99],
                "n_null_genes": 0,
                "n_alt_genes": 1,
                "run_null": False,
                "run_power": True,
                "include_baselines": True,
                "include_relax": True,
                "training_replicates": 20,
                "training_length_nt": 600,
                "kappa": 2.0,
            }
        return {
            "L_grid": [300, 600, 1200],
            "taxa_grid": [8, 16],
            "N_grid": [999],
            "n_null_genes": 0,
            "n_alt_genes": 500,
            "run_null": False,
            "run_power": True,
            "include_baselines": True,
            "include_relax": True,
            "training_replicates": 1000,
            "training_length_nt": 1800,
            "kappa": 2.0,
        }
    if p in {"null_full", "simulation_null_full"}:
        if fast:
            return {
                "L_grid": [300],
                "taxa_grid": [8],
                "N_grid": [99],
                "n_null_genes": 8,
                "n_alt_genes": 0,
                "run_null": True,
                "run_power": False,
                "include_baselines": False,
                "include_relax": False,
                "training_replicates": 20,
                "training_length_nt": 600,
                "kappa": 2.0,
            }
        return {
            "L_grid": [300, 600, 1200],
            "taxa_grid": [8, 16],
            "N_grid": [999],
            "n_null_genes": 2000,
            "n_alt_genes": 0,
            "run_null": True,
            "run_power": False,
            "include_baselines": False,
            "include_relax": False,
            "training_replicates": 800,
            "training_length_nt": 1800,
            "kappa": 2.0,
        }
    if p in {"power_full", "simulation_power_full"}:
        if fast:
            return {
                "L_grid": [600],
                "taxa_grid": [8],
                "N_grid": [99],
                "n_null_genes": 0,
                "n_alt_genes": 3,
                "run_null": False,
                "run_power": True,
                "include_baselines": True,
                "include_relax": True,
                "training_replicates": 20,
                "training_length_nt": 600,
                "kappa": 2.0,
            }
        return {
            "L_grid": [600],
            "taxa_grid": [8],
            "N_grid": [999],
            "n_null_genes": 0,
            "n_alt_genes": 2000,
            "run_null": False,
            "run_power": True,
            "include_baselines": True,
            "include_relax": True,
            "training_replicates": 800,
            "training_length_nt": 1800,
            "kappa": 2.0,
        }
    if p == "null_smoke":
        return {
            "L_grid": [300, 600],
            "taxa_grid": [8],
            "N_grid": [199, 999],
            "n_null_genes": 200,
            "n_alt_genes": 0,
            "run_null": True,
            "run_power": False,
            "include_baselines": False,
            "include_relax": False,
            "training_replicates": 100,
            "training_length_nt": 1200,
            "kappa": 2.0,
        }
    if p == "power_smoke":
        return {
            "L_grid": [600],
            "taxa_grid": [8],
            "N_grid": [199],
            "n_null_genes": 0,
            "n_alt_genes": 30,
            "run_null": False,
            "run_power": True,
            "include_baselines": True,
            "include_relax": False,
            "training_replicates": 80,
            "training_length_nt": 1200,
            "kappa": 2.0,
        }
    raise ValueError(f"Unknown simulation preset: {preset}")


def _run_dependence_sensitivity(
    *,
    outdir: Path,
    seed: int,
    tail: str,
    n_genes: int,
    n_calibration: int,
    write_plots: bool,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    spec_a = NeutralSpec(tree_newick="(T1:0.1,T2:0.1,T3:0.1,T4:0.1,T5:0.1,T6:0.1,T7:0.1,T8:0.1);", kappa=2.0, omega=1.0)
    spec_b = NeutralSpec(tree_newick="(T1:0.2,T2:0.2,T3:0.2,T4:0.2,T5:0.2,T6:0.2,T7:0.2,T8:0.2);", kappa=2.0, omega=1.0)

    model = train_energy_model_from_neutral_spec(
        neutral_spec=spec_a,
        sim_replicates=200,
        sim_length=1200,
        seed=int(rng.integers(0, 2**32 - 1)),
    )
    sim_a = GY94NeutralSimulator(spec_a)
    sim_b = GY94NeutralSimulator(spec_b)

    rows: list[dict[str, Any]] = []
    for i in range(n_genes):
        s = int(rng.integers(0, 2**32 - 1))
        aln_a = sim_a.simulate_alignment(length_nt=900, seed=s)
        aln_b = sim_b.simulate_alignment(length_nt=900, seed=int(rng.integers(0, 2**32 - 1)))
        obs = _mosaic_alignment(aln_a, aln_b, block_nt=90)

        mismatched = analyze_alignment(
            obs,
            model=model,
            calibration_size=n_calibration,
            calibration_mode="phylo",
            tail=tail,
            seed=s,
            name=f"dep_{i+1}",
            override_scaling=True,
        )
        rows.append(
            {
                "scenario": "mismatched_calibrator",
                "gene_index": i + 1,
                "p": mismatched.p_value,
                "tail": mismatched.tail,
            }
        )

        obs_feat = alignment_to_matrix(obs)
        d_obs, _ = compute_dispersion(obs_feat, model)
        null_d = np.empty(n_calibration, dtype=float)
        for j in range(n_calibration):
            a0 = sim_a.simulate_alignment(length_nt=900, seed=int(rng.integers(0, 2**32 - 1)))
            b0 = sim_b.simulate_alignment(length_nt=900, seed=int(rng.integers(0, 2**32 - 1)))
            m0 = _mosaic_alignment(a0, b0, block_nt=90)
            feat0 = alignment_to_matrix(m0)
            null_d[j], _ = compute_dispersion(feat0, model)
        p_matched = rank_p_value(d_obs, null_d, tail=tail)
        rows.append(
            {
                "scenario": "matched_mosaic_calibrator",
                "gene_index": i + 1,
                "p": p_matched,
                "tail": tail,
            }
        )

    dep_df = pd.DataFrame(rows)
    _write_tsv(outdir / "raw" / "dependence.tsv", dep_df)
    summary = (
        dep_df.groupby("scenario", as_index=False)
        .agg(
            n_genes=("p", "count"),
            mean_p=("p", "mean"),
            frac_p_lt_0_05=("p", lambda x: float(np.mean(np.asarray(x) <= 0.05))),
        )
        .sort_values("scenario")
    )
    _write_tsv(outdir / "tables" / "dependence_summary.tsv", summary)
    if write_plots:
        _render_dependence_plot(dep_df, outdir / "figures" / "S_dependence_calibration.pdf")
    return dep_df


def _simulation_finalize(outdir: Path, *, write_plots: bool) -> dict[str, Any]:
    raw = outdir / "raw"
    tables = outdir / "tables"
    figures = outdir / "figures"
    null_df = _read_tsv_or_empty(raw / "null_calibration.tsv")
    power_df = _read_tsv_or_empty(raw / "power.tsv")
    runtime_df = _read_tsv_or_empty(raw / "runtime.tsv")
    size_df = _read_tsv_or_empty(tables / "empirical_size.tsv")

    if not size_df.empty:
        _write_tsv(tables / "T1_null_calibration.tsv", size_df)

    if not power_df.empty:
        ok = power_df.copy()
        if "status" not in ok.columns:
            ok["status"] = "OK"
        if "p" not in ok.columns:
            ok["p"] = np.nan
        t2_rows: list[dict[str, Any]] = []
        for (family, method, taxa, length, n_calib), group in ok.groupby(
            ["family", "method", "taxa", "L", "N"], as_index=False
        ):
            n_total = int(len(group))
            sub = group[group["status"] == "OK"].copy()
            n_ok = int(len(sub))
            if n_ok > 0:
                p005 = float(np.mean(pd.to_numeric(sub["p"], errors="coerce") <= 0.05))
                p001 = float(np.mean(pd.to_numeric(sub["p"], errors="coerce") <= 0.01))
                se005 = float(np.sqrt(max(p005 * (1.0 - p005) / n_ok, 0.0)))
                se001 = float(np.sqrt(max(p001 * (1.0 - p001) / n_ok, 0.0)))
            else:
                p005 = 0.0
                p001 = 0.0
                se005 = 0.0
                se001 = 0.0
            t2_rows.append(
                {
                    "family": family,
                    "method": method,
                    "taxa": int(taxa),
                    "L": int(length),
                    "N": int(n_calib),
                    "n_total": n_total,
                    "n_ok": n_ok,
                    "power_alpha_0_05": p005,
                    "power_alpha_0_01": p001,
                    "ci95_low_0_05": max(p005 - 1.96 * se005, 0.0),
                    "ci95_high_0_05": min(p005 + 1.96 * se005, 1.0),
                    "ci95_low_0_01": max(p001 - 1.96 * se001, 0.0),
                    "ci95_high_0_01": min(p001 + 1.96 * se001, 1.0),
                }
            )
        t2 = pd.DataFrame(t2_rows).sort_values(["family", "method", "taxa", "L", "N"])
    else:
        t2 = pd.DataFrame(
            columns=[
                "family",
                "method",
                "taxa",
                "L",
                "N",
                "n_total",
                "n_ok",
                "power_alpha_0_05",
                "power_alpha_0_01",
                "ci95_low_0_05",
                "ci95_high_0_05",
                "ci95_low_0_01",
                "ci95_high_0_01",
            ]
        )
    _write_tsv(tables / "T2_power_grid.tsv", t2)

    if power_df.empty:
        t6 = pd.DataFrame(columns=["method", "status", "reason", "n"])
    else:
        if "status" not in power_df.columns:
            power_df["status"] = "OK"
        if "reason" not in power_df.columns:
            power_df["reason"] = "ok"
        t6 = (
            power_df.groupby(["method", "status", "reason"], as_index=False)
            .size()
            .rename(columns={"size": "n"})
            .sort_values(["method", "status", "n"], ascending=[True, True, False])
        )
    _write_tsv(tables / "T6_failure_rates.tsv", t6)

    if power_df.empty:
        fairness = pd.DataFrame(columns=["family", "method", "taxa", "L", "N", "n_total", "n_fail", "fail_rate"])
    else:
        if "status" not in power_df.columns:
            power_df["status"] = "OK"
        fairness = (
            power_df.groupby(["family", "method", "taxa", "L", "N"], as_index=False)
            .agg(
                n_total=("method", "count"),
                n_fail=("status", lambda x: int(np.count_nonzero(np.asarray(x) != "OK"))),
            )
            .sort_values(["family", "method", "taxa", "L", "N"])
        )
        fairness["fail_rate"] = fairness["n_fail"] / fairness["n_total"].clip(lower=1)
    _write_tsv(tables / "T2_baseline_fairness.tsv", fairness)

    if t2.empty:
        regime = pd.DataFrame(columns=["family", "taxa", "L", "N", "best_method_alpha_0_05", "best_power_alpha_0_05"])
    else:
        regime = (
            t2.sort_values("power_alpha_0_05", ascending=False)
            .groupby(["family", "taxa", "L", "N"], as_index=False)
            .first()[["family", "taxa", "L", "N", "method", "power_alpha_0_05"]]
            .rename(
                columns={
                    "method": "best_method_alpha_0_05",
                    "power_alpha_0_05": "best_power_alpha_0_05",
                }
            )
            .sort_values(["family", "taxa", "L", "N"])
        )
    _write_tsv(tables / "T2_regime_map.tsv", regime)

    if size_df.empty:
        size_dev = pd.DataFrame(columns=["taxa", "L", "N", "alpha", "size_hat", "size_deviation", "size_abs_deviation"])
    else:
        size_dev = size_df.copy()
        size_dev["size_deviation"] = size_dev["size_hat"] - size_dev["alpha"]
        size_dev["size_abs_deviation"] = np.abs(size_dev["size_deviation"])
    _write_tsv(tables / "T1_size_deviation.tsv", size_dev)

    if write_plots:
        plots = _plot_module()
        plots.plot_f1_null_calibration(null_df, size_df, figures / "F1_null_calibration.pdf")
        plots.plot_f2_power_and_regime(power_df, figures / "F2_power_regime_map.pdf")
        plots.plot_size_deviation(size_df, figures / "F1_size_deviation.pdf")
        plots.plot_discreteness_diagnostic(null_df, figures / "F1_discreteness_diagnostic.pdf")
        plots.plot_regime_map_table(regime, figures / "F2_regime_map_table.pdf")
        plots.plot_runtime_scaling(runtime_df, figures / "runtime_simulation.pdf", "Simulation Runtime Scaling")

    report_lines = [
        "Track: simulation",
        f"Null rows: {len(null_df)}",
        f"Power rows: {len(power_df)}",
        f"Runtime rows: {len(runtime_df)}",
        f"T1 rows: {len(size_df)}",
        f"T2 rows: {len(t2)}",
        f"T6 rows: {len(t6)}",
    ]
    plots = _plot_module()
    plots.plot_report_page(report_lines, outdir / "report" / "report.pdf", "BABAPPA Simulation Benchmark Report")
    return {
        "null_rows": int(len(null_df)),
        "power_rows": int(len(power_df)),
        "runtime_rows": int(len(runtime_df)),
    }


def _build_default_neutral_model_json(tree_path: Path, out_path: Path) -> Path:
    payload = {
        "schema_version": 1,
        "model_family": "GY94",
        "genetic_code_table": "standard",
        "tree_file": str(tree_path.resolve()),
        "kappa": 2.0,
        "omega": 1.0,
        "codon_frequencies_method": "F3x4",
        "frozen_values": True,
    }
    _write_json(out_path, payload)
    return out_path


def _real_preset_params(track: str, preset: str) -> dict[str, Any]:
    fast = os.environ.get("BABAPPA_BENCHMARK_FAST", "0") == "1"
    tr = track.lower()
    pr = preset.upper()
    if tr == "ortholog" and pr in {
        "ORTHOMAM_SMALL8_PAPER",
        "ORTHOMAM_SMALL8_FULL",
        "ORTHOLOG_SMALL8_FULL",
        "ORTHOLOG_REAL_V12",
    }:
        return {
            "taxa": 8,
            "min_taxa": 6,
            "max_genes": 8 if fast else (800 if pr.endswith("_FULL") else 500),
            "min_len_codons": 300,
            "calibration_size": 99 if fast else 999,
            "training_replicates": 30 if fast else 200,
            "training_length_nt": 1200,
            "methods": ("busted",),
        }
    if tr == "viral" and pr in {"HIV_ENV_B_PAPER", "HIV_ENV_B_FULL", "VIRAL_HIV_ENV_B_FULL", "HIV_ENV_B_REAL"}:
        return {
            "taxa": 24,
            "min_taxa": 6,
            "max_genes": 8 if fast else (600 if pr.endswith("_FULL") else 500),
            "min_len_codons": 120,
            "calibration_size": 99 if fast else 999,
            "training_replicates": 30 if fast else 200,
            "training_length_nt": 1200,
            "methods": ("busted",),
        }
    if tr == "viral" and pr in {"SARS_2020_PAPER", "SARS_2020_FULL", "VIRAL_SARS_2020_FULL", "SARS_COV2_REAL"}:
        return {
            "taxa": 32,
            "min_taxa": 8,
            "max_genes": 4 if fast else (1200 if pr.endswith("_FULL") else 1000),
            "min_len_codons": 80,
            "calibration_size": 49 if fast else 999,
            "training_replicates": 20 if fast else 200,
            "training_length_nt": 1200,
            "methods": ("busted",),
        }
    if tr == "ortholog" and pr == "ORTHOMAM_SMALL8":
        return {
            "taxa": 8,
            "min_taxa": 6,
            "max_genes": 8 if fast else 500,
            "min_len_codons": 300,
            "calibration_size": 99 if fast else 999,
            "training_replicates": 30 if fast else 400,
            "training_length_nt": 1200,
            "methods": ("busted",),
        }
    if tr == "ortholog" and pr == "ORTHOMAM_MEDIUM16":
        return {
            "taxa": 16,
            "min_taxa": 12,
            "max_genes": 8 if fast else 300,
            "min_len_codons": 300,
            "calibration_size": 99 if fast else 999,
            "training_replicates": 30 if fast else 500,
            "training_length_nt": 1200,
            "methods": ("busted",),
        }
    if tr == "viral" and pr == "HIV_ENV_B":
        return {
            "taxa": 24,
            "min_taxa": 6,
            "max_genes": 10 if fast else 400,
            "min_len_codons": 120,
            "calibration_size": 99 if fast else 999,
            "training_replicates": 30 if fast else 450,
            "training_length_nt": 1200,
            "methods": ("busted",),
        }
    if tr == "viral" and pr == "SARS_2020_GLOBAL":
        return {
            "taxa": 32,
            "min_taxa": 8,
            "max_genes": 10 if fast else 800,
            "min_len_codons": 80,
            "calibration_size": 99 if fast else 999,
            "training_replicates": 30 if fast else 350,
            "training_length_nt": 1200,
            "methods": ("busted",),
        }
    raise ValueError(f"Unknown preset for track={track}: {preset}")


def _create_synthetic_dataset(
    *,
    track: str,
    preset: str,
    outdir: Path,
    rng: np.random.Generator,
    taxa_n: int,
    gene_n: int,
    length_nt: int,
) -> tuple[dict[str, Any], Path]:
    ds_dir = outdir / "raw" / "synthetic_dataset"
    genes_dir = ds_dir / "genes"
    genes_dir.mkdir(parents=True, exist_ok=True)
    tree = _balanced_tree_newick(taxa_n, branch_length=0.1)
    tree_path = ds_dir / "tree.nwk"
    tree_path.write_text(tree + "\n", encoding="utf-8")

    neutral_spec = NeutralSpec(tree_newick=tree, kappa=2.0, omega=1.0)
    sim = GY94NeutralSimulator(neutral_spec)
    genes: list[dict[str, Any]] = []
    for i in range(gene_n):
        seed = int(rng.integers(0, 2**32 - 1))
        if track.lower() == "viral" and i % 5 == 0:
            aln = simulate_mixture_omega_alignment(
                neutral_spec=neutral_spec,
                length_nt=length_nt,
                selected_fraction=0.15,
                omega_alt=2.0,
                seed=seed,
            )
        else:
            aln = sim.simulate_alignment(length_nt=length_nt, seed=seed)
        gpath = genes_dir / f"gene_{i+1:04d}.fasta"
        _write_alignment_fasta(aln, gpath)
        genes.append({"gene_id": gpath.stem, "alignment_path": str(gpath.resolve())})

    neutral_model_path = _build_default_neutral_model_json(tree_path, ds_dir / "neutral_model.json")
    payload = {
        "schema_version": 1,
        "tree_path": str(tree_path.resolve()),
        "genes": genes,
        "neutral_model_json": str(neutral_model_path.resolve()),
        "foreground_taxon": "T1",
        "metadata": {
            "source": "synthetic_fallback",
            "track": track,
            "preset": preset,
        },
    }
    dataset_path = ds_dir / "dataset.json"
    _write_json(dataset_path, payload)
    return payload, dataset_path


def _resolve_real_dataset(
    *,
    config: BenchmarkRunConfig,
    track: str,
    preset: str,
    outdir: Path,
    rng: np.random.Generator,
    taxa_n: int,
    max_genes: int,
) -> tuple[dict[str, Any], Path, bool]:
    candidates: list[Path] = []
    if config.dataset_json:
        candidates.append(Path(config.dataset_json))
    env_track = "BABAPPA_ORTHOLOG_DATASET_JSON" if track == "ortholog" else "BABAPPA_VIRAL_DATASET_JSON"
    if os.environ.get(env_track):
        candidates.append(Path(str(os.environ.get(env_track))))
    if os.environ.get("BABAPPA_BENCHMARK_DATASET_JSON"):
        candidates.append(Path(str(os.environ.get("BABAPPA_BENCHMARK_DATASET_JSON"))))
    candidates.append(Path("datasets") / track / preset / "dataset.json")

    for c in candidates:
        if c.exists():
            payload = load_dataset_json(c)
            metadata = payload.get("metadata") if isinstance(payload, dict) else None
            synthetic = False
            if isinstance(metadata, dict):
                synthetic = bool(metadata.get("synthetic_fallback"))
                if str(metadata.get("source", "")).lower().strip() == "synthetic_fallback":
                    synthetic = True
            return payload, c.resolve(), synthetic

    if bool(config.require_real_data):
        raise FileNotFoundError(
            "Real-data benchmark requires an explicit dataset JSON. "
            "Use --dataset-json (or `benchmark realdata --data ...`) and ensure it exists."
        )

    payload, path = _create_synthetic_dataset(
        track=track,
        preset=preset,
        outdir=outdir,
        rng=rng,
        taxa_n=taxa_n,
        gene_n=max_genes,
        length_nt=1200,
    )
    return payload, path.resolve(), True


def _alignment_gap_fraction(aln: Alignment) -> float:
    total = int(aln.length) * int(aln.n_sequences)
    if total <= 0:
        return 1.0
    gaps = 0
    for seq in aln.sequences:
        gaps += int(seq.count("-"))
    return float(gaps) / float(total)


def _alignment_ambiguous_fraction(aln: Alignment) -> float:
    total = int(aln.length) * int(aln.n_sequences)
    if total <= 0:
        return 1.0
    ambiguous = 0
    for seq in aln.sequences:
        s = seq.upper().replace("U", "T")
        for ch in s:
            if ch not in {"A", "C", "G", "T", "-"}:
                ambiguous += 1
    return float(ambiguous) / float(total)


def _alignment_has_internal_stop(aln: Alignment) -> bool:
    stops = {"TAA", "TAG", "TGA"}
    for seq in aln.sequences:
        s = seq.upper().replace("U", "T")
        usable = len(s) - (len(s) % 3)
        if usable < 6:
            continue
        for i in range(0, usable - 3, 3):
            codon = s[i : i + 3]
            if any(ch not in {"A", "C", "G", "T"} for ch in codon):
                continue
            if codon in stops:
                return True
    return False


def _balanced_tree_for_taxa(names: list[str], branch_length: float = 0.1) -> str:
    if not names:
        raise ValueError("Cannot build balanced tree for empty taxa list.")

    def _build(items: list[str], root: bool = False) -> str:
        if len(items) == 1:
            return f"{items[0]}:{branch_length}"
        mid = len(items) // 2
        left = _build(items[:mid], False)
        right = _build(items[mid:], False)
        if root:
            return f"({left},{right})"
        return f"({left},{right}):{branch_length}"

    return _build(list(names), root=True) + ";"


def _prepare_gene_records(
    payload: dict[str, Any],
    *,
    default_tree_path: Path,
    min_len_codons: int,
    max_genes: int,
    min_taxa: int,
    require_full_completeness: bool,
    preprocessed_dir: Path,
) -> tuple[list[dict[str, Any]], pd.DataFrame, pd.DataFrame, dict[str, str]]:
    selected: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    taxa_map: dict[str, str] = {}
    genes = list(payload.get("genes", []))
    align_dir = preprocessed_dir / "alignments"
    tree_dir = preprocessed_dir / "trees"
    align_dir.mkdir(parents=True, exist_ok=True)
    tree_dir.mkdir(parents=True, exist_ok=True)

    def _add_row(
        *,
        unit_id: str,
        stage_name: str,
        status: str,
        reason_code: str,
        reason_detail: str,
        n_taxa_in_alignment: int | float = np.nan,
        n_taxa_in_tree: int | float = np.nan,
        n_taxa_intersection: int | float = np.nan,
        codon_length: int | float = np.nan,
        stop_codons_found: int | float = np.nan,
        ambiguous_fraction: float = np.nan,
        gap_fraction: float = np.nan,
        tree_alignment_taxa_mismatch: int | float = np.nan,
        file_path_alignment: str = "",
        file_path_tree: str = "",
    ) -> None:
        audit_rows.append(
            {
                "unit_id": unit_id,
                "stage_name": stage_name,
                "status": status,
                "reason_code": reason_code,
                "reason_detail": reason_detail,
                "n_taxa_in_alignment": n_taxa_in_alignment,
                "n_taxa_in_tree": n_taxa_in_tree,
                "n_taxa_intersection": n_taxa_intersection,
                "codon_length": codon_length,
                "stop_codons_found": stop_codons_found,
                "ambiguous_fraction": ambiguous_fraction,
                "gap_fraction": gap_fraction,
                "tree_alignment_taxa_mismatch": tree_alignment_taxa_mismatch,
                "file_path_alignment": file_path_alignment,
                "file_path_tree": file_path_tree,
            }
        )

    for idx, gene in enumerate(genes, start=1):
        unit_id = str(gene.get("gene_id", f"gene_{idx}"))
        raw_path = gene.get("alignment_path")
        if raw_path is None:
            _add_row(
                unit_id=unit_id,
                stage_name="DISCOVER_FILES",
                status="DROP",
                reason_code="MISSING_ALIGNMENT",
                reason_detail="alignment_path missing in dataset.json",
            )
            _add_row(
                unit_id=unit_id,
                stage_name="FINAL_KEEP",
                status="DROP",
                reason_code="MISSING_ALIGNMENT",
                reason_detail="alignment_path missing in dataset.json",
            )
            continue
        gpath = Path(str(raw_path)).resolve()
        tpath = (
            Path(str(gene.get("tree_path"))).resolve()
            if gene.get("tree_path") is not None
            else default_tree_path.resolve()
        )
        if len(selected) >= int(max_genes):
            _add_row(
                unit_id=unit_id,
                stage_name="DISCOVER_FILES",
                status="SKIP",
                reason_code="MAX_GENES_LIMIT",
                reason_detail=f"max_genes={int(max_genes)} reached",
                file_path_alignment=str(gpath),
                file_path_tree=str(tpath),
            )
            _add_row(
                unit_id=unit_id,
                stage_name="FINAL_KEEP",
                status="SKIP",
                reason_code="MAX_GENES_LIMIT",
                reason_detail=f"max_genes={int(max_genes)} reached",
                file_path_alignment=str(gpath),
                file_path_tree=str(tpath),
            )
            continue
        if not gpath.exists():
            _add_row(
                unit_id=unit_id,
                stage_name="DISCOVER_FILES",
                status="DROP",
                reason_code="MISSING_ALIGNMENT",
                reason_detail="alignment file does not exist",
                file_path_alignment=str(gpath),
                file_path_tree=str(tpath),
            )
            _add_row(
                unit_id=unit_id,
                stage_name="FINAL_KEEP",
                status="DROP",
                reason_code="MISSING_ALIGNMENT",
                reason_detail="alignment file does not exist",
                file_path_alignment=str(gpath),
                file_path_tree=str(tpath),
            )
            continue
        if not tpath.exists():
            _add_row(
                unit_id=unit_id,
                stage_name="DISCOVER_FILES",
                status="DROP",
                reason_code="MISSING_TREE",
                reason_detail="tree file does not exist",
                file_path_alignment=str(gpath),
                file_path_tree=str(tpath),
            )
            _add_row(
                unit_id=unit_id,
                stage_name="FINAL_KEEP",
                status="DROP",
                reason_code="MISSING_TREE",
                reason_detail="tree file does not exist",
                file_path_alignment=str(gpath),
                file_path_tree=str(tpath),
            )
            continue
        _add_row(
            unit_id=unit_id,
            stage_name="DISCOVER_FILES",
            status="KEEP",
            reason_code="DISCOVERED",
            reason_detail="alignment/tree files discovered",
            file_path_alignment=str(gpath),
            file_path_tree=str(tpath),
        )

        try:
            aln = read_fasta(gpath)
        except Exception as exc:
            _add_row(
                unit_id=unit_id,
                stage_name="PARSE_ALIGNMENT",
                status="DROP",
                reason_code="PARSE_ALIGNMENT_FAILED",
                reason_detail=str(exc),
                file_path_alignment=str(gpath),
                file_path_tree=str(tpath),
            )
            _add_row(
                unit_id=unit_id,
                stage_name="FINAL_KEEP",
                status="DROP",
                reason_code="PARSE_ALIGNMENT_FAILED",
                reason_detail=str(exc),
                file_path_alignment=str(gpath),
                file_path_tree=str(tpath),
            )
            continue

        aln_map: dict[str, str] = {}
        duplicate_norm_aln = False
        for nm, seq in zip(aln.names, aln.sequences):
            norm = normalize_taxon(str(nm))
            taxa_map[str(nm)] = norm
            if norm in aln_map:
                duplicate_norm_aln = True
            else:
                aln_map[norm] = str(seq)
        if duplicate_norm_aln:
            _add_row(
                unit_id=unit_id,
                stage_name="PARSE_ALIGNMENT",
                status="DROP",
                reason_code="DUPLICATE_ALIGNMENT_TAXA",
                reason_detail="duplicate taxa after normalization",
                n_taxa_in_alignment=int(aln.n_sequences),
                file_path_alignment=str(gpath),
                file_path_tree=str(tpath),
            )
            _add_row(
                unit_id=unit_id,
                stage_name="FINAL_KEEP",
                status="DROP",
                reason_code="DUPLICATE_ALIGNMENT_TAXA",
                reason_detail="duplicate taxa after normalization",
                n_taxa_in_alignment=int(aln.n_sequences),
                file_path_alignment=str(gpath),
                file_path_tree=str(tpath),
            )
            continue

        _add_row(
            unit_id=unit_id,
            stage_name="PARSE_ALIGNMENT",
            status="KEEP",
            reason_code="PARSED_ALIGNMENT",
            reason_detail="alignment parsed",
            n_taxa_in_alignment=int(len(aln_map)),
            file_path_alignment=str(gpath),
            file_path_tree=str(tpath),
        )

        try:
            tree_text = tpath.read_text(encoding="utf-8")
            tree = parse_newick(tree_text)
            tree_raw_names = tree.leaf_names()
        except Exception as exc:
            _add_row(
                unit_id=unit_id,
                stage_name="PARSE_TREE",
                status="DROP",
                reason_code="PARSE_TREE_FAILED",
                reason_detail=str(exc),
                n_taxa_in_alignment=int(len(aln_map)),
                file_path_alignment=str(gpath),
                file_path_tree=str(tpath),
            )
            _add_row(
                unit_id=unit_id,
                stage_name="FINAL_KEEP",
                status="DROP",
                reason_code="PARSE_TREE_FAILED",
                reason_detail=str(exc),
                n_taxa_in_alignment=int(len(aln_map)),
                file_path_alignment=str(gpath),
                file_path_tree=str(tpath),
            )
            continue

        tree_norm_names: list[str] = []
        duplicate_norm_tree = False
        for nm in tree_raw_names:
            norm = normalize_taxon(str(nm))
            taxa_map[str(nm)] = norm
            tree_norm_names.append(norm)
        if len(set(tree_norm_names)) != len(tree_norm_names):
            duplicate_norm_tree = True
        if duplicate_norm_tree:
            _add_row(
                unit_id=unit_id,
                stage_name="PARSE_TREE",
                status="DROP",
                reason_code="DUPLICATE_TREE_TAXA",
                reason_detail="duplicate tree taxa after normalization",
                n_taxa_in_alignment=int(len(aln_map)),
                n_taxa_in_tree=int(len(tree_norm_names)),
                file_path_alignment=str(gpath),
                file_path_tree=str(tpath),
            )
            _add_row(
                unit_id=unit_id,
                stage_name="FINAL_KEEP",
                status="DROP",
                reason_code="DUPLICATE_TREE_TAXA",
                reason_detail="duplicate tree taxa after normalization",
                n_taxa_in_alignment=int(len(aln_map)),
                n_taxa_in_tree=int(len(tree_norm_names)),
                file_path_alignment=str(gpath),
                file_path_tree=str(tpath),
            )
            continue

        _add_row(
            unit_id=unit_id,
            stage_name="PARSE_TREE",
            status="KEEP",
            reason_code="PARSED_TREE",
            reason_detail="tree parsed",
            n_taxa_in_alignment=int(len(aln_map)),
            n_taxa_in_tree=int(len(tree_norm_names)),
            file_path_alignment=str(gpath),
            file_path_tree=str(tpath),
        )

        aln_taxa = set(aln_map.keys())
        tree_taxa = set(tree_norm_names)
        inter = sorted(aln_taxa & tree_taxa)
        mismatch = int(len(inter) != len(aln_taxa) or len(inter) != len(tree_taxa))
        _add_row(
            unit_id=unit_id,
            stage_name="TAXA_MATCH",
            status="KEEP" if inter else "DROP",
            reason_code="TAXA_INTERSECTION" if inter else "NO_TAXA_INTERSECTION",
            reason_detail=f"intersection={len(inter)} aln={len(aln_taxa)} tree={len(tree_taxa)}",
            n_taxa_in_alignment=int(len(aln_taxa)),
            n_taxa_in_tree=int(len(tree_taxa)),
            n_taxa_intersection=int(len(inter)),
            tree_alignment_taxa_mismatch=mismatch,
            file_path_alignment=str(gpath),
            file_path_tree=str(tpath),
        )
        if not inter:
            _add_row(
                unit_id=unit_id,
                stage_name="FINAL_KEEP",
                status="DROP",
                reason_code="NO_TAXA_INTERSECTION",
                reason_detail="no taxa overlap between alignment and tree",
                n_taxa_in_alignment=int(len(aln_taxa)),
                n_taxa_in_tree=int(len(tree_taxa)),
                n_taxa_intersection=int(len(inter)),
                tree_alignment_taxa_mismatch=mismatch,
                file_path_alignment=str(gpath),
                file_path_tree=str(tpath),
            )
            continue

        if bool(require_full_completeness) and mismatch:
            _add_row(
                unit_id=unit_id,
                stage_name="PRUNE_POLICY",
                status="DROP",
                reason_code="REQUIRE_FULL_COMPLETENESS",
                reason_detail="tree/alignment completeness mismatch under strict completeness mode",
                n_taxa_in_alignment=int(len(aln_taxa)),
                n_taxa_in_tree=int(len(tree_taxa)),
                n_taxa_intersection=int(len(inter)),
                tree_alignment_taxa_mismatch=mismatch,
                file_path_alignment=str(gpath),
                file_path_tree=str(tpath),
            )
            _add_row(
                unit_id=unit_id,
                stage_name="FINAL_KEEP",
                status="DROP",
                reason_code="REQUIRE_FULL_COMPLETENESS",
                reason_detail="tree/alignment completeness mismatch under strict completeness mode",
                n_taxa_in_alignment=int(len(aln_taxa)),
                n_taxa_in_tree=int(len(tree_taxa)),
                n_taxa_intersection=int(len(inter)),
                tree_alignment_taxa_mismatch=mismatch,
                file_path_alignment=str(gpath),
                file_path_tree=str(tpath),
            )
            continue
        if len(inter) < int(min_taxa):
            _add_row(
                unit_id=unit_id,
                stage_name="PRUNE_POLICY",
                status="DROP",
                reason_code="BELOW_MIN_TAXA",
                reason_detail=f"intersection={len(inter)} < min_taxa={int(min_taxa)}",
                n_taxa_in_alignment=int(len(aln_taxa)),
                n_taxa_in_tree=int(len(tree_taxa)),
                n_taxa_intersection=int(len(inter)),
                tree_alignment_taxa_mismatch=mismatch,
                file_path_alignment=str(gpath),
                file_path_tree=str(tpath),
            )
            _add_row(
                unit_id=unit_id,
                stage_name="FINAL_KEEP",
                status="DROP",
                reason_code="BELOW_MIN_TAXA",
                reason_detail=f"intersection={len(inter)} < min_taxa={int(min_taxa)}",
                n_taxa_in_alignment=int(len(aln_taxa)),
                n_taxa_in_tree=int(len(tree_taxa)),
                n_taxa_intersection=int(len(inter)),
                tree_alignment_taxa_mismatch=mismatch,
                file_path_alignment=str(gpath),
                file_path_tree=str(tpath),
            )
            continue

        _add_row(
            unit_id=unit_id,
            stage_name="PRUNE_POLICY",
            status="KEEP",
            reason_code="PRUNED_TO_INTERSECTION",
            reason_detail=f"kept_taxa={len(inter)}",
            n_taxa_in_alignment=int(len(aln_taxa)),
            n_taxa_in_tree=int(len(tree_taxa)),
            n_taxa_intersection=int(len(inter)),
            tree_alignment_taxa_mismatch=mismatch,
            file_path_alignment=str(gpath),
            file_path_tree=str(tpath),
        )

        kept_taxa = list(inter)
        kept_seqs = [str(aln_map[t]) for t in kept_taxa]
        pruned = Alignment(names=tuple(kept_taxa), sequences=tuple(kept_seqs))
        codon_length = int(pruned.length // 3)
        stop_flag = int(_alignment_has_internal_stop(pruned))
        amb_frac = float(_alignment_ambiguous_fraction(pruned))
        gap_frac = float(_alignment_gap_fraction(pruned))
        _add_row(
            unit_id=unit_id,
            stage_name="CODON_QC",
            status="DROP" if stop_flag else "KEEP",
            reason_code="INTERNAL_STOP" if stop_flag else "CODON_QC_PASS",
            reason_detail="internal stop codon found" if stop_flag else "codon QC pass",
            n_taxa_in_alignment=int(len(aln_taxa)),
            n_taxa_in_tree=int(len(tree_taxa)),
            n_taxa_intersection=int(len(kept_taxa)),
            codon_length=int(codon_length),
            stop_codons_found=int(stop_flag),
            ambiguous_fraction=float(amb_frac),
            gap_fraction=float(gap_frac),
            tree_alignment_taxa_mismatch=mismatch,
            file_path_alignment=str(gpath),
            file_path_tree=str(tpath),
        )
        if stop_flag:
            _add_row(
                unit_id=unit_id,
                stage_name="FINAL_KEEP",
                status="DROP",
                reason_code="INTERNAL_STOP",
                reason_detail="internal stop codon found",
                n_taxa_in_alignment=int(len(aln_taxa)),
                n_taxa_in_tree=int(len(tree_taxa)),
                n_taxa_intersection=int(len(kept_taxa)),
                codon_length=int(codon_length),
                stop_codons_found=int(stop_flag),
                ambiguous_fraction=float(amb_frac),
                gap_fraction=float(gap_frac),
                tree_alignment_taxa_mismatch=mismatch,
                file_path_alignment=str(gpath),
                file_path_tree=str(tpath),
            )
            continue

        if pruned.length % 3 != 0:
            _add_row(
                unit_id=unit_id,
                stage_name="LENGTH_FILTER",
                status="DROP",
                reason_code="FRAME_NOT_MULTIPLE_OF_3",
                reason_detail=f"length_nt={pruned.length}",
                n_taxa_in_alignment=int(len(aln_taxa)),
                n_taxa_in_tree=int(len(tree_taxa)),
                n_taxa_intersection=int(len(kept_taxa)),
                codon_length=int(codon_length),
                stop_codons_found=int(stop_flag),
                ambiguous_fraction=float(amb_frac),
                gap_fraction=float(gap_frac),
                tree_alignment_taxa_mismatch=mismatch,
                file_path_alignment=str(gpath),
                file_path_tree=str(tpath),
            )
            _add_row(
                unit_id=unit_id,
                stage_name="FINAL_KEEP",
                status="DROP",
                reason_code="FRAME_NOT_MULTIPLE_OF_3",
                reason_detail=f"length_nt={pruned.length}",
                n_taxa_in_alignment=int(len(aln_taxa)),
                n_taxa_in_tree=int(len(tree_taxa)),
                n_taxa_intersection=int(len(kept_taxa)),
                codon_length=int(codon_length),
                stop_codons_found=int(stop_flag),
                ambiguous_fraction=float(amb_frac),
                gap_fraction=float(gap_frac),
                tree_alignment_taxa_mismatch=mismatch,
                file_path_alignment=str(gpath),
                file_path_tree=str(tpath),
            )
            continue
        if codon_length < int(min_len_codons):
            _add_row(
                unit_id=unit_id,
                stage_name="LENGTH_FILTER",
                status="DROP",
                reason_code="LENGTH_LT_MIN",
                reason_detail=f"codons={codon_length} < min={int(min_len_codons)}",
                n_taxa_in_alignment=int(len(aln_taxa)),
                n_taxa_in_tree=int(len(tree_taxa)),
                n_taxa_intersection=int(len(kept_taxa)),
                codon_length=int(codon_length),
                stop_codons_found=int(stop_flag),
                ambiguous_fraction=float(amb_frac),
                gap_fraction=float(gap_frac),
                tree_alignment_taxa_mismatch=mismatch,
                file_path_alignment=str(gpath),
                file_path_tree=str(tpath),
            )
            _add_row(
                unit_id=unit_id,
                stage_name="FINAL_KEEP",
                status="DROP",
                reason_code="LENGTH_LT_MIN",
                reason_detail=f"codons={codon_length} < min={int(min_len_codons)}",
                n_taxa_in_alignment=int(len(aln_taxa)),
                n_taxa_in_tree=int(len(tree_taxa)),
                n_taxa_intersection=int(len(kept_taxa)),
                codon_length=int(codon_length),
                stop_codons_found=int(stop_flag),
                ambiguous_fraction=float(amb_frac),
                gap_fraction=float(gap_frac),
                tree_alignment_taxa_mismatch=mismatch,
                file_path_alignment=str(gpath),
                file_path_tree=str(tpath),
            )
            continue

        _add_row(
            unit_id=unit_id,
            stage_name="LENGTH_FILTER",
            status="KEEP",
            reason_code="LENGTH_PASS",
            reason_detail=f"codons={codon_length}",
            n_taxa_in_alignment=int(len(aln_taxa)),
            n_taxa_in_tree=int(len(tree_taxa)),
            n_taxa_intersection=int(len(kept_taxa)),
            codon_length=int(codon_length),
            stop_codons_found=int(stop_flag),
            ambiguous_fraction=float(amb_frac),
            gap_fraction=float(gap_frac),
            tree_alignment_taxa_mismatch=mismatch,
            file_path_alignment=str(gpath),
            file_path_tree=str(tpath),
        )

        out_aln = align_dir / f"{unit_id}.fasta"
        out_tree = tree_dir / f"{unit_id}.nwk"
        _write_alignment_fasta(pruned, out_aln)
        out_tree.write_text(_balanced_tree_for_taxa(kept_taxa) + "\n", encoding="utf-8")
        _add_row(
            unit_id=unit_id,
            stage_name="FINAL_KEEP",
            status="KEEP",
            reason_code="KEEP",
            reason_detail="passed all filters",
            n_taxa_in_alignment=int(len(aln_taxa)),
            n_taxa_in_tree=int(len(tree_taxa)),
            n_taxa_intersection=int(len(kept_taxa)),
            codon_length=int(codon_length),
            stop_codons_found=int(stop_flag),
            ambiguous_fraction=float(amb_frac),
            gap_fraction=float(gap_frac),
            tree_alignment_taxa_mismatch=mismatch,
            file_path_alignment=str(out_aln.resolve()),
            file_path_tree=str(out_tree.resolve()),
        )
        selected.append(
            {
                "gene_id": unit_id,
                "alignment_path": str(out_aln.resolve()),
                "tree_path": str(out_tree.resolve()),
                "length_nt": int(pruned.length),
                "n_taxa": int(pruned.n_sequences),
            }
        )

    drop_df = pd.DataFrame(audit_rows)
    final_rows = drop_df[drop_df["stage_name"] == "FINAL_KEEP"].copy() if not drop_df.empty else pd.DataFrame()
    kept_n = int(np.count_nonzero(final_rows["status"] == "KEEP")) if not final_rows.empty else 0
    dropped = final_rows[final_rows["status"] == "DROP"] if not final_rows.empty else pd.DataFrame()
    reason_counts = (
        dropped.groupby("reason_code", as_index=False).size().rename(columns={"size": "count"})
        if not dropped.empty
        else pd.DataFrame(columns=["reason_code", "count"])
    )
    summary_rows: list[dict[str, Any]] = [
        {"metric": "total_candidates", "value": int(len(genes))},
        {"metric": "kept", "value": int(kept_n)},
    ]
    for row in reason_counts.itertuples(index=False):
        summary_rows.append(
            {
                "metric": f"dropped_reason_{str(getattr(row, 'reason_code'))}",
                "value": int(getattr(row, "count")),
            }
        )
    summary_df = pd.DataFrame(summary_rows)
    for row in summary_rows:
        _add_row(
            unit_id="__SUMMARY__",
            stage_name="SUMMARY",
            status="SKIP",
            reason_code=str(row["metric"]).upper(),
            reason_detail=str(row["value"]),
        )
    drop_df = pd.DataFrame(audit_rows)
    return selected, drop_df, summary_df, taxa_map


def run_dataset_audit(
    *,
    dataset_json: str | Path,
    outdir: str | Path,
    min_len_codons: int,
    min_taxa: int,
    require_full_completeness: bool = False,
    max_genes: int | None = None,
) -> dict[str, Any]:
    ds_path = Path(dataset_json).resolve()
    payload = load_dataset_json(ds_path)
    out = Path(outdir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    genes = list(payload.get("genes", []))
    limit = len(genes) if max_genes is None else int(max_genes)
    selected, drop_df, summary_df, taxa_map = _prepare_gene_records(
        payload,
        default_tree_path=Path(str(payload["tree_path"])).resolve(),
        min_len_codons=int(min_len_codons),
        max_genes=int(limit),
        min_taxa=int(min_taxa),
        require_full_completeness=bool(require_full_completeness),
        preprocessed_dir=out / "preprocessed_units",
    )
    _write_tsv(out / "drop_audit.tsv", drop_df)
    _write_tsv(out / "summary.tsv", summary_df)
    summary_payload: dict[str, Any] = {
        "schema_version": 1,
        "dataset_json": str(ds_path),
        "outdir": str(out),
        "total_candidates": int(len(genes)),
        "kept": int(len(selected)),
        "dropped_by_reason": {},
        "min_len_codons": int(min_len_codons),
        "min_taxa": int(min_taxa),
        "require_full_completeness": bool(require_full_completeness),
        "n_taxa_map_entries": int(len(taxa_map)),
    }
    final_rows = drop_df[drop_df["stage_name"] == "FINAL_KEEP"].copy() if not drop_df.empty else pd.DataFrame()
    if not final_rows.empty:
        d = final_rows[final_rows["status"] == "DROP"]
        if not d.empty:
            counts = d.groupby("reason_code", as_index=False).size().rename(columns={"size": "n"})
            summary_payload["dropped_by_reason"] = {str(r.reason_code): int(r.n) for r in counts.itertuples(index=False)}
    _write_json(out / "summary.json", summary_payload)
    _write_json(
        out / "taxa_map.json",
        {
            "schema_version": 1,
            "mapping_rule": "strip -> split('|','/') -> whitespace_to_underscore -> lowercase",
            "n_entries": int(len(taxa_map)),
            "map": taxa_map,
        },
    )
    return summary_payload


def _doctor_input_dir(genes: list[dict[str, Any]], outdir: Path) -> Path:
    gdir = outdir / "raw" / "doctor_genes"
    if gdir.exists():
        shutil.rmtree(gdir)
    gdir.mkdir(parents=True, exist_ok=True)
    for g in genes:
        src = Path(str(g["alignment_path"])).resolve()
        dst = gdir / f"{g['gene_id']}.fasta"
        shutil.copy2(src, dst)
    return gdir


def _split_train_test_genes(
    genes: list[dict[str, Any]],
    *,
    seed: int,
    train_fraction: float = 0.2,
    min_test_genes: int = 1,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not genes:
        return [], []
    ordered = sorted(genes, key=lambda g: str(g["gene_id"]))
    if len(ordered) == 1:
        return [ordered[0]], [ordered[0]]
    rng = np.random.default_rng(int(seed))
    idx = np.arange(len(ordered), dtype=int)
    rng.shuffle(idx)
    train_n = max(1, int(round(len(ordered) * float(train_fraction))))
    train_n = min(train_n, len(ordered) - 1)
    desired_test = max(1, int(min_test_genes))
    if len(ordered) > desired_test:
        train_n = min(train_n, len(ordered) - desired_test)
        train_n = max(1, int(train_n))
    train_idx = set(int(i) for i in idx[:train_n])
    train = [ordered[i] for i in range(len(ordered)) if i in train_idx]
    test = [ordered[i] for i in range(len(ordered)) if i not in train_idx]
    if not test:
        test = [ordered[-1]]
        train = ordered[:-1]
    return train, test


def _ensure_required_test_genes(
    train_genes: list[dict[str, Any]],
    test_genes: list[dict[str, Any]],
    required_tokens: tuple[str, ...],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train = list(train_genes)
    test = list(test_genes)
    req = tuple(str(t).lower() for t in required_tokens if str(t).strip())
    if not req:
        return train, test

    def _gid(g: dict[str, Any]) -> str:
        return str(g.get("gene_id", "")).lower()

    for token in req:
        if any(token in _gid(g) for g in test):
            continue
        src_idx = next((i for i, g in enumerate(train) if token in _gid(g)), None)
        if src_idx is None:
            continue
        moved = train.pop(int(src_idx))
        swap_idx = next((i for i, g in enumerate(test) if not any(t in _gid(g) for t in req)), None)
        if swap_idx is not None:
            train.append(test.pop(int(swap_idx)))
        test.append(moved)
    return train, test


def _method_col_name(method: str) -> str:
    raw = str(method).lower()
    out = []
    for ch in raw:
        if ch.isalnum():
            out.append(ch)
        else:
            out.append("_")
    norm = "".join(out).strip("_")
    return norm or "baseline"


def _reason_token(reason: Any, default: str = "unknown") -> str:
    raw = str(reason).strip().lower()
    if not raw:
        raw = str(default).strip().lower()
    out = []
    for ch in raw:
        if ch.isalnum():
            out.append(ch)
        else:
            out.append("_")
    token = "".join(out).strip("_")
    while "__" in token:
        token = token.replace("__", "_")
    return token or str(default).strip().lower() or "unknown"


def _normalize_method_status(
    status: Any,
    reason: Any,
    *,
    missing_reason: str = "not_run",
) -> str:
    st = str(status).strip().upper()
    rs = _reason_token(reason, default=missing_reason)
    if st == "OK":
        return "OK"
    if not st:
        return f"SKIP_{_reason_token(missing_reason, default='not_run')}"
    if st.startswith("SKIP") or st.startswith("DROP"):
        return f"SKIP_{rs}"
    return f"FAIL_{rs}"


def _bh_q_values_in_scope(p_values: pd.Series, scope_mask: pd.Series) -> pd.Series:
    p = pd.to_numeric(p_values, errors="coerce")
    scope = scope_mask.astype(bool)
    mask = scope & p.notna()
    out = pd.Series(np.nan, index=p.index, dtype=float)
    if bool(np.count_nonzero(mask)):
        q = benjamini_hochberg([float(x) for x in p.loc[mask].tolist()])
        out.loc[mask] = q
    return out


def _build_testable_set_diff(
    *,
    candidates: pd.DataFrame,
    orth: pd.DataFrame,
    babappa_seen: set[str],
    busted_seen: set[str],
) -> tuple[pd.DataFrame, bool]:
    cols = ["gene_id", "in_babappa", "in_busted", "babappa_status", "busted_status", "reason"]
    if orth.empty:
        return pd.DataFrame(columns=cols), False

    cand_ids = (
        set(candidates["gene_id"].astype(str).tolist())
        if not candidates.empty and "gene_id" in candidates.columns
        else set()
    )
    row_map: dict[str, dict[str, Any]] = {}
    for row in orth.itertuples(index=False):
        gid = str(getattr(row, "gene_id"))
        row_map[gid] = {
            "babappa_status": str(getattr(row, "babappa_status", "SKIP_not_run")),
            "busted_status": str(getattr(row, "busted_status", "SKIP_not_run")),
        }

    all_ids = sorted(set(row_map.keys()) | cand_ids | set(babappa_seen) | set(busted_seen))
    diff_rows: list[dict[str, Any]] = []
    for gid in all_ids:
        st = row_map.get(gid, {})
        bab = str(st.get("babappa_status", "SKIP_not_run"))
        bus = str(st.get("busted_status", "SKIP_not_run"))
        in_bab = bool(bab == "OK")
        in_bus = bool(bus == "OK")
        reasons: list[str] = []
        if gid not in cand_ids:
            reasons.append("not_in_candidates")
        if gid not in babappa_seen:
            reasons.append("missing_babappa_row")
        if gid not in busted_seen:
            reasons.append("missing_busted_row")
        if in_bab and not in_bus:
            reasons.append("babappa_only_ok")
        if in_bus and not in_bab:
            reasons.append("busted_only_ok")
        if reasons:
            diff_rows.append(
                {
                    "gene_id": gid,
                    "in_babappa": bool(in_bab),
                    "in_busted": bool(in_bus),
                    "babappa_status": bab,
                    "busted_status": bus,
                    "reason": ";".join(reasons),
                }
            )
    diff_df = pd.DataFrame(diff_rows, columns=cols)
    mismatch = bool(not diff_df.empty)
    return diff_df, mismatch


def _build_discovery_table(
    babappa_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
) -> pd.DataFrame:
    if babappa_df.empty:
        base = pd.DataFrame(columns=["gene_id", "babappa_p", "babappa_q", "babappa_status", "babappa_reason"])
    else:
        base = babappa_df[
            ["gene_id", "p", "q_value", "status", "reason"]
        ].rename(
            columns={
                "p": "babappa_p",
                "q_value": "babappa_q",
                "status": "babappa_status",
                "reason": "babappa_reason",
            }
        )

    if baseline_df.empty:
        return base

    out = base.copy()
    for method in sorted(set(str(x) for x in baseline_df["method"].tolist())):
        sub = baseline_df[baseline_df["method"] == method][
            ["gene_id", "p_value", "status", "reason"]
        ].copy()
        prefix = _method_col_name(method)
        sub = sub.rename(
            columns={
                "p_value": f"{prefix}_p",
                "status": f"{prefix}_status",
                "reason": f"{prefix}_reason",
            }
        )
        out = out.merge(sub, on="gene_id", how="left")
    return out


def _run_babappa_gene_worker(
    task: tuple[int, dict[str, Any], dict[str, Any], int, str, int]
) -> tuple[int, dict[str, Any], dict[str, Any] | None]:
    idx, gene, model_payload, calibration_size, tail, seed_gene = task
    from ..model import EnergyModel

    model = EnergyModel.from_dict(model_payload)
    gene_id = str(gene["gene_id"])
    gpath = Path(str(gene["alignment_path"])).resolve()
    try:
        aln = read_fasta(gpath)
    except Exception as exc:
        return (
            idx,
            {
                "gene_id": gene_id,
                "gene_index": idx,
                "L": np.nan,
                "D_obs": np.nan,
                "mu0_hat": np.nan,
                "sigma0_hat": np.nan,
                "p": np.nan,
                "q_value": np.nan,
                "tail": tail,
                "N": int(calibration_size),
                "n_used": 0,
                "model_hash": "",
                "phi_hash": "",
                "M0_hash": "",
                "seed_gene": None,
                "seed_calib_base": None,
                "status": "FAIL",
                "reason": f"read_error:{exc}",
                "alignment_path": str(gpath),
            },
            None,
        )

    start = time.perf_counter()
    try:
        res = analyze_alignment(
            aln,
            model=model,
            calibration_size=int(calibration_size),
            calibration_mode="phylo",
            tail=tail,
            seed=seed_gene,
            seed_gene=seed_gene,
            name=gene_id,
            override_scaling=True,
        )
        runtime = time.perf_counter() - start
        return (
            idx,
            {
                "gene_id": gene_id,
                "gene_index": idx,
                "L": res.gene_length,
                "D_obs": res.dispersion,
                "mu0_hat": res.mu0,
                "sigma0_hat": res.sigma0,
                "p": res.p_value,
                "q_value": np.nan,
                "tail": res.tail,
                "N": res.calibration_size,
                "n_used": res.n_used,
                "model_hash": res.model_hash,
                "phi_hash": res.phi_hash,
                "M0_hash": res.m0_hash,
                "seed_gene": res.seed_gene,
                "seed_calib_base": res.seed_calib_base,
                "status": "OK",
                "reason": "ok",
                "alignment_path": str(gpath),
            },
            {
                "method": "babappa",
                "gene_id": gene_id,
                "runtime_sec": runtime,
                "L": res.gene_length,
            },
        )
    except Exception as exc:
        runtime = time.perf_counter() - start
        return (
            idx,
            {
                "gene_id": gene_id,
                "gene_index": idx,
                "L": np.nan,
                "D_obs": np.nan,
                "mu0_hat": np.nan,
                "sigma0_hat": np.nan,
                "p": np.nan,
                "q_value": np.nan,
                "tail": tail,
                "N": int(calibration_size),
                "n_used": 0,
                "model_hash": "",
                "phi_hash": "",
                "M0_hash": "",
                "seed_gene": seed_gene,
                "seed_calib_base": seed_gene,
                "status": "FAIL",
                "reason": f"analysis_error:{exc}",
                "alignment_path": str(gpath),
            },
            {
                "method": "babappa",
                "gene_id": gene_id,
                "runtime_sec": runtime,
                "L": np.nan,
            },
        )


def _run_babappa_gene_worker_subprocess(
    task: tuple[int, dict[str, Any], dict[str, Any], int, str, int],
) -> tuple[int, dict[str, Any], dict[str, Any] | None]:
    idx, gene, model_payload, calibration_size, tail, seed_gene = task
    gene_id = str(gene["gene_id"])
    gpath = Path(str(gene["alignment_path"])).resolve()
    payload = {
        "idx": int(idx),
        "gene": gene,
        "model_payload": model_payload,
        "calibration_size": int(calibration_size),
        "tail": str(tail),
        "seed_gene": int(seed_gene),
    }
    start = time.perf_counter()
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _BABAPPA_GENE_SUBPROC_CODE],
            input=json.dumps(payload),
            text=True,
            capture_output=True,
            check=False,
            cwd=str(Path.cwd()),
        )
    except Exception as exc:
        return (
            idx,
            {
                "gene_id": gene_id,
                "gene_index": idx,
                "L": np.nan,
                "D_obs": np.nan,
                "mu0_hat": np.nan,
                "sigma0_hat": np.nan,
                "p": np.nan,
                "q_value": np.nan,
                "tail": tail,
                "N": int(calibration_size),
                "n_used": 0,
                "model_hash": "",
                "phi_hash": "",
                "M0_hash": "",
                "seed_gene": seed_gene,
                "seed_calib_base": seed_gene,
                "status": "FAIL",
                "reason": f"subprocess_launch_error:{exc}",
                "alignment_path": str(gpath),
            },
            {
                "method": "babappa",
                "gene_id": gene_id,
                "runtime_sec": time.perf_counter() - start,
                "L": np.nan,
            },
        )

    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip().replace("\n", " | ")
        if len(stderr) > 400:
            stderr = stderr[:400]
        return (
            idx,
            {
                "gene_id": gene_id,
                "gene_index": idx,
                "L": np.nan,
                "D_obs": np.nan,
                "mu0_hat": np.nan,
                "sigma0_hat": np.nan,
                "p": np.nan,
                "q_value": np.nan,
                "tail": tail,
                "N": int(calibration_size),
                "n_used": 0,
                "model_hash": "",
                "phi_hash": "",
                "M0_hash": "",
                "seed_gene": seed_gene,
                "seed_calib_base": seed_gene,
                "status": "FAIL",
                "reason": f"subprocess_error:{stderr or proc.returncode}",
                "alignment_path": str(gpath),
            },
            {
                "method": "babappa",
                "gene_id": gene_id,
                "runtime_sec": time.perf_counter() - start,
                "L": np.nan,
            },
        )

    stdout = (proc.stdout or "").strip().splitlines()
    if not stdout:
        return (
            idx,
            {
                "gene_id": gene_id,
                "gene_index": idx,
                "L": np.nan,
                "D_obs": np.nan,
                "mu0_hat": np.nan,
                "sigma0_hat": np.nan,
                "p": np.nan,
                "q_value": np.nan,
                "tail": tail,
                "N": int(calibration_size),
                "n_used": 0,
                "model_hash": "",
                "phi_hash": "",
                "M0_hash": "",
                "seed_gene": seed_gene,
                "seed_calib_base": seed_gene,
                "status": "FAIL",
                "reason": "subprocess_output_empty",
                "alignment_path": str(gpath),
            },
            {
                "method": "babappa",
                "gene_id": gene_id,
                "runtime_sec": time.perf_counter() - start,
                "L": np.nan,
            },
        )
    try:
        payload_out = json.loads(stdout[-1])
        row = dict(payload_out.get("row", {}))
        runtime = payload_out.get("runtime", None)
        idx_out = int(payload_out.get("idx", idx))
    except Exception as exc:
        return (
            idx,
            {
                "gene_id": gene_id,
                "gene_index": idx,
                "L": np.nan,
                "D_obs": np.nan,
                "mu0_hat": np.nan,
                "sigma0_hat": np.nan,
                "p": np.nan,
                "q_value": np.nan,
                "tail": tail,
                "N": int(calibration_size),
                "n_used": 0,
                "model_hash": "",
                "phi_hash": "",
                "M0_hash": "",
                "seed_gene": seed_gene,
                "seed_calib_base": seed_gene,
                "status": "FAIL",
                "reason": f"subprocess_parse_error:{exc}",
                "alignment_path": str(gpath),
            },
            {
                "method": "babappa",
                "gene_id": gene_id,
                "runtime_sec": time.perf_counter() - start,
                "L": np.nan,
            },
        )
    if runtime is not None:
        runtime_row = dict(runtime)
    else:
        runtime_row = {
            "method": "babappa",
            "gene_id": gene_id,
            "runtime_sec": time.perf_counter() - start,
            "L": row.get("L", np.nan),
        }
    return idx_out, row, runtime_row


def _run_babappa_for_genes(
    *,
    genes: list[dict[str, Any]],
    model: Any,
    calibration_size: int,
    tail: str,
    rng: np.random.Generator,
    jobs: int = 1,
    outdir: Path | None = None,
    shard_subdir: str = "babappa_shards",
    progress_filename: str = "progress_babappa.tsv",
    heartbeat_filename: str = "heartbeat.txt",
    resume: bool = False,
    heartbeat_interval_sec: int = 300,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    n_jobs = int(jobs)
    if n_jobs <= 0:
        n_jobs = max(1, min(16, (os.cpu_count() or 1)))

    seeds = [int(rng.integers(0, 2**32 - 1)) for _ in genes]
    model_payload = model.to_dict() if hasattr(model, "to_dict") else dict(model)

    tasks = [
        (idx, gene, model_payload, int(calibration_size), str(tail), int(seeds[idx - 1]))
        for idx, gene in enumerate(genes, start=1)
    ]
    checkpoint_enabled = outdir is not None
    shards_dir = (outdir / "raw" / str(shard_subdir)) if checkpoint_enabled else None
    progress_path = (outdir / "raw" / str(progress_filename)) if checkpoint_enabled else None
    heartbeat_path = (outdir / "logs" / str(heartbeat_filename)) if checkpoint_enabled else None
    interval_sec = max(1, int(heartbeat_interval_sec))

    if checkpoint_enabled:
        assert shards_dir is not None and progress_path is not None and heartbeat_path is not None
        if bool(resume):
            shards_dir.mkdir(parents=True, exist_ok=True)
        else:
            if shards_dir.exists():
                shutil.rmtree(shards_dir)
            shards_dir.mkdir(parents=True, exist_ok=True)
            if progress_path.exists():
                progress_path.unlink()
        heartbeat_path.parent.mkdir(parents=True, exist_ok=True)

    rows_by_idx: dict[int, dict[str, Any]] = {}
    runtime_by_idx: dict[int, dict[str, Any]] = {}
    if checkpoint_enabled and bool(resume) and shards_dir is not None:
        for idx, payload in _load_shards(shards_dir).items():
            if idx < 1 or idx > len(tasks):
                continue
            row = payload.get("row", {})
            if not isinstance(row, dict):
                continue
            gene_id = str(payload.get("gene_id", row.get("gene_id", "")))
            runtime_sec = _safe_float(payload.get("runtime_sec"))
            rows_by_idx[idx] = dict(row)
            runtime_by_idx[idx] = {
                "method": "babappa",
                "gene_id": gene_id,
                "runtime_sec": (runtime_sec if runtime_sec is not None else np.nan),
                "L": row.get("L", np.nan),
            }

    tasks_to_run = [task for task in tasks if int(task[0]) not in rows_by_idx]
    total_tasks = len(tasks)
    started = time.perf_counter()
    last_heartbeat = started
    last_gene = ""
    if rows_by_idx:
        last_idx = max(rows_by_idx)
        last_gene = str(rows_by_idx.get(last_idx, {}).get("gene_id", ""))

    if checkpoint_enabled and heartbeat_path is not None:
        _write_heartbeat(
            path=heartbeat_path,
            completed=len(rows_by_idx),
            total=total_tasks,
            active_workers=min(len(tasks_to_run), max(1, n_jobs)),
            elapsed_sec=time.perf_counter() - started,
            last_gene=last_gene,
        )

    def _on_complete(idx: int, row: dict[str, Any], rt: dict[str, Any] | None, active_workers: int) -> None:
        nonlocal last_heartbeat, last_gene

        idx_i = int(idx)
        row_out = dict(row)
        rows_by_idx[idx_i] = row_out

        if rt is not None:
            rt_out = dict(rt)
        else:
            rt_out = {
                "method": "babappa",
                "gene_id": str(row_out.get("gene_id", "")),
                "runtime_sec": np.nan,
                "L": row_out.get("L", np.nan),
            }
        runtime_by_idx[idx_i] = rt_out

        gene_id = str(row_out.get("gene_id", rt_out.get("gene_id", "")))
        runtime_sec_val = _safe_float(rt_out.get("runtime_sec"))
        runtime_sec = (runtime_sec_val if runtime_sec_val is not None else np.nan)

        if checkpoint_enabled and shards_dir is not None:
            _write_json_fsync(
                shards_dir / f"{idx_i:04d}.json",
                {
                    "idx": idx_i,
                    "gene_id": gene_id,
                    "runtime_sec": runtime_sec,
                    "row": row_out,
                },
            )
        if checkpoint_enabled and progress_path is not None:
            _append_progress_row(
                progress_path,
                {
                    "idx": idx_i,
                    "gene_id": gene_id,
                    "status": str(row_out.get("status", "")),
                    "runtime_sec": runtime_sec,
                    "p_value": row_out.get("p", ""),
                    "I": row_out.get("n_used", ""),
                    "D_obs": row_out.get("D_obs", ""),
                    "mu_null": row_out.get("mu0_hat", ""),
                    "sigma_null": row_out.get("sigma0_hat", ""),
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime()),
                },
            )
        last_gene = gene_id
        now = time.perf_counter()
        if checkpoint_enabled and heartbeat_path is not None and (
            (now - last_heartbeat) >= interval_sec or len(rows_by_idx) >= total_tasks
        ):
            _write_heartbeat(
                path=heartbeat_path,
                completed=len(rows_by_idx),
                total=total_tasks,
                active_workers=max(0, int(active_workers)),
                elapsed_sec=now - started,
                last_gene=last_gene,
            )
            last_heartbeat = now

    def _collect_futures(fut_map: dict[Any, int]) -> None:
        nonlocal last_heartbeat
        while fut_map:
            done, _ = wait(set(fut_map.keys()), timeout=float(interval_sec), return_when=FIRST_COMPLETED)
            if not done:
                if checkpoint_enabled and heartbeat_path is not None:
                    now = time.perf_counter()
                    _write_heartbeat(
                        path=heartbeat_path,
                        completed=len(rows_by_idx),
                        total=total_tasks,
                        active_workers=len(fut_map),
                        elapsed_sec=now - started,
                        last_gene=last_gene,
                    )
                    last_heartbeat = now
                continue
            for fut in done:
                idx, row, rt = fut.result()
                fut_map.pop(fut, None)
                _on_complete(int(idx), row, rt, active_workers=len(fut_map))

    if tasks_to_run:
        if n_jobs == 1 or len(tasks_to_run) <= 1:
            for j, task in enumerate(tasks_to_run, start=1):
                idx, row, rt = _run_babappa_gene_worker(task)
                _on_complete(int(idx), row, rt, active_workers=max(0, len(tasks_to_run) - j))
        else:
            try:
                with ProcessPoolExecutor(max_workers=n_jobs) as ex:
                    fut_map = {ex.submit(_run_babappa_gene_worker, task): int(task[0]) for task in tasks_to_run}
                    _collect_futures(fut_map)
            except Exception:
                force_pool = str(os.environ.get("BABAPPA_FORCE_PROCESSPOOL", "")).strip().lower() in {
                    "1",
                    "true",
                    "yes",
                    "on",
                }
                if force_pool:
                    raise
                # Some environments disallow ProcessPool semaphores; use
                # subprocess workers to retain true multi-core parallelism.
                try:
                    with ThreadPoolExecutor(max_workers=n_jobs) as ex:
                        fut_map = {
                            ex.submit(_run_babappa_gene_worker_subprocess, task): int(task[0])
                            for task in tasks_to_run
                        }
                        _collect_futures(fut_map)
                except Exception:
                    # Last-resort fallback: in-process threads.
                    with ThreadPoolExecutor(max_workers=n_jobs) as ex:
                        fut_map = {ex.submit(_run_babappa_gene_worker, task): int(task[0]) for task in tasks_to_run}
                        _collect_futures(fut_map)

    if checkpoint_enabled and heartbeat_path is not None:
        _write_heartbeat(
            path=heartbeat_path,
            completed=len(rows_by_idx),
            total=total_tasks,
            active_workers=0,
            elapsed_sec=time.perf_counter() - started,
            last_gene=last_gene,
        )

    ordered_idx = sorted(rows_by_idx)
    rows = [rows_by_idx[i] for i in ordered_idx]
    runtime_rows: list[dict[str, Any]] = [runtime_by_idx[i] for i in ordered_idx if i in runtime_by_idx]
    df = pd.DataFrame(rows)
    if not df.empty:
        ok_mask = (df["status"] == "OK") & pd.to_numeric(df["p"], errors="coerce").notna()
        if bool(np.count_nonzero(ok_mask)):
            qvals = benjamini_hochberg([float(x) for x in df.loc[ok_mask, "p"].tolist()])
            df.loc[ok_mask, "q_value"] = qvals
    return df, runtime_rows


def _run_baselines_for_genes(
    *,
    methods: list[str],
    genes: list[dict[str, Any]],
    tree_path: Path,
    outdir: Path,
    payload: dict[str, Any],
    timeout_sec: int,
    container: str,
    work_prefix: str,
    jobs: int,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    if not methods or not genes:
        return pd.DataFrame(), []
    runtime_rows: list[dict[str, Any]] = []
    frames: list[pd.DataFrame] = []
    dataset_payload = {
        "schema_version": 1,
        "tree_path": str(tree_path),
        "genes": [
            {
                "gene_id": str(g["gene_id"]),
                "alignment_path": str(g["alignment_path"]),
                "tree_path": (
                    str(Path(str(g["tree_path"])).resolve())
                    if g.get("tree_path") is not None
                    else str(tree_path)
                ),
            }
            for g in genes
        ],
        "method_options": {"foreground_taxon": payload.get("foreground_taxon", "T1")},
    }
    for method in methods:
        out_tsv = outdir / f"{work_prefix}_{method}.tsv"
        run_baseline_for_dataset(
            method=method,
            dataset=dataset_payload,
            out_tsv=out_tsv,
            work_dir=outdir / "baseline_work",
            foreground_taxon=str(payload.get("foreground_taxon", "T1")),
            foreground_branch_label=None,
            timeout_sec=int(timeout_sec),
            container=container,
            jobs=int(jobs),
        )
        bdf = _read_tsv_or_empty(out_tsv)
        if bdf.empty:
            continue
        frames.append(bdf)
        for _, row in bdf.iterrows():
            runtime_rows.append(
                {
                    "method": str(row.get("method", method)),
                    "gene_id": str(row.get("gene_id", "")),
                    "runtime_sec": float(row.get("runtime_sec", 0.0)),
                    "L": np.nan,
                }
            )
    merged = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return merged, runtime_rows


def _slice_alignment_window(aln: Alignment, start_nt: int, end_nt: int) -> Alignment:
    names = list(aln.names)
    seqs = [seq[start_nt:end_nt] for seq in aln.sequences]
    return Alignment(names=tuple(names), sequences=tuple(seqs))


def _window_target_genes(track: str, preset: str, genes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if track != "viral":
        return []
    p = preset.upper()
    if p.startswith("HIV_ENV_B"):
        return list(genes)
    if p.startswith("SARS_2020"):
        picks = [
            g
            for g in genes
            if ("SPIKE" in str(g["gene_id"]).upper() or "ORF1AB" in str(g["gene_id"]).upper())
        ]
        return picks
    return []


def _build_window_gene_records(
    *,
    genes: list[dict[str, Any]],
    window_size_codons: int,
    step_codons: int,
    windows_dir: Path,
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    windows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    w_nt = int(window_size_codons) * 3
    s_nt = int(step_codons) * 3
    windows_dir.mkdir(parents=True, exist_ok=True)
    for g in genes:
        gene_id = str(g["gene_id"])
        path = Path(str(g["alignment_path"])).resolve()
        try:
            aln = read_fasta(path)
        except Exception as exc:
            audit_rows.append({"gene_id": gene_id, "status": "DROP", "reason": f"read_error:{exc}"})
            continue
        if aln.length < w_nt:
            audit_rows.append({"gene_id": gene_id, "status": "DROP", "reason": "length_lt_window"})
            continue
        count = 0
        for start in range(0, aln.length - w_nt + 1, s_nt):
            end = start + w_nt
            win = _slice_alignment_window(aln, start, end)
            w_id = f"{gene_id}__w{start // 3 + 1:05d}_{end // 3:05d}"
            w_path = windows_dir / f"{w_id}.fasta"
            _write_alignment_fasta(win, w_path)
            windows.append(
                {
                    "gene_id": w_id,
                    "alignment_path": str(w_path.resolve()),
                    "parent_gene_id": gene_id,
                    "window_start_codon": int(start // 3 + 1),
                    "window_end_codon": int(end // 3),
                }
            )
            count += 1
        audit_rows.append({"gene_id": gene_id, "status": "KEEP", "reason": f"n_windows={count}"})
    return windows, pd.DataFrame(audit_rows)


def _write_explain_reports(
    *,
    model: Any,
    candidates: pd.DataFrame,
    calibration_size: int,
    tail: str,
    seed: int,
    outdir: Path,
    max_reports: int = 3,
) -> pd.DataFrame:
    from ..explain import explain_alignment

    explain_dir = outdir / "figures" / "explain"
    explain_dir.mkdir(parents=True, exist_ok=True)
    if candidates.empty:
        return pd.DataFrame(columns=["unit_id", "status", "reason", "pdf", "json"])
    if int(max_reports) <= 0:
        return pd.DataFrame(columns=["unit_id", "status", "reason", "pdf", "json"])

    view = candidates.copy()
    view["p"] = pd.to_numeric(view["p"], errors="coerce")
    ok = view.dropna(subset=["p"]).sort_values("p")
    if ok.empty:
        return pd.DataFrame(columns=["unit_id", "status", "reason", "pdf", "json"])
    if len(ok) <= int(max_reports):
        picks = ok.copy()
    else:
        # Anchor with strong-hit and null-like examples, then spread remaining picks across the rank range.
        anchor_idx: list[int] = [0]
        if len(ok) > 1 and int(max_reports) > 1:
            anchor_idx.append(1)
        if int(max_reports) > 2:
            anchor_idx.append(len(ok) - 1)
        chosen = list(dict.fromkeys(anchor_idx))
        remaining = int(max_reports) - len(chosen)
        if remaining > 0:
            q_positions = np.linspace(0, len(ok) - 1, num=remaining + 2)[1:-1]
            for q in q_positions:
                idx = int(round(float(q)))
                if idx not in chosen:
                    chosen.append(idx)
                if len(chosen) >= int(max_reports):
                    break
        if len(chosen) < int(max_reports):
            for idx in range(len(ok)):
                if idx not in chosen:
                    chosen.append(idx)
                if len(chosen) >= int(max_reports):
                    break
        picks = ok.iloc[sorted(chosen)[: int(max_reports)]].copy()
    picks = picks.drop_duplicates(subset=["gene_id"]).head(int(max_reports))

    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(int(seed) + 191)
    for i, row in enumerate(picks.itertuples(index=False), start=1):
        unit_id = str(getattr(row, "gene_id"))
        aln_path = Path(str(getattr(row, "alignment_path"))).resolve()
        pdf = explain_dir / f"{i:02d}_{unit_id}.pdf"
        js = explain_dir / f"{i:02d}_{unit_id}.json"
        try:
            aln = read_fasta(aln_path)
            explain_alignment(
                model=model,
                alignment=aln,
                calibration_size=int(calibration_size),
                tail=tail,
                seed=int(rng.integers(0, 2**32 - 1)),
                out_pdf=pdf,
                out_json=js,
                window_size=30,
            )
            rows.append(
                {
                    "unit_id": unit_id,
                    "status": "OK",
                    "reason": "ok",
                    "pdf": str(pdf.resolve()),
                    "json": str(js.resolve()),
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "unit_id": unit_id,
                    "status": "FAIL",
                    "reason": f"explain_error:{exc}",
                    "pdf": str(pdf.resolve()),
                    "json": str(js.resolve()),
                }
            )
    return pd.DataFrame(rows)


def _run_real_track(config: BenchmarkRunConfig, track: str) -> dict[str, Any]:
    params = _real_preset_params(track, config.preset)
    dirs = _ensure_dirs(config.outdir)
    _write_rebuild_script(config.outdir)
    rng = np.random.default_rng(config.seed)
    publication_mode = _is_publication_mode(config)

    if publication_mode and bool(config.allow_baseline_fail):
        raise ValueError("Publication mode forbids --allow-baseline-fail.")
    if publication_mode and bool(config.allow_qc_fail):
        raise ValueError("Publication mode forbids --allow-qc-fail.")
    if publication_mode and not bool(config.write_plots):
        raise ValueError("Publication mode requires figure/report generation; disable of plots is not allowed.")
    if publication_mode:
        config.require_real_data = True

    foreground_species = None
    if config.foreground_species is not None and str(config.foreground_species).strip():
        foreground_species = normalize_taxon(str(config.foreground_species))
    relax_skipped_reason = ""

    # Real-data profiles always require baseline methods to run (unless a method is explicitly omitted).
    method_list = _ordered_methods(list(params["methods"]))
    if not bool(config.require_real_data) and not bool(config.include_baselines):
        method_list = []
    if bool(config.include_relax) and "relax" not in method_list:
        method_list.append("relax")
    if "relax" in method_list and foreground_species is None:
        method_list = [m for m in method_list if m != "relax"]
        relax_skipped_reason = "RELAX skipped: no --foreground-species provided."
    method_list = _ordered_methods(method_list)
    if method_list:
        _enforce_baseline_doctor(
            outdir=config.outdir,
            methods=method_list,
            allow_baseline_fail=bool(config.allow_baseline_fail and not publication_mode),
            container=config.baseline_container,
            timeout_sec=int(max(30, min(config.baseline_timeout_sec, 300))),
        )

    payload, dataset_path, synthetic = _resolve_real_dataset(
        config=config,
        track=track,
        preset=config.preset,
        outdir=config.outdir,
        rng=rng,
        taxa_n=int(params["taxa"]),
        max_genes=int(params["max_genes"]),
    )
    if bool(config.require_real_data) and bool(synthetic):
        raise ValueError("Synthetic fallback is disabled for empirical real-data runs.")

    if foreground_species is not None:
        payload["foreground_taxon"] = foreground_species

    min_taxa = (
        int(config.min_taxa)
        if config.min_taxa is not None
        else int(params.get("min_taxa", max(2, int(params["taxa"]) - 2)))
    )
    genes_all, drop_audit_df, drop_summary_df, taxa_map = _prepare_gene_records(
        payload,
        default_tree_path=Path(str(payload["tree_path"])).resolve(),
        min_len_codons=int(params["min_len_codons"]),
        max_genes=int(params["max_genes"]),
        min_taxa=int(min_taxa),
        require_full_completeness=bool(config.require_full_completeness),
        preprocessed_dir=dirs["raw"] / "preprocessed_units",
    )
    _write_tsv(dirs["raw"] / "drop_audit.tsv", drop_audit_df)
    _write_tsv(dirs["tables"] / "drop_audit_summary.tsv", drop_summary_df)
    _write_json(
        dirs["manifests"] / "taxa_map.json",
        {
            "schema_version": 1,
            "mapping_rule": "strip -> split('|','/') -> whitespace_to_underscore -> lowercase",
            "n_entries": int(len(taxa_map)),
            "map": taxa_map,
        },
    )
    ingest_final = drop_audit_df[drop_audit_df["stage_name"] == "FINAL_KEEP"].copy() if not drop_audit_df.empty else pd.DataFrame()
    ingest_drop_df = (
        ingest_final[ingest_final["status"] != "KEEP"]
        .rename(
            columns={
                "unit_id": "gene_id",
                "reason_code": "reason",
                "file_path_alignment": "alignment_path",
            }
        )
        [["gene_id", "status", "reason", "alignment_path"]]
        if not ingest_final.empty
        else pd.DataFrame(columns=["gene_id", "status", "reason", "alignment_path"])
    )
    _write_tsv(dirs["raw"] / "gene_ingestion_audit.tsv", ingest_drop_df)
    if not genes_all:
        raise ValueError("No genes available after filtering for this real-data preset.")

    metadata_path = dataset_path.parent / "metadata.tsv"
    metadata_df = _read_tsv_or_empty(metadata_path)
    if not metadata_df.empty:
        _write_tsv(dirs["raw"] / "qc_by_gene.tsv", metadata_df)
    qc_row: dict[str, Any] = {
        "dataset_path": str(dataset_path),
        "metadata_rows": int(len(metadata_df)),
        "genes_after_ingestion": int(len(genes_all)),
        "ingestion_drops": int(len(ingest_drop_df)) if not ingest_drop_df.empty else 0,
        "min_taxa": int(min_taxa),
        "require_full_completeness": bool(config.require_full_completeness),
    }
    for row in drop_summary_df.itertuples(index=False):
        qc_row[str(getattr(row, "metric"))] = int(getattr(row, "value"))
    if not metadata_df.empty:
        if "length_nt" in metadata_df.columns:
            qc_row["length_nt_median"] = float(pd.to_numeric(metadata_df["length_nt"], errors="coerce").median())
        if "missingness_fraction" in metadata_df.columns:
            qc_row["missingness_fraction_mean"] = float(
                pd.to_numeric(metadata_df["missingness_fraction"], errors="coerce").mean()
            )
        if "n_input_sequences" in metadata_df.columns:
            qc_row["n_input_sequences_max"] = int(
                pd.to_numeric(metadata_df["n_input_sequences"], errors="coerce").fillna(0).max()
            )
        if "n_retained_sequences" in metadata_df.columns:
            qc_row["n_retained_sequences_max"] = int(
                pd.to_numeric(metadata_df["n_retained_sequences"], errors="coerce").fillna(0).max()
            )
        for col in metadata_df.columns:
            if str(col).startswith("dropped_"):
                qc_row[f"{col}_sum"] = int(pd.to_numeric(metadata_df[col], errors="coerce").fillna(0).sum())
        if "collection_date" in metadata_df.columns:
            filled = metadata_df["collection_date"].astype(str).str.strip()
            qc_row["collection_date_nonempty_rate"] = float(np.mean(filled != ""))
        if "country" in metadata_df.columns:
            filled = metadata_df["country"].astype(str).str.strip()
            qc_row["country_nonempty_rate"] = float(np.mean(filled != ""))
    _write_tsv(dirs["tables"] / "qc_summary.tsv", pd.DataFrame([qc_row]))

    min_test_required = 1
    if str(config.empirical_profile) in {"ortholog_small8_full", "ortholog_real_v12"}:
        min_test_required = 300
    train_genes, test_genes = _split_train_test_genes(
        genes_all,
        seed=int(config.seed),
        train_fraction=0.2,
        min_test_genes=min_test_required,
    )
    if str(config.empirical_profile) in {"sars_2020_full", "sars_cov2_real"}:
        train_genes, test_genes = _ensure_required_test_genes(
            train_genes,
            test_genes,
            required_tokens=("spike", "orf1ab"),
        )
    if not test_genes:
        raise ValueError("No test-set genes after train/test split.")

    candidate_genes = sorted(test_genes, key=lambda g: str(g["gene_id"]))
    candidate_rows = [
        {
            "gene_id": str(g["gene_id"]),
            "alignment_path": str(Path(str(g["alignment_path"])).resolve()),
            "tree_path": (
                str(Path(str(g["tree_path"])).resolve())
                if g.get("tree_path") is not None
                else str(Path(str(payload["tree_path"])).resolve())
            ),
            "reason_codes": "post_qc_keep;test_split",
        }
        for g in candidate_genes
    ]
    candidates_df = pd.DataFrame(candidate_rows)
    _write_tsv(dirs["raw"] / "candidates.tsv", candidates_df)

    tree_path = Path(
        str(
            candidate_genes[0].get("tree_path")
            if candidate_genes and candidate_genes[0].get("tree_path")
            else payload["tree_path"]
        )
    ).resolve()
    neutral_model_path = (
        Path(str(payload["neutral_model_json"])).resolve()
        if payload.get("neutral_model_json")
        else _build_default_neutral_model_json(tree_path, dirs["raw"] / "neutral_model.json")
    )

    doctor_dir = _doctor_input_dir(candidate_genes, config.outdir)
    doctor_report = run_doctor(
        genes_dir=doctor_dir,
        tree_file=str(tree_path),
        neutral_model_json=str(neutral_model_path),
        phi_spec=None,
        strict_taxa=True,
        taxa_policy="strict",
        allow_stop_codons=False,
        allow_polytomies=False,
        filter_to_multiple_of_3=False,
        existing_model_file=None,
        seed=config.seed,
        output_dir=str(dirs["root"]),
    )
    doctor_failures = int(np.count_nonzero([c.status == "FAIL" for c in doctor_report.checks]))
    doctor_warnings = int(np.count_nonzero([c.status == "WARN" for c in doctor_report.checks]))
    (dirs["logs"] / "doctor_report.txt").write_text(doctor_report.render() + "\n", encoding="utf-8")
    _write_json(
        dirs["manifests"] / "doctor_report.json",
        {
            "schema_version": 1,
            "track": track,
            "preset": config.preset,
            "dataset_path": str(dataset_path),
            "synthetic_fallback": synthetic,
            "has_failures": doctor_report.has_failures,
            "num_failures": doctor_failures,
            "num_warnings": doctor_warnings,
            "checks": [
                {"name": c.name, "status": c.status, "message": c.message, "fix": c.fix}
                for c in doctor_report.checks
            ],
        },
    )
    if doctor_report.has_failures and not bool(config.allow_qc_fail):
        raise ValueError(
            "Doctor reported failures. Re-run with --allow-qc-fail to continue and record failures."
        )

    neutral_spec, neutral_payload = load_neutral_model_spec(neutral_model_path)
    model = train_energy_model_from_neutral_spec(
        neutral_spec=neutral_spec,
        sim_replicates=int(params["training_replicates"]),
        sim_length=int(params["training_length_nt"]),
        seed=int(rng.integers(0, 2**32 - 1)),
    )
    from ..model import save_model

    model_path = dirs["raw"] / "frozen_model.json"
    save_model(model, model_path)
    m0_payload = model.neutral_spec.to_dict() if getattr(model, "neutral_spec", None) is not None else neutral_payload
    m0_hash = sha256_json(m0_payload)
    model_hash = sha256_json(model.to_dict())
    expected_phi_hash = sha256_json(
        {
            "schema_version": 1,
            "name": "phi_default_v1",
            "feature_names": list(FEATURE_NAMES),
        }
    )
    saved_model_hash = sha256_json(json.loads(model_path.read_text(encoding="utf-8")))
    model_hash_match = bool(saved_model_hash == model_hash)
    if publication_mode and not model_hash_match:
        raise ValueError(
            "Publication mode failed frozen-model immutability check: "
            "serialized model hash does not match trained model hash."
        )
    _write_json(
        dirs["manifests"] / "training_manifest.json",
        {
            "schema_version": 1,
            "track": track,
            "preset": config.preset,
            "model_path": str(model_path.resolve()),
            "model_hash": model_hash,
            "saved_model_hash": saved_model_hash,
            "model_hash_match": model_hash_match,
            "m0_hash": m0_hash,
            "phi_hash": expected_phi_hash,
            "training_samples_n": model.training_samples,
            "L_train": model.max_training_length,
            "training_mode": model.training_mode,
            "train_set_gene_ids": [str(g["gene_id"]) for g in train_genes],
            "test_set_gene_ids": [str(g["gene_id"]) for g in candidate_genes],
            "candidate_gene_ids": [str(g["gene_id"]) for g in candidate_genes],
            "neutral_model_payload": neutral_payload,
            "neutral_model_path": str(neutral_model_path),
            "m0_hash_source": "model.neutral_spec.to_dict",
            "leakage_policy": "train_set/test_set disjoint; energy model trained from frozen M0 simulations only",
        },
    )

    calib_n = int(config.calibration_size_override) if config.calibration_size_override else int(params["calibration_size"])
    if publication_mode and calib_n < 999:
        raise ValueError(f"Publication mode requires N >= 999; got N={calib_n}.")
    full_df, full_runtime_rows = _run_babappa_for_genes(
        genes=candidate_genes,
        model=model,
        calibration_size=calib_n,
        tail=config.tail,
        rng=rng,
        jobs=int(config.jobs),
        outdir=dirs["root"],
        shard_subdir="babappa_shards",
        progress_filename="progress_babappa.tsv",
        heartbeat_filename="heartbeat.txt",
        resume=bool(config.resume),
        heartbeat_interval_sec=300,
    )
    full_df["unit_kind"] = "full_gene"
    _write_tsv(dirs["raw"] / "babappa_results.tsv", full_df)
    if publication_mode:
        ok_full = full_df[full_df["status"] == "OK"].copy()
        if not ok_full.empty:
            bad = ok_full[ok_full["model_hash"].astype(str) != str(model_hash)]
            if not bad.empty:
                raise ValueError(
                    "Publication mode failed frozen-model immutability check: "
                    "analyzed rows contain model_hash different from frozen model."
                )
    runtime_hash_rows: list[dict[str, Any]] = [
        {
            "event": "after_training",
            "gene_id": "",
            "energy_model_hash": str(model_hash),
            "phi_hash": str(expected_phi_hash),
            "m0_hash": str(m0_hash),
            "status": "OK",
            "reason": "model_frozen",
        }
    ]
    for g in candidate_genes:
        runtime_hash_rows.append(
            {
                "event": "before_gene_eval",
                "gene_id": str(g["gene_id"]),
                "energy_model_hash": str(model_hash),
                "phi_hash": str(expected_phi_hash),
                "m0_hash": str(m0_hash),
                "status": "OK",
                "reason": "expected_frozen_hash",
            }
        )
    for row in full_df.itertuples(index=False):
        runtime_hash_rows.append(
            {
                "event": "after_gene_calibration",
                "gene_id": str(getattr(row, "gene_id", "")),
                "energy_model_hash": str(getattr(row, "model_hash", "")),
                "phi_hash": str(getattr(row, "phi_hash", "")),
                "m0_hash": str(getattr(row, "M0_hash", "")),
                "status": str(getattr(row, "status", "")),
                "reason": str(getattr(row, "reason", "")),
            }
        )
    _write_tsv(dirs["raw"] / "frozen_energy_runtime_hashes.tsv", pd.DataFrame(runtime_hash_rows))

    baseline_df_full, baseline_runtime_rows = _run_baselines_for_genes(
        methods=method_list,
        genes=candidate_genes,
        tree_path=tree_path,
        outdir=dirs["raw"],
        payload=payload,
        timeout_sec=int(config.baseline_timeout_sec),
        container=config.baseline_container,
        work_prefix="baseline_full",
        jobs=int(config.jobs),
    )
    if not baseline_df_full.empty:
        baseline_df_full["unit_kind"] = "full_gene"

    window_df = pd.DataFrame()
    window_baseline_df = pd.DataFrame()
    window_audit = pd.DataFrame(columns=["gene_id", "status", "reason"])
    window_runtime_rows: list[dict[str, Any]] = []
    window_targets = _window_target_genes(track, config.preset, candidate_genes)
    if window_targets:
        windows, window_audit = _build_window_gene_records(
            genes=window_targets,
            window_size_codons=300,
            step_codons=50,
            windows_dir=dirs["raw"] / "windows",
        )
        if windows:
            window_df, window_runtime_rows = _run_babappa_for_genes(
                genes=windows,
                model=model,
                calibration_size=calib_n,
                tail=config.tail,
                rng=rng,
                jobs=int(config.jobs),
            )
            meta = pd.DataFrame(windows)
            window_df = window_df.merge(
                meta[["gene_id", "parent_gene_id", "window_start_codon", "window_end_codon"]],
                on="gene_id",
                how="left",
            )
            window_df["unit_kind"] = "window"
            _write_tsv(dirs["raw"] / "window_babappa.tsv", window_df)
            top_windows = window_df.sort_values(["status", "p"], ascending=[True, True]).head(200)
            _write_tsv(dirs["tables"] / "top_windows.tsv", top_windows)

            window_baseline_methods = [m for m in method_list if m in {"busted", "relax"}]
            if window_baseline_methods:
                ok_windows = window_df[window_df["status"] == "OK"].copy()
                ok_windows["p"] = pd.to_numeric(ok_windows["p"], errors="coerce")
                topw = ok_windows.sort_values("p").head(20)["gene_id"].tolist()
                if topw:
                    top_genes = [w for w in windows if str(w["gene_id"]) in set(str(x) for x in topw)]
                    window_baseline_df, win_base_runtime = _run_baselines_for_genes(
                        methods=window_baseline_methods,
                        genes=top_genes,
                        tree_path=tree_path,
                        outdir=dirs["raw"],
                        payload=payload,
                        timeout_sec=int(config.baseline_timeout_sec),
                        container=config.baseline_container,
                        work_prefix="baseline_windows",
                        jobs=int(config.jobs),
                    )
                    if not window_baseline_df.empty:
                        window_baseline_df["unit_kind"] = "window"
                    baseline_runtime_rows.extend(win_base_runtime)
        _write_tsv(dirs["raw"] / "window_generation_audit.tsv", window_audit)

    baseline_df = (
        pd.concat([baseline_df_full, window_baseline_df], ignore_index=True)
        if not baseline_df_full.empty or not window_baseline_df.empty
        else pd.DataFrame()
    )
    _write_tsv(dirs["raw"] / "baseline_all.tsv", baseline_df)

    runtime_rows = []
    for row in full_runtime_rows:
        row2 = dict(row)
        row2["track"] = track
        row2["unit_kind"] = "full_gene"
        runtime_rows.append(row2)
    for row in window_runtime_rows:
        row2 = dict(row)
        row2["track"] = track
        row2["unit_kind"] = "window"
        runtime_rows.append(row2)
    for row in baseline_runtime_rows:
        row2 = dict(row)
        row2["track"] = track
        row2["unit_kind"] = "baseline"
        runtime_rows.append(row2)
    runtime_df = pd.DataFrame(runtime_rows)
    _write_tsv(dirs["raw"] / "runtime.tsv", runtime_df)

    discovery_full = _build_discovery_table(full_df, baseline_df_full)
    _write_tsv(dirs["tables"] / "discovery_full.tsv", discovery_full)
    top50 = discovery_full.sort_values("babappa_p").head(50) if not discovery_full.empty else discovery_full
    _write_tsv(dirs["tables"] / "top50_ranked_genes.tsv", top50)

    orth = pd.DataFrame()
    testable_n = 0
    testable_mismatch = False
    n_success_babappa = int(np.count_nonzero(full_df.get("status", pd.Series([], dtype=str)).astype(str) == "OK"))
    busted_full_status = (
        baseline_df_full[baseline_df_full["method"] == "busted"]["status"].astype(str)
        if not baseline_df_full.empty and "method" in baseline_df_full.columns
        else pd.Series([], dtype=str)
    )
    n_success_busted = int(np.count_nonzero(busted_full_status == "OK"))
    if track == "ortholog":
        candidate_ids = (
            candidates_df["gene_id"].astype(str).tolist()
            if not candidates_df.empty and "gene_id" in candidates_df.columns
            else full_df.get("gene_id", pd.Series([], dtype=str)).astype(str).tolist()
        )
        orth = pd.DataFrame({"gene_id": candidate_ids})
        taxa_map_full = {str(g["gene_id"]): int(g.get("n_taxa", 0)) for g in candidate_genes}
        full_one = full_df.copy()
        if not full_one.empty:
            full_one["gene_id"] = full_one["gene_id"].astype(str)
            full_one = full_one.drop_duplicates(subset=["gene_id"], keep="last")
        full_one = full_one.set_index("gene_id") if not full_one.empty else pd.DataFrame()
        orth["L_codons"] = pd.to_numeric(orth["gene_id"].map(full_one.get("L", pd.Series(dtype=float))), errors="coerce") / 3.0
        orth["n_taxa"] = orth["gene_id"].map(taxa_map_full).fillna(np.nan)
        orth["babappa_p"] = pd.to_numeric(orth["gene_id"].map(full_one.get("p", pd.Series(dtype=float))), errors="coerce")
        orth["babappa_reason"] = (
            orth["gene_id"].map(full_one.get("reason", pd.Series(dtype=str))).fillna("not_run").astype(str)
        )
        orth["babappa_status"] = [
            _normalize_method_status(
                orth["gene_id"].map(full_one.get("status", pd.Series(dtype=str))).iloc[i],
                orth["babappa_reason"].iloc[i],
            )
            for i in range(len(orth))
        ]
        b_rt = runtime_df[runtime_df["method"] == "babappa"][["gene_id", "runtime_sec"]].copy()
        b_rt["gene_id"] = b_rt["gene_id"].astype(str)
        b_rt = b_rt.drop_duplicates(subset=["gene_id"], keep="last")
        orth = orth.merge(
            b_rt.rename(columns={"runtime_sec": "babappa_runtime_sec"}),
            on="gene_id",
            how="left",
        )

        busted = baseline_df_full[baseline_df_full["method"] == "busted"].copy() if not baseline_df_full.empty else pd.DataFrame()
        if not busted.empty:
            busted["gene_id"] = busted["gene_id"].astype(str)
            busted["busted_p"] = pd.to_numeric(busted["p_value"], errors="coerce")
            busted = busted.drop_duplicates(subset=["gene_id"], keep="last")
            busted = busted.set_index("gene_id")
            orth["busted_p"] = pd.to_numeric(orth["gene_id"].map(busted.get("busted_p", pd.Series(dtype=float))), errors="coerce")
            orth["busted_reason"] = (
                orth["gene_id"].map(busted.get("reason", pd.Series(dtype=str))).fillna("not_run").astype(str)
            )
            orth["busted_status"] = [
                _normalize_method_status(
                    orth["gene_id"].map(busted.get("status", pd.Series(dtype=str))).iloc[i],
                    orth["busted_reason"].iloc[i],
                )
                for i in range(len(orth))
            ]
            busted_rt = busted.reset_index()[["gene_id", "runtime_sec"]].rename(columns={"runtime_sec": "busted_runtime_sec"})
            busted_rt = busted_rt.drop_duplicates(subset=["gene_id"], keep="last")
            orth = orth.merge(busted_rt, on="gene_id", how="left")
            busted_seen = set(str(x) for x in busted.index.tolist())
        else:
            orth["busted_p"] = np.nan
            orth["busted_reason"] = "not_run"
            orth["busted_status"] = "SKIP_not_run"
            orth["busted_runtime_sec"] = np.nan
            busted_seen = set()

        babappa_seen = set(str(x) for x in full_df.get("gene_id", pd.Series([], dtype=str)).astype(str).tolist())
        orth["in_testable_set"] = (
            (orth["babappa_status"].astype(str) == "OK")
            & (orth["busted_status"].astype(str) == "OK")
        )
        orth["babappa_q_all_candidates"] = _bh_q_values_in_scope(
            orth["babappa_p"],
            orth["babappa_status"].astype(str) == "OK",
        )
        orth["busted_q_all_candidates"] = _bh_q_values_in_scope(
            orth["busted_p"],
            orth["busted_status"].astype(str) == "OK",
        )
        orth["babappa_q_testable_set"] = _bh_q_values_in_scope(
            orth["babappa_p"],
            orth["in_testable_set"],
        )
        orth["busted_q_testable_set"] = _bh_q_values_in_scope(
            orth["busted_p"],
            orth["in_testable_set"],
        )
        # Default q columns are testable-set scoped to keep overlap metrics aligned.
        orth["babappa_q"] = orth["babappa_q_testable_set"]
        orth["busted_q"] = orth["busted_q_testable_set"]

        testable_diff_df, testable_mismatch = _build_testable_set_diff(
            candidates=candidates_df,
            orth=orth,
            babappa_seen=babappa_seen,
            busted_seen=busted_seen,
        )
        _write_tsv(dirs["tables"] / "testable_set_diff.tsv", testable_diff_df)

        testable_n = int(np.count_nonzero(orth["in_testable_set"]))
        n_success_babappa = int(np.count_nonzero(orth["babappa_status"].astype(str) == "OK"))
        n_success_busted = int(np.count_nonzero(orth["busted_status"].astype(str) == "OK"))
        overlap_testable = int(
            np.count_nonzero(
                (pd.to_numeric(orth["babappa_q_testable_set"], errors="coerce") <= 0.1)
                & (pd.to_numeric(orth["busted_q_testable_set"], errors="coerce") <= 0.1)
            )
        )
        method_scope_rows = [
            {
                "method": "babappa",
                "n_candidates": int(len(orth)),
                "n_ok_all_candidates": int(np.count_nonzero(orth["babappa_status"].astype(str) == "OK")),
                "n_testable_set": int(testable_n),
                "q_lt_0.1_on_testable_set": int(
                    np.count_nonzero(pd.to_numeric(orth["babappa_q_testable_set"], errors="coerce") <= 0.1)
                ),
                "q_lt_0.1_on_all_candidates": int(
                    np.count_nonzero(pd.to_numeric(orth["babappa_q_all_candidates"], errors="coerce") <= 0.1)
                ),
            },
            {
                "method": "busted",
                "n_candidates": int(len(orth)),
                "n_ok_all_candidates": int(np.count_nonzero(orth["busted_status"].astype(str) == "OK")),
                "n_testable_set": int(testable_n),
                "q_lt_0.1_on_testable_set": int(
                    np.count_nonzero(pd.to_numeric(orth["busted_q_testable_set"], errors="coerce") <= 0.1)
                ),
                "q_lt_0.1_on_all_candidates": int(
                    np.count_nonzero(pd.to_numeric(orth["busted_q_all_candidates"], errors="coerce") <= 0.1)
                ),
            },
            {
                "method": "overlap",
                "n_candidates": int(len(orth)),
                "n_ok_all_candidates": int(
                    np.count_nonzero(
                        (orth["babappa_status"].astype(str) == "OK")
                        & (orth["busted_status"].astype(str) == "OK")
                    )
                ),
                "n_testable_set": int(testable_n),
                "q_lt_0.1_on_testable_set": int(overlap_testable),
                "q_lt_0.1_on_all_candidates": np.nan,
            },
        ]
        _write_tsv(dirs["tables"] / "testable_set_summary.tsv", pd.DataFrame(method_scope_rows))

        orth["notes"] = ""
        _write_tsv(
            dirs["tables"] / "ortholog_results.tsv",
            orth[
                [
                    "gene_id",
                    "L_codons",
                    "n_taxa",
                    "in_testable_set",
                    "babappa_p",
                    "babappa_q",
                    "babappa_q_testable_set",
                    "babappa_q_all_candidates",
                    "busted_p",
                    "busted_q",
                    "busted_q_testable_set",
                    "busted_q_all_candidates",
                    "babappa_status",
                    "busted_status",
                    "babappa_reason",
                    "busted_reason",
                    "babappa_runtime_sec",
                    "busted_runtime_sec",
                    "notes",
                ]
            ],
        )

    if track == "ortholog" and not orth.empty:
        testable_mask = orth["in_testable_set"].astype(bool)
        t3_counts = pd.DataFrame(
            [
                {
                    "method": "babappa",
                    "n_total": int(len(orth)),
                    "n_testable_set": int(np.count_nonzero(testable_mask)),
                    "n_ok_all_candidates": int(np.count_nonzero(orth["babappa_status"].astype(str) == "OK")),
                    "n_sig_p_0_05": int(np.count_nonzero(pd.to_numeric(orth["babappa_p"], errors="coerce") <= 0.05)),
                    "n_sig_q_0_1": int(
                        np.count_nonzero(pd.to_numeric(orth["babappa_q_testable_set"], errors="coerce") <= 0.1)
                    ),
                    "q_lt_0.1_on_testable_set": int(
                        np.count_nonzero(pd.to_numeric(orth["babappa_q_testable_set"], errors="coerce") <= 0.1)
                    ),
                    "q_lt_0.1_on_all_candidates": int(
                        np.count_nonzero(pd.to_numeric(orth["babappa_q_all_candidates"], errors="coerce") <= 0.1)
                    ),
                },
                {
                    "method": "busted",
                    "n_total": int(len(orth)),
                    "n_testable_set": int(np.count_nonzero(testable_mask)),
                    "n_ok_all_candidates": int(np.count_nonzero(orth["busted_status"].astype(str) == "OK")),
                    "n_sig_p_0_05": int(np.count_nonzero(pd.to_numeric(orth["busted_p"], errors="coerce") <= 0.05)),
                    "n_sig_q_0_1": int(
                        np.count_nonzero(pd.to_numeric(orth["busted_q_testable_set"], errors="coerce") <= 0.1)
                    ),
                    "q_lt_0.1_on_testable_set": int(
                        np.count_nonzero(pd.to_numeric(orth["busted_q_testable_set"], errors="coerce") <= 0.1)
                    ),
                    "q_lt_0.1_on_all_candidates": int(
                        np.count_nonzero(pd.to_numeric(orth["busted_q_all_candidates"], errors="coerce") <= 0.1)
                    ),
                },
            ]
        )
        _write_tsv(dirs["tables"] / "discovery_counts.tsv", t3_counts)

        overlap = orth[["gene_id", "in_testable_set"]].copy()
        overlap["babappa_sig"] = pd.to_numeric(orth["babappa_q_testable_set"], errors="coerce") <= 0.1
        overlap["busted_sig"] = pd.to_numeric(orth["busted_q_testable_set"], errors="coerce") <= 0.1
        baseline_methods_overlap = ["busted"]
    else:
        t3_counts_rows = [
            {
                "method": "babappa",
                "n_total": int(len(full_df)),
                "n_sig_p_0_05": int(np.count_nonzero(pd.to_numeric(full_df["p"], errors="coerce") <= 0.05)),
                "n_sig_q_0_1": int(np.count_nonzero(pd.to_numeric(full_df["q_value"], errors="coerce") <= 0.1)),
            }
        ]
        if not baseline_df_full.empty:
            for method in sorted(set(str(x) for x in baseline_df_full["method"].tolist())):
                sub = baseline_df_full[baseline_df_full["method"] == method]
                ok = sub[sub["status"] == "OK"]
                t3_counts_rows.append(
                    {
                        "method": method,
                        "n_total": int(len(sub)),
                        "n_sig_p_0_05": int(np.count_nonzero(pd.to_numeric(ok["p_value"], errors="coerce") <= 0.05)),
                        "n_sig_q_0_1": np.nan,
                    }
                )
        t3_counts = pd.DataFrame(t3_counts_rows)
        _write_tsv(dirs["tables"] / "discovery_counts.tsv", t3_counts)

        baseline_methods_overlap = _ordered_methods([str(x) for x in baseline_df_full["method"].tolist()]) if not baseline_df_full.empty else []
        overlap = pd.DataFrame({"gene_id": full_df["gene_id"].astype(str)})
        overlap["babappa_sig"] = pd.to_numeric(full_df["q_value"], errors="coerce") <= 0.1
        for m in baseline_methods_overlap:
            sub = baseline_df_full[(baseline_df_full["method"] == m) & (baseline_df_full["status"] == "OK")]
            p_map = pd.to_numeric(sub["p_value"], errors="coerce")
            sig_map = {g: bool(p <= 0.05) for g, p in zip(sub["gene_id"], p_map)}
            overlap[f"{m}_sig"] = overlap["gene_id"].map(sig_map).fillna(False)

    combo_order = ["babappa"] + baseline_methods_overlap
    combo_columns = {"babappa": "babappa_sig"}
    for m in baseline_methods_overlap:
        combo_columns[m] = f"{m}_sig"

    overlap["combo"] = overlap.apply(
        lambda r: "|".join(
            (
                _combo_symbol(method)
                if bool(r[combo_columns[method]])
                else _combo_symbol(method).lower()
            )
            for method in combo_order
        ),
        axis=1,
    )
    overlap_for_combo = overlap[overlap["in_testable_set"].astype(bool)].copy() if "in_testable_set" in overlap.columns else overlap
    combos = overlap_for_combo.groupby("combo", as_index=False).size().rename(columns={"size": "count"})

    if track == "ortholog":
        _write_tsv(dirs["tables"] / "T3_ortholog_discovery.tsv", t3_counts)
        _write_tsv(dirs["tables"] / "T3_ortholog_overlap_matrix.tsv", overlap)
    else:
        _write_tsv(dirs["tables"] / "T5_viral_summary.tsv", t3_counts)
        _write_tsv(dirs["tables"] / "T5_viral_overlap_matrix.tsv", overlap)

    fail_rows: list[dict[str, Any]] = []
    if not ingest_drop_df.empty:
        for row in ingest_drop_df.itertuples(index=False):
            fail_rows.append(
                {
                    "method": "ingestion",
                    "status": str(getattr(row, "status", "DROP")),
                    "reason": str(getattr(row, "reason", "drop")),
                }
            )
    for row in full_df[full_df["status"] != "OK"].itertuples(index=False):
        fail_rows.append({"method": "babappa", "status": "FAIL", "reason": str(getattr(row, "reason", "unknown"))})
    if not baseline_df.empty:
        for row in baseline_df[baseline_df["status"] != "OK"].itertuples(index=False):
            fail_rows.append(
                {
                    "method": str(getattr(row, "method", "baseline")),
                    "status": str(getattr(row, "status", "FAIL")),
                    "reason": str(getattr(row, "reason", "unknown")),
                }
            )
    failure_df = pd.DataFrame(fail_rows)
    if failure_df.empty:
        t6 = pd.DataFrame(columns=["method", "status", "reason", "n"])
    else:
        t6 = (
            failure_df.groupby(["method", "status", "reason"], as_index=False)
            .size()
            .rename(columns={"size": "n"})
            .sort_values(["method", "status", "n"], ascending=[True, True, False])
        )
    _write_tsv(dirs["tables"] / "T6_failure_rates.tsv", t6)
    _write_tsv(dirs["tables"] / "failure_audit.tsv", t6)

    enrich_df = pd.DataFrame()
    if track == "ortholog":
        pos_set: set[str] = set()
        if isinstance(payload.get("selectome_positive_genes"), list):
            pos_set = {str(x) for x in payload["selectome_positive_genes"]}
        elif payload.get("selectome_positive_path"):
            sp = Path(str(payload["selectome_positive_path"]))
            if sp.exists():
                pos_set = {x.strip() for x in sp.read_text(encoding="utf-8").splitlines() if x.strip()}
        if pos_set:
            discovered = set(overlap[overlap["babappa_sig"]]["gene_id"].astype(str))
            all_genes = set(overlap["gene_id"].astype(str))
            a = len(discovered & pos_set)
            b = len(discovered - pos_set)
            c = len((all_genes - discovered) & pos_set)
            d = len((all_genes - discovered) - pos_set)
            odds, ci_low, ci_high, p_fisher = _fisher_enrichment(a, b, c, d)
            enrich_df = pd.DataFrame(
                [
                    {
                        "label": "Selectome enrichment",
                        "a_disc_pos": a,
                        "b_disc_neg": b,
                        "c_nondisc_pos": c,
                        "d_nondisc_neg": d,
                        "odds_ratio": odds,
                        "ci95_low": ci_low,
                        "ci95_high": ci_high,
                        "p_value_fisher_one_sided": p_fisher,
                    }
                ]
            )
        else:
            enrich_df = pd.DataFrame([{"label": "Selectome enrichment", "note": "No Selectome mapping provided."}])
        _write_tsv(dirs["tables"] / "T4_enrichment.tsv", enrich_df)

    if config.write_plots:
        explain_candidates = full_df[full_df["status"] == "OK"][["gene_id", "alignment_path", "p"]].copy()
        if not window_df.empty:
            ok_w = window_df[window_df["status"] == "OK"][["gene_id", "alignment_path", "p"]].copy()
            explain_candidates = pd.concat([explain_candidates, ok_w], ignore_index=True)
        explain_n = 5 if str(config.empirical_profile) in {"ortholog_small8_full", "ortholog_real_v12"} else 3
        explain_df = _write_explain_reports(
            model=model,
            candidates=explain_candidates,
            calibration_size=calib_n,
            tail=config.tail,
            seed=int(config.seed),
            outdir=dirs["root"],
            max_reports=explain_n,
        )
    else:
        explain_df = pd.DataFrame(columns=["unit_id", "status", "reason", "pdf", "json"])
    _write_tsv(dirs["tables"] / "explain_reports.tsv", explain_df)

    if config.write_plots:
        plots = _plot_module()
        if track == "ortholog":
            plots.plot_overlap_upset(combos, dirs["figures"] / "F3_ortholog_overlap_upset.pdf", "F3: OrthoMaM overlap")
            if not baseline_df_full.empty:
                for method in _ordered_methods([str(x) for x in baseline_df_full["method"].tolist()]):
                    fig_name = f"F3_scatter_babappa_vs_{method}.pdf"
                    sub = baseline_df_full[
                        (baseline_df_full["method"] == method) & (baseline_df_full["status"] == "OK")
                    ].copy()
                    merged = (
                        full_df.merge(
                            sub[["gene_id", "p_value"]].rename(columns={"p_value": "baseline_p"}),
                            on="gene_id",
                            how="inner",
                        )
                        if not sub.empty
                        else pd.DataFrame()
                    )
                    if not merged.empty:
                        merged["baseline_p"] = pd.to_numeric(merged["baseline_p"], errors="coerce")
                    plots.plot_scatter_neglog10(
                        merged.dropna(subset=["p", "baseline_p"]) if not merged.empty else pd.DataFrame(),
                        x_col="p",
                        y_col="baseline_p",
                        out_pdf=dirs["figures"] / fig_name,
                        title=f"F3: -log10(p) BABAPPA vs {method}",
                        x_label="-log10 p (BABAPPA)",
                        y_label=f"-log10 p ({method})",
                    )
            plots.plot_pvalue_histograms(full_df, baseline_df_full, dirs["figures"] / "pvalue_histograms.pdf")
            plots.plot_runtime_boxplot(runtime_df, dirs["figures"] / "runtime_boxplot.pdf", "Runtime per Gene by Method")
            plots.plot_enrichment(enrich_df, dirs["figures"] / "F4_selectome_enrichment.pdf")
        else:
            if config.preset.upper().startswith("HIV_ENV_B"):
                plots.plot_viral_signal(full_df, dirs["figures"] / "F5_hiv_stress_test.pdf", "F5: HIV full-gene signal")
            else:
                plots.plot_viral_signal(full_df, dirs["figures"] / "F6_sars_stress_test.pdf", "F6: SARS-CoV-2 ORF signal")
            plots.plot_pvalue_histograms(full_df, baseline_df_full, dirs["figures"] / "pvalue_histograms.pdf")
            plots.plot_runtime_boxplot(runtime_df, dirs["figures"] / "runtime_boxplot.pdf", "Runtime per Unit by Method")
            if not window_df.empty:
                plots.plot_manhattan_windows(
                    window_df,
                    window_baseline_df if not window_baseline_df.empty else pd.DataFrame(),
                    dirs["figures"] / "window_manhattan.pdf",
                    "Window-level Manhattan Plot",
                )
                window_overlap = pd.DataFrame({"gene_id": window_df["gene_id"].astype(str)})
                window_overlap["babappa_sig"] = pd.to_numeric(window_df["q_value"], errors="coerce") <= 0.1
                window_methods = (
                    _ordered_methods([str(x) for x in window_baseline_df["method"].tolist()])
                    if not window_baseline_df.empty
                    else []
                )
                for method in window_methods:
                    ok_method = window_baseline_df[
                        (window_baseline_df["method"] == method) & (window_baseline_df["status"] == "OK")
                    ]
                    sig_map = {
                        str(g): bool(float(p) <= 0.05)
                        for g, p in zip(
                            ok_method["gene_id"],
                            pd.to_numeric(ok_method["p_value"], errors="coerce"),
                        )
                    }
                    window_overlap[f"{method}_sig"] = window_overlap["gene_id"].map(sig_map).fillna(False)
                combo_methods = ["babappa"] + window_methods
                combo_cols = {"babappa": "babappa_sig"}
                for method in window_methods:
                    combo_cols[method] = f"{method}_sig"
                window_overlap["combo"] = window_overlap.apply(
                    lambda r: "|".join(
                        (
                            _combo_symbol(method)
                            if bool(r[combo_cols[method]])
                            else _combo_symbol(method).lower()
                        )
                        for method in combo_methods
                    ),
                    axis=1,
                )
                w_combos = window_overlap.groupby("combo", as_index=False).size().rename(columns={"size": "count"})
                _write_tsv(dirs["tables"] / "window_overlap.tsv", window_overlap)
                plots.plot_overlap_upset(w_combos, dirs["figures"] / "window_overlap_upset.pdf", "Window hit overlap")
            if config.preset.upper().startswith("SARS_2020"):
                plots.plot_orf_method_bar(full_df, baseline_df_full, dirs["figures"] / "orf_method_bar.pdf")

        if not metadata_df.empty:
            if "length_nt" in metadata_df.columns:
                plots.plot_length_distribution(
                    pd.to_numeric(metadata_df["length_nt"], errors="coerce").to_numpy(dtype=float),
                    dirs["figures"] / "qc_length_distribution.pdf",
                    title="QC: Alignment length distribution",
                )
            if "missingness_fraction" in metadata_df.columns:
                plots.plot_gap_fraction_distribution(
                    pd.to_numeric(metadata_df["missingness_fraction"], errors="coerce").to_numpy(dtype=float),
                    dirs["figures"] / "qc_gap_fraction_distribution.pdf",
                    title="QC: Gap/missingness distribution",
                )
        try:
            tree_text = tree_path.read_text(encoding="utf-8")
        except Exception:
            tree_text = ""
        plots.plot_branch_length_histogram(
            tree_text,
            dirs["figures"] / "qc_branch_length_histogram.pdf",
            title="QC: Branch length histogram",
        )

        plots.plot_runtime_scaling(runtime_df, dirs["figures"] / f"runtime_{track}.pdf", f"{track.title()} runtime")

    summary = {
        "track": track,
        "preset": config.preset,
        "mode": str(config.mode).strip().lower(),
        "publication_mode": publication_mode,
        "synthetic_fallback": bool(synthetic),
        "n_input_genes": int(len(genes_all)),
        "n_train_genes": int(len(train_genes)),
        "n_test_genes": int(len(candidate_genes)),
        "n_babappa_rows": int(len(full_df)),
        "n_window_rows": int(len(window_df)) if not window_df.empty else 0,
        "n_baseline_rows": int(len(baseline_df)),
        "calibration_size": int(calib_n),
        "doctor_failures": doctor_failures,
        "doctor_warnings": doctor_warnings,
        "n_explain_reports_ok": int(np.count_nonzero(explain_df["status"] == "OK")) if not explain_df.empty else 0,
        "m0_hash": m0_hash,
        "model_hash": model_hash,
        "min_taxa": int(min_taxa),
        "require_full_completeness": bool(config.require_full_completeness),
        "foreground_species": foreground_species,
        "relax_skipped_reason": relax_skipped_reason,
        "n_success_babappa": int(n_success_babappa),
        "n_success_busted": int(n_success_busted),
        "testable_set_size": int(testable_n),
        "testable_set_mismatch": bool(testable_mismatch),
    }

    if config.empirical_profile in {"ortholog_small8_full", "ortholog_real_v12"}:
        if int(len(candidate_genes)) < 300:
            raise ValueError(f"ortholog_small8_full requires >=300 analyzed genes; got {len(candidate_genes)}")
        for method in ("busted",):
            if baseline_df_full.empty or method not in set(str(x) for x in baseline_df_full["method"].tolist()):
                raise ValueError(f"ortholog_small8_full requires baseline method={method} results.")
            sub = baseline_df_full[baseline_df_full["method"] == method]
            ok_rate = float(np.mean(sub["status"] == "OK")) if len(sub) else 0.0
            if ok_rate < 0.95:
                raise ValueError(f"ortholog_small8_full requires >=95% baseline success for {method}; got {ok_rate:.3f}")

    if config.empirical_profile in {"hiv_env_b_full", "hiv_env_b_real", "sars_2020_full", "sars_cov2_real"}:
        metadata_path = dataset_path.parent / "metadata.tsv"
        retained = 0
        if metadata_path.exists():
            mdf = _read_tsv_or_empty(metadata_path)
            if not mdf.empty:
                if "n_retained_sequences" in mdf.columns:
                    retained = int(pd.to_numeric(mdf["n_retained_sequences"], errors="coerce").fillna(0).max())
                elif "n_taxa" in mdf.columns:
                    retained = int(pd.to_numeric(mdf["n_taxa"], errors="coerce").fillna(0).max())
        if config.empirical_profile in {"hiv_env_b_full", "hiv_env_b_real"} and retained < 200:
            raise ValueError(f"hiv_env_b_full requires >=200 retained sequences; got {retained}")
        if config.empirical_profile in {"sars_2020_full", "sars_cov2_real"} and retained < 500:
            raise ValueError(f"sars_2020_full requires >=500 retained genomes; got {retained}")
        if window_df.empty:
            raise ValueError(f"{config.empirical_profile} requires sliding-window outputs; none were produced.")
        full_gene_ids = {str(x).lower() for x in full_df["gene_id"].astype(str).tolist()}
        if config.empirical_profile in {"hiv_env_b_full", "hiv_env_b_real"}:
            if not any("env" in gid for gid in full_gene_ids):
                raise ValueError("hiv_env_b_full requires full-gene HIV env analysis outputs.")
        if config.empirical_profile in {"sars_2020_full", "sars_cov2_real"}:
            if not any("spike" in gid for gid in full_gene_ids):
                raise ValueError("sars_2020_full requires Spike ORF-level outputs.")
            if not any("orf1ab" in gid for gid in full_gene_ids):
                raise ValueError("sars_2020_full requires ORF1ab ORF-level outputs.")
            parent_col = "parent_gene_id" if "parent_gene_id" in window_df.columns else "gene_id"
            win_parents = {str(x).lower() for x in window_df[parent_col].astype(str).tolist()}
            if not any("spike" in gid for gid in win_parents):
                raise ValueError("sars_2020_full requires Spike sliding-window outputs.")
            if not any("orf1ab" in gid for gid in win_parents):
                raise ValueError("sars_2020_full requires ORF1ab sliding-window outputs.")

    explain_ok = int(np.count_nonzero(explain_df["status"] == "OK")) if not explain_df.empty else 0
    min_explain = 5 if str(config.empirical_profile) in {"ortholog_small8_full", "ortholog_real_v12"} else 3
    if config.require_real_data and config.write_plots and explain_ok < min_explain:
        raise ValueError(f"Real-data runs require at least {min_explain} explain reports; got {explain_ok}")

    _write_json(dirs["logs"] / "run.log", summary)
    plots = _plot_module()
    plots.plot_report_page(
        [
            f"Track: {track}",
            f"Preset: {config.preset}",
            f"Dataset: {dataset_path}",
            f"Synthetic fallback: {synthetic}",
            f"Input genes: {len(genes_all)}",
            f"Train genes: {len(train_genes)}",
            f"Test genes: {len(candidate_genes)}",
            f"BABAPPA rows: {len(full_df)}",
            f"Window rows: {len(window_df) if not window_df.empty else 0}",
            f"Baseline rows: {len(baseline_df)}",
            f"BABAPPA success: {n_success_babappa}",
            f"BUSTED success: {n_success_busted}",
            f"Testable set size: {testable_n}",
            f"Foreground species: {foreground_species if foreground_species is not None else 'none'}",
            f"RELAX note: {relax_skipped_reason if relax_skipped_reason else 'none'}",
            f"Doctor failures: {doctor_failures}",
            f"Doctor warnings: {doctor_warnings}",
        ],
        dirs["report"] / "report.pdf",
        f"BABAPPA {track.title()} Benchmark Report",
    )
    return summary


def _run_simulation_track(config: BenchmarkRunConfig) -> dict[str, Any]:
    params = _simulation_preset_params(config.preset)
    dirs = _ensure_dirs(config.outdir)
    _write_rebuild_script(config.outdir)
    publication_mode = _is_publication_mode(config)
    if publication_mode and bool(config.allow_baseline_fail):
        raise ValueError("Publication mode forbids --allow-baseline-fail.")
    if publication_mode and not bool(config.write_plots):
        raise ValueError("Publication mode requires figure/report generation; disable of plots is not allowed.")
    if publication_mode and min(int(x) for x in list(params["N_grid"])) < 999:
        raise ValueError(
            f"Publication mode requires N >= 999 in simulation presets; got N_grid={list(params['N_grid'])}."
        )
    if publication_mode and bool(params["include_baselines"]) and not bool(config.include_baselines):
        raise ValueError("Publication mode requires HyPhy baselines for this simulation preset.")
    if publication_mode and bool(params["include_relax"]) and not bool(config.include_relax):
        raise ValueError("Publication mode requires RELAX where the preset defines a foreground rule.")

    if bool(params["include_baselines"] and config.include_baselines):
        methods = ["busted"]
        if bool(params["include_relax"] and config.include_relax):
            methods.append("relax")
        _enforce_baseline_doctor(
            outdir=config.outdir,
            methods=methods,
            allow_baseline_fail=bool(config.allow_baseline_fail and not publication_mode),
            container=config.baseline_container,
            timeout_sec=int(max(30, min(config.baseline_timeout_sec, 300))),
        )

    bcfg = BenchmarkPackConfig(
        outdir=config.outdir,
        seed=config.seed,
        tail=config.tail,
        L_grid=list(params["L_grid"]),
        taxa_grid=list(params["taxa_grid"]),
        N_grid=list(params["N_grid"]),
        n_null_genes=int(params["n_null_genes"]),
        n_alt_genes=int(params["n_alt_genes"]),
        training_replicates=int(params["training_replicates"]),
        training_length_nt=int(params["training_length_nt"]),
        kappa=float(params["kappa"]),
        run_null=bool(params["run_null"]),
        run_power=bool(params["run_power"]),
        include_baselines=bool(params["include_baselines"] and config.include_baselines),
        include_relax=bool(params["include_relax"] and config.include_relax),
        write_figures=bool(config.write_plots),
        baseline_timeout_sec=int(config.baseline_timeout_sec),
        baseline_methods=("busted",),
    )
    with _baseline_backend_env(config.baseline_container):
        core_summary = run_benchmark_pack(bcfg)
    _write_json(dirs["manifests"] / "simulation_core_manifest.json", core_summary)

    if config.preset.lower() in {"power_full", "simulation_power_full", "power_smoke"}:
        n_genes = 3 if os.environ.get("BABAPPA_BENCHMARK_FAST", "0") == "1" else 300
    else:
        n_genes = 2 if os.environ.get("BABAPPA_BENCHMARK_FAST", "0") == "1" else 200
    dep_df = _run_dependence_sensitivity(
        outdir=config.outdir,
        seed=config.seed + 17,
        tail=config.tail,
        n_genes=n_genes,
        n_calibration=199,
        write_plots=bool(config.write_plots),
    )
    _write_json(
        dirs["manifests"] / "dependence_manifest.json",
        {
            "schema_version": 1,
            "n_rows": int(len(dep_df)),
            "n_genes": n_genes,
            "n_calibration": 199,
            "tail": config.tail,
        },
    )
    return _simulation_finalize(config.outdir, write_plots=bool(config.write_plots))


def _write_track_manifest(config: BenchmarkRunConfig, summary: dict[str, Any]) -> None:
    payload = {
        "schema_version": 1,
        "track": config.track,
        "preset": config.preset,
        "dataset_json": config.dataset_json,
        "mode": str(config.mode).strip().lower(),
        "seed": config.seed,
        "tail": config.tail,
        "include_baselines": config.include_baselines,
        "include_relax": config.include_relax,
        "allow_baseline_fail": config.allow_baseline_fail,
        "allow_qc_fail": config.allow_qc_fail,
        "write_plots": config.write_plots,
        "require_real_data": config.require_real_data,
        "calibration_size_override": config.calibration_size_override,
        "empirical_profile": config.empirical_profile,
        "min_taxa": config.min_taxa,
        "require_full_completeness": config.require_full_completeness,
        "foreground_species": config.foreground_species,
        "jobs": config.jobs,
        "resume": bool(config.resume),
        "baseline_container": config.baseline_container,
        "baseline_timeout_sec": config.baseline_timeout_sec,
        "system": get_system_metadata(Path.cwd()),
        "summary": summary,
    }
    _write_json(config.outdir / "manifests" / "benchmark_track_manifest.json", payload)


def run_benchmark_track(config: BenchmarkRunConfig) -> dict[str, Any]:
    track = config.track.lower()
    if _is_publication_mode(config):
        if bool(config.allow_baseline_fail):
            raise ValueError("Publication mode forbids --allow-baseline-fail.")
        if not bool(config.write_plots):
            raise ValueError("Publication mode requires figures and report outputs.")
        if track in {"ortholog", "viral"}:
            config.require_real_data = True
    config.outdir = config.outdir.resolve()
    _ensure_dirs(config.outdir)
    _write_rebuild_script(config.outdir)
    start = time.perf_counter()
    _write_track_manifest(
        config,
        {
            "status": "running",
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime()),
        },
    )

    if track == "simulation":
        summary = _run_simulation_track(config)
    elif track == "ortholog":
        summary = _run_real_track(config, "ortholog")
    elif track == "viral":
        summary = _run_real_track(config, "viral")
    else:
        raise ValueError(f"Unsupported benchmark track: {config.track}")

    summary = dict(summary)
    summary["runtime_total_sec"] = float(time.perf_counter() - start)
    _write_track_manifest(config, summary)
    _write_checksums(config.outdir)
    return summary


def resume_benchmark_track(outdir: str | Path, *, write_plots: bool | None = None) -> dict[str, Any]:
    root = Path(outdir).resolve()
    manifest = root / "manifests" / "benchmark_track_manifest.json"
    if not manifest.exists():
        raise FileNotFoundError(f"Cannot resume: benchmark manifest missing: {manifest}")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    track = str(payload.get("track", "")).strip().lower()
    if not track:
        raise ValueError(f"Cannot resume: invalid track in manifest: {manifest}")

    wp = bool(payload.get("write_plots", True)) if write_plots is None else bool(write_plots)
    dataset_json = payload.get("dataset_json", None)
    cfg = BenchmarkRunConfig(
        track=track,
        preset=str(payload.get("preset", "")),
        outdir=root,
        seed=int(payload.get("seed", 1)),
        tail=str(payload.get("tail", "right")),
        dataset_json=(None if dataset_json in {None, ""} else str(dataset_json)),
        include_baselines=bool(payload.get("include_baselines", True)),
        include_relax=bool(payload.get("include_relax", False)),
        allow_baseline_fail=bool(payload.get("allow_baseline_fail", False)),
        allow_qc_fail=bool(payload.get("allow_qc_fail", False)),
        baseline_container=str(payload.get("baseline_container", "auto")),
        baseline_timeout_sec=int(payload.get("baseline_timeout_sec", 1800)),
        write_plots=wp,
        require_real_data=bool(payload.get("require_real_data", False)),
        calibration_size_override=(
            None
            if payload.get("calibration_size_override", None) is None
            else int(payload.get("calibration_size_override"))
        ),
        empirical_profile=(
            None
            if payload.get("empirical_profile", None) in {None, ""}
            else str(payload.get("empirical_profile"))
        ),
        mode=str(payload.get("mode", "pilot")),
        min_taxa=(None if payload.get("min_taxa", None) is None else int(payload.get("min_taxa"))),
        require_full_completeness=bool(payload.get("require_full_completeness", False)),
        foreground_species=(
            None
            if payload.get("foreground_species", None) in {None, ""}
            else str(payload.get("foreground_species"))
        ),
        jobs=int(payload.get("jobs", 0)),
        resume=True,
    )
    return run_benchmark_track(cfg)


def _rebuild_simulation(outdir: Path, *, write_plots: bool) -> None:
    _simulation_finalize(outdir, write_plots=bool(write_plots))
    _write_checksums(outdir)


def _rebuild_real(outdir: Path, track: str, preset: str, *, write_plots: bool) -> None:
    # Rebuild from existing raw tables for ortholog/viral tracks.
    dirs = _ensure_dirs(outdir)
    babappa_df = _read_tsv_or_empty(dirs["raw"] / "babappa_results.tsv")
    baseline_df = _read_tsv_or_empty(dirs["raw"] / "baseline_all.tsv")
    runtime_df = _read_tsv_or_empty(dirs["raw"] / "runtime.tsv")

    overlap = _read_tsv_or_empty(
        dirs["tables"] / ("T3_ortholog_overlap_matrix.tsv" if track == "ortholog" else "T5_viral_overlap_matrix.tsv")
    )
    if overlap.empty and not babappa_df.empty:
        overlap = pd.DataFrame({"gene_id": babappa_df["gene_id"].astype(str)})
        overlap["babappa_sig"] = babappa_df["q_value"] <= 0.1
        method_list = (
            _ordered_methods([str(x) for x in baseline_df["method"].tolist()])
            if not baseline_df.empty and "method" in baseline_df.columns
            else []
        )
        for method in method_list:
            sub = baseline_df[(baseline_df["method"] == method) & (baseline_df["status"] == "OK")]
            sig_map = {
                str(g): bool(float(p) <= 0.05)
                for g, p in zip(sub["gene_id"], pd.to_numeric(sub["p_value"], errors="coerce"))
            }
            overlap[f"{method}_sig"] = overlap["gene_id"].map(sig_map).fillna(False)
        combo_methods = ["babappa"] + method_list
        combo_cols = {"babappa": "babappa_sig"}
        for method in method_list:
            combo_cols[method] = f"{method}_sig"
        overlap["combo"] = overlap.apply(
            lambda r: "|".join(
                (
                    _combo_symbol(method)
                    if bool(r[combo_cols[method]])
                    else _combo_symbol(method).lower()
                )
                for method in combo_methods
            ),
            axis=1,
        )
    combos = overlap.groupby("combo", as_index=False).size().rename(columns={"size": "count"}) if not overlap.empty else pd.DataFrame()
    window_df = _read_tsv_or_empty(dirs["raw"] / "window_babappa.tsv")
    if not baseline_df.empty and "unit_kind" in baseline_df.columns:
        window_baseline_df = baseline_df[baseline_df["unit_kind"] == "window"].copy()
    else:
        frames: list[pd.DataFrame] = []
        for method in ("busted", "relax"):
            tsv = dirs["raw"] / f"baseline_windows_{method}.tsv"
            df = _read_tsv_or_empty(tsv)
            if not df.empty:
                frames.append(df)
        window_baseline_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    qc_df = _read_tsv_or_empty(dirs["raw"] / "qc_by_gene.tsv")

    if write_plots:
        plots = _plot_module()
        if track == "ortholog":
            plots.plot_overlap_upset(combos, dirs["figures"] / "F3_ortholog_overlap_upset.pdf", "F3: OrthoMaM overlap")
            enrich = _read_tsv_or_empty(dirs["tables"] / "T4_enrichment.tsv")
            plots.plot_enrichment(enrich, dirs["figures"] / "F4_selectome_enrichment.pdf")
            plots.plot_pvalue_histograms(babappa_df, baseline_df, dirs["figures"] / "pvalue_histograms.pdf")
            plots.plot_runtime_boxplot(runtime_df, dirs["figures"] / "runtime_boxplot.pdf", "Runtime per Gene by Method")
        else:
            if preset.upper().startswith("HIV_ENV_B"):
                plots.plot_viral_signal(babappa_df, dirs["figures"] / "F5_hiv_stress_test.pdf", "F5: HIV stress test")
            else:
                plots.plot_viral_signal(babappa_df, dirs["figures"] / "F6_sars_stress_test.pdf", "F6: SARS-CoV-2 stress test")
            plots.plot_pvalue_histograms(babappa_df, baseline_df, dirs["figures"] / "pvalue_histograms.pdf")
            plots.plot_runtime_boxplot(runtime_df, dirs["figures"] / "runtime_boxplot.pdf", "Runtime per Unit by Method")
            if not window_df.empty:
                plots.plot_manhattan_windows(
                    window_df,
                    window_baseline_df if not window_baseline_df.empty else pd.DataFrame(),
                    dirs["figures"] / "window_manhattan.pdf",
                    "Window-level Manhattan Plot",
                )
            if preset.upper().startswith("SARS_2020"):
                plots.plot_orf_method_bar(babappa_df, baseline_df, dirs["figures"] / "orf_method_bar.pdf")

        if not qc_df.empty:
            if "length_nt" in qc_df.columns:
                plots.plot_length_distribution(
                    pd.to_numeric(qc_df["length_nt"], errors="coerce").to_numpy(dtype=float),
                    dirs["figures"] / "qc_length_distribution.pdf",
                    title="QC: Alignment length distribution",
                )
            if "missingness_fraction" in qc_df.columns:
                plots.plot_gap_fraction_distribution(
                    pd.to_numeric(qc_df["missingness_fraction"], errors="coerce").to_numpy(dtype=float),
                    dirs["figures"] / "qc_gap_fraction_distribution.pdf",
                    title="QC: Gap/missingness distribution",
                )
        tree_text = ""
        dr = dirs["manifests"] / "doctor_report.json"
        if dr.exists():
            try:
                dr_payload = json.loads(dr.read_text(encoding="utf-8"))
                ds_path = Path(str(dr_payload.get("dataset_path", ""))).resolve()
                if ds_path.exists():
                    ds = load_dataset_json(ds_path)
                    tpath = Path(str(ds.get("tree_path", ""))).resolve()
                    if tpath.exists():
                        tree_text = tpath.read_text(encoding="utf-8")
            except Exception:
                tree_text = ""
        plots.plot_branch_length_histogram(
            tree_text,
            dirs["figures"] / "qc_branch_length_histogram.pdf",
            title="QC: Branch length histogram",
        )

        plots.plot_runtime_scaling(runtime_df, dirs["figures"] / f"runtime_{track}.pdf", f"{track.title()} runtime")
    plots = _plot_module()
    plots.plot_report_page(
        [
            f"Track: {track}",
            f"Preset: {preset}",
            f"BABAPPA rows: {len(babappa_df)}",
            f"Baseline rows: {len(baseline_df)}",
            f"Runtime rows: {len(runtime_df)}",
        ],
        dirs["report"] / "report.pdf",
        f"BABAPPA {track.title()} Benchmark Report",
    )
    _write_checksums(outdir)


def rebuild_benchmark_pack(outdir: str | Path, *, write_plots: bool = True) -> None:
    root = Path(outdir).resolve()
    manifest = root / "manifests" / "benchmark_track_manifest.json"
    if not manifest.exists():
        raise FileNotFoundError(f"Track manifest missing for rebuild: {manifest}")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    track = str(payload.get("track", "")).lower()
    preset = str(payload.get("preset", ""))
    if track == "simulation":
        _rebuild_simulation(root, write_plots=bool(write_plots))
    elif track in {"ortholog", "viral"}:
        _rebuild_real(root, track, preset, write_plots=bool(write_plots))
    else:
        raise ValueError(f"Unsupported track in manifest: {track}")


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="python -m babappa.benchmark.orchestrator")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_run = sub.add_parser("run")
    p_run.add_argument("--track", choices=["simulation", "ortholog", "viral"], required=True)
    p_run.add_argument("--preset", required=True)
    p_run.add_argument("--outdir", required=True)
    p_run.add_argument("--seed", type=int, required=True)
    p_run.add_argument("--tail", choices=["right", "left", "two-sided"], default="right")
    p_run.add_argument("--dataset-json", default=None)
    p_run.add_argument("--include-baselines", action="store_true")
    p_run.add_argument("--include-relax", action="store_true")
    p_run.add_argument("--allow-baseline-fail", action="store_true")
    p_run.add_argument("--allow-qc-fail", action="store_true")
    p_run.add_argument("--require-real-data", action="store_true")
    p_run.add_argument("--calibration-size-override", type=int, default=None)
    p_run.add_argument("--empirical-profile", default=None)
    p_run.add_argument("--mode", choices=["pilot", "publication"], default="pilot")
    p_run.add_argument("--min-taxa", type=int, default=None)
    p_run.add_argument("--require-full-completeness", action="store_true")
    p_run.add_argument("--foreground-species", default=None)
    p_run.add_argument("--jobs", type=int, default=0)
    p_run.add_argument("--no-plots", action="store_true")
    p_run.add_argument("--baseline-container", choices=["auto", "docker", "singularity", "local"], default="auto")
    p_run.add_argument("--baseline-timeout", type=int, default=1800)

    p_rebuild = sub.add_parser("rebuild")
    p_rebuild.add_argument("--outdir", required=True)
    p_rebuild.add_argument("--no-plots", action="store_true")

    args = parser.parse_args(argv)
    if args.cmd == "rebuild":
        rebuild_benchmark_pack(args.outdir, write_plots=not bool(args.no_plots))
        return 0
    cfg = BenchmarkRunConfig(
        track=args.track,
        preset=args.preset,
        outdir=Path(args.outdir),
        seed=int(args.seed),
        tail=args.tail,
        dataset_json=args.dataset_json,
        include_baselines=bool(args.include_baselines),
        include_relax=bool(args.include_relax),
        allow_baseline_fail=bool(args.allow_baseline_fail),
        allow_qc_fail=bool(args.allow_qc_fail),
        baseline_container=str(args.baseline_container),
        baseline_timeout_sec=int(args.baseline_timeout),
        write_plots=not bool(args.no_plots),
        require_real_data=bool(args.require_real_data),
        calibration_size_override=args.calibration_size_override,
        empirical_profile=(None if args.empirical_profile is None else str(args.empirical_profile)),
        mode=str(args.mode),
        min_taxa=(None if args.min_taxa is None else int(args.min_taxa)),
        require_full_completeness=bool(args.require_full_completeness),
        foreground_species=(None if args.foreground_species is None else str(args.foreground_species)),
        jobs=int(args.jobs),
    )
    summary = run_benchmark_track(cfg)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
