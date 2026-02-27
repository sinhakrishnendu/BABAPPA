from __future__ import annotations

import json
import os
import stat
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ._plotting_env import configure_plotting_env
from .baseline_adapters import run_method_for_gene
from .engine import analyze_alignment, train_energy_model_from_neutral_spec
from .evaluation import mean_site_entropy, simulate_mixture_omega_alignment
from .hash_utils import sha256_json
from .io import Alignment
from .neutral import GY94NeutralSimulator, NeutralSpec
from .stats import rank_p_value


@dataclass
class BenchmarkPackConfig:
    outdir: Path
    seed: int
    tail: str
    L_grid: list[int]
    taxa_grid: list[int]
    N_grid: list[int]
    n_null_genes: int
    n_alt_genes: int
    training_replicates: int
    training_length_nt: int
    kappa: float
    run_null: bool = True
    run_power: bool = True
    include_baselines: bool = False
    include_relax: bool = False
    write_figures: bool = True
    baseline_timeout_sec: int = 1800
    baseline_methods: tuple[str, ...] = ("busted",)
    power_families: tuple[str, ...] = (
        "A_distributed_weak_selection",
        "B_constraint_shift",
        "C_episodic_branch_site_surrogate",
        "D_misspecification_stress",
    )
    family_a_pi: tuple[float, ...] = (0.05, 0.1, 0.2)
    family_a_omega: tuple[float, ...] = (1.05, 1.1, 1.2, 1.5)
    family_b_pi: tuple[float, ...] = (0.05, 0.1, 0.2)
    family_b_omega: tuple[float, ...] = (0.7, 0.8, 0.9)
    family_c_pi_fg: tuple[float, ...] = (0.005, 0.01, 0.05)
    family_c_omega_fg: tuple[float, ...] = (2.0, 5.0, 10.0)
    family_d_kappa_scale: tuple[float, ...] = (0.8, 1.2)
    family_d_branch_scale: tuple[float, ...] = (0.8, 1.2)
    alpha_levels: tuple[float, ...] = (0.1, 0.05, 0.01, 0.001)


def _star_tree_newick(taxa_n: int, branch_length: float = 0.1) -> str:
    leaves = [f"T{i+1}:{branch_length}" for i in range(taxa_n)]
    return f"({','.join(leaves)});"


def _ensure_pack_dirs(outdir: Path) -> dict[str, Path]:
    dirs = {
        "root": outdir,
        "manifests": outdir / "manifests",
        "raw": outdir / "raw",
        "figures": outdir / "figures",
        "tables": outdir / "tables",
        "logs": outdir / "logs",
        "scripts": outdir / "scripts",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def _write_rebuild_script(script_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    script_text = """#!/usr/bin/env bash
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
    script_text = script_text.replace("__BABAPPA_REPO_ROOT__", str(repo_root))
    script_path.write_text(script_text, encoding="utf-8")
    mode = script_path.stat().st_mode
    script_path.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _empirical_size_table(null_df: pd.DataFrame, alpha_levels: tuple[float, ...]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    grouped = null_df.groupby(["taxa", "L", "N"], as_index=False)
    for (taxa, length, n_calib), group in grouped:
        n = len(group)
        p = group["p"].to_numpy(dtype=float)
        for alpha in alpha_levels:
            hits = int(np.count_nonzero(p <= alpha))
            frac = hits / max(n, 1)
            # binomial normal approx CI
            se = np.sqrt(max(frac * (1.0 - frac) / max(n, 1), 0.0))
            rows.append(
                {
                    "taxa": int(taxa),
                    "L": int(length),
                    "N": int(n_calib),
                    "alpha": float(alpha),
                    "n_genes": int(n),
                    "size_hat": float(frac),
                    "ci95_low": float(max(frac - 1.96 * se, 0.0)),
                    "ci95_high": float(min(frac + 1.96 * se, 1.0)),
                }
            )
    return pd.DataFrame(rows)


def _get_plt() -> Any:
    configure_plotting_env()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _write_qq_plot(null_df: pd.DataFrame, out_pdf: Path) -> None:
    plt = _get_plt()
    p = np.sort(null_df["p"].to_numpy(dtype=float))
    n = len(p)
    if n == 0:
        return
    q = (np.arange(1, n + 1) - 0.5) / n
    plt.figure(figsize=(6, 6))
    plt.plot(q, p, ".", alpha=0.5, label="Observed p-values")
    plt.plot([0, 1], [0, 1], "r--", linewidth=1.5, label="Uniform(0,1)")
    plt.xlabel("Expected quantile")
    plt.ylabel("Observed p-value")
    plt.title("Null Calibration QQ Plot")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_pdf)
    plt.close()


def _write_power_heatmap(power_df: pd.DataFrame, out_pdf: Path) -> None:
    plt = _get_plt()
    if power_df.empty:
        return
    if "status" in power_df.columns:
        power_df = power_df[power_df["status"] == "OK"].copy()
    if power_df.empty:
        return
    summary = (
        power_df.groupby(["family", "method"], as_index=False)["significant"]
        .mean()
        .rename(columns={"significant": "power"})
    )
    fams = sorted(summary["family"].unique())
    methods = sorted(summary["method"].unique())
    matrix = np.zeros((len(fams), len(methods)), dtype=float)
    for i, fam in enumerate(fams):
        for j, method in enumerate(methods):
            sub = summary[(summary["family"] == fam) & (summary["method"] == method)]
            matrix[i, j] = 0.0 if sub.empty else float(sub.iloc[0]["power"])
    plt.figure(figsize=(0.9 * len(methods) + 3, 0.7 * len(fams) + 3))
    im = plt.imshow(matrix, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    plt.xticks(np.arange(len(methods)), methods, rotation=30, ha="right")
    plt.yticks(np.arange(len(fams)), fams)
    plt.title("Power by Alternative Family and Method")
    plt.colorbar(im, label="Power (alpha=0.05)")
    plt.tight_layout()
    plt.savefig(out_pdf)
    plt.close()


def _write_regime_map(power_summary: pd.DataFrame, out_pdf: Path) -> None:
    plt = _get_plt()
    if power_summary.empty:
        return
    view = power_summary.copy()
    if "power_alpha_0.05" not in view.columns:
        return
    best = (
        view.sort_values("power_alpha_0.05", ascending=False)
        .groupby(["family", "taxa", "L", "N"], as_index=False)
        .first()
    )
    fams = sorted(best["family"].unique())
    cell_labels = sorted({f"taxa={int(t)};L={int(l)};N={int(n)}" for t, l, n in zip(best["taxa"], best["L"], best["N"])})
    methods = sorted(best["method"].astype(str).unique())
    method_to_int = {m: i for i, m in enumerate(methods)}

    mat = np.full((len(fams), len(cell_labels)), fill_value=-1, dtype=int)
    for i, fam in enumerate(fams):
        sub = best[best["family"] == fam]
        for _, row in sub.iterrows():
            label = f"taxa={int(row['taxa'])};L={int(row['L'])};N={int(row['N'])}"
            j = cell_labels.index(label)
            mat[i, j] = method_to_int[str(row["method"])]

    plt.figure(figsize=(max(7, 0.8 * len(cell_labels) + 2), max(4, 0.7 * len(fams) + 2)))
    im = plt.imshow(mat, aspect="auto", cmap="tab20")
    plt.xticks(np.arange(len(cell_labels)), cell_labels, rotation=35, ha="right")
    plt.yticks(np.arange(len(fams)), fams)
    cbar = plt.colorbar(im, ticks=np.arange(len(methods)))
    cbar.ax.set_yticklabels(methods)
    plt.title("Regime Map (Method with Highest Power @ alpha=0.05)")
    plt.tight_layout()
    plt.savefig(out_pdf)
    plt.close()


def _write_size_deviation_plot(size_df: pd.DataFrame, out_pdf: Path) -> None:
    plt = _get_plt()
    if size_df.empty:
        return
    view = size_df[size_df["alpha"] == 0.05].copy()
    if view.empty:
        view = size_df.copy()
    view["size_deviation"] = view["size_hat"] - view["alpha"]
    plt.figure(figsize=(8, 5))
    x = np.arange(len(view))
    plt.axhline(0.0, color="black", linewidth=1)
    plt.plot(x, view["size_deviation"], marker="o")
    plt.xticks(
        x,
        [f"taxa={int(t)};L={int(l)};N={int(n)}" for t, l, n in zip(view["taxa"], view["L"], view["N"])],
        rotation=35,
        ha="right",
    )
    plt.ylabel("size_hat - alpha")
    plt.title("Size Deviation (alpha=0.05)")
    plt.tight_layout()
    plt.savefig(out_pdf)
    plt.close()


def _write_discreteness_plot(null_df: pd.DataFrame, out_pdf: Path) -> None:
    plt = _get_plt()
    if null_df.empty:
        return
    grouped = (
        null_df.groupby(["N"], as_index=False)["p"]
        .agg(
            n_unique=lambda x: int(np.unique(np.asarray(x, dtype=float)).size),
            n_total="count",
        )
    )
    grouped["grid_step"] = 1.0 / (grouped["N"] + 1.0)
    plt.figure(figsize=(8, 5))
    ax1 = plt.gca()
    ax1.plot(grouped["N"], grouped["n_unique"], marker="o", color="#4C78A8")
    ax1.set_xlabel("N (calibration replicates)")
    ax1.set_ylabel("Observed unique p-values")
    ax2 = ax1.twinx()
    ax2.plot(grouped["N"], grouped["grid_step"], marker="s", color="#F58518")
    ax2.set_ylabel("Theoretical p-value step 1/(N+1)")
    plt.title("Discreteness Diagnostic vs N")
    plt.tight_layout()
    plt.savefig(out_pdf)
    plt.close()


def _write_runtime_plot(runtime_df: pd.DataFrame, out_pdf: Path) -> None:
    plt = _get_plt()
    if runtime_df.empty:
        return
    grouped = (
        runtime_df.groupby(["taxa", "L", "N"], as_index=False)["runtime_sec"]
        .median()
        .rename(columns={"runtime_sec": "median_runtime_sec"})
    )
    plt.figure(figsize=(8, 5))
    tick_labels: list[str] = []
    for taxa in sorted(grouped["taxa"].unique()):
        sub = grouped[grouped["taxa"] == taxa].sort_values(["L", "N"])
        if not tick_labels:
            tick_labels = [f"L={int(l)};N={int(n)}" for l, n in zip(sub["L"], sub["N"])]
        plt.plot(
            np.arange(len(sub)),
            sub["median_runtime_sec"],
            marker="o",
            label=f"taxa={taxa}",
        )
    plt.xticks(np.arange(len(tick_labels)), tick_labels, rotation=35, ha="right")
    plt.ylabel("Median runtime per gene (sec)")
    plt.title("Runtime Scaling")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_pdf)
    plt.close()


def _run_null_grid(config: BenchmarkPackConfig, dirs: dict[str, Path], rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not config.run_null:
        return pd.DataFrame(), pd.DataFrame()

    null_rows: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []

    for taxa_n in config.taxa_grid:
        neutral_spec = NeutralSpec(
            tree_newick=_star_tree_newick(taxa_n),
            kappa=config.kappa,
            omega=1.0,
        )
        model_seed = int(rng.integers(0, 2**32 - 1))
        model = train_energy_model_from_neutral_spec(
            neutral_spec=neutral_spec,
            sim_replicates=config.training_replicates,
            sim_length=config.training_length_nt,
            seed=model_seed,
        )
        model_hash = sha256_json(model.to_dict())
        simulator = GY94NeutralSimulator(neutral_spec)

        for length in config.L_grid:
            for n_calib in config.N_grid:
                for gene_idx in range(config.n_null_genes):
                    sim_seed = int(rng.integers(0, 2**32 - 1))
                    alignment = simulator.simulate_alignment(length_nt=length, seed=sim_seed)
                    start = time.perf_counter()
                    result = analyze_alignment(
                        alignment,
                        model=model,
                        calibration_size=n_calib,
                        calibration_mode="phylo",
                        tail=config.tail,
                        seed=sim_seed,
                        name=f"null_taxa{taxa_n}_L{length}_{gene_idx+1}",
                    )
                    runtime_sec = time.perf_counter() - start
                    null_rows.append(
                        {
                            "taxa": taxa_n,
                            "L": length,
                            "N": n_calib,
                            "gene_index": gene_idx + 1,
                            "p": result.p_value,
                            "tail": result.tail,
                            "model_hash": model_hash,
                            "seed_gene": sim_seed,
                        }
                    )
                    runtime_rows.append(
                        {
                            "taxa": taxa_n,
                            "L": length,
                            "N": n_calib,
                            "runtime_sec": runtime_sec,
                            "phase": "null_test",
                            "method": "babappa",
                        }
                    )

    null_df = pd.DataFrame(null_rows)
    runtime_df = pd.DataFrame(runtime_rows)
    return null_df, runtime_df


def _perturb_codon_frequencies(base: dict[str, float] | None, rng: np.random.Generator) -> dict[str, float] | None:
    if base is None:
        return None
    keys = sorted(base.keys())
    v = np.array([base[k] for k in keys], dtype=float)
    noise = rng.normal(loc=0.0, scale=0.03, size=v.size)
    v2 = np.clip(v + noise, 1e-8, None)
    v2 /= v2.sum()
    return {k: float(x) for k, x in zip(keys, v2)}


def _write_alignment_fasta(alignment: Alignment, path: Path) -> None:
    lines: list[str] = []
    for name, seq in zip(alignment.names, alignment.sequences):
        lines.append(f">{name}")
        lines.append(seq)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_baseline_methods_for_alignment(
    *,
    config: BenchmarkPackConfig,
    alignment: Alignment,
    tree_newick: str,
    family: str,
    taxa_n: int,
    length: int,
    n_calib: int,
    rep: int,
    pi: float | None,
    omega: float | None,
    dirs: dict[str, Path],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []
    baseline_dir = dirs["raw"] / "baseline_runs"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    run_dir = baseline_dir / f"{family}_taxa{taxa_n}_L{length}_rep{rep+1}"
    run_dir.mkdir(parents=True, exist_ok=True)

    aln_path = run_dir / "alignment.fasta"
    tree_path = run_dir / "tree.nwk"
    _write_alignment_fasta(alignment, aln_path)
    tree_path.write_text(tree_newick + "\n", encoding="utf-8")

    methods = list(config.baseline_methods)
    if config.include_relax and "relax" not in methods:
        methods.append("relax")

    for method in methods:
        recs = run_method_for_gene(
            method=method,
            alignment_path=aln_path,
            tree_path=tree_path,
            workdir=run_dir / method,
            foreground_taxon=None,
            run_site_model=True,
            timeout_sec=int(config.baseline_timeout_sec),
        )
        for rec in recs:
            rows.append(
                {
                    "family": family,
                    "method": rec.method,
                    "taxa": taxa_n,
                    "L": length,
                    "N": n_calib,
                    "pi": np.nan if pi is None else pi,
                    "omega": np.nan if omega is None else omega,
                    "p": np.nan if rec.p_value is None else rec.p_value,
                    "significant": False if rec.p_value is None else rec.p_value <= 0.05,
                    "status": rec.status,
                    "reason": rec.reason,
                }
            )
            runtime_rows.append(
                {
                    "taxa": taxa_n,
                    "L": length,
                    "N": n_calib,
                    "runtime_sec": rec.runtime_sec,
                    "phase": "baseline_test",
                    "method": rec.method,
                }
            )
    return rows, runtime_rows


def _run_power_grid(
    config: BenchmarkPackConfig, dirs: dict[str, Path], rng: np.random.Generator
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not config.run_power:
        return pd.DataFrame(), pd.DataFrame()

    rows: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []
    families = set(config.power_families)

    for taxa_n in config.taxa_grid:
        tree_newick = _star_tree_newick(taxa_n)
        neutral_spec = NeutralSpec(
            tree_newick=tree_newick,
            kappa=config.kappa,
            omega=1.0,
        )
        model = train_energy_model_from_neutral_spec(
            neutral_spec=neutral_spec,
            sim_replicates=config.training_replicates,
            sim_length=config.training_length_nt,
            seed=int(rng.integers(0, 2**32 - 1)),
        )
        simulator = GY94NeutralSimulator(neutral_spec)

        for length in config.L_grid:
            n_calib = max(config.N_grid)
            if "A_distributed_weak_selection" in families:
                for pi in config.family_a_pi:
                    for omega_alt in config.family_a_omega:
                        for rep in range(config.n_alt_genes):
                            seed = int(rng.integers(0, 2**32 - 1))
                            aln = simulate_mixture_omega_alignment(
                                neutral_spec=neutral_spec,
                                length_nt=length,
                                selected_fraction=pi,
                                omega_alt=omega_alt,
                                seed=seed,
                            )
                            start = time.perf_counter()
                            res = analyze_alignment(
                                aln,
                                model=model,
                                calibration_size=n_calib,
                                calibration_mode="phylo",
                                tail=config.tail,
                                seed=seed,
                                name=f"A_{rep+1}",
                            )
                            runtime_rows.append(
                                {
                                    "taxa": taxa_n,
                                    "L": length,
                                    "N": n_calib,
                                    "runtime_sec": time.perf_counter() - start,
                                    "phase": "power_test",
                                    "method": "babappa",
                                }
                            )
                            rows.append(
                                {
                                    "family": "A_distributed_weak_selection",
                                    "method": "babappa",
                                    "taxa": taxa_n,
                                    "L": length,
                                    "N": n_calib,
                                    "pi": pi,
                                    "omega": omega_alt,
                                    "p": res.p_value,
                                    "significant": res.p_value <= 0.05,
                                    "status": "OK",
                                    "reason": "ok",
                                }
                            )
                            entropy_obs = mean_site_entropy(aln)
                            entropy_null = np.array(
                                [
                                    mean_site_entropy(
                                        simulator.simulate_alignment(
                                            length_nt=length, seed=int(rng.integers(0, 2**32 - 1))
                                        )
                                    )
                                    for _ in range(n_calib)
                                ],
                                dtype=float,
                            )
                            ep = rank_p_value(entropy_obs, entropy_null, tail=config.tail)
                            rows.append(
                                {
                                    "family": "A_distributed_weak_selection",
                                    "method": "entropy",
                                    "taxa": taxa_n,
                                    "L": length,
                                    "N": n_calib,
                                    "pi": pi,
                                    "omega": omega_alt,
                                    "p": ep,
                                    "significant": ep <= 0.05,
                                    "status": "OK",
                                    "reason": "ok",
                                }
                            )
                            if config.include_baselines:
                                brow, bruntime = _run_baseline_methods_for_alignment(
                                    config=config,
                                    alignment=aln,
                                    tree_newick=tree_newick,
                                    family="A_distributed_weak_selection",
                                    taxa_n=taxa_n,
                                    length=length,
                                    n_calib=n_calib,
                                    rep=rep,
                                    pi=float(pi),
                                    omega=float(omega_alt),
                                    dirs=dirs,
                                )
                                rows.extend(brow)
                                runtime_rows.extend(bruntime)

            if "B_constraint_shift" in families:
                for pi in config.family_b_pi:
                    for omega_alt in config.family_b_omega:
                        for rep in range(config.n_alt_genes):
                            seed = int(rng.integers(0, 2**32 - 1))
                            aln = simulate_mixture_omega_alignment(
                                neutral_spec=neutral_spec,
                                length_nt=length,
                                selected_fraction=pi,
                                omega_alt=omega_alt,
                                seed=seed,
                            )
                            start = time.perf_counter()
                            res = analyze_alignment(
                                aln,
                                model=model,
                                calibration_size=n_calib,
                                calibration_mode="phylo",
                                tail=config.tail,
                                seed=seed,
                                name=f"B_{rep+1}",
                            )
                            runtime_rows.append(
                                {
                                    "taxa": taxa_n,
                                    "L": length,
                                    "N": n_calib,
                                    "runtime_sec": time.perf_counter() - start,
                                    "phase": "power_test",
                                    "method": "babappa",
                                }
                            )
                            rows.append(
                                {
                                    "family": "B_constraint_shift",
                                    "method": "babappa",
                                    "taxa": taxa_n,
                                    "L": length,
                                    "N": n_calib,
                                    "pi": pi,
                                    "omega": omega_alt,
                                    "p": res.p_value,
                                    "significant": res.p_value <= 0.05,
                                    "status": "OK",
                                    "reason": "ok",
                                }
                            )
                            if config.include_baselines:
                                brow, bruntime = _run_baseline_methods_for_alignment(
                                    config=config,
                                    alignment=aln,
                                    tree_newick=tree_newick,
                                    family="B_constraint_shift",
                                    taxa_n=taxa_n,
                                    length=length,
                                    n_calib=n_calib,
                                    rep=rep,
                                    pi=float(pi),
                                    omega=float(omega_alt),
                                    dirs=dirs,
                                )
                                rows.extend(brow)
                                runtime_rows.extend(bruntime)

            if "C_episodic_branch_site_surrogate" in families:
                for pi_fg in config.family_c_pi_fg:
                    for omega_fg in config.family_c_omega_fg:
                        for rep in range(config.n_alt_genes):
                            seed = int(rng.integers(0, 2**32 - 1))
                            aln = simulate_mixture_omega_alignment(
                                neutral_spec=neutral_spec,
                                length_nt=length,
                                selected_fraction=pi_fg,
                                omega_alt=omega_fg,
                                seed=seed,
                            )
                            start = time.perf_counter()
                            res = analyze_alignment(
                                aln,
                                model=model,
                                calibration_size=n_calib,
                                calibration_mode="phylo",
                                tail=config.tail,
                                seed=seed,
                                name=f"C_{rep+1}",
                            )
                            runtime_rows.append(
                                {
                                    "taxa": taxa_n,
                                    "L": length,
                                    "N": n_calib,
                                    "runtime_sec": time.perf_counter() - start,
                                    "phase": "power_test",
                                    "method": "babappa",
                                }
                            )
                            rows.append(
                                {
                                    "family": "C_episodic_branch_site_surrogate",
                                    "method": "babappa",
                                    "taxa": taxa_n,
                                    "L": length,
                                    "N": n_calib,
                                    "pi": pi_fg,
                                    "omega": omega_fg,
                                    "p": res.p_value,
                                    "significant": res.p_value <= 0.05,
                                    "status": "OK",
                                    "reason": "ok",
                                }
                            )
                            if config.include_baselines:
                                brow, bruntime = _run_baseline_methods_for_alignment(
                                    config=config,
                                    alignment=aln,
                                    tree_newick=tree_newick,
                                    family="C_episodic_branch_site_surrogate",
                                    taxa_n=taxa_n,
                                    length=length,
                                    n_calib=n_calib,
                                    rep=rep,
                                    pi=float(pi_fg),
                                    omega=float(omega_fg),
                                    dirs=dirs,
                                )
                                rows.extend(brow)
                                runtime_rows.extend(bruntime)

            if "D_misspecification_stress" in families:
                for kappa_scale in config.family_d_kappa_scale:
                    for branch_scale in config.family_d_branch_scale:
                        for rep in range(config.n_alt_genes):
                            seed = int(rng.integers(0, 2**32 - 1))
                            stress_spec = NeutralSpec(
                                tree_newick=_star_tree_newick(taxa_n, branch_length=0.1 * branch_scale),
                                kappa=neutral_spec.kappa * kappa_scale,
                                omega=1.0,
                                codon_frequencies=_perturb_codon_frequencies(
                                    neutral_spec.codon_frequencies, rng
                                ),
                            )
                            stress_sim = GY94NeutralSimulator(stress_spec)
                            aln = stress_sim.simulate_alignment(length_nt=length, seed=seed)
                            start = time.perf_counter()
                            res = analyze_alignment(
                                aln,
                                model=model,
                                calibration_size=n_calib,
                                calibration_mode="phylo",
                                tail=config.tail,
                                seed=seed,
                                name=f"D_{rep+1}",
                            )
                            runtime_rows.append(
                                {
                                    "taxa": taxa_n,
                                    "L": length,
                                    "N": n_calib,
                                    "runtime_sec": time.perf_counter() - start,
                                    "phase": "power_test",
                                    "method": "babappa",
                                }
                            )
                            rows.append(
                                {
                                    "family": "D_misspecification_stress",
                                    "method": "babappa",
                                    "taxa": taxa_n,
                                    "L": length,
                                    "N": n_calib,
                                    "pi": np.nan,
                                    "omega": np.nan,
                                    "kappa_scale": kappa_scale,
                                    "branch_scale": branch_scale,
                                    "p": res.p_value,
                                    "significant": res.p_value <= 0.05,
                                    "status": "OK",
                                    "reason": "ok",
                                }
                            )

    power_df = pd.DataFrame(rows)
    runtime_df = pd.DataFrame(runtime_rows)
    return power_df, runtime_df


def _write_tables_and_figures(dirs: dict[str, Path], *, write_figures: bool = True) -> None:
    raw_dir = dirs["raw"]
    tables_dir = dirs["tables"]
    fig_dir = dirs["figures"]

    null_path = raw_dir / "null_calibration.tsv"
    power_path = raw_dir / "power.tsv"
    runtime_path = raw_dir / "runtime.tsv"
    if not null_path.exists() or not power_path.exists() or not runtime_path.exists():
        raise FileNotFoundError("Missing raw benchmark files required for rebuild.")

    null_df = pd.read_csv(null_path, sep="\t")
    power_df = pd.read_csv(power_path, sep="\t")
    runtime_df = pd.read_csv(runtime_path, sep="\t")

    size_df = _empirical_size_table(null_df, (0.1, 0.05, 0.01, 0.001))
    size_df.to_csv(tables_dir / "empirical_size.tsv", sep="\t", index=False)
    if not size_df.empty:
        size_dev = size_df.copy()
        size_dev["size_deviation"] = size_dev["size_hat"] - size_dev["alpha"]
        size_dev["size_abs_deviation"] = np.abs(size_dev["size_deviation"])
    else:
        size_dev = pd.DataFrame(
            columns=[
                "taxa",
                "L",
                "N",
                "alpha",
                "n_genes",
                "size_hat",
                "ci95_low",
                "ci95_high",
                "size_deviation",
                "size_abs_deviation",
            ]
        )
    size_dev.to_csv(tables_dir / "size_deviation.tsv", sep="\t", index=False)

    if power_df.empty:
        power_summary = pd.DataFrame(
            columns=[
                "family",
                "method",
                "taxa",
                "L",
                "N",
                "n_total",
                "n_ok",
                "power_alpha_0.05",
                "power_alpha_0.01",
                "ci95_low_0.05",
                "ci95_high_0.05",
                "ci95_low_0.01",
                "ci95_high_0.01",
            ]
        )
        fairness = pd.DataFrame(
            columns=["family", "method", "taxa", "L", "N", "n_total", "n_fail", "fail_rate"]
        )
    else:
        if "status" not in power_df.columns:
            power_df["status"] = "OK"
        if "p" not in power_df.columns:
            power_df["p"] = np.nan
        rows: list[dict[str, Any]] = []
        for (family, method, taxa, length, n_calib), group in power_df.groupby(
            ["family", "method", "taxa", "L", "N"], as_index=False
        ):
            n_total = int(len(group))
            ok = group[group["status"] == "OK"].copy()
            n_ok = int(len(ok))
            if n_ok > 0:
                p005 = float(np.mean(pd.to_numeric(ok["p"], errors="coerce") <= 0.05))
                p001 = float(np.mean(pd.to_numeric(ok["p"], errors="coerce") <= 0.01))
                se005 = float(np.sqrt(max(p005 * (1 - p005) / n_ok, 0.0)))
                se001 = float(np.sqrt(max(p001 * (1 - p001) / n_ok, 0.0)))
            else:
                p005 = 0.0
                p001 = 0.0
                se005 = 0.0
                se001 = 0.0
            rows.append(
                {
                    "family": family,
                    "method": method,
                    "taxa": int(taxa),
                    "L": int(length),
                    "N": int(n_calib),
                    "n_total": n_total,
                    "n_ok": n_ok,
                    "power_alpha_0.05": p005,
                    "power_alpha_0.01": p001,
                    "ci95_low_0.05": max(p005 - 1.96 * se005, 0.0),
                    "ci95_high_0.05": min(p005 + 1.96 * se005, 1.0),
                    "ci95_low_0.01": max(p001 - 1.96 * se001, 0.0),
                    "ci95_high_0.01": min(p001 + 1.96 * se001, 1.0),
                }
            )
        power_summary = pd.DataFrame(rows)
        fairness = (
            power_df.groupby(["family", "method", "taxa", "L", "N"], as_index=False)
            .agg(
                n_total=("method", "count"),
                n_fail=("status", lambda x: int(np.count_nonzero(np.asarray(x) != "OK"))),
            )
        )
        fairness["fail_rate"] = fairness["n_fail"] / fairness["n_total"].clip(lower=1)
    power_summary.to_csv(tables_dir / "power_summary.tsv", sep="\t", index=False)
    fairness.to_csv(tables_dir / "baseline_fairness.tsv", sep="\t", index=False)

    if runtime_df.empty:
        runtime_summary = pd.DataFrame(
            columns=[
                "taxa",
                "L",
                "N",
                "phase",
                "method",
                "median_runtime",
                "mean_runtime",
                "std_runtime",
            ]
        )
    else:
        if "phase" not in runtime_df.columns:
            runtime_df["phase"] = "unspecified"
        if "method" not in runtime_df.columns:
            runtime_df["method"] = "unknown"
        runtime_summary = runtime_df.groupby(
            ["taxa", "L", "N", "phase", "method"], as_index=False
        ).agg(
            median_runtime=("runtime_sec", "median"),
            mean_runtime=("runtime_sec", "mean"),
            std_runtime=("runtime_sec", "std"),
        )
    runtime_summary.to_csv(tables_dir / "runtime_summary.tsv", sep="\t", index=False)

    if write_figures:
        _write_qq_plot(null_df, fig_dir / "qq_null.pdf")
        _write_power_heatmap(power_df, fig_dir / "power_heatmap.pdf")
        _write_regime_map(power_summary, fig_dir / "regime_map.pdf")
        _write_runtime_plot(runtime_df, fig_dir / "runtime_scaling.pdf")
        _write_size_deviation_plot(size_df, fig_dir / "size_deviation.pdf")
        _write_discreteness_plot(null_df, fig_dir / "discreteness_vs_N.pdf")


def run_benchmark_pack(config: BenchmarkPackConfig) -> dict[str, Any]:
    dirs = _ensure_pack_dirs(config.outdir)
    _write_rebuild_script(dirs["scripts"] / "rebuild_all.sh")
    rng = np.random.default_rng(config.seed)

    start = time.perf_counter()
    null_df, null_runtime_df = _run_null_grid(config, dirs, rng)
    power_df, power_runtime_df = _run_power_grid(config, dirs, rng)
    runtime_df = pd.concat([null_runtime_df, power_runtime_df], ignore_index=True)

    if null_df.empty:
        null_df = pd.DataFrame(
            columns=["taxa", "L", "N", "gene_index", "p", "tail", "model_hash", "seed_gene"]
        )
    if power_df.empty:
        power_df = pd.DataFrame(
            columns=[
                "family",
                "method",
                "taxa",
                "L",
                "N",
                "pi",
                "omega",
                "p",
                "significant",
                "status",
                "reason",
            ]
        )
    if runtime_df.empty:
        runtime_df = pd.DataFrame(
            columns=["taxa", "L", "N", "runtime_sec", "phase", "method"]
        )

    null_df.to_csv(dirs["raw"] / "null_calibration.tsv", sep="\t", index=False)
    power_df.to_csv(dirs["raw"] / "power.tsv", sep="\t", index=False)
    runtime_df.to_csv(dirs["raw"] / "runtime.tsv", sep="\t", index=False)

    runtime_total = time.perf_counter() - start
    _write_tables_and_figures(dirs, write_figures=bool(config.write_figures))

    summary = {
        "run_null": bool(config.run_null),
        "run_power": bool(config.run_power),
        "include_baselines": bool(config.include_baselines),
        "write_figures": bool(config.write_figures),
        "n_null_rows": int(len(null_df)),
        "n_power_rows": int(len(power_df)),
        "runtime_total_sec": float(runtime_total),
    }
    (dirs["logs"] / "run.log").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (dirs["manifests"] / "benchmark_run.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "seed": config.seed,
                "tail": config.tail,
                "L_grid": config.L_grid,
                "taxa_grid": config.taxa_grid,
                "N_grid": config.N_grid,
                "n_null_genes": config.n_null_genes,
                "n_alt_genes": config.n_alt_genes,
                "training_replicates": config.training_replicates,
                "training_length_nt": config.training_length_nt,
                "kappa": config.kappa,
                "run_null": config.run_null,
                "run_power": config.run_power,
                "include_baselines": config.include_baselines,
                "write_figures": config.write_figures,
                "baseline_methods": list(config.baseline_methods),
                "power_families": list(config.power_families),
                "summary": summary,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return summary


def rebuild_from_raw(pack_dir: str | Path, *, write_figures: bool = True) -> None:
    dirs = _ensure_pack_dirs(Path(pack_dir))
    _write_tables_and_figures(dirs, write_figures=bool(write_figures))
