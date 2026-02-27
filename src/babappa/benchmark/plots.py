from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable

from .._plotting_env import configure_plotting_env

configure_plotting_env()

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _save_placeholder(path: Path, title: str, subtitle: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.text(0.5, 0.62, title, ha="center", va="center", fontsize=14, fontweight="bold")
    ax.text(0.5, 0.45, subtitle, ha="center", va="center", fontsize=10, wrap=True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_f1_null_calibration(null_df: pd.DataFrame, size_df: pd.DataFrame, out_pdf: Path) -> None:
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    if null_df.empty:
        _save_placeholder(out_pdf, "F1: Null Calibration", "No null-calibration rows available.")
        return

    p = np.sort(null_df["p"].to_numpy(dtype=float))
    q = (np.arange(1, len(p) + 1) - 0.5) / max(len(p), 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(q, p, ".", alpha=0.4, markersize=3)
    axes[0].plot([0, 1], [0, 1], "r--", linewidth=1.2)
    axes[0].set_title("QQ Plot (Null p-values)")
    axes[0].set_xlabel("Expected quantile")
    axes[0].set_ylabel("Observed p-value")

    if size_df.empty:
        axes[1].text(0.5, 0.5, "No size summary", ha="center", va="center")
        axes[1].axis("off")
    else:
        view = size_df[size_df["alpha"] == 0.05].copy()
        if view.empty:
            view = size_df.copy()
        view = view.sort_values(["N", "L", "taxa"])
        x = np.arange(len(view))
        axes[1].plot(x, view["size_hat"], marker="o", linewidth=1)
        axes[1].axhline(0.05, color="r", linestyle="--", linewidth=1, label="nominal 0.05")
        axes[1].set_ylim(0, 1)
        axes[1].set_title("Empirical Size")
        axes[1].set_ylabel("size_hat")
        axes[1].set_xlabel("cells")
        axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_pdf)
    plt.close(fig)


def plot_f2_power_and_regime(power_df: pd.DataFrame, out_pdf: Path) -> None:
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    if power_df.empty:
        _save_placeholder(out_pdf, "F2: Power Regimes", "No power-grid rows available.")
        return

    df = power_df.copy()
    if "status" in df.columns:
        df = df[df["status"] == "OK"]
    if df.empty:
        _save_placeholder(out_pdf, "F2: Power Regimes", "Power rows exist, but no successful methods.")
        return

    summary = (
        df.groupby(["family", "method"], as_index=False)["significant"]
        .mean()
        .rename(columns={"significant": "power"})
    )
    fams = sorted(summary["family"].unique())
    methods = sorted(summary["method"].unique())
    mat = np.zeros((len(fams), len(methods)), dtype=float)
    for i, fam in enumerate(fams):
        for j, method in enumerate(methods):
            sub = summary[(summary["family"] == fam) & (summary["method"] == method)]
            mat[i, j] = 0.0 if sub.empty else float(sub.iloc[0]["power"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    im = axes[0].imshow(mat, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    axes[0].set_xticks(np.arange(len(methods)))
    axes[0].set_xticklabels(methods, rotation=30, ha="right")
    axes[0].set_yticks(np.arange(len(fams)))
    axes[0].set_yticklabels(fams)
    axes[0].set_title("Power Heatmap (alpha=0.05)")
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

    winner = []
    for fam in fams:
        sub = summary[summary["family"] == fam].sort_values("power", ascending=False)
        winner.append("" if sub.empty else str(sub.iloc[0]["method"]))
    axes[1].barh(np.arange(len(fams)), [1.0] * len(fams), color="#4C78A8")
    axes[1].set_yticks(np.arange(len(fams)))
    axes[1].set_yticklabels([f"{fam}: {w}" for fam, w in zip(fams, winner)])
    axes[1].set_xlim(0, 1)
    axes[1].set_xticks([])
    axes[1].set_title("Regime Map (Best Method per Family)")

    fig.tight_layout()
    fig.savefig(out_pdf)
    plt.close(fig)


def plot_regime_map_table(regime_df: pd.DataFrame, out_pdf: Path) -> None:
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    if regime_df.empty:
        _save_placeholder(out_pdf, "Regime Map", "No regime rows available.")
        return
    rows = regime_df.copy()
    rows["cell"] = rows.apply(
        lambda r: f"taxa={int(r['taxa'])};L={int(r['L'])};N={int(r['N'])}", axis=1
    )
    fams = sorted(rows["family"].unique())
    cells = sorted(rows["cell"].unique())
    methods = sorted(rows["best_method_alpha_0_05"].astype(str).unique())
    m2i = {m: i for i, m in enumerate(methods)}
    mat = np.zeros((len(fams), len(cells)), dtype=int)
    for i, fam in enumerate(fams):
        sub = rows[rows["family"] == fam]
        for _, r in sub.iterrows():
            j = cells.index(str(r["cell"]))
            mat[i, j] = m2i[str(r["best_method_alpha_0_05"])]
    fig = plt.figure(figsize=(max(7, len(cells) * 0.7 + 2), max(4, len(fams) * 0.6 + 2)))
    ax = fig.add_subplot(111)
    im = ax.imshow(mat, aspect="auto", cmap="tab20")
    ax.set_xticks(np.arange(len(cells)))
    ax.set_xticklabels(cells, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(fams)))
    ax.set_yticklabels(fams)
    cbar = fig.colorbar(im, ax=ax, ticks=np.arange(len(methods)))
    cbar.ax.set_yticklabels(methods)
    ax.set_title("Regime Map (Best Method @ alpha=0.05)")
    fig.tight_layout()
    fig.savefig(out_pdf)
    plt.close(fig)


def plot_size_deviation(size_df: pd.DataFrame, out_pdf: Path) -> None:
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    if size_df.empty:
        _save_placeholder(out_pdf, "Size Deviation", "No size rows available.")
        return
    view = size_df[size_df["alpha"] == 0.05].copy()
    if view.empty:
        view = size_df.copy()
    view["size_deviation"] = view["size_hat"] - view["alpha"]
    x = np.arange(len(view))
    fig = plt.figure(figsize=(max(7, len(view) * 0.55 + 2), 4.8))
    ax = fig.add_subplot(111)
    ax.axhline(0.0, color="black", linewidth=1)
    ax.plot(x, view["size_deviation"], marker="o")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"taxa={int(t)};L={int(l)};N={int(n)}" for t, l, n in zip(view["taxa"], view["L"], view["N"])],
        rotation=35,
        ha="right",
    )
    ax.set_ylabel("size_hat - alpha")
    ax.set_title("Size Deviation (alpha=0.05)")
    fig.tight_layout()
    fig.savefig(out_pdf)
    plt.close(fig)


def plot_discreteness_diagnostic(null_df: pd.DataFrame, out_pdf: Path) -> None:
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    if null_df.empty:
        _save_placeholder(out_pdf, "Discreteness Diagnostic", "No null rows available.")
        return
    grouped = (
        null_df.groupby("N", as_index=False)["p"]
        .agg(
            n_unique=lambda x: int(np.unique(np.asarray(x, dtype=float)).size),
            n_total="count",
        )
        .sort_values("N")
    )
    grouped["grid_step"] = 1.0 / (grouped["N"] + 1.0)
    fig = plt.figure(figsize=(8, 5))
    ax1 = fig.add_subplot(111)
    ax1.plot(grouped["N"], grouped["n_unique"], marker="o", color="#4C78A8")
    ax1.set_xlabel("N")
    ax1.set_ylabel("Unique p-values")
    ax2 = ax1.twinx()
    ax2.plot(grouped["N"], grouped["grid_step"], marker="s", color="#F58518")
    ax2.set_ylabel("1/(N+1)")
    ax1.set_title("Discreteness Diagnostic")
    fig.tight_layout()
    fig.savefig(out_pdf)
    plt.close(fig)


def plot_runtime_scaling(runtime_df: pd.DataFrame, out_pdf: Path, title: str) -> None:
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    if runtime_df.empty:
        _save_placeholder(out_pdf, title, "No runtime rows available.")
        return

    grouped = (
        runtime_df.groupby(["method", "L"], as_index=False)["runtime_sec"]
        .median()
        .rename(columns={"runtime_sec": "median_runtime_sec"})
    )
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)
    for method in sorted(grouped["method"].unique()):
        sub = grouped[grouped["method"] == method].sort_values("L")
        ax.plot(sub["L"], sub["median_runtime_sec"], marker="o", label=method)
    ax.set_xlabel("Gene length (codons/nt units)")
    ax.set_ylabel("Median runtime (sec)")
    ax.set_title(title)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_pdf)
    plt.close(fig)


def plot_overlap_upset(combos_df: pd.DataFrame, out_pdf: Path, title: str) -> None:
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    if combos_df.empty:
        _save_placeholder(out_pdf, title, "No overlap combinations available.")
        return
    view = combos_df.sort_values("count", ascending=False).head(12)
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)
    ax.bar(np.arange(len(view)), view["count"])
    ax.set_xticks(np.arange(len(view)))
    ax.set_xticklabels(view["combo"], rotation=40, ha="right")
    ax.set_ylabel("Genes")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_pdf)
    plt.close(fig)


def plot_scatter_neglog10(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    out_pdf: Path,
    title: str,
    x_label: str,
    y_label: str,
) -> None:
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    if df.empty:
        _save_placeholder(out_pdf, title, "No overlapping p-values to plot.")
        return
    x = -np.log10(np.clip(df[x_col].to_numpy(dtype=float), 1e-300, 1.0))
    y = -np.log10(np.clip(df[y_col].to_numpy(dtype=float), 1e-300, 1.0))
    lim = max(float(np.max(x)) if len(x) else 1.0, float(np.max(y)) if len(y) else 1.0)
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    ax.plot(x, y, ".", alpha=0.5)
    ax.plot([0, lim], [0, lim], "r--", linewidth=1)
    ax.set_xlim(0, lim * 1.02)
    ax.set_ylim(0, lim * 1.02)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_pdf)
    plt.close(fig)


def plot_enrichment(enrich_df: pd.DataFrame, out_pdf: Path) -> None:
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    if enrich_df.empty:
        _save_placeholder(out_pdf, "F4: Selectome Enrichment", "No enrichment table available.")
        return
    if "odds_ratio" not in enrich_df.columns:
        _save_placeholder(
            out_pdf,
            "F4: Selectome Enrichment",
            "Enrichment table present but missing odds_ratio column.",
        )
        return
    labels = enrich_df["label"].astype(str).tolist() if "label" in enrich_df.columns else ["enrichment"]
    vals = enrich_df["odds_ratio"].to_numpy(dtype=float)
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111)
    ax.bar(np.arange(len(vals)), vals)
    ax.set_xticks(np.arange(len(vals)))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Odds ratio")
    ax.set_title("F4: Enrichment of BABAPPA discoveries in Selectome+")
    fig.tight_layout()
    fig.savefig(out_pdf)
    plt.close(fig)


def plot_viral_signal(
    df: pd.DataFrame, out_pdf: Path, title: str, index_col: str = "gene_index"
) -> None:
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    if df.empty or "p" not in df.columns:
        _save_placeholder(out_pdf, title, "No viral p-value rows available.")
        return
    if index_col not in df.columns:
        df = df.copy()
        df[index_col] = np.arange(1, len(df) + 1)
    x = df[index_col].to_numpy(dtype=float)
    y = -np.log10(np.clip(df["p"].to_numpy(dtype=float), 1e-300, 1.0))
    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(111)
    ax.plot(x, y, ".", markersize=4, alpha=0.7)
    ax.set_xlabel(index_col)
    ax.set_ylabel("-log10(p)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_pdf)
    plt.close(fig)


def plot_pvalue_histograms(babappa_df: pd.DataFrame, baseline_df: pd.DataFrame, out_pdf: Path) -> None:
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    series: list[tuple[str, np.ndarray]] = []
    if not babappa_df.empty and "p" in babappa_df.columns:
        ok = babappa_df.copy()
        if "status" in ok.columns:
            ok = ok[ok["status"] == "OK"]
        p = pd.to_numeric(ok["p"], errors="coerce").dropna().to_numpy(dtype=float)
        if p.size:
            series.append(("babappa", p))
    if not baseline_df.empty and "p_value" in baseline_df.columns:
        for method in sorted(set(str(x) for x in baseline_df["method"].tolist())):
            sub = baseline_df[baseline_df["method"] == method].copy()
            if "status" in sub.columns:
                sub = sub[sub["status"] == "OK"]
            p = pd.to_numeric(sub["p_value"], errors="coerce").dropna().to_numpy(dtype=float)
            if p.size:
                series.append((method, p))
    if not series:
        _save_placeholder(out_pdf, "P-value Histograms", "No valid p-values available.")
        return

    n = len(series)
    cols = 2
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(10, max(4, 3.2 * rows)))
    axes_a = np.atleast_1d(axes).ravel()
    for ax in axes_a[n:]:
        ax.axis("off")
    for ax, (name, pvals) in zip(axes_a, series):
        ax.hist(np.clip(pvals, 0.0, 1.0), bins=20, alpha=0.8)
        ax.set_title(str(name))
        ax.set_xlabel("p-value")
        ax.set_ylabel("count")
    fig.suptitle("P-value distributions by method", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_pdf)
    plt.close(fig)


def plot_runtime_boxplot(runtime_df: pd.DataFrame, out_pdf: Path, title: str) -> None:
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    if runtime_df.empty or "runtime_sec" not in runtime_df.columns:
        _save_placeholder(out_pdf, title, "No runtime rows available.")
        return
    view = runtime_df.copy()
    view["runtime_sec"] = pd.to_numeric(view["runtime_sec"], errors="coerce")
    view = view.dropna(subset=["runtime_sec"])
    if "method" not in view.columns:
        view["method"] = "unknown"
    groups: list[np.ndarray] = []
    labels: list[str] = []
    for method in sorted(set(str(x) for x in view["method"].tolist())):
        vals = view[view["method"] == method]["runtime_sec"].to_numpy(dtype=float)
        if vals.size == 0:
            continue
        groups.append(vals)
        labels.append(method)
    if not groups:
        _save_placeholder(out_pdf, title, "No runtime values after filtering.")
        return
    fig = plt.figure(figsize=(max(8, 1.2 * len(labels) + 3), 5))
    ax = fig.add_subplot(111)
    ax.boxplot(groups, tick_labels=labels, showfliers=False)
    ax.set_ylabel("runtime_sec")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(out_pdf)
    plt.close(fig)


def plot_manhattan_windows(
    window_df: pd.DataFrame,
    baseline_window_df: pd.DataFrame,
    out_pdf: Path,
    title: str,
) -> None:
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    if window_df.empty or "p" not in window_df.columns:
        _save_placeholder(out_pdf, title, "No window-level BABAPPA p-values available.")
        return
    view = window_df.copy()
    if "status" in view.columns:
        view = view[view["status"] == "OK"]
    view["p"] = pd.to_numeric(view["p"], errors="coerce")
    view = view.dropna(subset=["p"])
    if view.empty:
        _save_placeholder(out_pdf, title, "No valid BABAPPA window p-values after filtering.")
        return
    if "window_start_codon" in view.columns:
        x = pd.to_numeric(view["window_start_codon"], errors="coerce").fillna(0).to_numpy(dtype=float)
    else:
        x = np.arange(1, len(view) + 1, dtype=float)
    y = -np.log10(np.clip(view["p"].to_numpy(dtype=float), 1e-300, 1.0))

    fig = plt.figure(figsize=(12, 4.8))
    ax = fig.add_subplot(111)
    ax.plot(x, y, ".", label="BABAPPA", alpha=0.7)

    if not baseline_window_df.empty and "p_value" in baseline_window_df.columns:
        b = baseline_window_df.copy()
        if "status" in b.columns:
            b = b[b["status"] == "OK"]
        if not b.empty:
            b["p_value"] = pd.to_numeric(b["p_value"], errors="coerce")
            b = b.dropna(subset=["p_value"])
            if not b.empty:
                method = str(b.iloc[0].get("method", "baseline"))
                x_map = (
                    view.set_index("gene_id")["window_start_codon"].to_dict()
                    if "window_start_codon" in view.columns
                    else {gid: i + 1 for i, gid in enumerate(view["gene_id"])}
                )
                bx = np.asarray([float(x_map.get(str(g), np.nan)) for g in b["gene_id"]], dtype=float)
                by = -np.log10(np.clip(b["p_value"].to_numpy(dtype=float), 1e-300, 1.0))
                m = np.isfinite(bx)
                if np.any(m):
                    ax.plot(bx[m], by[m], "x", label=method, alpha=0.8)

    ax.set_xlabel("Window start codon")
    ax.set_ylabel("-log10(p)")
    ax.set_title(title)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_pdf)
    plt.close(fig)


def plot_orf_method_bar(babappa_df: pd.DataFrame, baseline_df: pd.DataFrame, out_pdf: Path) -> None:
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    if babappa_df.empty:
        _save_placeholder(out_pdf, "ORF-level Method Comparison", "No ORF-level rows available.")
        return
    b = babappa_df.copy()
    if "status" in b.columns:
        b = b[b["status"] == "OK"]
    b["p"] = pd.to_numeric(b["p"], errors="coerce")
    b = b.dropna(subset=["p"])
    if b.empty:
        _save_placeholder(out_pdf, "ORF-level Method Comparison", "No valid BABAPPA p-values.")
        return

    rows: list[dict[str, Any]] = []
    for _, r in b.iterrows():
        rows.append({"gene_id": str(r["gene_id"]), "method": "babappa", "score": -np.log10(max(float(r["p"]), 1e-300))})
    if not baseline_df.empty:
        for _, r in baseline_df.iterrows():
            if str(r.get("status", "")).upper() != "OK":
                continue
            pv = pd.to_numeric(pd.Series([r.get("p_value")]), errors="coerce").iloc[0]
            if not np.isfinite(pv):
                continue
            rows.append(
                {
                    "gene_id": str(r.get("gene_id", "")),
                    "method": str(r.get("method", "baseline")),
                    "score": -np.log10(max(float(pv), 1e-300)),
                }
            )
    view = pd.DataFrame(rows)
    if view.empty:
        _save_placeholder(out_pdf, "ORF-level Method Comparison", "No values available.")
        return
    top_orfs = (
        view[view["method"] == "babappa"]
        .sort_values("score", ascending=False)["gene_id"]
        .head(12)
        .tolist()
    )
    view = view[view["gene_id"].isin(top_orfs)]
    methods = sorted(set(str(x) for x in view["method"].tolist()))
    orfs = top_orfs
    if not orfs:
        _save_placeholder(out_pdf, "ORF-level Method Comparison", "No ORFs selected.")
        return

    width = 0.8 / max(len(methods), 1)
    x0 = np.arange(len(orfs), dtype=float)
    fig = plt.figure(figsize=(max(10, 0.8 * len(orfs) + 3), 5))
    ax = fig.add_subplot(111)
    for i, m in enumerate(methods):
        vals = []
        sub = view[view["method"] == m]
        score_map = {str(g): float(s) for g, s in zip(sub["gene_id"], sub["score"])}
        for g in orfs:
            vals.append(score_map.get(str(g), 0.0))
        ax.bar(x0 + i * width, vals, width=width, label=m)
    ax.set_xticks(x0 + width * (len(methods) - 1) / 2.0)
    ax.set_xticklabels(orfs, rotation=35, ha="right")
    ax.set_ylabel("-log10(p)")
    ax.set_title("ORF-level method comparison")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_pdf)
    plt.close(fig)


def plot_report_page(lines: Iterable[str], out_pdf: Path, title: str) -> None:
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(8.3, 11.7))
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.text(0.5, 0.97, title, ha="center", va="top", fontsize=15, fontweight="bold")
    y = 0.92
    for line in lines:
        ax.text(0.05, y, str(line), ha="left", va="top", fontsize=9, wrap=True)
        y -= 0.032
        if y < 0.05:
            break
    fig.tight_layout()
    fig.savefig(out_pdf)
    plt.close(fig)


def plot_length_distribution(length_values: np.ndarray, out_pdf: Path, title: str = "Length Distribution") -> None:
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    vals = np.asarray(length_values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        _save_placeholder(out_pdf, title, "No length values available.")
        return
    fig = plt.figure(figsize=(8, 4.8))
    ax = fig.add_subplot(111)
    ax.hist(vals, bins=20, alpha=0.85)
    ax.set_xlabel("Length (nt)")
    ax.set_ylabel("Count")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_pdf)
    plt.close(fig)


def plot_gap_fraction_distribution(gap_values: np.ndarray, out_pdf: Path, title: str = "Gap/Missingness Distribution") -> None:
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    vals = np.asarray(gap_values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        _save_placeholder(out_pdf, title, "No missingness values available.")
        return
    fig = plt.figure(figsize=(8, 4.8))
    ax = fig.add_subplot(111)
    ax.hist(np.clip(vals, 0.0, 1.0), bins=20, alpha=0.85)
    ax.set_xlabel("Gap/missingness fraction")
    ax.set_ylabel("Count")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_pdf)
    plt.close(fig)


def plot_branch_length_histogram(tree_newick: str, out_pdf: Path, title: str = "Branch Length Histogram") -> None:
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    parts = re.findall(r":([0-9]+(?:\\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)", str(tree_newick))
    vals = np.asarray([float(x) for x in parts], dtype=float) if parts else np.asarray([], dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        _save_placeholder(out_pdf, title, "No branch lengths parsed from tree.")
        return
    fig = plt.figure(figsize=(8, 4.8))
    ax = fig.add_subplot(111)
    ax.hist(vals, bins=20, alpha=0.85)
    ax.set_xlabel("Branch length")
    ax.set_ylabel("Count")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_pdf)
    plt.close(fig)
