from __future__ import annotations

import json
import os
import stat
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ._plotting_env import configure_plotting_env
from .engine import benjamini_hochberg


def _read_tsv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, sep="\t")
    except Exception:
        return pd.DataFrame()


def _dataset_label(pack: Path, manifest: dict[str, Any]) -> str:
    track = str(manifest.get("track", "")).strip().lower()
    preset = str(manifest.get("preset", "")).strip()
    if track and preset:
        return f"{track}:{preset}"
    return pack.name


def _full_gene_baseline(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    if "unit_kind" in out.columns:
        fg = out[out["unit_kind"] == "full_gene"].copy()
        if not fg.empty:
            out = fg
    return out


def _pack_summary(pack_dir: Path) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    manifest_path = pack_dir / "manifests" / "benchmark_track_manifest.json"
    manifest: dict[str, Any] = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    label = _dataset_label(pack_dir, manifest)

    babappa = _read_tsv(pack_dir / "raw" / "babappa_results.tsv")
    baseline = _full_gene_baseline(_read_tsv(pack_dir / "raw" / "baseline_all.tsv"))
    runtime = _read_tsv(pack_dir / "raw" / "runtime.tsv")

    n_units = int(len(babappa))
    bab_ok = babappa[babappa["status"] == "OK"].copy() if not babappa.empty and "status" in babappa.columns else pd.DataFrame()
    bab_p = pd.to_numeric(bab_ok["p"], errors="coerce") if not bab_ok.empty and "p" in bab_ok.columns else pd.Series(dtype=float)
    bab_q = pd.to_numeric(bab_ok["q_value"], errors="coerce") if not bab_ok.empty and "q_value" in bab_ok.columns else pd.Series(dtype=float)
    bab_sig = set(bab_ok.loc[bab_q <= 0.1, "gene_id"].astype(str).tolist()) if not bab_ok.empty else set()

    busted = baseline[baseline["method"] == "busted"].copy() if not baseline.empty and "method" in baseline.columns else pd.DataFrame()
    busted_ok = busted[busted["status"] == "OK"].copy() if not busted.empty and "status" in busted.columns else pd.DataFrame()
    busted_ok["p_value"] = pd.to_numeric(busted_ok["p_value"], errors="coerce")
    busted_ok = busted_ok.dropna(subset=["p_value"]) if not busted_ok.empty else busted_ok
    busted_sig: set[str] = set()
    if not busted_ok.empty:
        q = benjamini_hochberg([float(x) for x in busted_ok["p_value"].tolist()])
        busted_ok["q_value"] = q
        busted_sig = set(busted_ok.loc[pd.to_numeric(busted_ok["q_value"], errors="coerce") <= 0.1, "gene_id"].astype(str).tolist())

    relax = baseline[baseline["method"] == "relax"].copy() if not baseline.empty and "method" in baseline.columns else pd.DataFrame()
    relax_ok = relax[relax["status"] == "OK"].copy() if not relax.empty and "status" in relax.columns else pd.DataFrame()

    rt = runtime.copy()
    if not rt.empty and "unit_kind" in rt.columns:
        keep = rt[rt["unit_kind"] == "full_gene"].copy()
        if not keep.empty:
            rt = keep
    med_bab_runtime = float(pd.to_numeric(rt.loc[rt["method"] == "babappa", "runtime_sec"], errors="coerce").median()) if not rt.empty else np.nan
    med_bus_runtime = float(pd.to_numeric(rt.loc[rt["method"] == "busted", "runtime_sec"], errors="coerce").median()) if not rt.empty else np.nan

    row = {
        "dataset": label,
        "pack_dir": str(pack_dir),
        "n_units": n_units,
        "n_success_babappa": int(len(bab_ok)),
        "n_success_busted": int(len(busted_ok)),
        "n_success_relax": int(len(relax_ok)),
        "n_significant_babappa_q01": int(len(bab_sig)),
        "n_significant_busted_q01": int(len(busted_sig)),
        "overlap_q01": int(len(bab_sig & busted_sig)),
        "median_runtime_babappa": med_bab_runtime,
        "median_runtime_busted": med_bus_runtime,
        "failure_rate_babappa": float(1.0 - (len(bab_ok) / max(n_units, 1))),
        "failure_rate_busted": float(1.0 - (len(busted_ok) / max(n_units, 1))),
    }

    merged = pd.DataFrame()
    if not bab_ok.empty and not busted_ok.empty:
        merged = bab_ok[["gene_id", "p"]].rename(columns={"p": "babappa_p"}).merge(
            busted_ok[["gene_id", "p_value"]].rename(columns={"p_value": "busted_p"}),
            on="gene_id",
            how="inner",
        )
        if not merged.empty:
            merged["babappa_p"] = pd.to_numeric(merged["babappa_p"], errors="coerce")
            merged["busted_p"] = pd.to_numeric(merged["busted_p"], errors="coerce")
            merged = merged.dropna(subset=["babappa_p", "busted_p"])
    return row, merged, rt


def _write_rebuild_script(outdir: Path, inputs: list[Path]) -> None:
    script = outdir / "scripts" / "rebuild_all.sh"
    repo_root = Path(__file__).resolve().parents[2]
    cmd_inputs = " ".join(str(p.resolve()) for p in inputs)
    text = """#!/usr/bin/env bash
set -euo pipefail
OUT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
if python -c "import babappa" >/dev/null 2>&1; then
  python -m babappa.cli report compare --inputs __INPUTS__ --outdir "${OUT_DIR}"
elif [ -d "__REPO__/src" ]; then
  export PYTHONPATH="__REPO__/src:${PYTHONPATH:-}"
  python -m babappa.cli report compare --inputs __INPUTS__ --outdir "${OUT_DIR}"
else
  echo "Could not import babappa. Install BABAPPA or set PYTHONPATH." >&2
  exit 1
fi
"""
    text = text.replace("__INPUTS__", cmd_inputs).replace("__REPO__", str(repo_root))
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text(text, encoding="utf-8")
    script.chmod(script.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def run_compare_report(*, inputs: list[str | Path], outdir: str | Path) -> dict[str, Any]:
    configure_plotting_env()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    in_dirs = [Path(x).resolve() for x in inputs]
    if len(in_dirs) < 1:
        raise ValueError("At least one input pack is required.")
    for p in in_dirs:
        if not p.exists():
            raise FileNotFoundError(f"Input pack not found: {p}")

    out = Path(outdir).resolve()
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    (out / "report").mkdir(parents=True, exist_ok=True)
    (out / "manifests").mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    per_scatter: dict[str, pd.DataFrame] = {}
    per_runtime: dict[str, pd.DataFrame] = {}
    for p in in_dirs:
        row, merged, rt = _pack_summary(p)
        rows.append(row)
        per_scatter[str(row["dataset"])] = merged
        per_runtime[str(row["dataset"])] = rt

    summary = pd.DataFrame(rows)
    summary.to_csv(out / "tables" / "summary.tsv", sep="\t", index=False)

    for row in summary.itertuples(index=False):
        ds = str(getattr(row, "dataset"))
        fig = plt.figure(figsize=(6.5, 4.0))
        ax = fig.add_subplot(111)
        vals = [
            float(getattr(row, "n_significant_babappa_q01")),
            float(getattr(row, "n_significant_busted_q01")),
            float(getattr(row, "overlap_q01")),
        ]
        ax.bar(["BABAPPA", "BUSTED", "Overlap"], vals, color=["#4C78A8", "#F58518", "#54A24B"])
        ax.set_ylabel("Count")
        ax.set_title(f"{ds} overlap @ q<=0.1")
        fig.tight_layout()
        fig.savefig(out / "figures" / f"{ds.replace(':', '_')}_overlap_bar.pdf")
        plt.close(fig)

        rt = per_runtime.get(ds, pd.DataFrame())
        if not rt.empty:
            fig = plt.figure(figsize=(6.5, 4.0))
            ax = fig.add_subplot(111)
            data: list[np.ndarray] = []
            labels: list[str] = []
            for method in ["babappa", "busted"]:
                vals_m = pd.to_numeric(rt.loc[rt["method"] == method, "runtime_sec"], errors="coerce").dropna().to_numpy(dtype=float)
                if vals_m.size:
                    data.append(vals_m)
                    labels.append(method)
            if data:
                ax.boxplot(data, tick_labels=labels, showfliers=False)
                ax.set_ylabel("runtime_sec")
                ax.set_title(f"{ds} runtime comparison")
                fig.tight_layout()
                fig.savefig(out / "figures" / f"{ds.replace(':', '_')}_runtime_boxplot.pdf")
            plt.close(fig)

        sc = per_scatter.get(ds, pd.DataFrame())
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111)
        if not sc.empty:
            x = -np.log10(np.clip(pd.to_numeric(sc["babappa_p"], errors="coerce").to_numpy(dtype=float), 1e-300, 1.0))
            y = -np.log10(np.clip(pd.to_numeric(sc["busted_p"], errors="coerce").to_numpy(dtype=float), 1e-300, 1.0))
            ax.plot(x, y, ".", alpha=0.5)
            lim = max(float(np.max(x)) if len(x) else 1.0, float(np.max(y)) if len(y) else 1.0)
            ax.plot([0, lim], [0, lim], "r--", linewidth=1)
            ax.set_xlim(0, lim * 1.02)
            ax.set_ylim(0, lim * 1.02)
        ax.set_xlabel("-log10 p (BABAPPA)")
        ax.set_ylabel("-log10 p (BUSTED)")
        ax.set_title(f"{ds} scatter")
        fig.tight_layout()
        fig.savefig(out / "figures" / f"{ds.replace(':', '_')}_scatter.pdf")
        plt.close(fig)

    pdf_path = out / "report" / "report.pdf"
    with PdfPages(pdf_path) as pdf:
        fig = plt.figure(figsize=(8.3, 11.7))
        ax = fig.add_subplot(111)
        ax.axis("off")
        ax.text(0.5, 0.97, "BABAPPA vs HyPhy Comparison", ha="center", va="top", fontsize=15, fontweight="bold")
        y = 0.92
        for row in summary.itertuples(index=False):
            ax.text(
                0.04,
                y,
                (
                    f"{getattr(row, 'dataset')}: units={int(getattr(row, 'n_units'))}, "
                    f"babappa_ok={int(getattr(row, 'n_success_babappa'))}, "
                    f"busted_ok={int(getattr(row, 'n_success_busted'))}, "
                    f"sig_q01(babappa)={int(getattr(row, 'n_significant_babappa_q01'))}, "
                    f"sig_q01(busted)={int(getattr(row, 'n_significant_busted_q01'))}, "
                    f"overlap_q01={int(getattr(row, 'overlap_q01'))}"
                ),
                ha="left",
                va="top",
                fontsize=9,
                wrap=True,
            )
            y -= 0.04
        pdf.savefig(fig)
        plt.close(fig)

        for row in summary.itertuples(index=False):
            ds = str(getattr(row, "dataset"))
            fig = plt.figure(figsize=(8.3, 11.7))
            ax = fig.add_subplot(111)
            ax.axis("off")
            ax.text(0.5, 0.97, f"Dataset: {ds}", ha="center", va="top", fontsize=14, fontweight="bold")
            lines = [
                f"n_units: {int(getattr(row, 'n_units'))}",
                f"n_success_babappa: {int(getattr(row, 'n_success_babappa'))}",
                f"n_success_busted: {int(getattr(row, 'n_success_busted'))}",
                f"n_success_relax: {int(getattr(row, 'n_success_relax'))}",
                f"n_significant_babappa_q01: {int(getattr(row, 'n_significant_babappa_q01'))}",
                f"n_significant_busted_q01: {int(getattr(row, 'n_significant_busted_q01'))}",
                f"overlap_q01: {int(getattr(row, 'overlap_q01'))}",
                f"median_runtime_babappa: {float(getattr(row, 'median_runtime_babappa')):.4f}",
                f"median_runtime_busted: {float(getattr(row, 'median_runtime_busted')):.4f}",
                f"failure_rate_babappa: {float(getattr(row, 'failure_rate_babappa')):.4f}",
                f"failure_rate_busted: {float(getattr(row, 'failure_rate_busted')):.4f}",
            ]
            yy = 0.9
            for line in lines:
                ax.text(0.05, yy, line, ha="left", va="top", fontsize=10)
                yy -= 0.04
            pdf.savefig(fig)
            plt.close(fig)

    _write_rebuild_script(out, in_dirs)
    manifest = {
        "schema_version": 1,
        "inputs": [str(p) for p in in_dirs],
        "outdir": str(out),
        "n_inputs": int(len(in_dirs)),
        "summary_tsv": str((out / "tables" / "summary.tsv").resolve()),
        "report_pdf": str(pdf_path.resolve()),
    }
    (out / "manifests" / "compare_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest
