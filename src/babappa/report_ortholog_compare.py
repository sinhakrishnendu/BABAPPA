from __future__ import annotations

import json
import stat
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from ._plotting_env import configure_plotting_env


def _read_tsv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, sep="\t")
    except Exception:
        return pd.DataFrame()


def _hash_file(path: Path) -> str | None:
    import hashlib

    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_checksums(root: Path) -> None:
    rows: list[str] = []
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(root).as_posix()
        if rel == "checksums.txt":
            continue
        h = _hash_file(p)
        if h is None:
            continue
        rows.append(f"{h}  {rel}")
    (root / "checksums.txt").write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")


def _write_rebuild_script(pack: Path, audit: Path, outdir: Path) -> None:
    script = outdir / "scripts" / "rebuild_all.sh"
    repo_root = Path(__file__).resolve().parents[2]
    text = """#!/usr/bin/env bash
set -euo pipefail
OUT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PACK_DIR="__PACK__"
AUDIT_DIR="__AUDIT__"
if python -c "import babappa" >/dev/null 2>&1; then
  python -m babappa.cli report compare --pack "${PACK_DIR}" --audit "${AUDIT_DIR}" --outdir "${OUT_DIR}"
elif [ -d "__REPO__/src" ]; then
  export PYTHONPATH="__REPO__/src:${PYTHONPATH:-}"
  python -m babappa.cli report compare --pack "${PACK_DIR}" --audit "${AUDIT_DIR}" --outdir "${OUT_DIR}"
else
  echo "Could not import babappa. Install BABAPPA or set PYTHONPATH." >&2
  exit 1
fi
"""
    text = (
        text.replace("__PACK__", str(pack.resolve()))
        .replace("__AUDIT__", str(audit.resolve()))
        .replace("__REPO__", str(repo_root.resolve()))
    )
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text(text, encoding="utf-8")
    script.chmod(script.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _qq_data_with_envelope(p_values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    p = p_values[np.isfinite(p_values)]
    p = p[(p >= 0.0) & (p <= 1.0)]
    p = np.sort(p)
    n = int(p.size)
    if n == 0:
        z = np.array([], dtype=float)
        return z, z, z, z
    i = np.arange(1, n + 1, dtype=float)
    exp_p = i / float(n + 1)
    lo = stats.beta.ppf(0.025, i, n + 1 - i)
    hi = stats.beta.ppf(0.975, i, n + 1 - i)
    x = -np.log10(np.clip(exp_p, 1e-300, 1.0))
    y = -np.log10(np.clip(p, 1e-300, 1.0))
    y_lo = -np.log10(np.clip(hi, 1e-300, 1.0))
    y_hi = -np.log10(np.clip(lo, 1e-300, 1.0))
    return x, y, y_lo, y_hi


def run_ortholog_compare_report(*, pack: str | Path, audit: str | Path, outdir: str | Path) -> dict[str, Any]:
    configure_plotting_env()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    pack_dir = Path(pack).resolve()
    audit_dir = Path(audit).resolve()
    out = Path(outdir).resolve()
    for sub in ("tables", "figures", "report", "manifests", "logs", "scripts"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    orth = _read_tsv(pack_dir / "tables" / "ortholog_results.tsv")
    runtime = _read_tsv(pack_dir / "raw" / "runtime.tsv")
    pi0 = _read_tsv(audit_dir / "tables" / "pi0.tsv")
    null_size = _read_tsv(audit_dir / "tables" / "null_size.tsv")
    fdr_inputs = _read_tsv(audit_dir / "tables" / "fdr_inputs.tsv")
    fdr_qdiff = _read_tsv(audit_dir / "tables" / "fdr_q_diff.tsv")
    corr = _read_tsv(audit_dir / "tables" / "covariate_correlations.tsv")
    frozen = _read_tsv(audit_dir / "tables" / "frozen_energy_invariant.tsv")
    testable = _read_tsv(audit_dir / "tables" / "testable_set_alignment.tsv")
    null_uniformity = _read_tsv(audit_dir / "tables" / "null_uniformity.tsv")
    audit_manifest = (
        json.loads((audit_dir / "manifests" / "audit_manifest.json").read_text(encoding="utf-8"))
        if (audit_dir / "manifests" / "audit_manifest.json").exists()
        else {}
    )

    if orth.empty:
        raise ValueError(f"Missing or empty ortholog results: {pack_dir / 'tables' / 'ortholog_results.tsv'}")

    n_units = int(len(orth))
    testable_mask = (
        orth["in_testable_set"].astype(bool)
        if "in_testable_set" in orth.columns
        else (
            (orth.get("babappa_status", pd.Series([], dtype=str)).astype(str) == "OK")
            & (orth.get("busted_status", pd.Series([], dtype=str)).astype(str) == "OK")
        )
    )
    bq_col = "babappa_q_testable_set" if "babappa_q_testable_set" in orth.columns else "babappa_q"
    hq_col = "busted_q_testable_set" if "busted_q_testable_set" in orth.columns else "busted_q"
    bq_mask = testable_mask & (pd.to_numeric(orth[bq_col], errors="coerce") < 0.1)
    hq_mask = testable_mask & (pd.to_numeric(orth[hq_col], errors="coerce") < 0.1)
    bq_all_mask = pd.to_numeric(
        orth["babappa_q_all_candidates"] if "babappa_q_all_candidates" in orth.columns else orth["babappa_q"],
        errors="coerce",
    ) < 0.1
    hq_all_mask = pd.to_numeric(
        orth["busted_q_all_candidates"] if "busted_q_all_candidates" in orth.columns else orth["busted_q"],
        errors="coerce",
    ) < 0.1
    overlap = int(np.count_nonzero(bq_mask & hq_mask))
    bab_only = int(np.count_nonzero(bq_mask & (~hq_mask)))
    bus_only = int(np.count_nonzero((~bq_mask) & hq_mask))
    n_testable = int(np.count_nonzero(testable_mask))

    median_bab_rt = float(pd.to_numeric(orth["babappa_runtime_sec"], errors="coerce").median())
    median_bus_rt = float(pd.to_numeric(orth["busted_runtime_sec"], errors="coerce").median())
    busted_success_rate = float(np.mean(orth["busted_status"].astype(str) == "OK"))

    pi0_map = {str(r.method): float(r.pi0_hat) for r in pi0.itertuples(index=False)} if not pi0.empty else {}
    size05 = float("nan")
    if not null_size.empty:
        sub05 = null_size[np.isclose(pd.to_numeric(null_size["alpha"], errors="coerce"), 0.05)]
        if not sub05.empty:
            size05 = float(sub05.iloc[0]["size_hat"])
    null_ks_disc = float("nan")
    if not null_uniformity.empty:
        sub = null_uniformity[null_uniformity["metric"].astype(str) == "ks_discrete_grid_stat"]
        if not sub.empty:
            null_ks_disc = float(pd.to_numeric(sub.iloc[0]["value"], errors="coerce"))

    frozen_fail = bool(audit_manifest.get("frozen_energy_invariant_fail", False))
    testable_mismatch = bool(audit_manifest.get("testable_set_mismatch", False))
    severe_inflation = bool(audit_manifest.get("severe_inflation", False))

    n_shared_ok = float("nan")
    if not testable.empty:
        sub = testable[testable["metric"].astype(str) == "n_shared_ok_testable"]
        if not sub.empty:
            n_shared_ok = float(pd.to_numeric(sub.iloc[0]["value"], errors="coerce"))

    summary = pd.DataFrame(
        [
            {
                "n_units": n_units,
                "testable_set_size": int(n_testable),
                "babappa_q_lt_0.1": int(np.count_nonzero(bq_mask)),
                "busted_q_lt_0.1": int(np.count_nonzero(hq_mask)),
                "babappa_q_lt_0.1_on_all_candidates": int(np.count_nonzero(bq_all_mask)),
                "busted_q_lt_0.1_on_all_candidates": int(np.count_nonzero(hq_all_mask)),
                "overlap_q_lt_0.1": int(overlap),
                "babappa_only": int(bab_only),
                "busted_only": int(bus_only),
                "median_runtime_babappa_sec": median_bab_rt,
                "median_runtime_busted_sec": median_bus_rt,
                "pi0_babappa": float(pi0_map.get("babappa", float("nan"))),
                "pi0_busted": float(pi0_map.get("busted", float("nan"))),
                "size_hat_babappa_alpha_0.05": float(size05),
                "null_ks_discrete_stat": float(null_ks_disc),
                "frozen_energy_invariant_fail": bool(frozen_fail),
                "testable_set_mismatch": bool(testable_mismatch),
                "severe_inflation": bool(severe_inflation),
                "n_shared_ok_testable": int(n_shared_ok) if np.isfinite(n_shared_ok) else np.nan,
                "baseline_success_rate_busted": float(busted_success_rate),
            }
        ]
    )
    summary.to_csv(out / "tables" / "summary.tsv", sep="\t", index=False)

    # Figures.
    # 1) upset_q01
    fig = plt.figure(figsize=(6.5, 4.0))
    ax = fig.add_subplot(111)
    cats = ["BABAPPA-only", "Overlap", "BUSTED-only", "Neither"]
    neither = int(max(0, n_testable - bab_only - bus_only - overlap))
    vals = [bab_only, overlap, bus_only, neither]
    ax.bar(cats, vals, color=["#4C78A8", "#54A24B", "#F58518", "#999999"])
    ax.set_ylabel("count")
    ax.set_title("q<0.1 overlap counts")
    fig.tight_layout()
    fig.savefig(out / "figures" / "upset_q01.pdf")
    plt.close(fig)

    # 2) scatter_neglogp
    fig = plt.figure(figsize=(5.5, 5.5))
    ax = fig.add_subplot(111)
    x = -np.log10(np.clip(pd.to_numeric(orth["babappa_p"], errors="coerce").to_numpy(dtype=float), 1e-300, 1.0))
    y = -np.log10(np.clip(pd.to_numeric(orth["busted_p"], errors="coerce").to_numpy(dtype=float), 1e-300, 1.0))
    mask = np.isfinite(x) & np.isfinite(y)
    ax.plot(x[mask], y[mask], ".", alpha=0.6)
    if np.any(mask):
        lim = max(float(np.max(x[mask])), float(np.max(y[mask])))
        ax.plot([0, lim], [0, lim], "r--", linewidth=1)
        ax.set_xlim(0, lim * 1.02)
        ax.set_ylim(0, lim * 1.02)
    ax.set_xlabel("-log10(BABAPPA p)")
    ax.set_ylabel("-log10(BUSTED p)")
    ax.set_title("Per-gene p-value scatter")
    fig.tight_layout()
    fig.savefig(out / "figures" / "scatter_neglogp.pdf")
    plt.close(fig)

    # 3) runtime_compare
    fig = plt.figure(figsize=(6.5, 4.0))
    ax = fig.add_subplot(111)
    rt_data = []
    rt_labels = []
    for m, c in [("BABAPPA", "babappa_runtime_sec"), ("BUSTED", "busted_runtime_sec")]:
        vv = pd.to_numeric(orth[c], errors="coerce").dropna().to_numpy(dtype=float)
        if vv.size:
            rt_data.append(vv)
            rt_labels.append(m)
    if rt_data:
        ax.boxplot(rt_data, tick_labels=rt_labels, showfliers=False)
    ax.set_ylabel("runtime_sec")
    ax.set_title("Runtime comparison")
    fig.tight_layout()
    fig.savefig(out / "figures" / "runtime_compare.pdf")
    plt.close(fig)

    # 4) qq_babappa_and_busted
    fig = plt.figure(figsize=(9.5, 4.5))
    axes = fig.subplots(1, 2)
    for ax, vals, title in [
        (axes[0], pd.to_numeric(orth["babappa_p"], errors="coerce").to_numpy(dtype=float), "BABAPPA"),
        (axes[1], pd.to_numeric(orth["busted_p"], errors="coerce").to_numpy(dtype=float), "BUSTED"),
    ]:
        xx, yy, yl, yh = _qq_data_with_envelope(vals)
        if xx.size:
            ax.fill_between(xx, yl, yh, color="#DDDDDD", alpha=0.7)
            ax.plot(xx, yy, ".", alpha=0.8)
            lim = max(float(np.max(xx)), float(np.max(yy)))
            ax.plot([0, lim], [0, lim], "r--", linewidth=1)
            ax.set_xlim(0, lim * 1.02)
            ax.set_ylim(0, lim * 1.02)
        ax.set_title(title)
        ax.set_xlabel("Expected -log10(p)")
        ax.set_ylabel("Observed -log10(p)")
    fig.tight_layout()
    fig.savefig(out / "figures" / "qq_babappa_and_busted.pdf")
    plt.close(fig)

    # Report.
    track_manifest_path = pack_dir / "benchmark_track_manifest.json"
    if not track_manifest_path.exists():
        alt = pack_dir / "manifests" / "benchmark_track_manifest.json"
        if alt.exists():
            track_manifest_path = alt

    core_files = [
        track_manifest_path,
        pack_dir / "checksums.txt",
        pack_dir / "report" / "report.pdf",
        pack_dir / "raw" / "babappa_results.tsv",
        pack_dir / "raw" / "baseline_all.tsv",
        pack_dir / "tables" / "ortholog_results.tsv",
    ]
    core_missing = [str(p) for p in core_files if not p.exists()]
    core_ok = len(core_missing) == 0

    comparison_pdf = out / "report" / "comparison_report.pdf"
    with PdfPages(comparison_pdf) as pdf:
        fig = plt.figure(figsize=(8.3, 11.7))
        ax = fig.add_subplot(111)
        ax.axis("off")
        ax.text(0.5, 0.97, "BABAPPA vs HyPhy (Ortholog v9)", ha="center", va="top", fontsize=15, fontweight="bold")
        y = 0.92
        sections = [
            ("Section 1: pack integrity", [f"core_ok={core_ok}", f"missing_count={len(core_missing)}"]),
            (
                "Section 2: FDR verification",
                [
                    f"fdr_inputs_rows={len(fdr_inputs)}",
                    f"fdr_qdiff_max_babappa={float(fdr_qdiff.loc[fdr_qdiff['method']=='babappa','max_abs_diff'].max()) if not fdr_qdiff.empty else float('nan'):.3e}",
                    f"fdr_qdiff_max_busted={float(fdr_qdiff.loc[fdr_qdiff['method']=='busted','max_abs_diff'].max()) if not fdr_qdiff.empty else float('nan'):.3e}",
                ],
            ),
            (
                "Section 3: p/QQ + pi0",
                [
                    f"pi0_babappa={float(pi0_map.get('babappa', float('nan'))):.6f}",
                    f"pi0_busted={float(pi0_map.get('busted', float('nan'))):.6f}",
                ],
            ),
            (
                "Section 4: confounders",
                [
                    f"covariate_tests={len(corr)}",
                    f"strong_flags(|rho|>0.5 for BABAPPA L_codons/GC3)={int(np.count_nonzero((corr.get('method', pd.Series([], dtype=str)).astype(str)=='babappa') & (corr.get('covariate', pd.Series([], dtype=str)).astype(str).isin(['L_codons','GC3'])) & (pd.to_numeric(corr.get('spearman_rho', pd.Series([], dtype=float)), errors='coerce').abs()>0.5))) if not corr.empty else 0}",
                ],
            ),
            (
                "Section 5: matched-null size",
                [
                    f"size_hat_alpha_0.05={size05:.6f}",
                    f"null_ks_discrete_stat={null_ks_disc:.6f}",
                    f"calibration_fail={bool(audit_manifest.get('calibration_fail', False))}",
                    f"severe_inflation={severe_inflation}",
                ],
            ),
            (
                "Section 6: frozen-energy and testable-set",
                [
                    f"frozen_energy_invariant_fail={frozen_fail}",
                    f"testable_set_mismatch={testable_mismatch}",
                    f"frozen_checks={len(frozen)}",
                    f"testable_rows={len(testable)}",
                ],
            ),
            (
                "Section 7: BABAPPA vs BUSTED summary",
                [
                    f"n_units={n_units}",
                    f"testable_set_size={n_testable}",
                    f"babappa_q_lt_0.1={int(np.count_nonzero(bq_mask))}",
                    f"busted_q_lt_0.1={int(np.count_nonzero(hq_mask))}",
                    f"babappa_q_lt_0.1_on_all_candidates={int(np.count_nonzero(bq_all_mask))}",
                    f"busted_q_lt_0.1_on_all_candidates={int(np.count_nonzero(hq_all_mask))}",
                    f"overlap_q_lt_0.1={overlap}",
                    f"babappa_only={bab_only}",
                    f"busted_only={bus_only}",
                    f"median_runtime_babappa_sec={median_bab_rt:.3f}",
                    f"median_runtime_busted_sec={median_bus_rt:.3f}",
                    f"baseline_success_rate_busted={busted_success_rate:.6f}",
                ],
            ),
        ]
        for title, lines in sections:
            ax.text(0.04, y, title, ha="left", va="top", fontsize=11, fontweight="bold")
            y -= 0.028
            for line in lines:
                ax.text(0.06, y, line, ha="left", va="top", fontsize=9)
                y -= 0.024
            y -= 0.01
        pdf.savefig(fig)
        plt.close(fig)

    manifest = {
        "schema_version": 1,
        "command": "report_compare_ortholog_pack",
        "pack": str(pack_dir),
        "audit": str(audit_dir),
        "outdir": str(out),
        "summary_tsv": str((out / "tables" / "summary.tsv").resolve()),
        "comparison_report_pdf": str(comparison_pdf.resolve()),
        "figures": {
            "upset_q01": str((out / "figures" / "upset_q01.pdf").resolve()),
            "scatter_neglogp": str((out / "figures" / "scatter_neglogp.pdf").resolve()),
            "runtime_compare": str((out / "figures" / "runtime_compare.pdf").resolve()),
            "qq_babappa_and_busted": str((out / "figures" / "qq_babappa_and_busted.pdf").resolve()),
        },
    }
    (out / "manifests" / "compare_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_rebuild_script(pack_dir, audit_dir, out)
    _write_checksums(out)
    return manifest
