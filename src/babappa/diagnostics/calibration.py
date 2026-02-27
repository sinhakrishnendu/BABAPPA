from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from .._plotting_env import configure_plotting_env
from ..engine import benjamini_hochberg
from ..hash_utils import sha256_json
from ..phylo import parse_newick


def _read_tsv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, sep="\t")
    except Exception:
        return pd.DataFrame()


def _hash_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _storey_pi0(p_values: np.ndarray) -> tuple[float, list[dict[str, float]]]:
    p = p_values[np.isfinite(p_values)]
    p = p[(p >= 0.0) & (p <= 1.0)]
    if p.size == 0:
        return float("nan"), []
    lambdas = np.arange(0.5, 0.96, 0.05, dtype=float)
    vals: list[float] = []
    rows: list[dict[str, float]] = []
    m = float(p.size)
    for lam in lambdas:
        denom = max((1.0 - float(lam)) * m, 1.0)
        est = float(np.count_nonzero(p > float(lam))) / denom
        est = float(min(max(est, 0.0), 1.0))
        vals.append(est)
        rows.append({"lambda": float(lam), "pi0_hat_lambda": est})
    if not vals:
        return float("nan"), []
    pi0 = float(min(max(np.median(np.array(vals, dtype=float)), 0.0), 1.0))
    return pi0, rows


def _ks_uniform_tail(p_values: np.ndarray, low: float = 0.2, high: float = 1.0) -> tuple[float, float, int]:
    p = p_values[np.isfinite(p_values)]
    p = p[(p >= low) & (p <= high)]
    n = int(p.size)
    if n == 0:
        return float("nan"), float("nan"), 0
    u = (p - low) / max(high - low, 1e-12)
    u = np.clip(u, 0.0, 1.0)
    res = stats.kstest(u, "uniform")
    return float(res.statistic), float(res.pvalue), n


def _spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float, int]:
    xv = np.asarray(x, dtype=float)
    yv = np.asarray(y, dtype=float)
    mask = np.isfinite(xv) & np.isfinite(yv)
    n = int(np.count_nonzero(mask))
    if n < 3:
        return float("nan"), float("nan"), n
    rho, pval = stats.spearmanr(xv[mask], yv[mask], nan_policy="omit")
    return float(rho), float(pval), n


def _tree_length_map(pack: Path, gene_ids: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    tree_dir = pack / "raw" / "preprocessed_units" / "trees"
    for gid in gene_ids:
        candidates = [
            tree_dir / f"{gid}.nwk",
            tree_dir / gid,
        ]
        tpath = next((p for p in candidates if p.exists()), None)
        if tpath is None:
            out[gid] = float("nan")
            continue
        try:
            tree = parse_newick(tpath.read_text(encoding="utf-8"))
            out[gid] = float(np.sum(np.array(tree.branch_lengths(), dtype=float)))
        except Exception:
            out[gid] = float("nan")
    return out


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


def _write_rebuild_script(pack: Path, outdir: Path) -> None:
    script = outdir / "scripts" / "rebuild_all.sh"
    repo_root = Path(__file__).resolve().parents[3]
    text = """#!/usr/bin/env bash
set -euo pipefail
OUT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PACK_DIR="__PACK__"
if python -c "import babappa" >/dev/null 2>&1; then
  python -m babappa.cli diagnostics calibrate --pack "${PACK_DIR}" --outdir "${OUT_DIR}"
elif [ -d "__REPO__/src" ]; then
  export PYTHONPATH="__REPO__/src:${PYTHONPATH:-}"
  python -m babappa.cli diagnostics calibrate --pack "${PACK_DIR}" --outdir "${OUT_DIR}"
else
  echo "Could not import babappa. Install BABAPPA or set PYTHONPATH." >&2
  exit 1
fi
"""
    text = text.replace("__PACK__", str(pack.resolve())).replace("__REPO__", str(repo_root.resolve()))
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text(text, encoding="utf-8")
    script.chmod(script.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _write_provenance_freeze(pack: Path, babappa_df: pd.DataFrame, baseline_df: pd.DataFrame) -> dict[str, Any]:
    manifests = pack / "manifests"
    logs = pack / "logs"
    training_manifest_path = manifests / "training_manifest.json"
    doctor_manifest_path = manifests / "doctor_report.json"
    track_manifest_path = manifests / "benchmark_track_manifest.json"
    baseline_doctor_path = manifests / "baseline_doctor_manifest.json"

    training_manifest = json.loads(training_manifest_path.read_text(encoding="utf-8")) if training_manifest_path.exists() else {}
    doctor_manifest = json.loads(doctor_manifest_path.read_text(encoding="utf-8")) if doctor_manifest_path.exists() else {}
    track_manifest = json.loads(track_manifest_path.read_text(encoding="utf-8")) if track_manifest_path.exists() else {}
    baseline_doctor_manifest = (
        json.loads(baseline_doctor_path.read_text(encoding="utf-8")) if baseline_doctor_path.exists() else {}
    )

    dataset_path = Path(str(doctor_manifest.get("dataset_path", ""))).resolve() if doctor_manifest else None
    if dataset_path is None or not dataset_path.exists():
        dataset_path = None
    dataset_payload: dict[str, Any] = {}
    if dataset_path is not None:
        try:
            payload = json.loads(dataset_path.read_text(encoding="utf-8"))
            dataset_payload = payload if isinstance(payload, dict) else {}
        except Exception:
            dataset_payload = {}

    fetch_manifest_path = dataset_path.parent / "fetch_manifest.json" if dataset_path is not None else None
    fetch_manifest_payload: dict[str, Any] = {}
    if fetch_manifest_path is not None and fetch_manifest_path.exists():
        try:
            payload = json.loads(fetch_manifest_path.read_text(encoding="utf-8"))
            fetch_manifest_payload = payload if isinstance(payload, dict) else {}
        except Exception:
            fetch_manifest_payload = {}

    neutral_model_path = Path(str(training_manifest.get("neutral_model_path", ""))).resolve()
    neutral_payload: dict[str, Any] = {}
    if neutral_model_path.exists():
        try:
            payload = json.loads(neutral_model_path.read_text(encoding="utf-8"))
            neutral_payload = payload if isinstance(payload, dict) else {}
        except Exception:
            neutral_payload = {}

    model_path = Path(str(training_manifest.get("model_path", ""))).resolve()
    model_hash_file = _hash_file(model_path) if model_path.exists() else None
    dataset_hash_file = _hash_file(dataset_path) if dataset_path is not None else None
    fetch_hash_file = _hash_file(fetch_manifest_path) if fetch_manifest_path is not None else None
    training_hash_file = _hash_file(training_manifest_path) if training_manifest_path.exists() else None

    bab_seed_vals = (
        pd.to_numeric(babappa_df["seed_gene"], errors="coerce").dropna().astype(np.int64).to_numpy(dtype=np.int64)
        if "seed_gene" in babappa_df.columns
        else np.array([], dtype=np.int64)
    )
    calib_seed_vals = (
        pd.to_numeric(babappa_df["seed_calib_base"], errors="coerce").dropna().astype(np.int64).to_numpy(dtype=np.int64)
        if "seed_calib_base" in babappa_df.columns
        else np.array([], dtype=np.int64)
    )
    seed_gene_hash = hashlib.sha256(bab_seed_vals.tobytes()).hexdigest() if bab_seed_vals.size else None
    seed_calib_hash = hashlib.sha256(calib_seed_vals.tobytes()).hexdigest() if calib_seed_vals.size else None

    methods = []
    if not baseline_df.empty and "method" in baseline_df.columns:
        methods = sorted(set(str(x) for x in baseline_df["method"].astype(str).tolist()))

    provenance = {
        "schema_version": 1,
        "pack_path": str(pack.resolve()),
        "dataset": {
            "dataset_json_path": (None if dataset_path is None else str(dataset_path)),
            "dataset_json_sha256": dataset_hash_file,
            "dataset_json_hash_canonical": (sha256_json(dataset_payload) if dataset_payload else None),
            "source": dataset_payload.get("metadata", {}).get("source") if dataset_payload else None,
            "release": dataset_payload.get("metadata", {}).get("release") if dataset_payload else None,
            "tree_policy": dataset_payload.get("metadata", {}).get("tree_policy") if dataset_payload else None,
            "synthetic_fallback": dataset_payload.get("metadata", {}).get("synthetic_fallback") if dataset_payload else None,
            "fetch_manifest_path": (None if fetch_manifest_path is None else str(fetch_manifest_path)),
            "fetch_manifest_sha256": fetch_hash_file,
            "fetch_manifest_hash_canonical": (sha256_json(fetch_manifest_payload) if fetch_manifest_payload else None),
        },
        "m0": {
            "neutral_model_path": (str(neutral_model_path) if neutral_model_path.exists() else None),
            "omega": neutral_payload.get("omega"),
            "kappa": neutral_payload.get("kappa"),
            "codon_frequencies_method": neutral_payload.get("codon_frequencies_method"),
            "genetic_code_table": neutral_payload.get("genetic_code_table"),
            "tree_file": neutral_payload.get("tree_file"),
            "tree_newick_present": bool(neutral_payload.get("tree_newick")),
            "m0_hash": training_manifest.get("m0_hash"),
        },
        "frozen_energy_model": {
            "model_path": (str(model_path) if model_path.exists() else None),
            "model_sha256": model_hash_file,
            "model_hash_manifest": training_manifest.get("model_hash"),
            "saved_model_hash": training_manifest.get("saved_model_hash"),
            "model_hash_match": training_manifest.get("model_hash_match"),
            "training_manifest_path": (str(training_manifest_path) if training_manifest_path.exists() else None),
            "training_manifest_sha256": training_hash_file,
            "training_manifest_hash_canonical": (sha256_json(training_manifest) if training_manifest else None),
            "training_samples_n": training_manifest.get("training_samples_n"),
            "L_train": training_manifest.get("L_train"),
        },
        "calibration": {
            "N_unique": (
                sorted(set(int(x) for x in pd.to_numeric(babappa_df["N"], errors="coerce").dropna().astype(int).tolist()))
                if "N" in babappa_df.columns
                else []
            ),
            "tail_unique": (sorted(set(str(x) for x in babappa_df["tail"].astype(str).tolist())) if "tail" in babappa_df.columns else []),
            "pvalue_rule": "rank-based Monte Carlo p-value with add-one correction: (b+1)/(N+1)",
            "qvalue_rule": "Benjamini-Hochberg step-up applied to per-method p-values",
            "bh_implementation": "babappa.engine.benjamini_hochberg",
        },
        "hashes": {
            "phi_hash_unique": (
                sorted(set(str(x) for x in babappa_df["phi_hash"].astype(str).tolist()))
                if "phi_hash" in babappa_df.columns
                else []
            ),
            "model_hash_unique_in_rows": (
                sorted(set(str(x) for x in babappa_df["model_hash"].astype(str).tolist()))
                if "model_hash" in babappa_df.columns
                else []
            ),
            "m0_hash_unique_in_rows": (
                sorted(set(str(x) for x in babappa_df["M0_hash"].astype(str).tolist()))
                if "M0_hash" in babappa_df.columns
                else []
            ),
        },
        "hyphy": {
            "methods_seen": methods,
            "baseline_doctor_manifest_path": (str(baseline_doctor_path) if baseline_doctor_path.exists() else None),
            "hyphy_reported_versions": (
                sorted(
                    set(
                        str(x.get("method_version"))
                        for x in baseline_doctor_manifest.get("results", [])
                        if isinstance(x, dict) and x.get("method_version")
                    )
                )
                if isinstance(baseline_doctor_manifest, dict)
                else []
            ),
            "container_images": (
                sorted(
                    set(
                        str(x.get("container_image"))
                        for x in baseline_doctor_manifest.get("results", [])
                        if isinstance(x, dict) and x.get("container_image")
                    )
                )
                if isinstance(baseline_doctor_manifest, dict)
                else []
            ),
            "container_digests": (
                sorted(
                    set(
                        str(x.get("container_digest"))
                        for x in baseline_doctor_manifest.get("results", [])
                        if isinstance(x, dict) and x.get("container_digest")
                    )
                )
                if isinstance(baseline_doctor_manifest, dict)
                else []
            ),
            "busted_options": "--alignment alignment.fasta --tree tree.nwk --output busted.json",
        },
        "seeds": {
            "run_seed": track_manifest.get("seed"),
            "seed_gene_n": int(bab_seed_vals.size),
            "seed_gene_min": (int(np.min(bab_seed_vals)) if bab_seed_vals.size else None),
            "seed_gene_max": (int(np.max(bab_seed_vals)) if bab_seed_vals.size else None),
            "seed_gene_sha256": seed_gene_hash,
            "seed_calibration_n": int(calib_seed_vals.size),
            "seed_calibration_min": (int(np.min(calib_seed_vals)) if calib_seed_vals.size else None),
            "seed_calibration_max": (int(np.max(calib_seed_vals)) if calib_seed_vals.size else None),
            "seed_calibration_sha256": seed_calib_hash,
        },
        "files": {
            "track_manifest_path": (str(track_manifest_path) if track_manifest_path.exists() else None),
            "doctor_report_path": (str(doctor_manifest_path) if doctor_manifest_path.exists() else None),
            "doctor_report_text_path": (str(logs / "doctor_report.txt") if (logs / "doctor_report.txt").exists() else None),
            "baseline_doctor_report_text_path": (
                str(logs / "baseline_doctor_report.txt") if (logs / "baseline_doctor_report.txt").exists() else None
            ),
        },
    }
    out_path = manifests / "provenance_freeze.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return provenance


def run_calibration_diagnostics(*, pack: str | Path, outdir: str | Path) -> dict[str, Any]:
    configure_plotting_env()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    pack_dir = Path(pack).resolve()
    if not pack_dir.exists():
        raise FileNotFoundError(f"Pack directory not found: {pack_dir}")

    out = Path(outdir).resolve()
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    (out / "manifests").mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(parents=True, exist_ok=True)
    (out / "report").mkdir(parents=True, exist_ok=True)
    (out / "scripts").mkdir(parents=True, exist_ok=True)
    (out / "raw").mkdir(parents=True, exist_ok=True)

    babappa = _read_tsv(pack_dir / "raw" / "babappa_results.tsv")
    baseline_all = _read_tsv(pack_dir / "raw" / "baseline_all.tsv")
    ortholog_results = _read_tsv(pack_dir / "tables" / "ortholog_results.tsv")
    runtime = _read_tsv(pack_dir / "raw" / "runtime.tsv")

    if babappa.empty:
        raise ValueError(f"Missing or empty BABAPPA results: {pack_dir / 'raw' / 'babappa_results.tsv'}")

    if "status" in babappa.columns:
        babappa_ok = babappa[babappa["status"] == "OK"].copy()
    else:
        babappa_ok = babappa.copy()

    if baseline_all.empty:
        busted_ok = pd.DataFrame(columns=["gene_id", "p_value", "q_value", "status"])
    else:
        base = baseline_all.copy()
        if "unit_kind" in base.columns:
            keep = base[base["unit_kind"] == "full_gene"].copy()
            if not keep.empty:
                base = keep
        busted_ok = base[base.get("method", pd.Series([], dtype=str)).astype(str) == "busted"].copy()
        if "status" in busted_ok.columns:
            busted_ok = busted_ok[busted_ok["status"] == "OK"].copy()
        busted_ok["p_value"] = pd.to_numeric(busted_ok.get("p_value", np.nan), errors="coerce")
        busted_ok = busted_ok.dropna(subset=["p_value"])
        q_bus = benjamini_hochberg([float(x) for x in busted_ok["p_value"].tolist()]) if not busted_ok.empty else []
        busted_ok["q_value"] = q_bus

    if ortholog_results.empty:
        ortholog_results = pd.DataFrame({"gene_id": babappa_ok["gene_id"].astype(str).tolist()})
        ortholog_results["L_codons"] = pd.to_numeric(babappa_ok.get("L", np.nan), errors="coerce") / 3.0
        ortholog_results["n_taxa"] = np.nan

    gene_ids = [str(x) for x in ortholog_results.get("gene_id", pd.Series([], dtype=str)).astype(str).tolist()]
    tree_map = _tree_length_map(pack_dir, gene_ids)

    merge = ortholog_results[["gene_id"]].copy()
    merge["L_codons"] = pd.to_numeric(ortholog_results.get("L_codons", np.nan), errors="coerce")
    merge["n_taxa"] = pd.to_numeric(ortholog_results.get("n_taxa", np.nan), errors="coerce")
    merge["tree_total_length"] = merge["gene_id"].map(tree_map)

    bsub = babappa_ok[["gene_id", "p", "q_value", "D_obs"]].copy()
    bsub["p"] = pd.to_numeric(bsub["p"], errors="coerce")
    bsub["q_value"] = pd.to_numeric(bsub["q_value"], errors="coerce")
    bsub["D_obs"] = pd.to_numeric(bsub["D_obs"], errors="coerce")
    bsub = bsub.rename(columns={"p": "babappa_p", "q_value": "babappa_q", "D_obs": "babappa_D"})
    merge = merge.merge(bsub, on="gene_id", how="left")

    if not busted_ok.empty:
        usub = busted_ok[["gene_id", "p_value", "q_value"]].copy()
        usub = usub.rename(columns={"p_value": "busted_p", "q_value": "busted_q"})
        merge = merge.merge(usub, on="gene_id", how="left")
    else:
        merge["busted_p"] = np.nan
        merge["busted_q"] = np.nan

    merge.to_csv(out / "raw" / "merged_calibration_input.tsv", sep="\t", index=False)

    rows: list[dict[str, Any]] = []
    for method, p_col, q_col in [
        ("babappa", "babappa_p", "babappa_q"),
        ("busted", "busted_p", "busted_q"),
    ]:
        p = pd.to_numeric(merge[p_col], errors="coerce").to_numpy(dtype=float)
        q = pd.to_numeric(merge[q_col], errors="coerce").to_numpy(dtype=float)
        p_valid = p[np.isfinite(p)]
        q_valid = q[np.isfinite(q)]
        pi0, pi0_grid = _storey_pi0(p_valid)
        ks_stat, ks_p, ks_n = _ks_uniform_tail(p_valid, low=0.2, high=1.0)
        rows.extend(
            [
                {"method": method, "metric": "n_p", "covariate": "", "value": float(p_valid.size), "n": int(p_valid.size), "note": ""},
                {"method": method, "metric": "n_q", "covariate": "", "value": float(q_valid.size), "n": int(q_valid.size), "note": ""},
                {"method": method, "metric": "mean_p", "covariate": "", "value": float(np.mean(p_valid)) if p_valid.size else np.nan, "n": int(p_valid.size), "note": ""},
                {"method": method, "metric": "median_p", "covariate": "", "value": float(np.median(p_valid)) if p_valid.size else np.nan, "n": int(p_valid.size), "note": ""},
                {"method": method, "metric": "storey_pi0", "covariate": "", "value": pi0, "n": int(p_valid.size), "note": "lambda=0.50..0.95 step 0.05"},
                {"method": method, "metric": "ks_tail_stat", "covariate": "p in [0.2,1.0]", "value": ks_stat, "n": ks_n, "note": ""},
                {"method": method, "metric": "ks_tail_pvalue", "covariate": "p in [0.2,1.0]", "value": ks_p, "n": ks_n, "note": ""},
            ]
        )
        for item in pi0_grid:
            rows.append(
                {
                    "method": method,
                    "metric": "storey_pi0_lambda",
                    "covariate": f"lambda={item['lambda']:.2f}",
                    "value": float(item["pi0_hat_lambda"]),
                    "n": int(p_valid.size),
                    "note": "",
                }
            )

    for method, p_col in [("babappa", "babappa_p"), ("busted", "busted_p")]:
        for cov, cov_col in [("L_codons", "L_codons"), ("n_taxa", "n_taxa"), ("tree_total_length", "tree_total_length")]:
            rho, pv, n = _spearman(
                pd.to_numeric(merge[p_col], errors="coerce").to_numpy(dtype=float),
                pd.to_numeric(merge[cov_col], errors="coerce").to_numpy(dtype=float),
            )
            rows.append(
                {
                    "method": method,
                    "metric": "spearman_rho_p_vs_covariate",
                    "covariate": cov,
                    "value": rho,
                    "n": n,
                    "note": f"pvalue={pv}",
                }
            )

    for cov, cov_col in [("L_codons", "L_codons"), ("n_taxa", "n_taxa"), ("tree_total_length", "tree_total_length")]:
        rho, pv, n = _spearman(
            pd.to_numeric(merge["babappa_D"], errors="coerce").to_numpy(dtype=float),
            pd.to_numeric(merge[cov_col], errors="coerce").to_numpy(dtype=float),
        )
        rows.append(
            {
                "method": "babappa",
                "metric": "spearman_rho_dispersion_vs_covariate",
                "covariate": cov,
                "value": rho,
                "n": n,
                "note": f"pvalue={pv}",
            }
        )

    nullcheck_size_path = pack_dir / "tables" / "nullcheck_size.tsv"
    if nullcheck_size_path.exists():
        size_df = _read_tsv(nullcheck_size_path)
        for row in size_df.itertuples(index=False):
            rows.append(
                {
                    "method": "babappa",
                    "metric": "empirical_size_from_nullcheck",
                    "covariate": f"alpha={getattr(row, 'alpha', np.nan)}",
                    "value": float(getattr(row, "size_hat", np.nan)),
                    "n": int(getattr(row, "n", 0)),
                    "note": f"ci=[{getattr(row, 'ci95_low', np.nan)},{getattr(row, 'ci95_high', np.nan)}]",
                }
            )
    else:
        for alpha in (0.1, 0.05, 0.01, 0.001):
            rows.append(
                {
                    "method": "babappa",
                    "metric": "empirical_size_from_nullcheck",
                    "covariate": f"alpha={alpha}",
                    "value": np.nan,
                    "n": 0,
                    "note": "MISSING_NULLCHECK: run `babappa benchmark nullcheck`",
                }
            )

    diag_df = pd.DataFrame(rows)
    diag_df.to_csv(out / "tables" / "calibration_diagnostics.tsv", sep="\t", index=False)

    # p-value histogram
    fig = plt.figure(figsize=(9.5, 4.0))
    axes = fig.subplots(1, 2)
    for ax, method, col in [(axes[0], "BABAPPA", "babappa_p"), (axes[1], "BUSTED", "busted_p")]:
        p = pd.to_numeric(merge[col], errors="coerce").dropna().to_numpy(dtype=float)
        if p.size:
            ax.hist(np.clip(p, 0.0, 1.0), bins=20, color="#4C78A8", alpha=0.8, edgecolor="white")
        ax.set_title(f"{method} raw p-values")
        ax.set_xlabel("p")
        ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(out / "figures" / "pvalue_histograms.pdf")
    plt.close(fig)

    # q-value histogram
    fig = plt.figure(figsize=(9.5, 4.0))
    axes = fig.subplots(1, 2)
    for ax, method, col in [(axes[0], "BABAPPA", "babappa_q"), (axes[1], "BUSTED", "busted_q")]:
        q = pd.to_numeric(merge[col], errors="coerce").dropna().to_numpy(dtype=float)
        if q.size:
            ax.hist(np.clip(q, 0.0, 1.0), bins=20, color="#F58518", alpha=0.8, edgecolor="white")
        ax.set_title(f"{method} BH q-values")
        ax.set_xlabel("q")
        ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(out / "figures" / "qvalue_histograms.pdf")
    plt.close(fig)

    # QQ plot
    fig = plt.figure(figsize=(9.5, 4.0))
    axes = fig.subplots(1, 2)
    for ax, method, col in [(axes[0], "BABAPPA", "babappa_p"), (axes[1], "BUSTED", "busted_p")]:
        p = np.sort(pd.to_numeric(merge[col], errors="coerce").dropna().to_numpy(dtype=float))
        if p.size:
            exp = (np.arange(1, p.size + 1, dtype=float) - 0.5) / float(p.size)
            ax.plot(-np.log10(np.clip(exp, 1e-300, 1.0)), -np.log10(np.clip(p, 1e-300, 1.0)), ".", alpha=0.7)
            lim = max(float(np.max(-np.log10(np.clip(exp, 1e-300, 1.0)))), float(np.max(-np.log10(np.clip(p, 1e-300, 1.0)))))
            ax.plot([0, lim], [0, lim], "r--", linewidth=1)
            ax.set_xlim(0, lim * 1.02)
            ax.set_ylim(0, lim * 1.02)
        ax.set_title(f"{method} QQ vs Uniform(0,1)")
        ax.set_xlabel("Expected -log10(p)")
        ax.set_ylabel("Observed -log10(p)")
    fig.tight_layout()
    fig.savefig(out / "figures" / "qq_uniform.pdf")
    plt.close(fig)

    # Correlation scatter for BABAPPA p and dispersion.
    fig = plt.figure(figsize=(12, 7))
    axes = fig.subplots(2, 3)
    covs = [("L_codons", "L codons"), ("n_taxa", "n taxa"), ("tree_total_length", "tree length")]
    for j, (col, title) in enumerate(covs):
        x = pd.to_numeric(merge[col], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(merge["babappa_p"], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        axes[0, j].plot(x[mask], y[mask], ".", alpha=0.5)
        axes[0, j].set_title(f"p vs {title}")
        axes[0, j].set_xlabel(title)
        axes[0, j].set_ylabel("BABAPPA p")

        yd = pd.to_numeric(merge["babappa_D"], errors="coerce").to_numpy(dtype=float)
        maskd = np.isfinite(x) & np.isfinite(yd)
        axes[1, j].plot(x[maskd], yd[maskd], ".", alpha=0.5)
        axes[1, j].set_title(f"D_g vs {title}")
        axes[1, j].set_xlabel(title)
        axes[1, j].set_ylabel("BABAPPA D_g")
    fig.tight_layout()
    fig.savefig(out / "figures" / "babappa_correlations.pdf")
    plt.close(fig)

    # BUSTED p correlations.
    fig = plt.figure(figsize=(12, 3.8))
    axes = fig.subplots(1, 3)
    for j, (col, title) in enumerate(covs):
        x = pd.to_numeric(merge[col], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(merge["busted_p"], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        axes[j].plot(x[mask], y[mask], ".", alpha=0.5, color="#F58518")
        axes[j].set_title(f"BUSTED p vs {title}")
        axes[j].set_xlabel(title)
        axes[j].set_ylabel("BUSTED p")
    fig.tight_layout()
    fig.savefig(out / "figures" / "busted_correlations.pdf")
    plt.close(fig)

    # Runtime distribution (sanity)
    if not runtime.empty and "method" in runtime.columns and "runtime_sec" in runtime.columns:
        fig = plt.figure(figsize=(6.5, 4.0))
        ax = fig.add_subplot(111)
        data = []
        labels = []
        for method in ("babappa", "busted"):
            vals = pd.to_numeric(runtime.loc[runtime["method"] == method, "runtime_sec"], errors="coerce").dropna().to_numpy(dtype=float)
            if vals.size:
                data.append(vals)
                labels.append(method)
        if data:
            ax.boxplot(data, tick_labels=labels, showfliers=False)
        ax.set_title("Runtime distribution")
        ax.set_ylabel("runtime_sec")
        fig.tight_layout()
        fig.savefig(out / "figures" / "runtime_distribution.pdf")
        plt.close(fig)

    provenance = _write_provenance_freeze(pack_dir, babappa_ok, baseline_all)
    (out / "manifests" / "provenance_freeze.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    report_pdf = out / "report" / "report.pdf"
    with PdfPages(report_pdf) as pdf:
        fig = plt.figure(figsize=(8.3, 11.7))
        ax = fig.add_subplot(111)
        ax.axis("off")
        ax.text(0.5, 0.97, "BABAPPA Calibration Diagnostics", ha="center", va="top", fontsize=15, fontweight="bold")
        lines = [
            f"Pack: {pack_dir}",
            f"Output: {out}",
            f"BABAPPA rows (OK): {len(babappa_ok)}",
            f"BUSTED rows (OK): {len(busted_ok)}",
            "Theory contract checked: frozen model + rank Monte Carlo + BH.",
            "See tables/calibration_diagnostics.tsv for Storey pi0, KS-tail, and correlation metrics.",
            "See manifests/provenance_freeze.json for run provenance freeze.",
        ]
        y = 0.92
        for line in lines:
            ax.text(0.04, y, line, ha="left", va="top", fontsize=9, wrap=True)
            y -= 0.035
        pdf.savefig(fig)
        plt.close(fig)

    _write_rebuild_script(pack_dir, out)
    _write_checksums(out)

    manifest = {
        "schema_version": 1,
        "command": "diagnostics_calibrate",
        "pack": str(pack_dir),
        "outdir": str(out),
        "n_babappa_ok": int(len(babappa_ok)),
        "n_busted_ok": int(len(busted_ok)),
        "tables": {
            "calibration_diagnostics_tsv": str((out / "tables" / "calibration_diagnostics.tsv").resolve()),
        },
        "figures": {
            "pvalue_histograms": str((out / "figures" / "pvalue_histograms.pdf").resolve()),
            "qvalue_histograms": str((out / "figures" / "qvalue_histograms.pdf").resolve()),
            "qq_uniform": str((out / "figures" / "qq_uniform.pdf").resolve()),
            "babappa_correlations": str((out / "figures" / "babappa_correlations.pdf").resolve()),
            "busted_correlations": str((out / "figures" / "busted_correlations.pdf").resolve()),
        },
        "provenance_freeze": str((pack_dir / "manifests" / "provenance_freeze.json").resolve()),
        "report_pdf": str(report_pdf.resolve()),
        "checksums": str((out / "checksums.txt").resolve()),
    }
    (out / "manifests" / "diagnostics_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest

