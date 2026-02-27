from __future__ import annotations

import hashlib
import json
import os
import stat
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from ._plotting_env import configure_plotting_env
from .engine import benjamini_hochberg, compute_dispersion
from .hash_utils import sha256_json
from .io import Alignment, read_fasta
from .model import EnergyModel, load_model
from .neutral import GY94NeutralSimulator, NeutralSpec
from .phylo import parse_newick
from .representation import alignment_to_matrix
from .stats import rank_p_value


_NULL_MODEL: EnergyModel | None = None


def _read_tsv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, sep="\t")
    except Exception:
        return pd.DataFrame()


def _write_tsv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False)


def _hash_file(path: Path) -> str | None:
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


def _write_rebuild_script(pack: Path, outdir: Path, seed: int, null_N: int, null_G: int, jobs: int) -> None:
    script = outdir / "scripts" / "rebuild_all.sh"
    repo_root = Path(__file__).resolve().parents[2]
    text = """#!/usr/bin/env bash
set -euo pipefail
OUT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PACK_DIR="__PACK__"
if python -c "import babappa" >/dev/null 2>&1; then
  python -m babappa.cli audit ortholog --pack "${PACK_DIR}" --outdir "${OUT_DIR}" --seed __SEED__ --null_N __NULL_N__ --null_G __NULL_G__ --jobs __JOBS__
elif [ -d "__REPO__/src" ]; then
  export PYTHONPATH="__REPO__/src:${PYTHONPATH:-}"
  python -m babappa.cli audit ortholog --pack "${PACK_DIR}" --outdir "${OUT_DIR}" --seed __SEED__ --null_N __NULL_N__ --null_G __NULL_G__ --jobs __JOBS__
else
  echo "Could not import babappa. Install BABAPPA or set PYTHONPATH." >&2
  exit 1
fi
"""
    text = (
        text.replace("__PACK__", str(pack.resolve()))
        .replace("__REPO__", str(repo_root.resolve()))
        .replace("__SEED__", str(int(seed)))
        .replace("__NULL_N__", str(int(null_N)))
        .replace("__NULL_G__", str(int(null_G)))
        .replace("__JOBS__", str(int(jobs)))
    )
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text(text, encoding="utf-8")
    script.chmod(script.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _storey_pi0(p_values: np.ndarray) -> tuple[float, list[float]]:
    p = p_values[np.isfinite(p_values)]
    p = p[(p >= 0.0) & (p <= 1.0)]
    if p.size == 0:
        return float("nan"), []
    lambdas = np.arange(0.5, 0.96, 0.05, dtype=float)
    vals: list[float] = []
    m = float(p.size)
    for lam in lambdas:
        denom = max((1.0 - float(lam)) * m, 1.0)
        est = float(np.count_nonzero(p > float(lam))) / denom
        vals.append(float(min(max(est, 0.0), 1.0)))
    pi0 = float(min(max(np.median(np.array(vals, dtype=float)), 0.0), 1.0))
    return pi0, [float(x) for x in lambdas.tolist()]


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


def _spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float, int]:
    xv = np.asarray(x, dtype=float)
    yv = np.asarray(y, dtype=float)
    mask = np.isfinite(xv) & np.isfinite(yv)
    n = int(np.count_nonzero(mask))
    if n < 3:
        return float("nan"), float("nan"), n
    rho, pval = stats.spearmanr(xv[mask], yv[mask], nan_policy="omit")
    return float(rho), float(pval), n


def _alignment_covariates(aln: Alignment) -> dict[str, float]:
    total = int(aln.length) * int(aln.n_sequences)
    if total <= 0:
        return {
            "gap_fraction": float("nan"),
            "ambiguous_fraction": float("nan"),
            "GC": float("nan"),
            "GC3": float("nan"),
            "mean_pairwise_divergence": float("nan"),
        }

    gap = 0
    ambig = 0
    gc = 0
    valid = 0
    gc3 = 0
    valid3 = 0

    for seq in aln.sequences:
        s = seq.upper().replace("U", "T")
        for ch in s:
            if ch == "-":
                gap += 1
                continue
            if ch not in {"A", "C", "G", "T"}:
                ambig += 1
                continue
            valid += 1
            if ch in {"G", "C"}:
                gc += 1
        usable = len(s) - (len(s) % 3)
        for i in range(0, usable, 3):
            c3 = s[i + 2]
            if c3 in {"A", "C", "G", "T"}:
                valid3 += 1
                if c3 in {"G", "C"}:
                    gc3 += 1

    pair_div: list[float] = []
    seqs = [x.upper().replace("U", "T") for x in aln.sequences]
    for i in range(len(seqs)):
        for j in range(i + 1, len(seqs)):
            mism = 0
            denom = 0
            s1 = seqs[i]
            s2 = seqs[j]
            for a, b in zip(s1, s2):
                if a not in {"A", "C", "G", "T"}:
                    continue
                if b not in {"A", "C", "G", "T"}:
                    continue
                denom += 1
                if a != b:
                    mism += 1
            if denom > 0:
                pair_div.append(float(mism) / float(denom))

    return {
        "gap_fraction": float(gap) / float(total),
        "ambiguous_fraction": float(ambig) / float(total),
        "GC": (float(gc) / float(valid) if valid > 0 else float("nan")),
        "GC3": (float(gc3) / float(valid3) if valid3 > 0 else float("nan")),
        "mean_pairwise_divergence": (float(np.mean(np.array(pair_div, dtype=float))) if pair_div else float("nan")),
    }


def _init_null_worker(model_payload: dict[str, Any]) -> None:
    global _NULL_MODEL
    _NULL_MODEL = EnergyModel.from_dict(model_payload)


def _calibration_worker(task: tuple[str, int, str, float, dict[str, float] | None, int, int]) -> dict[str, Any]:
    gene_id, length_nt, tree_newick, kappa, codon_freq, null_N, seed_calib = task
    global _NULL_MODEL
    if _NULL_MODEL is None:
        raise RuntimeError("Null worker was not initialized with model payload.")
    t0 = time.perf_counter()
    spec = NeutralSpec(
        tree_newick=tree_newick,
        kappa=float(kappa),
        omega=1.0,
        codon_frequencies=codon_freq,
        simulator="gy94",
    )
    simulator = GY94NeutralSimulator(spec)

    rng = np.random.default_rng(int(seed_calib))
    null_vals = np.empty(int(null_N), dtype=float)
    for j in range(int(null_N)):
        s = int(rng.integers(0, 2**32 - 1))
        aln_j = simulator.simulate_alignment(length_nt=int(length_nt), seed=s)
        null_vals[j], _ = compute_dispersion(alignment_to_matrix(aln_j), _NULL_MODEL)
    return {
        "gene_id": str(gene_id),
        "L_codons": int(length_nt) // 3,
        "seed_calib": int(seed_calib),
        "null_values": null_vals.tolist(),
        "runtime_sec": float(time.perf_counter() - t0),
        "status": "OK",
        "reason": "ok",
    }


def _observed_worker(task: tuple[int, str, int, str, float, dict[str, float] | None, int]) -> dict[str, Any]:
    rep_id, gene_id, length_nt, tree_newick, kappa, codon_freq, seed_obs = task
    global _NULL_MODEL
    if _NULL_MODEL is None:
        raise RuntimeError("Null worker was not initialized with model payload.")
    t0 = time.perf_counter()
    spec = NeutralSpec(
        tree_newick=tree_newick,
        kappa=float(kappa),
        omega=1.0,
        codon_frequencies=codon_freq,
        simulator="gy94",
    )
    simulator = GY94NeutralSimulator(spec)
    obs = simulator.simulate_alignment(length_nt=int(length_nt), seed=int(seed_obs))
    obs_disp, _ = compute_dispersion(alignment_to_matrix(obs), _NULL_MODEL)
    return {
        "replicate_id": int(rep_id),
        "gene_id": str(gene_id),
        "L_codons": int(length_nt) // 3,
        "obs_dispersion": float(obs_disp),
        "seed_obs": int(seed_obs),
        "runtime_sec": float(time.perf_counter() - t0),
        "status": "OK",
        "reason": "ok",
    }


def _null_ci(k: int, n: int) -> tuple[float, float]:
    if n <= 0:
        return float("nan"), float("nan")
    ci = stats.binomtest(int(k), int(n)).proportion_ci(confidence_level=0.95, method="exact")
    return float(ci.low), float(ci.high)


def _frozen_energy_invariant_table(
    *,
    pack_dir: Path,
    babappa_raw: pd.DataFrame,
) -> tuple[pd.DataFrame, bool, dict[str, Any]]:
    training_manifest_path = pack_dir / "manifests" / "training_manifest.json"
    model_path = pack_dir / "raw" / "frozen_model.json"
    runtime_hash_path = pack_dir / "raw" / "frozen_energy_runtime_hashes.tsv"
    training_manifest: dict[str, Any] = {}
    if training_manifest_path.exists():
        try:
            payload = json.loads(training_manifest_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                training_manifest = payload
        except Exception:
            training_manifest = {}

    expected_model_hash = str(training_manifest.get("model_hash", "") or "")
    expected_m0_hash = str(training_manifest.get("m0_hash", "") or "")
    expected_phi_hash = str(training_manifest.get("phi_hash", "") or "")
    expected_m0_hash_frozen_spec = ""
    if model_path.exists():
        try:
            model_payload = json.loads(model_path.read_text(encoding="utf-8"))
            if isinstance(model_payload, dict) and isinstance(model_payload.get("neutral_spec"), dict):
                expected_m0_hash_frozen_spec = str(sha256_json(model_payload["neutral_spec"]))
        except Exception:
            expected_m0_hash_frozen_spec = ""

    ok = babappa_raw.copy()
    if "status" in ok.columns:
        ok = ok[ok["status"].astype(str) == "OK"].copy()
    if "p" in ok.columns:
        ok = ok[pd.to_numeric(ok["p"], errors="coerce").notna()].copy()

    model_vals = (
        sorted(
            set(
                str(x)
                for x in ok.get("model_hash", pd.Series([], dtype=str)).astype(str).tolist()
                if str(x).strip() and str(x).strip().lower() != "nan"
            )
        )
        if not ok.empty
        else []
    )
    m0_vals = (
        sorted(
            set(
                str(x)
                for x in ok.get("M0_hash", pd.Series([], dtype=str)).astype(str).tolist()
                if str(x).strip() and str(x).strip().lower() != "nan"
            )
        )
        if not ok.empty
        else []
    )
    phi_vals = (
        sorted(
            set(
                str(x)
                for x in ok.get("phi_hash", pd.Series([], dtype=str)).astype(str).tolist()
                if str(x).strip() and str(x).strip().lower() != "nan"
            )
        )
        if not ok.empty
        else []
    )
    runtime_hash = _read_tsv(runtime_hash_path) if runtime_hash_path.exists() else pd.DataFrame()
    runtime_after_training_vals: list[str] = []
    runtime_before_vals: list[str] = []
    runtime_after_vals: list[str] = []
    runtime_phi_before_vals: list[str] = []
    runtime_phi_after_vals: list[str] = []
    runtime_m0_before_vals: list[str] = []
    runtime_m0_after_vals: list[str] = []
    if not runtime_hash.empty:
        ev = runtime_hash.get("event", pd.Series([], dtype=str)).astype(str)
        eh = runtime_hash.get("energy_model_hash", pd.Series([], dtype=str)).astype(str)
        ph = runtime_hash.get("phi_hash", pd.Series([], dtype=str)).astype(str)
        mh = runtime_hash.get("m0_hash", pd.Series([], dtype=str)).astype(str)
        st = runtime_hash.get("status", pd.Series([], dtype=str)).astype(str)
        keep_h = (eh.str.strip() != "") & (eh.str.strip().str.lower() != "nan")
        keep_p = (ph.str.strip() != "") & (ph.str.strip().str.lower() != "nan")
        keep_m = (mh.str.strip() != "") & (mh.str.strip().str.lower() != "nan")
        runtime_after_training_vals = sorted(set(eh[(ev == "after_training") & keep_h].tolist()))
        runtime_before_vals = sorted(set(eh[(ev == "before_gene_eval") & keep_h].tolist()))
        runtime_after_vals = sorted(set(eh[(ev == "after_gene_calibration") & (st == "OK") & keep_h].tolist()))
        runtime_phi_before_vals = sorted(set(ph[(ev == "before_gene_eval") & keep_p].tolist()))
        runtime_phi_after_vals = sorted(set(ph[(ev == "after_gene_calibration") & (st == "OK") & keep_p].tolist()))
        runtime_m0_before_vals = sorted(set(mh[(ev == "before_gene_eval") & keep_m].tolist()))
        runtime_m0_after_vals = sorted(set(mh[(ev == "after_gene_calibration") & (st == "OK") & keep_m].tolist()))

    checks: list[dict[str, Any]] = []

    def _append(check: str, passed: bool, detail: str, value: str = "") -> None:
        checks.append(
            {
                "check": check,
                "passed": bool(passed),
                "status": "PASS" if passed else "FAIL",
                "detail": detail,
                "value": value,
            }
        )

    _append(
        "frozen_model_file_exists",
        bool(model_path.exists()),
        "frozen_model.json must exist for audit integrity.",
        str(model_path),
    )
    _append(
        "training_manifest_exists",
        bool(training_manifest_path.exists()),
        "training_manifest.json should be present to pin frozen hashes.",
        str(training_manifest_path),
    )
    _append(
        "single_model_hash_in_results",
        len(model_vals) == 1,
        "All analyzed rows should reference one immutable model hash.",
        ",".join(model_vals),
    )
    _append(
        "single_m0_hash_in_results",
        len(m0_vals) == 1,
        "All analyzed rows should reference one immutable neutral model hash.",
        ",".join(m0_vals),
    )
    _append(
        "single_phi_hash_in_results",
        len(phi_vals) <= 1,
        "Calibration feature map (phi) should be fixed across analyzed rows.",
        ",".join(phi_vals),
    )
    _append(
        "model_hash_matches_training_manifest",
        bool(expected_model_hash) and len(model_vals) == 1 and model_vals[0] == expected_model_hash,
        "Result model_hash must match training manifest model_hash.",
        f"expected={expected_model_hash};observed={','.join(model_vals)}",
    )
    _append(
        "m0_hash_matches_training_manifest",
        (not expected_m0_hash)
        or (
            len(m0_vals) == 1
            and (
                m0_vals[0] == expected_m0_hash
                or (expected_m0_hash_frozen_spec and m0_vals[0] == expected_m0_hash_frozen_spec)
            )
        ),
        "Result M0_hash should match manifest or frozen-model neutral_spec hash.",
        (
            f"expected_manifest={expected_m0_hash};"
            f"expected_frozen_spec={expected_m0_hash_frozen_spec};"
            f"observed={','.join(m0_vals)}"
        ),
    )
    _append(
        "phi_hash_matches_training_manifest",
        (not expected_phi_hash) or (len(phi_vals) == 1 and phi_vals[0] == expected_phi_hash),
        "Result phi_hash should match training manifest phi_hash when provided.",
        f"expected={expected_phi_hash};observed={','.join(phi_vals)}",
    )
    _append(
        "runtime_hash_log_exists",
        bool(runtime_hash_path.exists()),
        "Runtime frozen-energy hash log should exist.",
        str(runtime_hash_path),
    )
    _append(
        "runtime_hash_after_training_constant",
        (not expected_model_hash)
        or (len(runtime_after_training_vals) == 1 and runtime_after_training_vals[0] == expected_model_hash),
        "Hash logged immediately after training/load should match expected model hash.",
        f"expected={expected_model_hash};observed={','.join(runtime_after_training_vals)}",
    )
    _append(
        "runtime_hash_before_eval_constant",
        (not expected_model_hash) or (len(runtime_before_vals) == 1 and runtime_before_vals[0] == expected_model_hash),
        "Hash logged before each gene evaluation should stay fixed.",
        f"expected={expected_model_hash};observed={','.join(runtime_before_vals)}",
    )
    _append(
        "runtime_hash_after_calibration_constant",
        (not expected_model_hash) or (len(runtime_after_vals) == 1 and runtime_after_vals[0] == expected_model_hash),
        "Hash logged after each calibration should stay fixed.",
        f"expected={expected_model_hash};observed={','.join(runtime_after_vals)}",
    )
    _append(
        "runtime_phi_hash_constant",
        (not expected_phi_hash)
        or (
            (len(runtime_phi_before_vals) == 1 and runtime_phi_before_vals[0] == expected_phi_hash)
            and (len(runtime_phi_after_vals) == 1 and runtime_phi_after_vals[0] == expected_phi_hash)
        ),
        "Runtime phi_hash should stay fixed before/after calibration.",
        (
            f"expected={expected_phi_hash};"
            f"before={','.join(runtime_phi_before_vals)};"
            f"after={','.join(runtime_phi_after_vals)}"
        ),
    )
    _append(
        "runtime_m0_hash_constant",
        (not expected_m0_hash)
        or (
            (
                len(runtime_m0_before_vals) == 1
                and (
                    runtime_m0_before_vals[0] == expected_m0_hash
                    or (expected_m0_hash_frozen_spec and runtime_m0_before_vals[0] == expected_m0_hash_frozen_spec)
                )
            )
            and (
                len(runtime_m0_after_vals) == 1
                and (
                    runtime_m0_after_vals[0] == expected_m0_hash
                    or (expected_m0_hash_frozen_spec and runtime_m0_after_vals[0] == expected_m0_hash_frozen_spec)
                )
            )
        ),
        "Runtime M0 hash should stay fixed before/after calibration.",
        (
            f"expected={expected_m0_hash};"
            f"expected_frozen_spec={expected_m0_hash_frozen_spec};"
            f"before={','.join(runtime_m0_before_vals)};"
            f"after={','.join(runtime_m0_after_vals)}"
        ),
    )

    fail = bool(np.count_nonzero([not bool(r["passed"]) for r in checks]) > 0)
    meta = {
        "training_manifest_path": str(training_manifest_path) if training_manifest_path.exists() else None,
        "runtime_hash_path": str(runtime_hash_path) if runtime_hash_path.exists() else None,
        "training_mode": training_manifest.get("training_mode"),
        "training_samples_n": training_manifest.get("training_samples_n"),
        "L_train": training_manifest.get("L_train"),
        "leakage_policy": training_manifest.get("leakage_policy"),
        "expected_model_hash": expected_model_hash,
        "expected_m0_hash": expected_m0_hash,
        "expected_phi_hash": expected_phi_hash,
        "expected_m0_hash_frozen_spec": expected_m0_hash_frozen_spec,
        "model_hash_unique": model_vals,
        "m0_hash_unique": m0_vals,
        "phi_hash_unique": phi_vals,
        "runtime_model_hash_after_training_unique": runtime_after_training_vals,
        "runtime_model_hash_before_eval_unique": runtime_before_vals,
        "runtime_model_hash_after_calibration_unique": runtime_after_vals,
    }
    return pd.DataFrame(checks), fail, meta


def _testable_set_alignment_table(
    *,
    orth: pd.DataFrame,
    baseline_all: pd.DataFrame,
) -> tuple[pd.DataFrame, bool, pd.DataFrame, int]:
    diff_cols = ["gene_id", "in_babappa", "in_busted", "babappa_status", "busted_status", "reason"]
    if "gene_id" in orth.columns:
        orth_gene_ids = set(orth["gene_id"].astype(str).tolist())
        bab_status = orth["babappa_status"].astype(str) if "babappa_status" in orth.columns else pd.Series([""] * len(orth), index=orth.index)
        busted_status = orth["busted_status"].astype(str) if "busted_status" in orth.columns else pd.Series([""] * len(orth), index=orth.index)
        babappa_ok = set(orth.loc[bab_status == "OK", "gene_id"].astype(str).tolist())
        busted_ok = set(orth.loc[busted_status == "OK", "gene_id"].astype(str).tolist())
    else:
        orth_gene_ids = set()
        babappa_ok = set()
        busted_ok = set()

    baseline = baseline_all.copy()
    if not baseline.empty and "unit_kind" in baseline.columns:
        fg = baseline[baseline["unit_kind"].astype(str) == "full_gene"].copy()
        if not fg.empty:
            baseline = fg
    if not baseline.empty:
        baseline = baseline[baseline.get("method", pd.Series([], dtype=str)).astype(str) == "busted"].copy()
    if "gene_id" in baseline.columns:
        baseline_gene_ids = set(baseline["gene_id"].astype(str).tolist())
        status = baseline["status"].astype(str) if "status" in baseline.columns else pd.Series([""] * len(baseline), index=baseline.index)
        baseline_ok_gene_ids = set(baseline.loc[status == "OK", "gene_id"].astype(str).tolist())
    else:
        baseline_gene_ids = set()
        baseline_ok_gene_ids = set()

    only_babappa = sorted(babappa_ok - busted_ok)
    only_busted = sorted(busted_ok - babappa_ok)
    baseline_not_in_orth = sorted(baseline_gene_ids - orth_gene_ids)
    orth_not_in_baseline = sorted(orth_gene_ids - baseline_gene_ids)
    shared_ok = sorted(babappa_ok & busted_ok)
    testable_set_size = int(len(shared_ok))

    orth_bab_map: dict[str, str] = {}
    orth_bus_map: dict[str, str] = {}
    if "gene_id" in orth.columns:
        for row in orth.itertuples(index=False):
            gid = str(getattr(row, "gene_id", ""))
            orth_bab_map[gid] = str(getattr(row, "babappa_status", "SKIP_not_run"))
            orth_bus_map[gid] = str(getattr(row, "busted_status", "SKIP_not_run"))

    baseline_bus_map: dict[str, str] = {}
    if not baseline.empty and "gene_id" in baseline.columns:
        for row in baseline.itertuples(index=False):
            gid = str(getattr(row, "gene_id", ""))
            st = str(getattr(row, "status", ""))
            rs = str(getattr(row, "reason", "not_run"))
            if st == "OK":
                baseline_bus_map[gid] = "OK"
            elif st:
                baseline_bus_map[gid] = f"FAIL_{rs}"
            else:
                baseline_bus_map[gid] = "SKIP_not_run"

    diff_rows: list[dict[str, Any]] = []
    for gid in sorted(orth_gene_ids | baseline_gene_ids):
        bab = orth_bab_map.get(gid, "SKIP_not_in_ortholog")
        bus = orth_bus_map.get(gid, baseline_bus_map.get(gid, "SKIP_not_in_baseline"))
        in_bab = bool(bab == "OK")
        in_bus = bool(bus == "OK")
        reasons: list[str] = []
        if gid not in orth_gene_ids:
            reasons.append("not_in_ortholog_table")
        if gid not in baseline_gene_ids:
            reasons.append("not_in_busted_baseline_table")
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

    rows = [
        {"metric": "n_orth_rows", "value": int(len(orth_gene_ids)), "detail": ""},
        {"metric": "n_babappa_ok", "value": int(len(babappa_ok)), "detail": ""},
        {"metric": "n_busted_ok", "value": int(len(busted_ok)), "detail": ""},
        {"metric": "n_shared_ok_testable", "value": int(testable_set_size), "detail": ""},
        {"metric": "n_babappa_only_ok", "value": int(len(only_babappa)), "detail": ",".join(only_babappa[:20])},
        {"metric": "n_busted_only_ok", "value": int(len(only_busted)), "detail": ",".join(only_busted[:20])},
        {
            "metric": "n_baseline_gene_ids_not_in_ortholog_table",
            "value": int(len(baseline_not_in_orth)),
            "detail": ",".join(baseline_not_in_orth[:20]),
        },
        {
            "metric": "n_ortholog_gene_ids_not_in_baseline_table",
            "value": int(len(orth_not_in_baseline)),
            "detail": ",".join(orth_not_in_baseline[:20]),
        },
        {"metric": "n_baseline_ok_rows", "value": int(len(baseline_ok_gene_ids)), "detail": ""},
        {"metric": "n_testable_set_diff_rows", "value": int(len(diff_rows)), "detail": ""},
    ]
    mismatch = bool(len(diff_rows) > 0)
    return pd.DataFrame(rows), mismatch, pd.DataFrame(diff_rows, columns=diff_cols), testable_set_size


def _null_uniformity_table(*, null_p: np.ndarray, null_N: int) -> tuple[pd.DataFrame, bool]:
    p = np.asarray(null_p, dtype=float)
    p = p[np.isfinite(p)]
    p = p[(p >= 0.0) & (p <= 1.0)]
    n = int(p.size)
    if n <= 0:
        rows = [
            {"metric": "n_null", "value": 0.0, "detail": ""},
            {"metric": "null_N", "value": float(null_N), "detail": ""},
            {"metric": "ks_uniform_stat", "value": float("nan"), "detail": ""},
            {"metric": "ks_uniform_pvalue", "value": float("nan"), "detail": ""},
            {"metric": "ks_discrete_grid_stat", "value": float("nan"), "detail": ""},
            {"metric": "tail_mass_p_le_0.05", "value": float("nan"), "detail": ""},
            {"metric": "tail_mass_expected", "value": 0.05, "detail": ""},
            {"metric": "tail_mass_zscore", "value": float("nan"), "detail": ""},
        ]
        return pd.DataFrame(rows), False

    ks_res = stats.kstest(p, "uniform")
    support = np.arange(1, int(null_N) + 2, dtype=float) / float(int(null_N) + 1)
    p_sorted = np.sort(p)
    obs_cdf = np.searchsorted(p_sorted, support, side="right") / float(n)
    exp_cdf = np.arange(1, support.size + 1, dtype=float) / float(support.size)
    ks_disc = float(np.max(np.abs(obs_cdf - exp_cdf))) if support.size else float("nan")

    tail_mass = float(np.mean(p <= 0.05))
    tail_expected = 0.05
    tail_sd = float(np.sqrt(tail_expected * (1.0 - tail_expected) / float(max(n, 1))))
    tail_z = (tail_mass - tail_expected) / tail_sd if tail_sd > 0 else float("nan")
    severe_inflation = bool((np.isfinite(tail_z) and tail_z > 4.0 and tail_mass > tail_expected) or (np.isfinite(ks_disc) and ks_disc > 0.08))

    rows = [
        {"metric": "n_null", "value": float(n), "detail": ""},
        {"metric": "null_N", "value": float(null_N), "detail": ""},
        {"metric": "ks_uniform_stat", "value": float(ks_res.statistic), "detail": ""},
        {"metric": "ks_uniform_pvalue", "value": float(ks_res.pvalue), "detail": ""},
        {"metric": "ks_discrete_grid_stat", "value": float(ks_disc), "detail": ""},
        {"metric": "tail_mass_p_le_0.05", "value": float(tail_mass), "detail": ""},
        {"metric": "tail_mass_expected", "value": float(tail_expected), "detail": ""},
        {"metric": "tail_mass_zscore", "value": float(tail_z), "detail": ""},
    ]
    return pd.DataFrame(rows), severe_inflation


def _run_matched_null(
    *,
    model: EnergyModel,
    contexts: list[dict[str, Any]],
    outdir: Path,
    seed: int,
    null_N: int,
    null_G: int,
    jobs: int,
) -> tuple[pd.DataFrame, pd.DataFrame, bool]:
    outdir.mkdir(parents=True, exist_ok=True)
    raw_path = outdir / "raw" / "null_pvalues.tsv"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = outdir / "logs" / "null_progress.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(int(seed))
    n_ctx = len(contexts)
    if n_ctx <= 0:
        raise ValueError("No gene contexts available for matched-null run.")

    order: list[int] = []
    while len(order) < int(null_G):
        order.extend([int(x) for x in rng.permutation(n_ctx).tolist()])
    order = order[: int(null_G)]

    base_spec = model.neutral_spec
    if base_spec is None:
        raise ValueError("Frozen model has no neutral specification; cannot run phylogenetic matched-null.")
    kappa = float(base_spec.kappa)
    codon_freq = base_spec.codon_frequencies
    ctx_by_gene = {str(ctx["gene_id"]): ctx for ctx in contexts}
    unique_gene_ids = sorted(ctx_by_gene.keys())
    cal_tasks: list[tuple[str, int, str, float, dict[str, float] | None, int, int]] = []
    for gid in unique_gene_ids:
        ctx = ctx_by_gene[gid]
        cal_tasks.append(
            (
                str(gid),
                int(ctx["length_nt"]),
                str(ctx["tree_newick"]),
                kappa,
                codon_freq,
                int(null_N),
                int(rng.integers(0, 2**32 - 1)),
            )
        )

    obs_tasks: list[tuple[int, str, int, str, float, dict[str, float] | None, int]] = []
    for i, idx in enumerate(order, start=1):
        ctx = contexts[int(idx)]
        obs_tasks.append(
            (
                int(i),
                str(ctx["gene_id"]),
                int(ctx["length_nt"]),
                str(ctx["tree_newick"]),
                kappa,
                codon_freq,
                int(rng.integers(0, 2**32 - 1)),
            )
        )

    rows: list[dict[str, Any]] = []
    jobs_eff = int(jobs) if int(jobs) > 0 else int(max(1, (os.cpu_count() or 1)))
    start = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        header = {
            "started_at": time.time(),
            "null_G": int(null_G),
            "null_N": int(null_N),
            "jobs": int(jobs_eff),
            "calibration_policy": "per-gene-context null calibration reused across pooled observed null replicates",
            "n_contexts": int(len(unique_gene_ids)),
        }
        log.write(json.dumps(header) + "\n")
        log.flush()

        # Phase 1: per-context calibration nulls.
        cal_rows: list[dict[str, Any]] = []
        if jobs_eff <= 1:
            _init_null_worker(model.to_dict())
            for i, task in enumerate(cal_tasks, start=1):
                cal_rows.append(_calibration_worker(task))
                if i % 10 == 0 or i == len(cal_tasks):
                    msg = f"null_calibration_contexts done={i}/{len(cal_tasks)} elapsed_sec={time.perf_counter() - start:.1f}"
                    print(msg)
                    log.write(msg + "\n")
                    log.flush()
        else:
            try:
                with ProcessPoolExecutor(
                    max_workers=int(jobs_eff),
                    initializer=_init_null_worker,
                    initargs=(model.to_dict(),),
                ) as ex:
                    fut_map = {ex.submit(_calibration_worker, task): str(task[0]) for task in cal_tasks}
                    done = 0
                    for fut in as_completed(fut_map):
                        gid = str(fut_map[fut])
                        try:
                            row = fut.result()
                        except Exception as exc:
                            row = {
                                "gene_id": gid,
                                "L_codons": np.nan,
                                "seed_calib": np.nan,
                                "null_values": [],
                                "runtime_sec": np.nan,
                                "status": "FAIL",
                                "reason": f"null_calibration_error:{exc}",
                            }
                        cal_rows.append(row)
                        done += 1
                        if done % 10 == 0 or done == len(cal_tasks):
                            msg = f"null_calibration_contexts done={done}/{len(cal_tasks)} elapsed_sec={time.perf_counter() - start:.1f}"
                            print(msg)
                            log.write(msg + "\n")
                            log.flush()
            except PermissionError as exc:
                msg = f"null_parallel_fallback reason={exc}"
                print(msg)
                log.write(msg + "\n")
                log.flush()
                _init_null_worker(model.to_dict())
                for i, task in enumerate(cal_tasks, start=1):
                    cal_rows.append(_calibration_worker(task))
                    if i % 10 == 0 or i == len(cal_tasks):
                        pmsg = f"null_calibration_contexts done={i}/{len(cal_tasks)} elapsed_sec={time.perf_counter() - start:.1f}"
                        print(pmsg)
                        log.write(pmsg + "\n")
                        log.flush()

        null_map: dict[str, np.ndarray] = {}
        for row in cal_rows:
            if str(row.get("status", "")) != "OK":
                continue
            gid = str(row.get("gene_id", ""))
            vals = np.asarray(row.get("null_values", []), dtype=float)
            if vals.size > 0:
                null_map[gid] = vals

        # Phase 2: pooled observed null replicates.
        if jobs_eff <= 1:
            _init_null_worker(model.to_dict())
            for i, task in enumerate(obs_tasks, start=1):
                row = _observed_worker(task)
                gid = str(row["gene_id"])
                null_vals = null_map.get(gid)
                if null_vals is None or null_vals.size == 0:
                    row["status"] = "FAIL"
                    row["reason"] = "missing_context_calibration"
                    row["p_value"] = np.nan
                    row["seed_calib"] = np.nan
                else:
                    row["p_value"] = float(rank_p_value(float(row["obs_dispersion"]), null_vals, tail="right"))
                    row["seed_calib"] = int(next((x[6] for x in cal_tasks if str(x[0]) == gid), -1))
                rows.append(row)
                if i % 25 == 0 or i == len(obs_tasks):
                    msg = f"null_observed done={i}/{len(obs_tasks)} elapsed_sec={time.perf_counter() - start:.1f}"
                    print(msg)
                    log.write(msg + "\n")
                    log.flush()
        else:
            try:
                with ProcessPoolExecutor(
                    max_workers=int(jobs_eff),
                    initializer=_init_null_worker,
                    initargs=(model.to_dict(),),
                ) as ex:
                    fut_map = {ex.submit(_observed_worker, task): int(task[0]) for task in obs_tasks}
                    done = 0
                    for fut in as_completed(fut_map):
                        rep_id = int(fut_map[fut])
                        try:
                            row = fut.result()
                            gid = str(row["gene_id"])
                            null_vals = null_map.get(gid)
                            if null_vals is None or null_vals.size == 0:
                                row["status"] = "FAIL"
                                row["reason"] = "missing_context_calibration"
                                row["p_value"] = np.nan
                                row["seed_calib"] = np.nan
                            else:
                                row["p_value"] = float(rank_p_value(float(row["obs_dispersion"]), null_vals, tail="right"))
                                row["seed_calib"] = int(next((x[6] for x in cal_tasks if str(x[0]) == gid), -1))
                        except Exception as exc:
                            row = {
                                "replicate_id": rep_id,
                                "gene_id": "",
                                "L_codons": np.nan,
                                "obs_dispersion": np.nan,
                                "p_value": np.nan,
                                "seed_obs": np.nan,
                                "seed_calib": np.nan,
                                "runtime_sec": np.nan,
                                "status": "FAIL",
                                "reason": f"null_observed_error:{exc}",
                            }
                        rows.append(row)
                        done += 1
                        if done % 25 == 0 or done == len(obs_tasks):
                            msg = f"null_observed done={done}/{len(obs_tasks)} elapsed_sec={time.perf_counter() - start:.1f}"
                            print(msg)
                            log.write(msg + "\n")
                            log.flush()
            except PermissionError as exc:
                msg = f"null_parallel_fallback reason={exc}"
                print(msg)
                log.write(msg + "\n")
                log.flush()
                _init_null_worker(model.to_dict())
                for i, task in enumerate(obs_tasks, start=1):
                    row = _observed_worker(task)
                    gid = str(row["gene_id"])
                    null_vals = null_map.get(gid)
                    if null_vals is None or null_vals.size == 0:
                        row["status"] = "FAIL"
                        row["reason"] = "missing_context_calibration"
                        row["p_value"] = np.nan
                        row["seed_calib"] = np.nan
                    else:
                        row["p_value"] = float(rank_p_value(float(row["obs_dispersion"]), null_vals, tail="right"))
                        row["seed_calib"] = int(next((x[6] for x in cal_tasks if str(x[0]) == gid), -1))
                    rows.append(row)
                    if i % 25 == 0 or i == len(obs_tasks):
                        pmsg = f"null_observed done={i}/{len(obs_tasks)} elapsed_sec={time.perf_counter() - start:.1f}"
                        print(pmsg)
                        log.write(pmsg + "\n")
                        log.flush()

    null_df = pd.DataFrame(rows).sort_values("replicate_id").reset_index(drop=True)
    _write_tsv(raw_path, null_df)
    ok = null_df[null_df["status"] == "OK"].copy() if not null_df.empty and "status" in null_df.columns else pd.DataFrame()
    p_ok = pd.to_numeric(ok.get("p_value", np.nan), errors="coerce").dropna().to_numpy(dtype=float)
    n_ok = int(p_ok.size)

    size_rows: list[dict[str, Any]] = []
    for alpha in (0.1, 0.05, 0.01, 0.001):
        k = int(np.count_nonzero(p_ok <= float(alpha)))
        size_hat = (float(k) / float(n_ok)) if n_ok > 0 else float("nan")
        ci_low, ci_high = _null_ci(k, n_ok)
        size_rows.append(
            {
                "alpha": float(alpha),
                "size_hat": float(size_hat),
                "ci_low": float(ci_low),
                "ci_high": float(ci_high),
                "n_null": int(n_ok),
                "k_reject": int(k),
                "nominal_in_ci": bool(ci_low <= float(alpha) <= ci_high) if n_ok > 0 else False,
            }
        )
    size_df = pd.DataFrame(size_rows)
    _write_tsv(outdir / "tables" / "null_size.tsv", size_df)
    fail = False
    row05 = size_df[np.isclose(pd.to_numeric(size_df["alpha"], errors="coerce"), 0.05)]
    if not row05.empty:
        r = row05.iloc[0]
        fail = bool(not bool(r["nominal_in_ci"]) and float(r["size_hat"]) > 0.05)
    return null_df, size_df, fail


def run_ortholog_audit(
    *,
    pack: str | Path,
    outdir: str | Path,
    seed: int = 1,
    null_N: int = 999,
    null_G: int = 2000,
    jobs: int = 0,
) -> dict[str, Any]:
    configure_plotting_env()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    pack_dir = Path(pack).resolve()
    out = Path(outdir).resolve()
    for sub in ("raw", "tables", "figures", "report", "manifests", "logs", "scripts"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    track_manifest_path = pack_dir / "benchmark_track_manifest.json"
    if not track_manifest_path.exists():
        alt = pack_dir / "manifests" / "benchmark_track_manifest.json"
        if alt.exists():
            track_manifest_path = alt

    required = [
        pack_dir / "raw" / "babappa_results.tsv",
        pack_dir / "raw" / "baseline_all.tsv",
        pack_dir / "tables" / "ortholog_results.tsv",
        track_manifest_path,
    ]
    miss = [str(p) for p in required if not p.exists()]
    if miss:
        raise FileNotFoundError("Missing required pack files:\n" + "\n".join(miss))

    babappa_raw = _read_tsv(pack_dir / "raw" / "babappa_results.tsv")
    baseline_all = _read_tsv(pack_dir / "raw" / "baseline_all.tsv")
    orth = _read_tsv(pack_dir / "tables" / "ortholog_results.tsv")
    qc = _read_tsv(pack_dir / "raw" / "qc_by_gene.tsv")

    frozen_df, frozen_invariant_fail, frozen_meta = _frozen_energy_invariant_table(
        pack_dir=pack_dir,
        babappa_raw=babappa_raw,
    )
    _write_tsv(out / "tables" / "frozen_energy_invariant.tsv", frozen_df)

    testable_set_df, testable_set_mismatch, testable_set_diff_df, testable_set_size = _testable_set_alignment_table(
        orth=orth,
        baseline_all=baseline_all,
    )
    _write_tsv(out / "tables" / "testable_set_alignment.tsv", testable_set_df)
    _write_tsv(out / "tables" / "testable_set_diff.tsv", testable_set_diff_df)

    # FDR inputs.
    fdr_rows: list[dict[str, Any]] = []
    qdiff_rows: list[dict[str, Any]] = []
    method_specs = [
        (
            "babappa",
            "babappa_p",
            ("babappa_q_all_candidates" if "babappa_q_all_candidates" in orth.columns else "babappa_q"),
            "babappa_status",
        ),
        (
            "busted",
            "busted_p",
            ("busted_q_all_candidates" if "busted_q_all_candidates" in orth.columns else "busted_q"),
            "busted_status",
        ),
    ]
    for method, p_col, q_col, st_col in method_specs:
        df = orth.copy()
        p = pd.to_numeric(df[p_col], errors="coerce")
        q = pd.to_numeric(df[q_col], errors="coerce")
        st = df[st_col].astype(str)
        ok_mask = (st == "OK") & p.notna()
        p_ok = p[ok_mask].to_numpy(dtype=float)
        q_ok_stored = q[ok_mask].to_numpy(dtype=float)
        gene_ok = df.loc[ok_mask, "gene_id"].astype(str).to_numpy(dtype=str)
        q_ok_recalc = np.array(benjamini_hochberg([float(x) for x in p_ok.tolist()]), dtype=float) if p_ok.size else np.array([], dtype=float)
        abs_diff = np.abs(q_ok_recalc - q_ok_stored) if p_ok.size else np.array([], dtype=float)
        max_diff = float(np.max(abs_diff)) if abs_diff.size else 0.0
        n_diff = int(np.count_nonzero(abs_diff > 1e-9)) if abs_diff.size else 0
        worst_gene = str(gene_ok[int(np.argmax(abs_diff))]) if abs_diff.size else ""
        fdr_rows.append(
            {
                "method": method,
                "n_total_units": int(len(df)),
                "n_p_nonnull": int(p_ok.size),
                "n_p_missing": int(len(df) - p_ok.size),
                "n_status_fail": int(np.count_nonzero(st != "OK")),
                "bh_scope": (
                    "OK rows with non-null p over all candidates"
                    if q_col.endswith("_all_candidates")
                    else "OK rows with non-null p over ortholog_results.tsv"
                ),
            }
        )
        qdiff_rows.append(
            {
                "method": method,
                "max_abs_diff": max_diff,
                "n_diff_gt_1e-9": n_diff,
                "worst_gene_id": worst_gene,
            }
        )
    fdr_inputs_df = pd.DataFrame(fdr_rows)
    fdr_qdiff_df = pd.DataFrame(qdiff_rows)
    _write_tsv(out / "tables" / "fdr_inputs.tsv", fdr_inputs_df)
    _write_tsv(out / "tables" / "fdr_q_diff.tsv", fdr_qdiff_df)

    # Method vectors.
    b_p = pd.to_numeric(orth["babappa_p"], errors="coerce").to_numpy(dtype=float)
    h_p = pd.to_numeric(orth["busted_p"], errors="coerce").to_numpy(dtype=float)
    b_q = pd.to_numeric(orth["babappa_q"], errors="coerce").to_numpy(dtype=float)
    h_q = pd.to_numeric(orth["busted_q"], errors="coerce").to_numpy(dtype=float)

    # Histograms.
    for name, vals, color in [
        ("p_hist_babappa.pdf", b_p, "#4C78A8"),
        ("p_hist_busted.pdf", h_p, "#F58518"),
        ("q_hist_babappa.pdf", b_q, "#4C78A8"),
        ("q_hist_busted.pdf", h_q, "#F58518"),
    ]:
        fig = plt.figure(figsize=(6.5, 4.0))
        ax = fig.add_subplot(111)
        vv = vals[np.isfinite(vals)]
        if vv.size:
            ax.hist(np.clip(vv, 0.0, 1.0), bins=20, color=color, edgecolor="white")
        ax.set_xlabel("value")
        ax.set_ylabel("count")
        ax.set_title(name.replace(".pdf", "").replace("_", " "))
        fig.tight_layout()
        fig.savefig(out / "figures" / name)
        plt.close(fig)

    # QQ plots.
    for name, vals, label in [
        ("qq_babappa.pdf", b_p, "BABAPPA"),
        ("qq_busted.pdf", h_p, "BUSTED"),
    ]:
        x, y, y_lo, y_hi = _qq_data_with_envelope(vals)
        fig = plt.figure(figsize=(6.5, 5.0))
        ax = fig.add_subplot(111)
        if x.size:
            ax.fill_between(x, y_lo, y_hi, color="#DDDDDD", alpha=0.7, label="95% envelope")
            ax.plot(x, y, ".", color="#1F77B4", alpha=0.8, label=label)
            lim = max(float(np.max(x)), float(np.max(y)))
            ax.plot([0, lim], [0, lim], "r--", linewidth=1)
            ax.set_xlim(0, lim * 1.02)
            ax.set_ylim(0, lim * 1.02)
        ax.set_xlabel("Expected -log10(p)")
        ax.set_ylabel("Observed -log10(p)")
        ax.set_title(f"{label} QQ")
        ax.legend(loc="lower right", fontsize=8)
        fig.tight_layout()
        fig.savefig(out / "figures" / name)
        plt.close(fig)

    # Storey pi0.
    pi0_rows: list[dict[str, Any]] = []
    for method, vals in [("babappa", b_p), ("busted", h_p)]:
        pi0_hat, grid = _storey_pi0(vals)
        n = int(np.count_nonzero(np.isfinite(vals)))
        pi0_rows.append(
            {
                "method": method,
                "pi0_hat": float(pi0_hat),
                "nonnull_hat": float((1.0 - float(pi0_hat)) * float(n)) if np.isfinite(pi0_hat) else float("nan"),
                "lambda_grid_used": ",".join(f"{x:.2f}" for x in grid),
            }
        )
    pi0_df = pd.DataFrame(pi0_rows)
    _write_tsv(out / "tables" / "pi0.tsv", pi0_df)

    # Gene covariates.
    bmap = babappa_raw.set_index("gene_id").to_dict(orient="index")
    qc_map = qc.set_index("gene_id").to_dict(orient="index") if not qc.empty and "gene_id" in qc.columns else {}
    cov_rows: list[dict[str, Any]] = []
    for row in orth.itertuples(index=False):
        gid = str(getattr(row, "gene_id"))
        b = bmap.get(gid, {})
        aln_path = Path(str(b.get("alignment_path", "")))
        tree_candidates = [
            pack_dir / "raw" / "preprocessed_units" / "trees" / f"{gid}.nwk",
            Path(str(qc_map.get(gid, {}).get("tree_path", ""))),
        ]
        tree_text = ""
        for t in tree_candidates:
            if t and t.exists():
                try:
                    tree_text = t.read_text(encoding="utf-8")
                    break
                except Exception:
                    tree_text = ""
        tree_total = float("nan")
        if tree_text:
            try:
                tree_total = float(np.sum(np.array(parse_newick(tree_text).branch_lengths(), dtype=float)))
            except Exception:
                tree_total = float("nan")
        cov = {
            "gap_fraction": float("nan"),
            "ambiguous_fraction": float("nan"),
            "GC": float("nan"),
            "GC3": float("nan"),
            "mean_pairwise_divergence": float("nan"),
        }
        if aln_path.exists():
            try:
                cov = _alignment_covariates(read_fasta(aln_path))
            except Exception:
                pass
        cov_rows.append(
            {
                "gene_id": gid,
                "L_codons": float(getattr(row, "L_codons")),
                "n_taxa": float(getattr(row, "n_taxa")),
                "gap_fraction": float(cov["gap_fraction"]),
                "ambiguous_fraction": float(cov["ambiguous_fraction"]),
                "GC": float(cov["GC"]),
                "GC3": float(cov["GC3"]),
                "total_tree_length": float(tree_total),
                "mean_pairwise_divergence": float(cov["mean_pairwise_divergence"]),
            }
        )
    cov_df = pd.DataFrame(cov_rows)
    _write_tsv(out / "tables" / "gene_covariates.tsv", cov_df)

    merged = orth.merge(cov_df, on=["gene_id", "L_codons", "n_taxa"], how="left")
    corr_rows: list[dict[str, Any]] = []
    for method, p_col in [("babappa", "babappa_p"), ("busted", "busted_p")]:
        y = -np.log10(np.clip(pd.to_numeric(merged[p_col], errors="coerce").to_numpy(dtype=float), 1e-300, 1.0))
        for cov_col in [
            "L_codons",
            "n_taxa",
            "gap_fraction",
            "ambiguous_fraction",
            "GC",
            "GC3",
            "total_tree_length",
            "mean_pairwise_divergence",
        ]:
            x = pd.to_numeric(merged[cov_col], errors="coerce").to_numpy(dtype=float)
            rho, pval, n = _spearman(y, x)
            corr_rows.append(
                {
                    "method": method,
                    "covariate": cov_col,
                    "spearman_rho": float(rho),
                    "p_value": float(pval),
                    "n": int(n),
                }
            )
    corr_df = pd.DataFrame(corr_rows)
    _write_tsv(out / "tables" / "covariate_correlations.tsv", corr_df)

    # Required confounder plots.
    def _scatter_plot(x_col: str, y_col: str, out_name: str, title: str) -> None:
        x = pd.to_numeric(merged[x_col], errors="coerce").to_numpy(dtype=float)
        y = -np.log10(np.clip(pd.to_numeric(merged[y_col], errors="coerce").to_numpy(dtype=float), 1e-300, 1.0))
        mask = np.isfinite(x) & np.isfinite(y)
        fig = plt.figure(figsize=(6.5, 4.2))
        ax = fig.add_subplot(111)
        ax.plot(x[mask], y[mask], ".", alpha=0.6)
        ax.set_xlabel(x_col)
        ax.set_ylabel(f"-log10({y_col})")
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(out / "figures" / out_name)
        plt.close(fig)

    _scatter_plot("L_codons", "babappa_p", "babappa_vs_length.pdf", "BABAPPA vs length")
    _scatter_plot("GC3", "babappa_p", "babappa_vs_GC3.pdf", "BABAPPA vs GC3")
    _scatter_plot("gap_fraction", "babappa_p", "babappa_vs_gap.pdf", "BABAPPA vs gap fraction")
    _scatter_plot("L_codons", "busted_p", "busted_vs_length.pdf", "BUSTED vs length")

    # Matched-null contexts.
    model = load_model(pack_dir / "raw" / "frozen_model.json")
    ctx_rows: list[dict[str, Any]] = []
    for row in orth.itertuples(index=False):
        gid = str(getattr(row, "gene_id"))
        b = bmap.get(gid, {})
        length_nt = int(round(float(b.get("L", float(getattr(row, "L_codons")) * 3.0))))
        tree_path = pack_dir / "raw" / "preprocessed_units" / "trees" / f"{gid}.nwk"
        if not tree_path.exists():
            t2 = Path(str(qc_map.get(gid, {}).get("tree_path", "")))
            if t2.exists():
                tree_path = t2
        if not tree_path.exists():
            continue
        tree_newick = tree_path.read_text(encoding="utf-8")
        ctx_rows.append({"gene_id": gid, "length_nt": int(length_nt), "tree_newick": tree_newick})
    if not ctx_rows:
        raise ValueError("No tree/length contexts available for matched-null.")

    null_df, null_size_df, calibration_fail = _run_matched_null(
        model=model,
        contexts=ctx_rows,
        outdir=out,
        seed=int(seed),
        null_N=int(null_N),
        null_G=int(null_G),
        jobs=int(jobs),
    )

    # Null plots.
    null_p = pd.to_numeric(null_df.loc[null_df["status"] == "OK", "p_value"], errors="coerce").dropna().to_numpy(dtype=float)
    null_uniformity_df, severe_inflation = _null_uniformity_table(
        null_p=null_p,
        null_N=int(null_N),
    )
    _write_tsv(out / "tables" / "null_uniformity.tsv", null_uniformity_df)
    fig = plt.figure(figsize=(6.5, 4.0))
    ax = fig.add_subplot(111)
    if null_p.size:
        ax.hist(np.clip(null_p, 0.0, 1.0), bins=20, color="#4C78A8", edgecolor="white")
    ax.set_xlabel("null p")
    ax.set_ylabel("count")
    ax.set_title("Matched-null p-value histogram (BABAPPA)")
    fig.tight_layout()
    fig.savefig(out / "figures" / "null_p_hist_babappa.pdf")
    plt.close(fig)

    x, y, y_lo, y_hi = _qq_data_with_envelope(null_p)
    fig = plt.figure(figsize=(6.5, 5.0))
    ax = fig.add_subplot(111)
    if x.size:
        ax.fill_between(x, y_lo, y_hi, color="#DDDDDD", alpha=0.7, label="95% envelope")
        ax.plot(x, y, ".", color="#1F77B4", alpha=0.8)
        lim = max(float(np.max(x)), float(np.max(y)))
        ax.plot([0, lim], [0, lim], "r--", linewidth=1)
        ax.set_xlim(0, lim * 1.02)
        ax.set_ylim(0, lim * 1.02)
    ax.set_xlabel("Expected -log10(p)")
    ax.set_ylabel("Observed -log10(p)")
    ax.set_title("Matched-null QQ (BABAPPA)")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "figures" / "null_qq_babappa.pdf")
    plt.close(fig)

    fig = plt.figure(figsize=(6.5, 4.0))
    ax = fig.add_subplot(111)
    ax.axis("off")
    cell_text = [
        [
            f"{float(r.alpha):.3f}",
            f"{float(r.size_hat):.6f}",
            f"[{float(r.ci_low):.6f}, {float(r.ci_high):.6f}]",
            f"{int(r.n_null)}",
        ]
        for r in null_size_df.itertuples(index=False)
    ]
    table = ax.table(
        cellText=cell_text,
        colLabels=["alpha", "size_hat", "95% CI", "n_null"],
        loc="center",
    )
    table.scale(1.0, 1.5)
    ax.set_title("Matched-null empirical size")
    fig.tight_layout()
    fig.savefig(out / "figures" / "size_table_visual.pdf")
    plt.close(fig)

    strong_flags = corr_df[
        (corr_df["method"] == "babappa")
        & (corr_df["covariate"].isin(["L_codons", "GC3"]))
        & (pd.to_numeric(corr_df["spearman_rho"], errors="coerce").abs() > 0.5)
    ].copy()

    # Report.
    report_path = out / "report" / "audit_report.pdf"
    with PdfPages(report_path) as pdf:
        fig = plt.figure(figsize=(8.3, 11.7))
        ax = fig.add_subplot(111)
        ax.axis("off")
        ax.text(0.5, 0.97, "ORTHOLOG AUDIT REPORT", ha="center", va="top", fontsize=15, fontweight="bold")
        y = 0.92
        lines = [
            f"pack={pack_dir}",
            f"outdir={out}",
            f"n_units={len(orth)}",
            f"n_success_babappa={int(np.count_nonzero(orth.get('babappa_status', pd.Series([], dtype=str)).astype(str) == 'OK'))}",
            f"n_success_busted={int(np.count_nonzero(orth.get('busted_status', pd.Series([], dtype=str)).astype(str) == 'OK'))}",
            f"testable_set_size={int(testable_set_size)}",
            f"null_N={int(null_N)}",
            f"null_G={int(null_G)}",
            f"fdr_diff_max_babappa={float(fdr_qdiff_df.loc[fdr_qdiff_df['method']=='babappa','max_abs_diff'].max()):.3e}",
            f"fdr_diff_max_busted={float(fdr_qdiff_df.loc[fdr_qdiff_df['method']=='busted','max_abs_diff'].max()):.3e}",
            f"frozen_energy_invariant_fail={bool(frozen_invariant_fail)}",
            f"testable_set_mismatch={bool(testable_set_mismatch)}",
            f"severe_inflation={bool(severe_inflation)}",
        ]
        for line in lines:
            ax.text(0.04, y, line, ha="left", va="top", fontsize=9)
            y -= 0.03
        if calibration_fail or severe_inflation:
            ax.text(
                0.04,
                y - 0.01,
                "CALIBRATION FAIL",
                ha="left",
                va="top",
                fontsize=16,
                color="red",
                fontweight="bold",
            )
            y -= 0.06
            for line in [
                "matched-null diagnostics indicate inflation risk.",
                "Corrective actions:",
                "- revise neutral model M0 (pi/kappa/branch lengths)",
                "- stratified calibration by L_codons bins",
                "- revise phi to reduce compositional sensitivity",
                "- increase training n and/or tighten L/n guardrail",
            ]:
                ax.text(0.04, y, line, ha="left", va="top", fontsize=9, color="red")
                y -= 0.028
        else:
            ax.text(0.04, y - 0.01, "CALIBRATION PASS", ha="left", va="top", fontsize=14, color="green", fontweight="bold")
            y -= 0.04
        if not strong_flags.empty:
            ax.text(0.04, y - 0.01, "FLAG: |rho|>0.5 for BABAPPA vs L_codons/GC3", ha="left", va="top", fontsize=11, color="red")
            y -= 0.035
            for r in strong_flags.itertuples(index=False):
                ax.text(0.04, y, f"{r.covariate}: rho={float(r.spearman_rho):.3f}, p={float(r.p_value):.3e}", ha="left", va="top", fontsize=9, color="red")
                y -= 0.028
        pdf.savefig(fig)
        plt.close(fig)

    # Provenance copy/read.
    prov_src = pack_dir / "manifests" / "provenance_freeze.json"
    prov_payload = {}
    if prov_src.exists():
        prov_payload = json.loads(prov_src.read_text(encoding="utf-8"))
        (out / "manifests" / "provenance_freeze.json").write_text(
            json.dumps(prov_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    # Manifest + rebuild + checksums.
    core_files = [
        track_manifest_path,
        pack_dir / "checksums.txt",
        pack_dir / "report" / "report.pdf",
        pack_dir / "raw" / "babappa_results.tsv",
        pack_dir / "raw" / "baseline_all.tsv",
        pack_dir / "tables" / "ortholog_results.tsv",
    ]
    core_missing = [str(p) for p in core_files if not p.exists()]
    size_hat_005 = (
        float(null_size_df.loc[np.isclose(pd.to_numeric(null_size_df["alpha"], errors="coerce"), 0.05), "size_hat"].iloc[0])
        if not null_size_df.empty
        else float("nan")
    )
    ks_uniform_pvalue = float("nan")
    tail_mass_zscore = float("nan")
    if not null_uniformity_df.empty:
        ks_row = null_uniformity_df[null_uniformity_df["metric"].astype(str) == "ks_uniform_pvalue"]
        tail_row = null_uniformity_df[null_uniformity_df["metric"].astype(str) == "tail_mass_zscore"]
        if not ks_row.empty:
            ks_uniform_pvalue = float(pd.to_numeric(ks_row.iloc[0]["value"], errors="coerce"))
        if not tail_row.empty:
            tail_mass_zscore = float(pd.to_numeric(tail_row.iloc[0]["value"], errors="coerce"))

    n_success_babappa = int(np.count_nonzero(orth.get("babappa_status", pd.Series([], dtype=str)).astype(str) == "OK"))
    n_success_busted = int(np.count_nonzero(orth.get("busted_status", pd.Series([], dtype=str)).astype(str) == "OK"))
    manifest = {
        "schema_version": 1,
        "command": "audit_ortholog",
        "pack": str(pack_dir),
        "outdir": str(out),
        "seed": int(seed),
        "null_N": int(null_N),
        "null_G": int(null_G),
        "jobs": int(jobs if int(jobs) > 0 else max(1, (os.cpu_count() or 1))),
        "pack_core_ok": bool(len(core_missing) == 0),
        "pack_core_missing": core_missing,
        "n_units": int(len(orth)),
        "n_success_babappa": int(n_success_babappa),
        "n_success_busted": int(n_success_busted),
        "testable_set_size": int(testable_set_size),
        "fdr_q_diff_max": {
            "babappa": float(fdr_qdiff_df.loc[fdr_qdiff_df["method"] == "babappa", "max_abs_diff"].max()),
            "busted": float(fdr_qdiff_df.loc[fdr_qdiff_df["method"] == "busted", "max_abs_diff"].max()),
        },
        "pi0": {str(r.method): float(r.pi0_hat) for r in pi0_df.itertuples(index=False)},
        "null_size_alpha_0_05": float(size_hat_005),
        "size_hat@0.05": float(size_hat_005),
        "ks_uniform_pvalue": float(ks_uniform_pvalue),
        "tail_mass_zscore": float(tail_mass_zscore),
        "frozen_energy_invariant_fail": bool(frozen_invariant_fail),
        "testable_set_mismatch": bool(testable_set_mismatch),
        "severe_inflation": bool(severe_inflation),
        "calibration_fail": bool(calibration_fail),
        "reproducibility": {
            "phi_hash_unique": frozen_meta.get("phi_hash_unique", []),
            "model_hash_unique": frozen_meta.get("model_hash_unique", []),
            "m0_hash_unique": frozen_meta.get("m0_hash_unique", []),
            "training_mode": frozen_meta.get("training_mode"),
            "training_samples_n": frozen_meta.get("training_samples_n"),
            "L_train": frozen_meta.get("L_train"),
            "leakage_policy": frozen_meta.get("leakage_policy"),
            "training_manifest_path": frozen_meta.get("training_manifest_path"),
            "runtime_hash_path": frozen_meta.get("runtime_hash_path"),
            "runtime_model_hash_after_training_unique": frozen_meta.get("runtime_model_hash_after_training_unique", []),
            "runtime_model_hash_before_eval_unique": frozen_meta.get("runtime_model_hash_before_eval_unique", []),
            "runtime_model_hash_after_calibration_unique": frozen_meta.get(
                "runtime_model_hash_after_calibration_unique", []
            ),
            "N_unique": sorted(
                set(
                    int(x)
                    for x in pd.to_numeric(babappa_raw.get("N", pd.Series([], dtype=float)), errors="coerce")
                    .dropna()
                    .astype(int)
                    .tolist()
                )
            ),
            "n_used_unique": sorted(
                set(
                    int(x)
                    for x in pd.to_numeric(babappa_raw.get("n_used", pd.Series([], dtype=float)), errors="coerce")
                    .dropna()
                    .astype(int)
                    .tolist()
                )
            ),
            "seed_gene_min": (
                int(
                    np.min(
                        pd.to_numeric(babappa_raw.get("seed_gene", pd.Series([], dtype=float)), errors="coerce")
                        .dropna()
                        .to_numpy(dtype=float)
                    )
                )
                if np.count_nonzero(
                    np.isfinite(
                        pd.to_numeric(babappa_raw.get("seed_gene", pd.Series([], dtype=float)), errors="coerce")
                        .to_numpy(dtype=float)
                    )
                )
                else None
            ),
            "seed_gene_max": (
                int(
                    np.max(
                        pd.to_numeric(babappa_raw.get("seed_gene", pd.Series([], dtype=float)), errors="coerce")
                        .dropna()
                        .to_numpy(dtype=float)
                    )
                )
                if np.count_nonzero(
                    np.isfinite(
                        pd.to_numeric(babappa_raw.get("seed_gene", pd.Series([], dtype=float)), errors="coerce")
                        .to_numpy(dtype=float)
                    )
                )
                else None
            ),
            "seed_calib_base_min": (
                int(
                    np.min(
                        pd.to_numeric(babappa_raw.get("seed_calib_base", pd.Series([], dtype=float)), errors="coerce")
                        .dropna()
                        .to_numpy(dtype=float)
                    )
                )
                if np.count_nonzero(
                    np.isfinite(
                        pd.to_numeric(babappa_raw.get("seed_calib_base", pd.Series([], dtype=float)), errors="coerce")
                        .to_numpy(dtype=float)
                    )
                )
                else None
            ),
            "seed_calib_base_max": (
                int(
                    np.max(
                        pd.to_numeric(babappa_raw.get("seed_calib_base", pd.Series([], dtype=float)), errors="coerce")
                        .dropna()
                        .to_numpy(dtype=float)
                    )
                )
                if np.count_nonzero(
                    np.isfinite(
                        pd.to_numeric(babappa_raw.get("seed_calib_base", pd.Series([], dtype=float)), errors="coerce")
                        .to_numpy(dtype=float)
                    )
                )
                else None
            ),
        },
        "tables": {
            "fdr_inputs": str((out / "tables" / "fdr_inputs.tsv").resolve()),
            "fdr_q_diff": str((out / "tables" / "fdr_q_diff.tsv").resolve()),
            "pi0": str((out / "tables" / "pi0.tsv").resolve()),
            "gene_covariates": str((out / "tables" / "gene_covariates.tsv").resolve()),
            "covariate_correlations": str((out / "tables" / "covariate_correlations.tsv").resolve()),
            "null_size": str((out / "tables" / "null_size.tsv").resolve()),
            "null_uniformity": str((out / "tables" / "null_uniformity.tsv").resolve()),
            "frozen_energy_invariant": str((out / "tables" / "frozen_energy_invariant.tsv").resolve()),
            "testable_set_alignment": str((out / "tables" / "testable_set_alignment.tsv").resolve()),
            "testable_set_diff": str((out / "tables" / "testable_set_diff.tsv").resolve()),
        },
        "report_pdf": str(report_path.resolve()),
    }
    (out / "manifests" / "audit_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    _write_rebuild_script(pack_dir, out, seed=int(seed), null_N=int(null_N), null_G=int(null_G), jobs=int(jobs))
    _write_checksums(out)
    return manifest
