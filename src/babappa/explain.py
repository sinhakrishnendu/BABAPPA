from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ._plotting_env import configure_plotting_env

configure_plotting_env()

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .engine import calibrate_dispersion, compute_dispersion
from .io import Alignment
from .model import EnergyModel
from .neutral import GY94NeutralSimulator
from .representation import alignment_to_matrix


def _window_means(values: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    if values.size == 0:
        return np.array([], dtype=float), np.array([], dtype=float)
    w = max(int(window), 1)
    xs: list[float] = []
    ys: list[float] = []
    for start in range(0, values.size, w):
        end = min(start + w, values.size)
        xs.append(float((start + end) / 2.0))
        ys.append(float(np.mean(values[start:end])))
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


def _null_site_energies(
    *,
    model: EnergyModel,
    gene_length: int,
    n_reps: int,
    seed: int | None,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    energies: list[np.ndarray] = []
    if model.neutral_spec is not None:
        sim = GY94NeutralSimulator(model.neutral_spec)
        for _ in range(n_reps):
            s = int(rng.integers(0, 2**32 - 1))
            aln = sim.simulate_alignment(length_nt=gene_length, seed=s)
            feat = alignment_to_matrix(aln)
            energies.append(model.energy(feat))
    else:
        for _ in range(n_reps):
            feat = rng.multivariate_normal(
                mean=model.mean,
                cov=model.covariance,
                size=gene_length,
                check_valid="ignore",
            )
            energies.append(model.energy(feat))
    if not energies:
        return np.array([], dtype=float)
    out = np.concatenate(energies, axis=0)
    if out.size > 50000:
        idx = rng.choice(out.size, size=50000, replace=False)
        out = out[idx]
    return out


def explain_alignment(
    *,
    model: EnergyModel,
    alignment: Alignment,
    calibration_size: int,
    tail: str,
    seed: int | None,
    out_pdf: str | Path,
    out_json: str | Path | None = None,
    window_size: int = 30,
) -> dict[str, Any]:
    features = alignment_to_matrix(alignment)
    dispersion, energies = compute_dispersion(features, model)
    p, mu0, sigma0, null_disp, used_mode, used_tail = calibrate_dispersion(
        observed_dispersion=dispersion,
        gene_length=alignment.length,
        model=model,
        calibration_size=calibration_size,
        seed=seed,
        calibration_mode="auto",
        tail=tail,
    )
    null_energies = _null_site_energies(
        model=model,
        gene_length=alignment.length,
        n_reps=min(max(calibration_size // 10, 10), 100),
        seed=seed,
    )

    centered = energies - float(np.mean(energies))
    contrib = centered * centered
    order = np.argsort(contrib)[::-1]
    contrib_sorted = contrib[order]
    csum = np.cumsum(contrib_sorted) / max(float(np.sum(contrib_sorted)), 1e-12)

    wx, wy = _window_means(energies, window=max(window_size, 1))

    payload = {
        "gene_length": int(alignment.length),
        "dispersion": float(dispersion),
        "mu0_hat": float(mu0),
        "sigma0_hat": float(sigma0),
        "identifiability_index": float(0.0 if sigma0 == 0 else (dispersion - mu0) / sigma0),
        "p_value": float(p),
        "tail": used_tail,
        "calibration_mode": used_mode,
        "calibration_size": int(calibration_size),
        "n_sites": int(features.shape[0]),
        "top_site_indices_by_contribution_1_based": [int(i + 1) for i in order[:20]],
    }

    out_pdf_p = Path(out_pdf)
    out_pdf_p.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    axes[0, 0].hist(energies, bins=40, alpha=0.6, label="Observed site energies")
    if null_energies.size:
        axes[0, 0].hist(null_energies, bins=40, alpha=0.5, label="Null site energies")
    axes[0, 0].set_title("Site Energy Histogram")
    axes[0, 0].legend()

    axes[0, 1].hist(null_disp, bins=40, alpha=0.7, color="#4C78A8")
    axes[0, 1].axvline(dispersion, color="red", linestyle="--", linewidth=1.5, label="Observed D")
    axes[0, 1].set_title("Dispersion vs Null")
    axes[0, 1].set_xlabel("D")
    axes[0, 1].legend()

    axes[1, 0].plot(np.arange(1, contrib_sorted.size + 1), csum, linewidth=1.3)
    axes[1, 0].set_ylim(0, 1.0)
    axes[1, 0].set_title("Cumulative Dispersion Contribution")
    axes[1, 0].set_xlabel("Ranked sites")
    axes[1, 0].set_ylabel("Cumulative fraction")

    if wx.size:
        axes[1, 1].plot(wx, wy, marker="o", linewidth=1.2)
    axes[1, 1].set_title(f"Windowed Mean Energy (window={max(window_size,1)})")
    axes[1, 1].set_xlabel("Site index (center)")
    axes[1, 1].set_ylabel("Mean energy")

    fig.suptitle(
        (
            "BABAPPA Explain Report\n"
            f"L={alignment.length} | D={dispersion:.6g} | mu0={mu0:.6g} | "
            f"sigma0={sigma0:.6g} | p={p:.6g} ({used_tail})"
        ),
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_pdf_p)
    plt.close(fig)

    if out_json is not None:
        out_json_p = Path(out_json)
        out_json_p.parent.mkdir(parents=True, exist_ok=True)
        out_json_p.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload
