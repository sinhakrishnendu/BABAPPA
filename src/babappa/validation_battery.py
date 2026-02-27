from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import time
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from ._plotting_env import configure_plotting_env
from .codon import SENSE_CODONS
from .engine import benjamini_hochberg, compute_dispersion
from .hash_utils import sha256_json
from .io import Alignment, read_fasta
from .model import EnergyModel, load_model
from .neutral import GY94NeutralSimulator, NeutralSpec
from .phylo import TreeNode, parse_newick
from .representation import FEATURE_NAMES, alignment_to_matrix
from .stats import rank_p_value


DEFAULT_CONFIG_NAME = "config.yaml"


@dataclass(frozen=True)
class M0Setting:
    name: str
    omega_mode: str
    omega_value: float
    rate_heterogeneity: str
    gamma_k: int
    gamma_alpha: float
    codon_freqs: str
    mixture_weight: float = 0.8
    mixture_omega_purifying: float = 0.1
    mixture_omega_neutral: float = 1.0


@dataclass(frozen=True)
class FrozenContext:
    model: EnergyModel
    model_hash: str
    phi_hash: str
    m0_hash: str
    global_seed: int


@dataclass(frozen=True)
class EvalResult:
    gene_id: str
    L_codons: int
    n_taxa: int
    D_obs: float
    mu_null: float
    sigma_null: float
    I: float
    p_babappa: float
    q_babappa: float
    gap_fraction: float
    ambiguous_fraction: float
    runtime_babappa: float
    energy_model_hash: str
    phi_hash: str
    M0_hash: str
    global_seed: int
    per_gene_seed: int
    per_calibration_seed: int
    notes: str


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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


def _load_yaml_or_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    text = path.read_text(encoding="utf-8")
    try:
        payload = json.loads(text)
        if isinstance(payload, dict):
            return payload
    except Exception:
        pass
    try:
        import yaml  # type: ignore

        payload = yaml.safe_load(text)
    except Exception as exc:  # pragma: no cover
        raise ValueError(f"Could not parse config/grid file: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Config/grid must be a mapping object: {path}")
    return payload


def _load_model_from_config(config: dict[str, Any]) -> EnergyModel:
    model_key = str(config.get("model") or config.get("frozen_model") or "").strip()
    if not model_key:
        raise ValueError("Config must provide 'model' (path to frozen model JSON).")
    model_path = Path(model_key).expanduser().resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Frozen model not found: {model_path}")
    return load_model(model_path)


def _load_m0_from_config(model: EnergyModel, config: dict[str, Any]) -> NeutralSpec:
    base = model.neutral_spec
    m0_cfg = config.get("m0") if isinstance(config.get("m0"), dict) else {}
    tree_newick = ""
    if isinstance(m0_cfg, dict):
        tree_newick = str(m0_cfg.get("tree_newick") or "").strip()
    if not tree_newick and isinstance(m0_cfg, dict):
        tree_file = str(m0_cfg.get("tree_file") or "").strip()
        if tree_file:
            tree_newick = Path(tree_file).expanduser().read_text(encoding="utf-8").strip()
    if not tree_newick and base is not None:
        tree_newick = str(base.tree_newick)
    if not tree_newick:
        raise ValueError("M0 requires a tree (model.neutral_spec.tree_newick or config.m0.tree_newick/tree_file).")

    def _num(key: str, fallback: float) -> float:
        if isinstance(m0_cfg, dict) and key in m0_cfg:
            return float(m0_cfg[key])
        return fallback

    kappa = _num("kappa", float(base.kappa) if base is not None else 2.0)
    omega = _num("omega", float(base.omega) if base is not None else 1.0)
    codon_frequencies: dict[str, float] | None = None
    if isinstance(m0_cfg, dict) and m0_cfg.get("codon_frequencies") is not None:
        raw = m0_cfg.get("codon_frequencies")
        if isinstance(raw, dict):
            codon_frequencies = {str(k).upper(): float(v) for k, v in raw.items()}
    elif base is not None:
        codon_frequencies = base.codon_frequencies
    return NeutralSpec(
        tree_newick=tree_newick,
        kappa=float(kappa),
        omega=float(omega),
        codon_frequencies=codon_frequencies,
        simulator="gy94",
    )


def _build_frozen_context(config: dict[str, Any]) -> FrozenContext:
    model = _load_model_from_config(config)
    m0 = _load_m0_from_config(model, config)
    phi_hash = sha256_json(
        {
            "schema_version": 1,
            "name": "phi_default_v1",
            "feature_names": list(FEATURE_NAMES),
        }
    )
    return FrozenContext(
        model=model,
        model_hash=sha256_json(model.to_dict()),
        phi_hash=phi_hash,
        m0_hash=sha256_json(m0.to_dict()),
        global_seed=int(config.get("global_seed", config.get("seed", 1))),
    )


def _normalize_length_nt(length_nt: int) -> int:
    l = int(length_nt)
    if l < 3:
        return 3
    return l - (l % 3)


def _collect_alignment_paths(gene_set: Path) -> list[Path]:
    if gene_set.is_file():
        return [gene_set]
    exts = ("*.fa", "*.fasta", "*.fna", "*.fas")
    paths: list[Path] = []
    for ext in exts:
        paths.extend(gene_set.glob(ext))
    uniq = sorted(set(p.resolve() for p in paths))
    if not uniq:
        raise ValueError(f"No FASTA alignments found in gene-set: {gene_set}")
    return uniq


def _load_gene_set(gene_set: str | Path) -> list[tuple[str, Alignment, Path]]:
    root = Path(gene_set).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"gene-set path does not exist: {root}")
    rows: list[tuple[str, Alignment, Path]] = []
    for path in _collect_alignment_paths(root):
        aln = read_fasta(path)
        gid = path.stem
        rows.append((gid, aln, path))
    rows.sort(key=lambda x: x[0])
    return rows


def _gap_ambiguous_fraction(aln: Alignment) -> tuple[float, float]:
    total = aln.length * aln.n_sequences
    if total <= 0:
        return float("nan"), float("nan")
    gap = 0
    amb = 0
    for seq in aln.sequences:
        s = seq.upper()
        gap += sum(ch == "-" for ch in s)
        amb += sum(ch not in {"A", "C", "G", "T", "-"} for ch in s)
    return float(gap) / float(total), float(amb) / float(total)


def _empirical_codon_freqs(aln: Alignment) -> dict[str, float] | None:
    pseudo = 1e-6
    counts = {c: pseudo for c in SENSE_CODONS}
    total = float(pseudo * len(SENSE_CODONS))
    for seq in aln.sequences:
        s = seq.upper().replace("U", "T")
        usable = len(s) - (len(s) % 3)
        for i in range(0, usable, 3):
            codon = s[i : i + 3]
            if codon in counts:
                counts[codon] += 1.0
                total += 1.0
    if total <= 0:
        return None
    return {c: counts[c] / total for c in SENSE_CODONS}


def _f3x4_codon_freqs(aln: Alignment) -> dict[str, float] | None:
    pos_counts = [{"A": 0.0, "C": 0.0, "G": 0.0, "T": 0.0} for _ in range(3)]
    for seq in aln.sequences:
        s = seq.upper().replace("U", "T")
        usable = len(s) - (len(s) % 3)
        for i in range(0, usable, 3):
            codon = s[i : i + 3]
            if codon not in SENSE_CODONS:
                continue
            for j, nt in enumerate(codon):
                pos_counts[j][nt] += 1.0
    pos_freqs: list[dict[str, float]] = []
    for d in pos_counts:
        tot = sum(d.values())
        if tot <= 0:
            return None
        pos_freqs.append({k: float(v) / float(tot) for k, v in d.items()})

    pseudo = 1e-6
    raw = {}
    tot = 0.0
    for codon in SENSE_CODONS:
        p = pos_freqs[0][codon[0]] * pos_freqs[1][codon[1]] * pos_freqs[2][codon[2]]
        p += pseudo
        raw[codon] = float(p)
        tot += float(p)
    if tot <= 0:
        return None
    return {c: raw[c] / tot for c in SENSE_CODONS}


def _scaled_tree(node: TreeNode, factor: float) -> TreeNode:
    return TreeNode(
        name=node.name,
        length=float(node.length) * float(factor),
        children=[_scaled_tree(c, factor) for c in node.children],
    )


def _tree_to_newick(node: TreeNode, *, is_root: bool = False) -> str:
    if node.children:
        left = "(" + ",".join(_tree_to_newick(c, is_root=False) for c in node.children) + ")"
        label = node.name or ""
        core = left + label
    else:
        if not node.name:
            raise ValueError("Leaf node is missing name while serializing tree.")
        core = node.name
    if not is_root:
        return f"{core}:{node.length:.12f}"
    if node.length > 0:
        return f"{core}:{node.length:.12f}"
    return core


def _scale_newick_lengths(tree_newick: str, factor: float) -> str:
    tree = parse_newick(tree_newick)
    scaled = _scaled_tree(tree, factor)
    return _tree_to_newick(scaled, is_root=True) + ";"


def _omega_components(setting: M0Setting) -> list[tuple[float, float]]:
    mode = setting.omega_mode
    if mode == "fixed_1":
        return [(1.0, 1.0)]
    if mode == "fixed_omega_lt1":
        return [(float(setting.omega_value), 1.0)]
    if mode == "mixture_purifying":
        w = float(setting.mixture_weight)
        w = min(max(w, 0.0), 1.0)
        return [
            (float(setting.mixture_omega_purifying), w),
            (float(setting.mixture_omega_neutral), 1.0 - w),
        ]
    raise ValueError(f"Unsupported omega_mode: {mode}")


def _rate_components(setting: M0Setting) -> list[tuple[float, float]]:
    if setting.rate_heterogeneity == "none":
        return [(1.0, 1.0)]
    if setting.rate_heterogeneity != "gamma":
        raise ValueError(f"Unsupported rate_heterogeneity: {setting.rate_heterogeneity}")
    k = max(1, int(setting.gamma_k))
    alpha = float(setting.gamma_alpha)
    if alpha <= 0:
        raise ValueError("gamma alpha must be > 0")
    rates: list[float] = []
    for i in range(1, k + 1):
        q = (i - 0.5) / float(k)
        r = float(stats.gamma.ppf(q, a=alpha, scale=1.0 / alpha))
        if not np.isfinite(r) or r <= 0:
            r = 1.0
        rates.append(r)
    w = 1.0 / float(k)
    return [(r, w) for r in rates]


def _allocate_codons(total_codons: int, weights: list[float]) -> list[int]:
    if total_codons <= 0:
        return [0 for _ in weights]
    ww = np.array(weights, dtype=float)
    ww = np.clip(ww, 0.0, np.inf)
    if float(np.sum(ww)) <= 0:
        ww = np.ones_like(ww)
    ww = ww / float(np.sum(ww))
    raw = ww * float(total_codons)
    base = np.floor(raw).astype(int)
    rem = int(total_codons - int(np.sum(base)))
    if rem > 0:
        frac_order = np.argsort(-(raw - base))
        for idx in frac_order[:rem]:
            base[int(idx)] += 1
    return [int(x) for x in base.tolist()]


def _concat_alignments(segments: list[Alignment]) -> Alignment:
    if not segments:
        raise ValueError("No segments to concatenate")
    names = segments[0].names
    seqs = ["" for _ in names]
    for seg in segments:
        if seg.names != names:
            raise ValueError("Alignment segment taxa mismatch")
        for i, s in enumerate(seg.sequences):
            seqs[i] += s
    return Alignment(names=names, sequences=tuple(seqs))


def _codon_freq_mode(setting: M0Setting, aln: Alignment) -> dict[str, float] | None:
    mode = str(setting.codon_freqs).strip().lower()
    if mode in {"uniform", "equal"}:
        return None
    if mode in {"empirical_from_alignment", "empirical"}:
        return _empirical_codon_freqs(aln)
    if mode == "f3x4":
        return _f3x4_codon_freqs(aln)
    raise ValueError(f"Unsupported codon_freqs mode: {setting.codon_freqs}")


def _simulate_under_setting(
    *,
    base_m0: NeutralSpec,
    length_nt: int,
    seed: int,
    setting: M0Setting,
    template_alignment: Alignment | None,
) -> Alignment:
    length_nt = _normalize_length_nt(length_nt)
    total_codons = max(1, length_nt // 3)

    omega_comp = _omega_components(setting)
    rate_comp = _rate_components(setting)
    combos: list[tuple[float, float, float]] = []
    for (omega, w_om), (rate, w_rt) in product(omega_comp, rate_comp):
        weight = float(w_om) * float(w_rt)
        if weight <= 0:
            continue
        combos.append((float(omega), float(rate), float(weight)))
    if not combos:
        combos = [(1.0, 1.0, 1.0)]

    counts = _allocate_codons(total_codons, [c[2] for c in combos])
    rng = np.random.default_rng(int(seed))

    codon_freq = None
    if template_alignment is not None:
        codon_freq = _codon_freq_mode(setting, template_alignment)

    segments: list[Alignment] = []
    for idx, ((omega, rate, _w), n_codons) in enumerate(zip(combos, counts, strict=False)):
        if n_codons <= 0:
            continue
        tree_scaled = _scale_newick_lengths(base_m0.tree_newick, float(rate))
        spec = NeutralSpec(
            tree_newick=tree_scaled,
            kappa=float(base_m0.kappa),
            omega=float(omega),
            codon_frequencies=(codon_freq if codon_freq is not None else base_m0.codon_frequencies),
            simulator="gy94",
        )
        sim = GY94NeutralSimulator(spec)
        seg_seed = int(rng.integers(0, 2**32 - 1))
        seg = sim.simulate_alignment(length_nt=int(n_codons) * 3, seed=seg_seed)
        segments.append(seg)

    if not segments:
        spec = NeutralSpec(
            tree_newick=base_m0.tree_newick,
            kappa=float(base_m0.kappa),
            omega=float(base_m0.omega),
            codon_frequencies=base_m0.codon_frequencies,
            simulator="gy94",
        )
        return GY94NeutralSimulator(spec).simulate_alignment(length_nt=length_nt, seed=int(seed))
    return _concat_alignments(segments)


def _evaluate_alignment(
    *,
    gene_id: str,
    alignment: Alignment,
    frozen: FrozenContext,
    base_m0: NeutralSpec,
    N: int,
    seed_gene: int,
    seed_calibration: int,
    setting: M0Setting,
) -> EvalResult:
    t0 = time.perf_counter()
    features = alignment_to_matrix(alignment)
    d_obs, _ = compute_dispersion(features, frozen.model)

    rng = np.random.default_rng(int(seed_calibration))
    null_vals = np.empty(int(N), dtype=float)
    for j in range(int(N)):
        s = int(rng.integers(0, 2**32 - 1))
        sim_aln = _simulate_under_setting(
            base_m0=base_m0,
            length_nt=int(alignment.length),
            seed=s,
            setting=setting,
            template_alignment=alignment,
        )
        null_vals[j], _ = compute_dispersion(alignment_to_matrix(sim_aln), frozen.model)

    mu = float(np.mean(null_vals))
    sigma = float(np.std(null_vals, ddof=0))
    p = float(rank_p_value(d_obs, null_vals, tail="right"))
    I = 0.0 if sigma == 0 else float((d_obs - mu) / sigma)
    gap_fraction, ambiguous_fraction = _gap_ambiguous_fraction(alignment)

    return EvalResult(
        gene_id=str(gene_id),
        L_codons=int(alignment.length // 3),
        n_taxa=int(alignment.n_sequences),
        D_obs=float(d_obs),
        mu_null=float(mu),
        sigma_null=float(sigma),
        I=float(I),
        p_babappa=float(p),
        q_babappa=float("nan"),
        gap_fraction=float(gap_fraction),
        ambiguous_fraction=float(ambiguous_fraction),
        runtime_babappa=float(time.perf_counter() - t0),
        energy_model_hash=str(frozen.model_hash),
        phi_hash=str(frozen.phi_hash),
        M0_hash=str(frozen.m0_hash),
        global_seed=int(frozen.global_seed),
        per_gene_seed=int(seed_gene),
        per_calibration_seed=int(seed_calibration),
        notes=f"setting={setting.name}",
    )


def _results_to_df(rows: list[EvalResult]) -> pd.DataFrame:
    records = [r.__dict__.copy() for r in rows]
    df = pd.DataFrame(records)
    if df.empty:
        return df
    q = benjamini_hochberg([float(x) for x in pd.to_numeric(df["p_babappa"], errors="coerce").fillna(1.0)])
    df["q_babappa"] = np.array(q, dtype=float)
    return df


def _qq_with_envelope(p_values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    p = p_values[np.isfinite(p_values)]
    p = p[(p >= 0.0) & (p <= 1.0)]
    p = np.sort(p)
    n = int(p.size)
    if n <= 0:
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
    mask = np.isfinite(x) & np.isfinite(y)
    n = int(np.count_nonzero(mask))
    if n < 3:
        return float("nan"), float("nan"), n
    rho, p = stats.spearmanr(x[mask], y[mask], nan_policy="omit")
    return float(rho), float(p), n


def _validate_dirs(outdir: Path) -> None:
    for sub in ("raw", "report", "manifests", "figures", "tables", "diagnostics", "scripts"):
        (outdir / sub).mkdir(parents=True, exist_ok=True)


def _write_provenance(
    *,
    outdir: Path,
    command: str,
    frozen: FrozenContext,
    m0: NeutralSpec,
    N: int,
    extra: dict[str, Any] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "command": command,
        "global_seed": int(frozen.global_seed),
        "N": int(N),
        "energy_model_hash": str(frozen.model_hash),
        "phi_hash": str(frozen.phi_hash),
        "M0_hash": str(frozen.m0_hash),
        "m0": m0.to_dict(),
        "p_value_rule": "(1 + count) / (N + 1)",
        "deterministic": True,
    }
    if extra:
        payload.update(extra)
    _write_json(outdir / "manifests" / "provenance_freeze.json", payload)


def _write_rebuild_script(outdir: Path, command_line: str) -> None:
    script = outdir / "scripts" / "rebuild_all.sh"
    repo_root = Path(__file__).resolve().parents[2]
    text = f"""#!/usr/bin/env bash
set -euo pipefail
OUT_DIR=\"$(cd \"$(dirname \"$0\")/..\" && pwd)\"
if python -c \"import babappa\" >/dev/null 2>&1; then
  {command_line}
elif [ -d \"{repo_root}/src\" ]; then
  export PYTHONPATH=\"{repo_root}/src:${{PYTHONPATH:-}}\"
  {command_line}
else
  echo \"Could not import babappa. Install BABAPPA or set PYTHONPATH.\" >&2
  exit 1
fi
"""
    script.write_text(text, encoding="utf-8")
    script.chmod(script.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _default_setting() -> M0Setting:
    return M0Setting(
        name="default_fixed_1_none_uniform",
        omega_mode="fixed_1",
        omega_value=1.0,
        rate_heterogeneity="none",
        gamma_k=1,
        gamma_alpha=1.0,
        codon_freqs="uniform",
    )


def _make_report_validate_null(
    *,
    outdir: Path,
    df: pd.DataFrame,
    N: int,
    title: str,
) -> dict[str, float]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    p = pd.to_numeric(df["p_babappa"], errors="coerce").dropna().to_numpy(dtype=float)
    pmin = 1.0 / float(int(N) + 1)
    mean_p = float(np.mean(p)) if p.size else float("nan")
    f001 = float(np.mean(p <= 0.01)) if p.size else float("nan")
    f005 = float(np.mean(p <= 0.05)) if p.size else float("nan")
    f01 = float(np.mean(p <= 0.1)) if p.size else float("nan")
    frac_pmin = float(np.mean(np.isclose(p, pmin))) if p.size else float("nan")

    fig = plt.figure(figsize=(6.5, 4.0))
    ax = fig.add_subplot(111)
    if p.size:
        ax.hist(np.clip(p, 0.0, 1.0), bins=20, color="#4C78A8", edgecolor="white")
    ax.set_title("p-value histogram")
    ax.set_xlabel("p-value")
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(outdir / "figures" / "p_hist.pdf")
    plt.close(fig)

    x, y, ylo, yhi = _qq_with_envelope(p)
    fig = plt.figure(figsize=(6.5, 5.0))
    ax = fig.add_subplot(111)
    if x.size:
        ax.fill_between(x, ylo, yhi, color="#DDDDDD", alpha=0.7)
        ax.plot(x, y, ".", color="#1F77B4", alpha=0.8)
        lim = max(float(np.max(x)), float(np.max(y)))
        ax.plot([0, lim], [0, lim], "r--", linewidth=1)
        ax.set_xlim(0, lim * 1.02)
        ax.set_ylim(0, lim * 1.02)
    ax.set_title("QQ vs Uniform(0,1)")
    ax.set_xlabel("Expected -log10(p)")
    ax.set_ylabel("Observed -log10(p)")
    fig.tight_layout()
    fig.savefig(outdir / "figures" / "qq_uniform.pdf")
    plt.close(fig)

    with PdfPages(outdir / "report" / "report.pdf") as pdf:
        fig = plt.figure(figsize=(8.3, 11.7))
        ax = fig.add_subplot(111)
        ax.axis("off")
        ax.text(0.5, 0.97, title, ha="center", va="top", fontsize=15, fontweight="bold")
        y0 = 0.92
        for line in [
            f"n_genes={len(df)}",
            f"N={int(N)}",
            f"mean_p={mean_p:.6f}",
            f"frac_p_le_0.01={f001:.6f}",
            f"frac_p_le_0.05={f005:.6f}",
            f"frac_p_le_0.1={f01:.6f}",
            f"p_min={pmin:.6f}",
            f"frac_at_pmin={frac_pmin:.6f}",
        ]:
            ax.text(0.05, y0, line, ha="left", va="top", fontsize=10)
            y0 -= 0.032
        pdf.savefig(fig)
        plt.close(fig)

        img1 = plt.figure(figsize=(8.3, 5.0))
        ax1 = img1.add_subplot(111)
        if p.size:
            ax1.hist(np.clip(p, 0.0, 1.0), bins=20, color="#4C78A8", edgecolor="white")
        ax1.set_title("p-value histogram")
        ax1.set_xlabel("p-value")
        ax1.set_ylabel("count")
        img1.tight_layout()
        pdf.savefig(img1)
        plt.close(img1)

        img2 = plt.figure(figsize=(8.3, 5.5))
        ax2 = img2.add_subplot(111)
        if x.size:
            ax2.fill_between(x, ylo, yhi, color="#DDDDDD", alpha=0.7)
            ax2.plot(x, y, ".", color="#1F77B4", alpha=0.8)
            lim = max(float(np.max(x)), float(np.max(y)))
            ax2.plot([0, lim], [0, lim], "r--", linewidth=1)
            ax2.set_xlim(0, lim * 1.02)
            ax2.set_ylim(0, lim * 1.02)
        ax2.set_title("QQ vs Uniform(0,1)")
        ax2.set_xlabel("Expected -log10(p)")
        ax2.set_ylabel("Observed -log10(p)")
        img2.tight_layout()
        pdf.savefig(img2)
        plt.close(img2)

    return {
        "mean_p": mean_p,
        "frac_p_le_0.01": f001,
        "frac_p_le_0.05": f005,
        "frac_p_le_0.1": f01,
        "p_min": float(pmin),
        "frac_at_pmin": frac_pmin,
    }


def _alignment_lengths_from_config(config: dict[str, Any], G: int) -> list[int]:
    lengths: list[int] = []
    if config.get("gene_set"):
        rows = _load_gene_set(str(config["gene_set"]))
        lengths = [_normalize_length_nt(aln.length) for _gid, aln, _p in rows]
    if not lengths:
        if "length_nt" in config:
            lengths = [_normalize_length_nt(int(config["length_nt"]))]
        elif "length_codons" in config:
            lengths = [_normalize_length_nt(int(config["length_codons"]) * 3)]
        else:
            lengths = [900]
    out: list[int] = []
    i = 0
    while len(out) < int(G):
        out.append(int(lengths[i % len(lengths)]))
        i += 1
    return out


def run_validate_null(*, config_path: str | Path, outdir: str | Path, G: int, N: int) -> dict[str, Any]:
    configure_plotting_env()
    out = Path(outdir).resolve()
    _validate_dirs(out)
    config = _load_yaml_or_json(Path(config_path).expanduser().resolve())
    frozen = _build_frozen_context(config)
    m0 = _load_m0_from_config(frozen.model, config)

    lengths = _alignment_lengths_from_config(config, int(G))
    rng = np.random.default_rng(int(frozen.global_seed))
    setting = _default_setting()

    rows: list[EvalResult] = []
    for i in range(int(G)):
        seed_gene = int(rng.integers(0, 2**32 - 1))
        seed_cal = int(rng.integers(0, 2**32 - 1))
        aln = _simulate_under_setting(
            base_m0=m0,
            length_nt=int(lengths[i]),
            seed=seed_gene,
            setting=setting,
            template_alignment=None,
        )
        res = _evaluate_alignment(
            gene_id=f"null_{i+1:04d}",
            alignment=aln,
            frozen=frozen,
            base_m0=m0,
            N=int(N),
            seed_gene=seed_gene,
            seed_calibration=seed_cal,
            setting=setting,
        )
        rows.append(res)

    df = _results_to_df(rows)
    _write_tsv(out / "raw" / "results.tsv", df)
    summary = _make_report_validate_null(outdir=out, df=df, N=int(N), title="BABAPPA Validate-Null")
    _write_tsv(out / "tables" / "summary.tsv", pd.DataFrame([summary]))

    _write_provenance(
        outdir=out,
        command="validate-null",
        frozen=frozen,
        m0=m0,
        N=int(N),
        extra={"G": int(G), "config_path": str(Path(config_path).resolve())},
    )
    cmd = (
        f"python -m babappa.cli validate-null --G {int(G)} --N {int(N)} "
        f"--config {Path(config_path).resolve()} --out $OUT_DIR"
    )
    _write_rebuild_script(out, cmd)
    _write_checksums(out)

    payload = {
        "status": "ok",
        "outdir": str(out),
        "G": int(G),
        "N": int(N),
        **summary,
    }
    _write_json(out / "manifests" / "run_manifest.json", payload)
    return payload


def run_validate_freeze(*, config_path: str | Path, outdir: str | Path) -> dict[str, Any]:
    configure_plotting_env()
    out = Path(outdir).resolve()
    _validate_dirs(out)
    config = _load_yaml_or_json(Path(config_path).expanduser().resolve())
    frozen = _build_frozen_context(config)
    m0 = _load_m0_from_config(frozen.model, config)

    from . import engine as engine_mod

    original_train = engine_mod.train_energy_model
    original_train_sim = engine_mod.train_energy_model_from_neutral_spec

    def _forbid(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("FROZEN_MODEL violation: training/update attempted after freeze")

    engine_mod.train_energy_model = _forbid
    engine_mod.train_energy_model_from_neutral_spec = _forbid

    setting = _default_setting()
    rng = np.random.default_rng(int(frozen.global_seed))
    rows: list[dict[str, Any]] = []
    retrain_violations = 0

    try:
        for i in range(5):
            seed_gene = int(rng.integers(0, 2**32 - 1))
            seed_cal = int(rng.integers(0, 2**32 - 1))
            before = sha256_json(frozen.model.to_dict())
            try:
                aln = _simulate_under_setting(
                    base_m0=m0,
                    length_nt=900,
                    seed=seed_gene,
                    setting=setting,
                    template_alignment=None,
                )
                res = _evaluate_alignment(
                    gene_id=f"freeze_{i+1:03d}",
                    alignment=aln,
                    frozen=frozen,
                    base_m0=m0,
                    N=50,
                    seed_gene=seed_gene,
                    seed_calibration=seed_cal,
                    setting=setting,
                )
                status = "OK"
                reason = "ok"
            except RuntimeError as exc:
                if "FROZEN_MODEL violation" in str(exc):
                    retrain_violations += 1
                status = "FAIL"
                reason = str(exc)
                res = None
            after = sha256_json(frozen.model.to_dict())
            rows.append(
                {
                    "gene_id": f"freeze_{i+1:03d}",
                    "status": status,
                    "reason": reason,
                    "energy_model_hash_before": before,
                    "energy_model_hash_after": after,
                    "hash_constant": bool(before == after),
                    "global_seed": int(frozen.global_seed),
                    "per_gene_seed": int(seed_gene),
                    "per_calibration_seed": int(seed_cal),
                    "p_babappa": (float(res.p_babappa) if res is not None else float("nan")),
                }
            )
    finally:
        engine_mod.train_energy_model = original_train
        engine_mod.train_energy_model_from_neutral_spec = original_train_sim

    df = pd.DataFrame(rows)
    _write_tsv(out / "raw" / "results.tsv", df)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    pass_hash = bool(df["hash_constant"].all()) if not df.empty else False
    pass_no_retrain = int(retrain_violations) == 0

    with PdfPages(out / "report" / "report.pdf") as pdf:
        fig = plt.figure(figsize=(8.3, 11.7))
        ax = fig.add_subplot(111)
        ax.axis("off")
        ax.text(0.5, 0.97, "BABAPPA Validate-Freeze", ha="center", va="top", fontsize=15, fontweight="bold")
        y0 = 0.92
        for line in [
            f"runs=5",
            f"retrain_violations={retrain_violations}",
            f"hash_constant_all={pass_hash}",
            f"PASS_no_retrain={pass_no_retrain}",
        ]:
            ax.text(0.05, y0, line, ha="left", va="top", fontsize=11)
            y0 -= 0.035
        pdf.savefig(fig)
        plt.close(fig)

    _write_provenance(
        outdir=out,
        command="validate-freeze",
        frozen=frozen,
        m0=m0,
        N=50,
        extra={
            "guard_flag": True,
            "retrain_violations": int(retrain_violations),
            "hash_constant_all": bool(pass_hash),
        },
    )
    cmd = f"python -m babappa.cli validate-freeze --config {Path(config_path).resolve()} --out $OUT_DIR"
    _write_rebuild_script(out, cmd)
    _write_checksums(out)

    payload = {
        "status": "ok" if pass_hash and pass_no_retrain else "fail",
        "outdir": str(out),
        "retrain_violations": int(retrain_violations),
        "hash_constant_all": bool(pass_hash),
    }
    _write_json(out / "manifests" / "run_manifest.json", payload)
    return payload


def run_pmin_audit(
    *,
    gene_set: str | Path,
    config_path: str | Path,
    outdir: str | Path,
    N: int,
) -> dict[str, Any]:
    configure_plotting_env()
    out = Path(outdir).resolve()
    _validate_dirs(out)

    config = _load_yaml_or_json(Path(config_path).expanduser().resolve())
    frozen = _build_frozen_context(config)
    m0 = _load_m0_from_config(frozen.model, config)
    genes = _load_gene_set(gene_set)

    setting = _default_setting()
    rng = np.random.default_rng(int(frozen.global_seed))

    real_rows: list[EvalResult] = []
    null_rows: list[EvalResult] = []

    for gid, aln, _path in genes:
        seed_gene = int(rng.integers(0, 2**32 - 1))
        seed_cal = int(rng.integers(0, 2**32 - 1))
        res = _evaluate_alignment(
            gene_id=gid,
            alignment=aln,
            frozen=frozen,
            base_m0=m0,
            N=int(N),
            seed_gene=seed_gene,
            seed_calibration=seed_cal,
            setting=setting,
        )
        real_rows.append(res)

        n_seed_gene = int(rng.integers(0, 2**32 - 1))
        n_seed_cal = int(rng.integers(0, 2**32 - 1))
        sim_aln = _simulate_under_setting(
            base_m0=m0,
            length_nt=aln.length,
            seed=n_seed_gene,
            setting=setting,
            template_alignment=aln,
        )
        nres = _evaluate_alignment(
            gene_id=f"null_{gid}",
            alignment=sim_aln,
            frozen=frozen,
            base_m0=m0,
            N=int(N),
            seed_gene=n_seed_gene,
            seed_calibration=n_seed_cal,
            setting=setting,
        )
        null_rows.append(nres)

    real_df = _results_to_df(real_rows)
    null_df = _results_to_df(null_rows)
    real_df["cohort"] = "real"
    null_df["cohort"] = "m0_sim"
    all_df = pd.concat([real_df, null_df], ignore_index=True)
    _write_tsv(out / "raw" / "results.tsv", all_df)

    pmin = 1.0 / float(int(N) + 1)

    def _fractions(df: pd.DataFrame) -> dict[str, float]:
        p = pd.to_numeric(df["p_babappa"], errors="coerce").dropna().to_numpy(dtype=float)
        return {
            "n": float(p.size),
            "frac_at_pmin": float(np.mean(np.isclose(p, pmin))) if p.size else float("nan"),
            "frac_at_p001": float(np.mean(p <= 0.002)) if p.size else float("nan"),
            "mean_p": float(np.mean(p)) if p.size else float("nan"),
        }

    real_sum = _fractions(real_df)
    null_sum = _fractions(null_df)
    summary_df = pd.DataFrame(
        [
            {"cohort": "real", **real_sum},
            {"cohort": "m0_sim", **null_sum},
        ]
    )
    _write_tsv(out / "tables" / "summary.tsv", summary_df)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    fig = plt.figure(figsize=(6.5, 4.0))
    ax = fig.add_subplot(111)
    rp = pd.to_numeric(real_df["p_babappa"], errors="coerce").dropna().to_numpy(dtype=float)
    npv = pd.to_numeric(null_df["p_babappa"], errors="coerce").dropna().to_numpy(dtype=float)
    if rp.size:
        ax.hist(np.clip(rp, 0.0, 1.0), bins=20, alpha=0.6, label="real")
    if npv.size:
        ax.hist(np.clip(npv, 0.0, 1.0), bins=20, alpha=0.6, label="m0_sim")
    ax.legend()
    ax.set_xlabel("p")
    ax.set_ylabel("count")
    ax.set_title("P-value histogram: real vs M0 simulated")
    fig.tight_layout()
    fig.savefig(out / "figures" / "pmin_hist_compare.pdf")
    plt.close(fig)

    with PdfPages(out / "report" / "report.pdf") as pdf:
        fig = plt.figure(figsize=(8.3, 11.7))
        ax = fig.add_subplot(111)
        ax.axis("off")
        ax.text(0.5, 0.97, "BABAPPA P_MIN Audit", ha="center", va="top", fontsize=15, fontweight="bold")
        y0 = 0.92
        for line in [
            f"N={int(N)} p_min={pmin:.6f}",
            f"real_frac_at_pmin={real_sum['frac_at_pmin']:.6f}",
            f"real_frac_at_p001={real_sum['frac_at_p001']:.6f}",
            f"m0sim_frac_at_pmin={null_sum['frac_at_pmin']:.6f}",
            f"m0sim_frac_at_p001={null_sum['frac_at_p001']:.6f}",
        ]:
            ax.text(0.05, y0, line, ha="left", va="top", fontsize=11)
            y0 -= 0.035
        pdf.savefig(fig)
        plt.close(fig)

        fig2 = plt.figure(figsize=(8.3, 5.0))
        ax2 = fig2.add_subplot(111)
        if rp.size:
            ax2.hist(np.clip(rp, 0.0, 1.0), bins=20, alpha=0.6, label="real")
        if npv.size:
            ax2.hist(np.clip(npv, 0.0, 1.0), bins=20, alpha=0.6, label="m0_sim")
        ax2.legend()
        ax2.set_xlabel("p")
        ax2.set_ylabel("count")
        ax2.set_title("P-value histogram: real vs M0 simulated")
        fig2.tight_layout()
        pdf.savefig(fig2)
        plt.close(fig2)

    _write_provenance(
        outdir=out,
        command="pmin-audit",
        frozen=frozen,
        m0=m0,
        N=int(N),
        extra={"gene_set": str(Path(gene_set).resolve())},
    )
    cmd = (
        f"python -m babappa.cli pmin-audit --gene-set {Path(gene_set).resolve()} --N {int(N)} "
        f"--config {Path(config_path).resolve()} --out $OUT_DIR"
    )
    _write_rebuild_script(out, cmd)
    _write_checksums(out)

    payload = {
        "status": "ok",
        "outdir": str(out),
        "N": int(N),
        "real_frac_at_pmin": float(real_sum["frac_at_pmin"]),
        "m0sim_frac_at_pmin": float(null_sum["frac_at_pmin"]),
    }
    _write_json(out / "manifests" / "run_manifest.json", payload)
    return payload


def run_bias_audit(
    *,
    gene_set: str | Path,
    config_path: str | Path,
    outdir: str | Path,
    N: int,
) -> dict[str, Any]:
    configure_plotting_env()
    out = Path(outdir).resolve()
    _validate_dirs(out)

    config = _load_yaml_or_json(Path(config_path).expanduser().resolve())
    frozen = _build_frozen_context(config)
    m0 = _load_m0_from_config(frozen.model, config)
    genes = _load_gene_set(gene_set)

    setting = _default_setting()
    rng = np.random.default_rng(int(frozen.global_seed))

    rows: list[EvalResult] = []
    for gid, aln, _path in genes:
        seed_gene = int(rng.integers(0, 2**32 - 1))
        seed_cal = int(rng.integers(0, 2**32 - 1))
        rows.append(
            _evaluate_alignment(
                gene_id=gid,
                alignment=aln,
                frozen=frozen,
                base_m0=m0,
                N=int(N),
                seed_gene=seed_gene,
                seed_calibration=seed_cal,
                setting=setting,
            )
        )

    df = _results_to_df(rows)
    _write_tsv(out / "raw" / "results.tsv", df)

    p = pd.to_numeric(df["p_babappa"], errors="coerce").to_numpy(dtype=float)
    I = pd.to_numeric(df["I"], errors="coerce").to_numpy(dtype=float)
    L = pd.to_numeric(df["L_codons"], errors="coerce").to_numpy(dtype=float)
    gap = pd.to_numeric(df["gap_fraction"], errors="coerce").to_numpy(dtype=float)
    amb = pd.to_numeric(df["ambiguous_fraction"], errors="coerce").to_numpy(dtype=float)

    rho_L, p_L, n_L = _spearman(L, I)
    rho_gap, p_gap, n_gap = _spearman(gap, I)
    rho_amb, p_amb, n_amb = _spearman(amb, I)

    corr_df = pd.DataFrame(
        [
            {"x": "L_codons", "y": "I", "rho": rho_L, "p_value": p_L, "n": n_L},
            {"x": "gap_fraction", "y": "I", "rho": rho_gap, "p_value": p_gap, "n": n_gap},
            {"x": "ambiguous_fraction", "y": "I", "rho": rho_amb, "p_value": p_amb, "n": n_amb},
        ]
    )
    _write_tsv(out / "tables" / "correlations.tsv", corr_df)

    q = pd.to_numeric(df["q_babappa"], errors="coerce").to_numpy(dtype=float)
    rej = q <= 0.1

    def _quartile_rejection(x: np.ndarray, label: str) -> pd.DataFrame:
        mask = np.isfinite(x) & np.isfinite(q)
        if int(np.count_nonzero(mask)) < 4:
            return pd.DataFrame(columns=["metric", "quartile", "n", "rejection_rate_q01"])
        bins = pd.qcut(x[mask], q=4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop")
        out_rows: list[dict[str, Any]] = []
        rej_sub = rej[mask]
        for qlab in ["Q1", "Q2", "Q3", "Q4"]:
            m = bins == qlab
            n = int(np.count_nonzero(m))
            rr = float(np.mean(rej_sub[m])) if n > 0 else float("nan")
            out_rows.append({"metric": label, "quartile": qlab, "n": n, "rejection_rate_q01": rr})
        return pd.DataFrame(out_rows)

    qlen = _quartile_rejection(L, "L_codons")
    qgap = _quartile_rejection(gap, "gap_fraction")
    quartile_df = pd.concat([qlen, qgap], ignore_index=True)
    _write_tsv(out / "tables" / "quartile_rejections.tsv", quartile_df)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    def _scatter(x: np.ndarray, y: np.ndarray, xlab: str, out_name: str) -> None:
        m = np.isfinite(x) & np.isfinite(y)
        fig = plt.figure(figsize=(6.5, 4.0))
        ax = fig.add_subplot(111)
        ax.plot(x[m], y[m], ".", alpha=0.7)
        ax.set_xlabel(xlab)
        ax.set_ylabel("I")
        ax.set_title(f"{xlab} vs I")
        fig.tight_layout()
        fig.savefig(out / "figures" / out_name)
        plt.close(fig)

    _scatter(L, I, "L_codons", "L_vs_I.pdf")
    _scatter(gap, I, "gap_fraction", "gap_vs_I.pdf")

    with PdfPages(out / "report" / "report.pdf") as pdf:
        fig = plt.figure(figsize=(8.3, 11.7))
        ax = fig.add_subplot(111)
        ax.axis("off")
        ax.text(0.5, 0.97, "BABAPPA Bias Audit", ha="center", va="top", fontsize=15, fontweight="bold")
        y0 = 0.92
        for line in [
            f"corr(L_codons, I) rho={rho_L:.4f} p={p_L:.3e} n={n_L}",
            f"corr(gap_fraction, I) rho={rho_gap:.4f} p={p_gap:.3e} n={n_gap}",
            f"corr(ambiguous_fraction, I) rho={rho_amb:.4f} p={p_amb:.3e} n={n_amb}",
        ]:
            ax.text(0.05, y0, line, ha="left", va="top", fontsize=10)
            y0 -= 0.033
        pdf.savefig(fig)
        plt.close(fig)

        for fig_name, xarr, xlab in [
            ("L vs I", L, "L_codons"),
            ("gap vs I", gap, "gap_fraction"),
        ]:
            f = plt.figure(figsize=(8.3, 5.0))
            a = f.add_subplot(111)
            m = np.isfinite(xarr) & np.isfinite(I)
            a.plot(xarr[m], I[m], ".", alpha=0.7)
            a.set_xlabel(xlab)
            a.set_ylabel("I")
            a.set_title(fig_name)
            f.tight_layout()
            pdf.savefig(f)
            plt.close(f)

    _write_provenance(
        outdir=out,
        command="bias-audit",
        frozen=frozen,
        m0=m0,
        N=int(N),
        extra={"gene_set": str(Path(gene_set).resolve())},
    )
    cmd = (
        f"python -m babappa.cli bias-audit --gene-set {Path(gene_set).resolve()} --N {int(N)} "
        f"--config {Path(config_path).resolve()} --out $OUT_DIR"
    )
    _write_rebuild_script(out, cmd)
    _write_checksums(out)

    payload = {
        "status": "ok",
        "outdir": str(out),
        "rho_length_I": float(rho_L),
        "rho_gap_I": float(rho_gap),
        "rho_ambiguous_I": float(rho_amb),
    }
    _write_json(out / "manifests" / "run_manifest.json", payload)
    return payload


def _grid_to_settings(grid: dict[str, Any]) -> list[M0Setting]:
    omega_modes = grid.get("omega_mode", ["fixed_1"])
    if isinstance(omega_modes, str):
        omega_modes = [omega_modes]
    rate_modes = grid.get("rate_heterogeneity", ["none"])
    if isinstance(rate_modes, str):
        rate_modes = [rate_modes]
    codon_modes = grid.get("codon_freqs", ["uniform"])
    if isinstance(codon_modes, str):
        codon_modes = [codon_modes]

    fixed_lt1_vals = grid.get("fixed_omega_lt1_values", [0.2])
    if isinstance(fixed_lt1_vals, (int, float)):
        fixed_lt1_vals = [float(fixed_lt1_vals)]

    gamma_alphas = grid.get("gamma_alpha", [0.5])
    if isinstance(gamma_alphas, (int, float)):
        gamma_alphas = [float(gamma_alphas)]
    gamma_k = int(grid.get("gamma_k", 4))

    mix = grid.get("mixture_purifying", {}) if isinstance(grid.get("mixture_purifying"), dict) else {}
    mix_w = float(mix.get("w_purifying", 0.8))
    mix_om_p = float(mix.get("omega_purifying", 0.1))
    mix_om_n = float(mix.get("omega_neutral", 1.0))

    settings: list[M0Setting] = []
    for om_mode in [str(x) for x in omega_modes]:
        om_values = [1.0]
        if om_mode == "fixed_omega_lt1":
            om_values = [float(x) for x in fixed_lt1_vals]
        for om_val in om_values:
            for rt_mode in [str(x) for x in rate_modes]:
                alphas = [1.0]
                if rt_mode == "gamma":
                    alphas = [float(x) for x in gamma_alphas]
                for alpha in alphas:
                    for cf in [str(x) for x in codon_modes]:
                        name = f"omega={om_mode}:{om_val:g}|rate={rt_mode}:{alpha:g}|codon={cf}"
                        settings.append(
                            M0Setting(
                                name=name,
                                omega_mode=om_mode,
                                omega_value=float(om_val),
                                rate_heterogeneity=rt_mode,
                                gamma_k=int(gamma_k),
                                gamma_alpha=float(alpha),
                                codon_freqs=cf,
                                mixture_weight=float(mix_w),
                                mixture_omega_purifying=float(mix_om_p),
                                mixture_omega_neutral=float(mix_om_n),
                            )
                        )
    if not settings:
        settings = [_default_setting()]
    return settings


def run_sensitivity(
    *,
    gene_set: str | Path,
    grid_path: str | Path,
    config_path: str | Path,
    outdir: str | Path,
) -> dict[str, Any]:
    configure_plotting_env()
    out = Path(outdir).resolve()
    _validate_dirs(out)

    config = _load_yaml_or_json(Path(config_path).expanduser().resolve())
    grid = _load_yaml_or_json(Path(grid_path).expanduser().resolve())
    frozen = _build_frozen_context(config)
    base_m0 = _load_m0_from_config(frozen.model, config)
    genes = _load_gene_set(gene_set)

    N = int(config.get("N", config.get("null_N", 199)))
    settings = _grid_to_settings(grid)
    rng = np.random.default_rng(int(frozen.global_seed))

    per_setting_rows: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []

    for setting in settings:
        gene_rows: list[EvalResult] = []
        t0 = time.perf_counter()
        for gid, aln, _path in genes:
            seed_gene = int(rng.integers(0, 2**32 - 1))
            seed_cal = int(rng.integers(0, 2**32 - 1))
            gene_rows.append(
                _evaluate_alignment(
                    gene_id=gid,
                    alignment=aln,
                    frozen=frozen,
                    base_m0=base_m0,
                    N=N,
                    seed_gene=seed_gene,
                    seed_calibration=seed_cal,
                    setting=setting,
                )
            )
        sdf = _results_to_df(gene_rows)
        sdf["setting"] = setting.name
        all_rows.append(sdf)

        pvals = pd.to_numeric(sdf["p_babappa"], errors="coerce").dropna().to_numpy(dtype=float)
        qvals = pd.to_numeric(sdf["q_babappa"], errors="coerce").dropna().to_numpy(dtype=float)
        pmin = 1.0 / float(N + 1)
        per_setting_rows.append(
            {
                "setting": setting.name,
                "n_genes": int(len(sdf)),
                "frac_q_lt_0_1": float(np.mean(qvals < 0.1)) if qvals.size else float("nan"),
                "mean_p": float(np.mean(pvals)) if pvals.size else float("nan"),
                "mean_q": float(np.mean(qvals)) if qvals.size else float("nan"),
                "frac_at_pmin": float(np.mean(np.isclose(pvals, pmin))) if pvals.size else float("nan"),
                "runtime_sec": float(time.perf_counter() - t0),
            }
        )

    full_df = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    summary_df = pd.DataFrame(per_setting_rows)
    _write_tsv(out / "raw" / "results.tsv", full_df)
    _write_tsv(out / "tables" / "sensitivity_summary.tsv", summary_df)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    fig = plt.figure(figsize=(8.5, max(3.5, 0.33 * max(1, len(summary_df)))))
    ax = fig.add_subplot(111)
    if not summary_df.empty:
        order = np.arange(len(summary_df))
        vals = pd.to_numeric(summary_df["frac_q_lt_0_1"], errors="coerce").to_numpy(dtype=float)
        ax.barh(order, vals, color="#4C78A8")
        ax.set_yticks(order)
        ax.set_yticklabels(summary_df["setting"].astype(str).tolist(), fontsize=7)
        ax.set_xlabel("fraction q<0.1")
        ax.set_title("Sensitivity rejection fraction by M0 setting")
    fig.tight_layout()
    fig.savefig(out / "figures" / "rejection_fraction_by_setting.pdf")
    plt.close(fig)

    with PdfPages(out / "report" / "report.pdf") as pdf:
        fig = plt.figure(figsize=(8.3, 11.7))
        ax = fig.add_subplot(111)
        ax.axis("off")
        ax.text(0.5, 0.97, "BABAPPA Sensitivity Grid", ha="center", va="top", fontsize=15, fontweight="bold")
        ax.text(0.05, 0.92, f"settings={len(summary_df)} genes={len(genes)} N={N}", ha="left", va="top", fontsize=10)
        y0 = 0.88
        for row in summary_df.itertuples(index=False):
            line = (
                f"{row.setting}: frac_q<0.1={float(row.frac_q_lt_0_1):.4f}, "
                f"mean_p={float(row.mean_p):.4f}, mean_q={float(row.mean_q):.4f}, "
                f"frac_at_pmin={float(row.frac_at_pmin):.4f}, runtime={float(row.runtime_sec):.1f}s"
            )
            ax.text(0.05, y0, line, ha="left", va="top", fontsize=8)
            y0 -= 0.023
            if y0 < 0.08:
                break
        pdf.savefig(fig)
        plt.close(fig)

        g2 = plt.figure(figsize=(8.3, max(3.5, 0.33 * max(1, len(summary_df)))))
        a2 = g2.add_subplot(111)
        if not summary_df.empty:
            order = np.arange(len(summary_df))
            vals = pd.to_numeric(summary_df["frac_q_lt_0_1"], errors="coerce").to_numpy(dtype=float)
            a2.barh(order, vals, color="#4C78A8")
            a2.set_yticks(order)
            a2.set_yticklabels(summary_df["setting"].astype(str).tolist(), fontsize=7)
            a2.set_xlabel("fraction q<0.1")
            a2.set_title("Sensitivity rejection fraction by M0 setting")
        g2.tight_layout()
        pdf.savefig(g2)
        plt.close(g2)

    _write_provenance(
        outdir=out,
        command="sensitivity",
        frozen=frozen,
        m0=base_m0,
        N=int(N),
        extra={
            "gene_set": str(Path(gene_set).resolve()),
            "grid_path": str(Path(grid_path).resolve()),
            "n_settings": int(len(settings)),
        },
    )
    cmd = (
        f"python -m babappa.cli sensitivity --gene-set {Path(gene_set).resolve()} "
        f"--grid {Path(grid_path).resolve()} --config {Path(config_path).resolve()} --out $OUT_DIR"
    )
    _write_rebuild_script(out, cmd)
    _write_checksums(out)

    payload = {
        "status": "ok",
        "outdir": str(out),
        "n_settings": int(len(settings)),
        "n_genes": int(len(genes)),
        "N": int(N),
    }
    _write_json(out / "manifests" / "run_manifest.json", payload)
    return payload
