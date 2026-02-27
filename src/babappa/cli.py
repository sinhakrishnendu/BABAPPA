from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from . import __version__


def _parse_int_list(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _write_json_file(path: str | Path, payload: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _emit_json(payload: Any) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def _load_tree_text(tree: str | None, tree_file: str | None) -> str | None:
    if tree and tree_file:
        raise ValueError("Use either --tree or --tree-file, not both.")
    if tree:
        return tree.strip()
    if tree_file:
        return Path(tree_file).read_text(encoding="utf-8").strip()
    return None


def _load_codon_frequencies(value: str) -> dict[str, float] | None:
    if value.lower() == "equal":
        return None
    path = Path(value)
    if not path.exists():
        raise ValueError(
            f"Codon frequency file not found: {value}. Use 'equal' or a JSON file path."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Codon frequency JSON must contain an object mapping codon->value.")
    return {str(k): float(v) for k, v in payload.items()}


def _base_manifest(args: argparse.Namespace, command_name: str, seed: int | None) -> dict[str, object]:
    from .system_info import get_system_metadata

    return {
        "schema_version": 1,
        "command": command_name,
        "command_line": "babappa " + " ".join(getattr(args, "_argv", [])),
        "tool_version": __version__,
        "system": get_system_metadata(Path.cwd()),
        "seed": seed,
        "seed_policy": "numpy.default_rng(seed); per-gene seeds derived via rng.integers",
    }


def _write_validated_manifest(path: str | Path, payload: dict[str, object], manifest_kind: str) -> None:
    from .schemas import validate_manifest_payload

    validate_manifest_payload(payload, manifest_kind)
    _write_json_file(path, payload)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="babappa",
        description="BABAPPA: distributed evolutionary burden testing with frozen-energy Monte Carlo calibration.",
        epilog=(
            "Docs: docs/biologist_quickstart.md | docs/model_spec.md | "
            "docs/benchmarking.md | docs/interpretation.md"
        ),
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # doctor
    doctor = subparsers.add_parser("doctor", help="Validate biological and reproducibility readiness.")
    doctor.add_argument("--genes-dir", required=True, metavar="DIR")
    doctor.add_argument("--tree-file", required=True, metavar="NEWICK")
    doctor.add_argument("--neutral-model-json", required=True, metavar="JSON")
    doctor.add_argument("--phi-spec", default=None, metavar="JSON")
    doctor.add_argument("--strict-taxa", action="store_true", default=True)
    doctor.add_argument("--taxa-policy", choices=["strict", "intersection"], default="strict")
    doctor.add_argument("--allow-stop-codons", action="store_true")
    doctor.add_argument("--allow-polytomies", action="store_true")
    doctor.add_argument("--filter-to-multiple-of-3", action="store_true")
    doctor.add_argument("--existing-model", default=None, metavar="JSON")
    doctor.add_argument("--seed", type=int, default=None)
    doctor.add_argument("--output-dir", default=None, metavar="DIR")

    # train
    train = subparsers.add_parser("train", help="Train frozen neutral energy model.")
    train.add_argument("--neutral", nargs="+", default=None, metavar="FASTA")
    train.add_argument("--neutral-model-json", default=None, metavar="JSON")
    train.add_argument("--tree", default=None, metavar="NEWICK")
    train.add_argument("--tree-file", default=None, metavar="PATH")
    train.add_argument("--kappa", type=float, default=2.0)
    train.add_argument("--omega", type=float, default=1.0)
    train.add_argument("--codon-frequencies", default="equal", metavar="equal|JSON")
    train.add_argument("--sim-replicates", type=int, default=200)
    train.add_argument("--sim-length", type=int, default=900)
    train.add_argument("--max-columns", type=int, default=None)
    train.add_argument("--ridge", type=float, default=1e-5)
    train.add_argument("--seed", type=int, default=None)
    train.add_argument("--phi-spec", default=None, metavar="JSON")
    train.add_argument("--model", required=True, metavar="JSON")
    train.add_argument("--json", action="store_true")
    train.add_argument("--manifest", default=None, metavar="JSON")

    # analyze
    analyze = subparsers.add_parser("analyze", help="Analyze one gene alignment.")
    analyze.add_argument("--model", required=True, metavar="JSON")
    analyze.add_argument("--alignment", required=True, metavar="FASTA")
    analyze.add_argument("--calibration-size", type=int, default=1000)
    analyze.add_argument("--calibration-mode", choices=["auto", "phylo", "gaussian"], default="auto")
    analyze.add_argument("--tail", choices=["right", "left", "two-sided"], default="right")
    analyze.add_argument("--seed", type=int, default=None)
    analyze.add_argument("--strict-scaling", action="store_true")
    analyze.add_argument("--override-scaling", action="store_true")
    analyze.add_argument("--output", default=None, metavar="JSON")
    analyze.add_argument("--phi-spec", default=None, metavar="JSON")
    analyze.add_argument("--json", action="store_true")
    analyze.add_argument("--manifest", default=None, metavar="JSON")

    # batch
    batch = subparsers.add_parser("batch", help="Analyze many gene alignments.")
    batch.add_argument("--model", required=True, metavar="JSON")
    batch.add_argument("--alignments", nargs="+", required=True, metavar="FASTA")
    batch.add_argument("--calibration-size", type=int, default=1000)
    batch.add_argument("--calibration-mode", choices=["auto", "phylo", "gaussian"], default="auto")
    batch.add_argument("--tail", choices=["right", "left", "two-sided"], default="right")
    batch.add_argument("--seed", type=int, default=None)
    batch.add_argument("--strict-scaling", action="store_true")
    batch.add_argument("--override-scaling", action="store_true")
    batch.add_argument("--output", required=True, metavar="TSV")
    batch.add_argument("--json-output", default=None, metavar="JSON")
    batch.add_argument("--phi-spec", default=None, metavar="JSON")
    batch.add_argument("--manifest", default=None, metavar="JSON")

    # fdr
    fdr = subparsers.add_parser("fdr", help="Apply Benjamini-Hochberg FDR correction to a TSV.")
    fdr.add_argument("--input", required=True, metavar="TSV")
    fdr.add_argument("--output", required=True, metavar="TSV")
    fdr.add_argument("--p-column", default="p", metavar="NAME")
    fdr.add_argument("--q-column", default="q_value", metavar="NAME")

    # explain
    explain = subparsers.add_parser(
        "explain",
        help="Generate an interpretability report (energy histograms, dispersion, window contributions).",
    )
    explain.add_argument("--model", required=True, metavar="JSON")
    explain.add_argument("--alignment", required=True, metavar="FASTA")
    explain.add_argument("--calibration-size", type=int, default=999)
    explain.add_argument("--tail", choices=["right", "left", "two-sided"], default="right")
    explain.add_argument("--seed", type=int, default=None)
    explain.add_argument("--window-size", type=int, default=30)
    explain.add_argument("--out", required=True, metavar="PDF")
    explain.add_argument("--json-out", default=None, metavar="JSON")
    explain.add_argument("--manifest", default=None, metavar="JSON")

    # baseline
    baseline = subparsers.add_parser("baseline", help="Run external baseline methods.")
    baseline_sub = baseline.add_subparsers(dest="baseline_command", required=True)
    baseline_run = baseline_sub.add_parser("run", help="Run one baseline method on dataset.json")
    baseline_run.add_argument("--method", choices=["codeml", "busted", "relax"], required=True)
    baseline_run.add_argument("--input", required=True, metavar="dataset.json")
    baseline_run.add_argument("--out", required=True, metavar="TSV")
    baseline_run.add_argument("--work-dir", default=None, metavar="DIR")
    baseline_run.add_argument("--foreground", default=None, metavar="TAXON")
    baseline_run.add_argument("--foreground-branch-label", default=None, metavar="LABEL")
    baseline_run.add_argument("--container", choices=["auto", "docker", "singularity", "local"], default="auto")
    baseline_run.add_argument("--timeout", "--timeout-sec", dest="timeout", type=int, default=1800)
    baseline_run.add_argument("--jobs", type=int, default=1, help="Parallel worker count for per-gene baseline runs.")
    baseline_run.add_argument("--manifest", default=None, metavar="JSON")
    baseline_doctor = baseline_sub.add_parser(
        "doctor",
        help="Validate baseline backends with a toy run and p-value parsing check.",
    )
    baseline_doctor.add_argument("--methods", default="busted")
    baseline_doctor.add_argument(
        "--hyphy",
        action="store_true",
        help="Shortcut for HyPhy backend checks (equivalent to --methods busted,relax).",
    )
    baseline_doctor.add_argument("--container", choices=["auto", "docker", "singularity", "local"], default="auto")
    baseline_doctor.add_argument("--timeout", "--timeout-sec", dest="timeout", type=int, default=5)
    baseline_doctor.add_argument("--work-dir", default=None, metavar="DIR")
    baseline_doctor.add_argument("--no-pull-images", action="store_true")
    baseline_doctor.add_argument("--json", action="store_true")
    baseline_doctor.add_argument("--manifest", default=None, metavar="JSON")

    # benchmark
    benchmark = subparsers.add_parser("benchmark", help="Run benchmark and produce results_pack.")
    benchmark.add_argument("--results-pack", default=None, metavar="DIR")
    benchmark.add_argument(
        "--preset",
        choices=["null_smoke", "power_distributed_smoke"],
        default=None,
    )
    benchmark.add_argument("--seed", type=int, default=123)
    benchmark.add_argument("--tail", choices=["right", "left", "two-sided"], default="right")
    benchmark.add_argument("--L-grid", default="150,300,600,1200")
    benchmark.add_argument("--taxa-grid", default="8,16")
    benchmark.add_argument("--N-grid", default="199,999,4999")
    benchmark.add_argument("--n-null-genes", type=int, default=200)
    benchmark.add_argument("--n-alt-genes", type=int, default=80)
    benchmark.add_argument("--training-replicates", type=int, default=200)
    benchmark.add_argument("--training-length", type=int, default=1200)
    benchmark.add_argument("--kappa", type=float, default=2.0)
    benchmark.add_argument("--include-baselines", action="store_true")
    benchmark.add_argument("--include-relax", action="store_true")
    benchmark.add_argument("--allow-baseline-fail", action="store_true")
    benchmark.add_argument("--no-plots", action="store_true")
    benchmark.add_argument("--baseline-methods", default="busted")
    benchmark.add_argument(
        "--power-families",
        default=(
            "A_distributed_weak_selection,"
            "B_constraint_shift,"
            "C_episodic_branch_site_surrogate,"
            "D_misspecification_stress"
        ),
    )
    benchmark.add_argument("--rebuild-only", action="store_true")
    benchmark.add_argument("--manifest", default=None, metavar="JSON")

    benchmark_sub = benchmark.add_subparsers(dest="benchmark_command", required=False)
    benchmark_run = benchmark_sub.add_parser(
        "run",
        help="Integrated benchmark orchestrator across simulation/ortholog/viral tracks.",
    )
    benchmark_run.add_argument("--track", choices=["simulation", "ortholog", "viral"], required=True)
    benchmark_run.add_argument("--preset", required=True)
    benchmark_run.add_argument("--outdir", required=True, metavar="DIR")
    benchmark_run.add_argument("--seed", type=int, required=True)
    benchmark_run.add_argument("--tail", choices=["right", "left", "two-sided"], default="right")
    benchmark_run.add_argument("--dataset-json", default=None, metavar="JSON")
    benchmark_run.add_argument("--include-baselines", action="store_true")
    benchmark_run.add_argument("--include-relax", action="store_true")
    benchmark_run.add_argument("--allow-baseline-fail", action="store_true")
    benchmark_run.add_argument("--allow-qc-fail", action="store_true")
    benchmark_run.add_argument("--no-plots", action="store_true")
    benchmark_run.add_argument("--mode", choices=["pilot", "publication"], default="pilot")
    benchmark_run.add_argument("--min-taxa", type=int, default=None)
    benchmark_run.add_argument("--require-full-completeness", action="store_true")
    benchmark_run.add_argument("--foreground-species", default=None, metavar="TAXON")
    benchmark_run.add_argument("--jobs", type=int, default=0, help="Parallel baseline worker count (0=auto).")
    benchmark_run.add_argument("--baseline-container", choices=["auto", "docker", "singularity", "local"], default="auto")
    benchmark_run.add_argument("--baseline-timeout", type=int, default=1800)
    benchmark_run.add_argument("--manifest", default=None, metavar="JSON")

    benchmark_rebuild = benchmark_sub.add_parser(
        "rebuild",
        help="Rebuild tables/figures/checksums from existing raw benchmark outputs.",
    )
    benchmark_rebuild.add_argument("--outdir", required=True, metavar="DIR")
    benchmark_rebuild.add_argument("--no-plots", action="store_true")

    benchmark_resume = benchmark_sub.add_parser(
        "resume",
        help="Resume an interrupted benchmark run from shard checkpoints in an existing outdir.",
    )
    benchmark_resume.add_argument("--outdir", required=True, metavar="DIR")
    benchmark_resume.add_argument("--no-plots", action="store_true")
    benchmark_resume.add_argument("--manifest", default=None, metavar="JSON")

    benchmark_realdata = benchmark_sub.add_parser(
        "realdata",
        help="Empirical benchmark runner for curated ortholog/HIV/SARS datasets.",
    )
    benchmark_realdata.add_argument(
        "--preset",
        choices=[
            "ortholog_small8_full",
            "hiv_env_b_full",
            "sars_2020_full",
            "ortholog_real_v12",
            "hiv_env_b_real",
            "sars_cov2_real",
        ],
        required=True,
    )
    benchmark_realdata.add_argument("--data", required=True, metavar="DATASET_JSON_OR_DIR")
    benchmark_realdata.add_argument(
        "--outdir",
        default=None,
        metavar="DIR",
        help="Results pack directory (default: results/<preset>).",
    )
    benchmark_realdata.add_argument("--N", type=int, default=999)
    benchmark_realdata.add_argument("--seed", type=int, default=1)
    benchmark_realdata.add_argument("--tail", choices=["right", "left", "two-sided"], default="right")
    benchmark_realdata.add_argument("--allow-qc-fail", action="store_true")
    benchmark_realdata.add_argument("--allow-baseline-fail", action="store_true")
    benchmark_realdata.add_argument("--include-relax", action="store_true")
    benchmark_realdata.add_argument("--mode", choices=["pilot", "publication"], default="pilot")
    benchmark_realdata.add_argument("--min-taxa", type=int, default=None)
    benchmark_realdata.add_argument("--require-full-completeness", action="store_true")
    benchmark_realdata.add_argument("--foreground-species", default=None, metavar="TAXON")
    benchmark_realdata.add_argument("--jobs", type=int, default=0, help="Parallel baseline worker count (0=auto).")
    benchmark_realdata.add_argument("--baseline-container", choices=["auto", "docker", "singularity", "local"], default="auto")
    benchmark_realdata.add_argument("--baseline-timeout", type=int, default=1800)
    benchmark_realdata.add_argument("--no-plots", action="store_true")
    benchmark_realdata.add_argument("--results-only", action="store_true", help="Alias for --no-plots.")
    benchmark_realdata.add_argument("--manifest", default=None, metavar="JSON")

    # dataset
    dataset = subparsers.add_parser("dataset", help="Prepare/fetch dataset bundles for benchmark tracks.")
    dataset_sub = dataset.add_subparsers(dest="dataset_command", required=True)

    dataset_cache = dataset_sub.add_parser("cache", help="Show local dataset cache status.")
    dataset_cache.add_argument("--show", action="store_true")

    dataset_clear = dataset_sub.add_parser("clear-cache", help="Remove one dataset cache entry by name.")
    dataset_clear.add_argument("--name", required=True)

    dataset_audit = dataset_sub.add_parser(
        "audit",
        help="Run fast dataset preflight (discovery/parse/taxa match/prune) without BABAPPA/HyPhy.",
    )
    dataset_audit.add_argument("--dataset", required=True, metavar="DATASET_JSON_OR_DIR")
    dataset_audit.add_argument("--out", required=True, metavar="DIR")
    dataset_audit.add_argument("--min-length-codons", type=int, default=300)
    dataset_audit.add_argument("--min-taxa", type=int, default=6)
    dataset_audit.add_argument("--require-full-completeness", action="store_true")
    dataset_audit.add_argument("--max-genes", type=int, default=None)

    dataset_fetch = dataset_sub.add_parser("fetch", help="Fetch or synthesize a dataset bundle.")
    dataset_fetch_sub = dataset_fetch.add_subparsers(dest="dataset_source", required=True)

    ds_orth = dataset_fetch_sub.add_parser("orthomam", help="Fetch real OrthoMaM-style ortholog dataset.")
    ds_orth.add_argument("--release", "--version", default="v12")
    ds_orth.add_argument("--species-set", choices=["small8", "medium16"], default="small8")
    ds_orth.add_argument("--mode", choices=["cache", "cached", "remote", "small"], default="cache")
    ds_orth.add_argument("--base-url", default=None, metavar="URL")
    ds_orth.add_argument("--min-length", type=int, default=300)
    ds_orth.add_argument("--max-genes", "--n-markers", dest="max_genes", type=int, default=800)
    ds_orth.add_argument("--marker-ids-file", default=None, metavar="TXT")
    ds_orth.add_argument("--outdir", default=None, metavar="DIR")
    ds_orth.add_argument("--seed", type=int, default=42)
    ds_orth.add_argument("--retries", type=int, default=3)
    ds_orth.add_argument("--timeout-sec", type=float, default=60.0)
    ds_orth.add_argument("--tree-file", default=None, metavar="NEWICK")

    ds_hiv = dataset_fetch_sub.add_parser("hiv", help="Fetch curated HIV-1 dataset (cache-backed).")
    ds_hiv.add_argument("--source", default="lanl")
    ds_hiv.add_argument("--gene", default="env")
    ds_hiv.add_argument("--subtype", default="B")
    ds_hiv.add_argument("--recombination-policy", choices=["subtype_filter", "gard_filter"], default="subtype_filter")
    ds_hiv.add_argument("--alignment-id", default=None, metavar="ID")
    ds_hiv.add_argument("--outdir", default=None, metavar="DIR")
    ds_hiv.add_argument("--tree-file", default=None, metavar="NEWICK")
    ds_hiv.add_argument("--max-ambiguous-fraction", type=float, default=0.05)

    ds_sars = dataset_fetch_sub.add_parser(
        "sarscov2",
        aliases=["sars"],
        help="Fetch curated SARS-CoV-2 dataset (cache-backed).",
    )
    ds_sars.add_argument("--source", default="ncbi")
    ds_sars.add_argument("--date-range", default="2020-01-01:2020-12-31")
    ds_sars.add_argument("--max-samples", "--n", dest="max_samples", type=int, default=2000)
    ds_sars.add_argument("--outdir", default=None, metavar="DIR")
    ds_sars.add_argument("--tree-file", default=None, metavar="NEWICK")
    ds_sars.add_argument("--max-n-fraction", type=float, default=0.01)
    ds_sars.add_argument("--host", default="human")
    ds_sars.add_argument("--complete-only", action=argparse.BooleanOptionalAction, default=True)
    ds_sars.add_argument("--include-cds", action=argparse.BooleanOptionalAction, default=True)
    ds_sars.add_argument("--stratify", default="month,country")
    ds_sars.add_argument("--seed", type=int, default=1)
    ds_sars.add_argument("--datasets-timeout", type=int, default=1800)

    dataset_import = dataset_sub.add_parser("import", help="Import curated real dataset inputs.")
    dataset_import_sub = dataset_import.add_subparsers(dest="dataset_import_source", required=True)

    imp_orth = dataset_import_sub.add_parser("orthomam", help="Import curated ortholog alignments from local files.")
    imp_orth.add_argument("--source-dir", required=True, metavar="DIR")
    imp_orth.add_argument("--release", required=True)
    imp_orth.add_argument("--species-set", choices=["small8", "medium16"], default="small8")
    imp_orth.add_argument("--min-length", type=int, default=300)
    imp_orth.add_argument("--max-genes", type=int, default=800)
    imp_orth.add_argument("--outdir", required=True, metavar="DIR")
    imp_orth.add_argument("--seed", type=int, default=42)
    imp_orth.add_argument("--tree-file", default=None, metavar="NEWICK")
    imp_orth.add_argument("--provenance", default=None, metavar="JSON")

    imp_hiv = dataset_import_sub.add_parser("hiv", help="Import curated HIV codon alignment + provenance.")
    imp_hiv.add_argument("--alignment", "--alignment-fasta", required=True, metavar="FASTA")
    imp_hiv.add_argument("--outdir", required=True, metavar="DIR")
    imp_hiv.add_argument("--provenance", default=None, metavar="JSON")
    imp_hiv.add_argument("--alignment-id", default=None, metavar="ID")
    imp_hiv.add_argument("--download-steps-text", default=None, metavar="TEXT")
    imp_hiv.add_argument("--gene", default="env")
    imp_hiv.add_argument("--subtype", default="B")
    imp_hiv.add_argument("--recombination-policy", choices=["subtype_filter", "gard_filter"], default="subtype_filter")
    imp_hiv.add_argument("--tree-file", default=None, metavar="NEWICK")
    imp_hiv.add_argument("--max-ambiguous-fraction", type=float, default=0.05)

    imp_sars = dataset_import_sub.add_parser("sarscov2", help="Import curated SARS-CoV-2 FASTA + metadata TSV.")
    imp_sars.add_argument("--fasta", required=True, metavar="FASTA")
    imp_sars.add_argument("--metadata", required=True, metavar="TSV")
    imp_sars.add_argument("--outdir", required=True, metavar="DIR")
    imp_sars.add_argument("--provenance", default=None, metavar="JSON")
    imp_sars.add_argument("--tree-file", default=None, metavar="NEWICK")
    imp_sars.add_argument("--max-n-fraction", type=float, default=0.01)

    # reproduce
    reproduce = subparsers.add_parser(
        "reproduce",
        help="Run paper-oriented benchmark sequence and write a consolidated summary.",
    )
    reproduce.add_argument("--paper", choices=["all", "paper_ready"], default="all")
    reproduce.add_argument("--paper_ready", action="store_true")
    reproduce.add_argument("--mode", choices=["pilot", "publication"], default="pilot")
    reproduce.add_argument("--outdir", required=True, metavar="DIR")
    reproduce.add_argument("--seed", type=int, default=42)
    reproduce.add_argument("--tail", choices=["right", "left", "two-sided"], default="right")
    reproduce.add_argument("--allow-baseline-fail", action="store_true")
    reproduce.add_argument("--baseline-container", choices=["auto", "docker", "singularity", "local"], default="auto")
    reproduce.add_argument("--baseline-timeout", type=int, default=1800)
    reproduce.add_argument("--include-sars", action="store_true")
    reproduce.add_argument("--no-plots", action="store_true")
    reproduce.add_argument("--dataset-json-ortholog", default=None, metavar="JSON")
    reproduce.add_argument("--dataset-json-viral-hiv", default=None, metavar="JSON")
    reproduce.add_argument("--dataset-json-viral-sars", default=None, metavar="JSON")
    reproduce.add_argument("--foreground-species", default=None, metavar="TAXON")
    reproduce.add_argument("--jobs", type=int, default=0, help="Parallel baseline worker count (0=auto).")
    reproduce.add_argument("--fast", action="store_true")
    reproduce.add_argument("--manifest", default=None, metavar="JSON")

    # validation battery
    validate_null = subparsers.add_parser(
        "validate-null",
        help="Test-1 null uniformity validation under frozen M0/model.",
    )
    validate_null.add_argument("--G", type=int, default=300)
    validate_null.add_argument("--N", type=int, default=999)
    validate_null.add_argument("--config", default="config.yaml", metavar="YAML")
    validate_null.add_argument("--out", required=True, metavar="DIR")
    validate_null.add_argument("--manifest", default=None, metavar="JSON")

    validate_freeze = subparsers.add_parser(
        "validate-freeze",
        help="Test-2 frozen-model leak trap (hash constancy and retrain guard).",
    )
    validate_freeze.add_argument("--config", default="config.yaml", metavar="YAML")
    validate_freeze.add_argument("--out", required=True, metavar="DIR")
    validate_freeze.add_argument("--manifest", default=None, metavar="JSON")

    pmin_audit = subparsers.add_parser(
        "pmin-audit",
        help="Test-3 p_min mass audit on real genes vs matched M0-simulated genes.",
    )
    pmin_audit.add_argument("--gene-set", required=True, metavar="DIR_OR_FASTA")
    pmin_audit.add_argument("--N", type=int, default=999)
    pmin_audit.add_argument("--config", default="config.yaml", metavar="YAML")
    pmin_audit.add_argument("--out", required=True, metavar="DIR")
    pmin_audit.add_argument("--manifest", default=None, metavar="JSON")

    bias_audit = subparsers.add_parser(
        "bias-audit",
        help="Test-4 length/gap/ambiguity dependence audit.",
    )
    bias_audit.add_argument("--gene-set", required=True, metavar="DIR_OR_FASTA")
    bias_audit.add_argument("--N", type=int, default=199)
    bias_audit.add_argument("--config", default="config.yaml", metavar="YAML")
    bias_audit.add_argument("--out", required=True, metavar="DIR")
    bias_audit.add_argument("--manifest", default=None, metavar="JSON")

    sensitivity = subparsers.add_parser(
        "sensitivity",
        help="Test-5 realistic M0 sensitivity grid audit.",
    )
    sensitivity.add_argument("--gene-set", required=True, metavar="DIR_OR_FASTA")
    sensitivity.add_argument("--grid", required=True, metavar="YAML")
    sensitivity.add_argument("--config", default="config.yaml", metavar="YAML")
    sensitivity.add_argument("--out", required=True, metavar="DIR")
    sensitivity.add_argument("--manifest", default=None, metavar="JSON")

    # report
    report = subparsers.add_parser("report", help="Cross-pack reporting utilities.")
    report_sub = report.add_subparsers(dest="report_command", required=True)
    report_compare = report_sub.add_parser(
        "compare",
        help="Build consolidated BABAPPA vs HyPhy comparison report from completed packs.",
    )
    report_compare.add_argument("--inputs", nargs="+", default=None, metavar="PACK_DIR")
    report_compare.add_argument("--pack", default=None, metavar="PACK_DIR")
    report_compare.add_argument("--audit", default=None, metavar="AUDIT_DIR")
    report_compare.add_argument("--outdir", required=True, metavar="DIR")
    report_compare.add_argument("--manifest", default=None, metavar="JSON")

    # audit
    audit = subparsers.add_parser("audit", help="Post-run audit utilities.")
    audit_sub = audit.add_subparsers(dest="audit_command", required=True)
    audit_orth = audit_sub.add_parser(
        "ortholog",
        help="Run ortholog audit+nullcheck diagnostics from an existing completed pack.",
    )
    audit_orth.add_argument("--pack", required=True, metavar="PACK_DIR")
    audit_orth.add_argument("--outdir", required=True, metavar="DIR")
    audit_orth.add_argument("--seed", type=int, default=1)
    audit_orth.add_argument("--null_N", type=int, default=999)
    audit_orth.add_argument("--null_G", type=int, default=2000)
    audit_orth.add_argument("--jobs", type=int, default=0)
    audit_orth.add_argument("--manifest", default=None, metavar="JSON")

    return parser


def _cmd_doctor(args: argparse.Namespace) -> int:
    from .doctor import run_doctor

    report = run_doctor(
        genes_dir=args.genes_dir,
        tree_file=args.tree_file,
        neutral_model_json=args.neutral_model_json,
        phi_spec=args.phi_spec,
        strict_taxa=args.strict_taxa,
        taxa_policy=args.taxa_policy,
        allow_stop_codons=args.allow_stop_codons,
        allow_polytomies=args.allow_polytomies,
        filter_to_multiple_of_3=args.filter_to_multiple_of_3,
        existing_model_file=args.existing_model,
        seed=args.seed,
        output_dir=args.output_dir,
    )
    print(report.render())
    return 1 if report.has_failures else 0


def _cmd_train(args: argparse.Namespace) -> int:
    from .engine import train_energy_model, train_energy_model_from_neutral_spec
    from .hash_utils import sha256_json
    from .io import read_alignments
    from .model import save_model
    from .neutral import NeutralSpec
    from .specs import load_neutral_model_spec, load_phi_spec, m0_hash, phi_hash

    phi_spec = load_phi_spec(args.phi_spec)
    phi_h = phi_hash(phi_spec)

    neutral_spec: NeutralSpec | None = None
    neutral_payload: dict[str, Any] | None = None
    if args.neutral_model_json:
        neutral_spec, neutral_payload = load_neutral_model_spec(args.neutral_model_json)
    else:
        tree_text = _load_tree_text(args.tree, args.tree_file)
        if tree_text:
            neutral_payload = {
                "schema_version": 1,
                "model_family": "GY94",
                "genetic_code_table": "standard",
                "tree_newick": tree_text,
                "kappa": float(args.kappa),
                "omega": float(args.omega),
                "codon_frequencies": _load_codon_frequencies(args.codon_frequencies),
                "frozen_values": True,
            }
            neutral_spec = NeutralSpec(
                tree_newick=tree_text,
                kappa=float(args.kappa),
                omega=float(args.omega),
                codon_frequencies=neutral_payload["codon_frequencies"],  # type: ignore[index]
            )

    if neutral_spec is not None and neutral_spec.omega != 1.0:
        raise ValueError("Training neutral model must use omega=1.")

    if args.neutral:
        alignments = read_alignments(args.neutral)
        model = train_energy_model(
            alignments,
            max_columns=args.max_columns,
            ridge=args.ridge,
            seed=args.seed,
            neutral_spec=neutral_spec,
            training_mode="alignment",
        )
    else:
        if neutral_spec is None:
            raise ValueError(
                "Provide --neutral alignments, --neutral-model-json, or --tree/--tree-file for simulation."
            )
        model = train_energy_model_from_neutral_spec(
            neutral_spec=neutral_spec,
            sim_replicates=args.sim_replicates,
            sim_length=args.sim_length,
            max_columns=args.max_columns,
            ridge=args.ridge,
            seed=args.seed,
        )

    save_model(model, args.model)
    model_h = sha256_json(model.to_dict())
    m0_h = "NA" if neutral_payload is None else m0_hash(neutral_payload)

    summary = {
        "model_path": str(Path(args.model).resolve()),
        "training_mode": model.training_mode,
        "training_alignments": model.training_alignments,
        "training_samples_n": model.training_samples,
        "max_training_length_L_train": model.max_training_length,
        "energy_family": model.energy_family,
        "energy_capacity": model.energy_capacity,
        "optimizer": model.optimizer,
        "objective": model.objective,
        "model_hash": model_h,
        "phi_hash": phi_h,
        "M0_hash": m0_h,
    }
    if args.json:
        _emit_json(summary)
    else:
        print("BABAPPA model trained.")
        for key, value in summary.items():
            print(f"{key}: {value}")

    if args.manifest:
        manifest = _base_manifest(args, "train", args.seed)
        manifest.update(
            {
                "m0_hash": m0_h,
                "phi_hash": phi_h,
                "model_hash": model_h,
                "M0_spec": neutral_payload,
                "energy_spec": {
                    "family": model.energy_family,
                    "capacity": model.energy_capacity,
                    "optimizer": model.optimizer,
                    "ridge": model.ridge,
                },
                "n": model.training_samples,
                "L_train": model.max_training_length,
                "model_path": str(Path(args.model).resolve()),
                "tails": None,
                "N": None,
                "taxa_policy": None,
            }
        )
        _write_validated_manifest(args.manifest, manifest, "training")
    return 0


def _cmd_analyze(args: argparse.Namespace) -> int:
    from .engine import analyze_alignment
    from .io import read_fasta
    from .model import load_model
    from .specs import load_phi_spec

    model = load_model(args.model)
    _ = load_phi_spec(args.phi_spec)
    alignment = read_fasta(args.alignment)
    result = analyze_alignment(
        alignment,
        model=model,
        calibration_size=args.calibration_size,
        calibration_mode=args.calibration_mode,
        tail=args.tail,
        seed=args.seed,
        seed_gene=args.seed,
        strict_scaling=args.strict_scaling,
        override_scaling=args.override_scaling,
        name=Path(args.alignment).name,
    )
    payload = result.to_dict()
    if args.output:
        _write_json_file(args.output, payload)
    if args.json:
        _emit_json(payload)
    else:
        for key, value in payload.items():
            print(f"{key}: {value}")

    if args.manifest:
        manifest = _base_manifest(args, "analyze", args.seed)
        manifest.update(
            {
                "m0_hash": result.m0_hash,
                "phi_hash": result.phi_hash,
                "model_hash": result.model_hash,
                "tail": result.tail,
                "N": result.calibration_size,
                "n": result.n_used,
                "alignment_path": str(Path(args.alignment).resolve()),
                "model_path": str(Path(args.model).resolve()),
                "override_scaling": bool(args.override_scaling),
                "strict_scaling": bool(args.strict_scaling),
                "warnings": list(result.warnings),
            }
        )
        _write_validated_manifest(args.manifest, manifest, "analysis")
    return 0


def _cmd_batch(args: argparse.Namespace) -> int:
    import pandas as pd

    from .engine import analyze_batch
    from .hash_utils import sha256_json
    from .io import read_fasta
    from .model import load_model
    from .specs import load_phi_spec, phi_hash

    model = load_model(args.model)
    _ = load_phi_spec(args.phi_spec)
    alignments = [(Path(p).stem, read_fasta(p)) for p in args.alignments]
    results = analyze_batch(
        alignments,
        model=model,
        calibration_size=args.calibration_size,
        calibration_mode=args.calibration_mode,
        tail=args.tail,
        seed=args.seed,
        strict_scaling=args.strict_scaling,
        override_scaling=args.override_scaling,
    )

    # required publishable columns + q_value
    required_cols = [
        "gene_id",
        "L",
        "D_obs",
        "mu0_hat",
        "sigma0_hat",
        "p",
        "tail",
        "N",
        "n_used",
        "model_hash",
        "phi_hash",
        "M0_hash",
        "seed_gene",
        "seed_calib_base",
        "q_value",
    ]
    out_rows: list[dict[str, Any]] = []
    for r in results:
        out_rows.append(
            {
                "gene_id": r.alignment_name,
                "L": r.gene_length,
                "D_obs": r.dispersion,
                "mu0_hat": r.mu0,
                "sigma0_hat": r.sigma0,
                "p": r.p_value,
                "tail": r.tail,
                "N": r.calibration_size,
                "n_used": r.n_used,
                "model_hash": r.model_hash,
                "phi_hash": r.phi_hash,
                "M0_hash": r.m0_hash,
                "seed_gene": r.seed_gene,
                "seed_calib_base": r.seed_calib_base,
                "q_value": r.q_value,
            }
        )
    df = pd.DataFrame(out_rows, columns=required_cols)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, sep="\t", index=False)

    if args.json_output:
        _write_json_file(args.json_output, [r.to_dict() for r in results])

    print(f"Analyzed {len(results)} genes.")
    print(f"TSV output: {out_path.resolve()}")
    if args.json_output:
        print(f"JSON output: {Path(args.json_output).resolve()}")

    if args.manifest:
        m0_h = "NA" if model.neutral_spec is None else sha256_json(model.neutral_spec.to_dict())
        phi_h = phi_hash(load_phi_spec(args.phi_spec))
        model_h = sha256_json(model.to_dict())
        manifest = _base_manifest(args, "batch", args.seed)
        manifest.update(
            {
                "m0_hash": m0_h,
                "phi_hash": phi_h,
                "model_hash": model_h,
                "tail": args.tail,
                "N": int(args.calibration_size),
                "n": int(model.training_samples),
                "n_genes": len(results),
                "model_path": str(Path(args.model).resolve()),
                "output_tsv": str(out_path.resolve()),
                "override_scaling": bool(args.override_scaling),
                "strict_scaling": bool(args.strict_scaling),
            }
        )
        _write_validated_manifest(args.manifest, manifest, "analysis")
    return 0


def _cmd_fdr(args: argparse.Namespace) -> int:
    import pandas as pd

    from .engine import benjamini_hochberg

    df = pd.read_csv(args.input, sep="\t")
    if args.p_column not in df.columns:
        raise ValueError(f"p-column '{args.p_column}' not found in {args.input}")
    pvals = [float(x) for x in df[args.p_column].to_list()]
    qvals = benjamini_hochberg(pvals)
    df[args.q_column] = qvals
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, sep="\t", index=False)
    print(f"FDR-adjusted output: {out.resolve()}")
    return 0


def _cmd_explain(args: argparse.Namespace) -> int:
    from .explain import explain_alignment
    from .io import read_fasta
    from .model import load_model

    model = load_model(args.model)
    alignment = read_fasta(args.alignment)
    payload = explain_alignment(
        model=model,
        alignment=alignment,
        calibration_size=int(args.calibration_size),
        tail=str(args.tail),
        seed=args.seed,
        out_pdf=args.out,
        out_json=args.json_out,
        window_size=int(args.window_size),
    )
    print(f"Explain report PDF: {Path(args.out).resolve()}")
    if args.json_out:
        print(f"Explain summary JSON: {Path(args.json_out).resolve()}")
    if args.manifest:
        manifest = _base_manifest(args, "explain", args.seed)
        manifest.update(
            {
                "model_path": str(Path(args.model).resolve()),
                "alignment_path": str(Path(args.alignment).resolve()),
                "out_pdf": str(Path(args.out).resolve()),
                "out_json": None if args.json_out is None else str(Path(args.json_out).resolve()),
                "tail": args.tail,
                "N": int(args.calibration_size),
                "summary": payload,
            }
        )
        _write_json_file(args.manifest, manifest)
    return 0


def _cmd_baseline_run(args: argparse.Namespace) -> int:
    from .baseline import load_dataset_json, run_baseline_for_dataset

    dataset = load_dataset_json(args.input)
    records = run_baseline_for_dataset(
        method=args.method,
        dataset=dataset,
        out_tsv=args.out,
        work_dir=args.work_dir,
        foreground_taxon=args.foreground,
        foreground_branch_label=args.foreground_branch_label,
        timeout_sec=int(args.timeout),
        container=str(args.container),
        jobs=int(args.jobs),
    )
    ok_n = sum(1 for r in records if r.status == "OK")
    fail_n = len(records) - ok_n
    print(f"Baseline method={args.method} complete. OK={ok_n} FAIL={fail_n}")
    print(f"Output TSV: {Path(args.out).resolve()}")
    if args.manifest:
        manifest = _base_manifest(args, "baseline_run", None)
        manifest.update(
            {
                "method": args.method,
                "dataset_path": str(Path(args.input).resolve()),
                "out_tsv": str(Path(args.out).resolve()),
                "n_records": len(records),
                "foreground_taxon": args.foreground,
                "foreground_branch_label": args.foreground_branch_label,
                "container": args.container,
                "timeout_sec": int(args.timeout),
                "jobs": int(args.jobs),
            }
        )
        _write_json_file(args.manifest, manifest)
    return 0


def _cmd_baseline_doctor(args: argparse.Namespace) -> int:
    from .baseline import run_baseline_doctor

    methods = tuple(x.strip().lower() for x in str(args.methods).split(",") if x.strip())
    if bool(getattr(args, "hyphy", False)):
        methods = ("busted", "relax")
    if not methods:
        methods = ("busted",)
    report = run_baseline_doctor(
        methods=methods,
        timeout_sec=int(args.timeout),
        work_dir=args.work_dir,
        pull_images=not bool(args.no_pull_images),
        container=str(args.container),
    )
    if args.json:
        _emit_json(
            {
                "docker_available": report.docker_available,
                "singularity_available": report.singularity_available,
                "timeout_sec": report.timeout_sec,
                "has_failures": report.has_failures,
                "methods": [m.to_dict() for m in report.methods],
            }
        )
    else:
        print(report.render())
    if args.manifest:
        _write_json_file(
            args.manifest,
            {
                "schema_version": 1,
                "command": "baseline_doctor",
                "methods": list(methods),
                "timeout_sec": int(args.timeout),
                "container": args.container,
                "docker_available": report.docker_available,
                "singularity_available": report.singularity_available,
                "has_failures": report.has_failures,
                "results": [m.to_dict() for m in report.methods],
            },
        )
    return 1 if report.has_failures else 0


def _print_dataset_result(source: str, payload: dict[str, object]) -> None:
    print(f"Dataset source: {source}")
    print(f"dataset_json: {payload['dataset_json']}")
    print(f"metadata_tsv: {payload['metadata_tsv']}")
    print(f"fetch_manifest: {payload['fetch_manifest']}")
    print(f"synthetic_fallback: {payload['synthetic_fallback']}")
    print(f"n_genes: {payload['n_genes']}")


def _cmd_dataset_cache(args: argparse.Namespace) -> int:
    from .dataset_real import dataset_cache_show

    _ = bool(args.show)  # explicit switch retained for backward-compatible CLI shape
    payload = dataset_cache_show()
    print(f"cache_root: {payload['cache_root']}")
    print(f"n_entries: {payload['n_entries']}")
    for row in payload["entries"]:
        print(
            "entry: "
            f"name={row['name']} files={row['n_files']} size_bytes={row['size_bytes']} path={row['path']}"
        )
    return 0


def _cmd_dataset_clear_cache(args: argparse.Namespace) -> int:
    from .dataset_real import dataset_cache_clear

    payload = dataset_cache_clear(str(args.name))
    print(f"name: {payload['name']}")
    print(f"path: {payload['path']}")
    print(f"removed: {payload['removed']}")
    return 0


def _cmd_dataset_audit(args: argparse.Namespace) -> int:
    from .benchmark.orchestrator import run_dataset_audit

    ds_arg = Path(str(args.dataset)).resolve()
    dataset_json = ds_arg / "dataset.json" if ds_arg.is_dir() else ds_arg
    if not dataset_json.exists():
        raise FileNotFoundError(
            f"dataset.json not found for --dataset input: {dataset_json}. "
            "Provide a dataset JSON path or a dataset directory containing dataset.json."
        )
    payload = run_dataset_audit(
        dataset_json=str(dataset_json),
        outdir=str(Path(str(args.out)).resolve()),
        min_len_codons=int(args.min_length_codons),
        min_taxa=int(args.min_taxa),
        require_full_completeness=bool(args.require_full_completeness),
        max_genes=(None if args.max_genes is None else int(args.max_genes)),
    )
    print(f"audit_summary_json: {Path(str(args.out)).resolve() / 'summary.json'}")
    print(f"total_candidates: {payload['total_candidates']}")
    print(f"kept: {payload['kept']}")
    print(f"dropped_by_reason: {json.dumps(payload['dropped_by_reason'], sort_keys=True)}")
    return 0


def _cmd_dataset_fetch(args: argparse.Namespace) -> int:
    from .dataset_real import fetch_hiv_dataset, fetch_orthomam_dataset, fetch_sarscov2_dataset

    source = str(args.dataset_source).lower()
    if source == "orthomam":
        outdir = (
            str(args.outdir)
            if args.outdir is not None
            else str((Path.cwd() / "data" / f"orthomam_{args.release}_{args.species_set}").resolve())
        )
        res = fetch_orthomam_dataset(
            release=str(args.release),
            species_set=str(args.species_set),
            min_length_codons=int(args.min_length),
            max_genes=int(args.max_genes),
            outdir=outdir,
            seed=int(args.seed),
            tree_file=args.tree_file,
            mode=str(args.mode),
            base_url=args.base_url,
            n_markers=int(args.max_genes),
            marker_ids_file=args.marker_ids_file,
            retries=int(args.retries),
            timeout_sec=float(args.timeout_sec),
        )
    elif source == "hiv":
        outdir = (
            str(args.outdir)
            if args.outdir is not None
            else str((Path.cwd() / "data" / f"hiv_{args.gene}_{args.subtype}").resolve())
        )
        res = fetch_hiv_dataset(
            source=str(args.source),
            gene=str(args.gene),
            subtype=str(args.subtype),
            outdir=outdir,
            recombination_policy=str(args.recombination_policy),
            tree_file=args.tree_file,
            max_ambiguous_fraction=float(args.max_ambiguous_fraction),
            alignment_id=args.alignment_id,
        )
    elif source in {"sarscov2", "sars"}:
        key = str(args.date_range).replace(":", "_")
        outdir = (
            str(args.outdir)
            if args.outdir is not None
            else str((Path.cwd() / "data" / f"sarscov2_{key}").resolve())
        )
        res = fetch_sarscov2_dataset(
            source=str(args.source),
            date_range=str(args.date_range),
            max_samples=int(args.max_samples),
            outdir=outdir,
            tree_file=args.tree_file,
            max_n_fraction=float(args.max_n_fraction),
            host=str(args.host),
            complete_only=bool(args.complete_only),
            include_cds=bool(args.include_cds),
            stratify=str(args.stratify),
            seed=int(args.seed),
            datasets_timeout_sec=int(args.datasets_timeout),
        )
    else:
        raise ValueError(f"Unsupported dataset source: {args.dataset_source}")

    _print_dataset_result(
        source,
        {
            "dataset_json": str(res.dataset_json.resolve()),
            "metadata_tsv": str(res.metadata_tsv.resolve()),
            "fetch_manifest": str(res.fetch_manifest_json.resolve()),
            "synthetic_fallback": bool(res.synthetic_fallback),
            "n_genes": int(res.n_genes),
        },
    )
    return 0


def _cmd_dataset_import(args: argparse.Namespace) -> int:
    from .dataset_real import import_hiv_dataset, import_ortholog_dataset, import_sarscov2_dataset

    source = str(args.dataset_import_source).lower()
    if source == "orthomam":
        res = import_ortholog_dataset(
            source_dir=str(args.source_dir),
            outdir=str(args.outdir),
            source_name="orthomam_import",
            release=str(args.release),
            species_set=str(args.species_set),
            min_length_codons=int(args.min_length),
            max_genes=int(args.max_genes),
            tree_file=args.tree_file,
            provenance_json=args.provenance,
            seed=int(args.seed),
        )
    elif source == "hiv":
        res = import_hiv_dataset(
            alignment=str(args.alignment),
            outdir=str(args.outdir),
            provenance_json=args.provenance,
            recombination_policy=str(args.recombination_policy),
            alignment_id=args.alignment_id,
            download_steps_text=args.download_steps_text,
            gene=str(args.gene),
            subtype=str(args.subtype),
            tree_file=args.tree_file,
            max_ambiguous_fraction=float(args.max_ambiguous_fraction),
        )
    elif source == "sarscov2":
        res = import_sarscov2_dataset(
            fasta=str(args.fasta),
            metadata_tsv=str(args.metadata),
            outdir=str(args.outdir),
            provenance_json=args.provenance,
            tree_file=args.tree_file,
            max_n_fraction=float(args.max_n_fraction),
        )
    else:
        raise ValueError(f"Unsupported dataset import source: {args.dataset_import_source}")

    _print_dataset_result(
        source,
        {
            "dataset_json": str(res.dataset_json.resolve()),
            "metadata_tsv": str(res.metadata_tsv.resolve()),
            "fetch_manifest": str(res.fetch_manifest_json.resolve()),
            "synthetic_fallback": bool(res.synthetic_fallback),
            "n_genes": int(res.n_genes),
        },
    )
    return 0


def _cmd_benchmark_track_run(args: argparse.Namespace) -> int:
    from .benchmark.orchestrator import BenchmarkRunConfig, run_benchmark_track

    mode = str(getattr(args, "mode", "pilot")).strip().lower()
    if mode == "publication" and bool(args.allow_baseline_fail):
        raise ValueError("Publication mode forbids --allow-baseline-fail.")
    if mode == "publication" and bool(args.no_plots):
        raise ValueError("Publication mode requires figures/report; --no-plots is not allowed.")

    cfg = BenchmarkRunConfig(
        track=str(args.track),
        preset=str(args.preset),
        outdir=Path(str(args.outdir)).resolve(),
        seed=int(args.seed),
        tail=str(args.tail),
        dataset_json=(None if args.dataset_json is None else str(args.dataset_json)),
        include_baselines=bool(args.include_baselines),
        include_relax=bool(args.include_relax),
        allow_baseline_fail=bool(args.allow_baseline_fail),
        allow_qc_fail=bool(args.allow_qc_fail),
        baseline_container=str(args.baseline_container),
        baseline_timeout_sec=int(args.baseline_timeout),
        write_plots=not bool(args.no_plots),
        mode=mode,
        min_taxa=(None if args.min_taxa is None else int(args.min_taxa)),
        require_full_completeness=bool(args.require_full_completeness),
        foreground_species=(None if args.foreground_species is None else str(args.foreground_species)),
        jobs=int(args.jobs),
    )
    summary = run_benchmark_track(cfg)
    print("Integrated benchmark results pack created.")
    print(f"Path: {cfg.outdir}")
    print(f"track: {cfg.track}")
    print(f"preset: {cfg.preset}")
    for key, value in summary.items():
        print(f"{key}: {value}")
    if args.manifest:
        _write_json_file(
            args.manifest,
            {
                "schema_version": 1,
                "command": "benchmark_run",
                "track": cfg.track,
                "preset": cfg.preset,
                "outdir": str(cfg.outdir),
                "seed": cfg.seed,
                "tail": cfg.tail,
                "include_baselines": cfg.include_baselines,
                "include_relax": cfg.include_relax,
                "allow_baseline_fail": cfg.allow_baseline_fail,
                "allow_qc_fail": cfg.allow_qc_fail,
                "write_plots": cfg.write_plots,
                "mode": cfg.mode,
                "min_taxa": cfg.min_taxa,
                "require_full_completeness": cfg.require_full_completeness,
                "foreground_species": cfg.foreground_species,
                "jobs": cfg.jobs,
                "baseline_container": cfg.baseline_container,
                "baseline_timeout_sec": cfg.baseline_timeout_sec,
                "summary": summary,
            },
        )
    return 0


def _cmd_benchmark_realdata(args: argparse.Namespace) -> int:
    from .benchmark.orchestrator import BenchmarkRunConfig, run_benchmark_track

    mode = str(getattr(args, "mode", "pilot")).strip().lower()
    if mode == "publication" and bool(args.allow_baseline_fail):
        raise ValueError("Publication mode forbids --allow-baseline-fail.")
    if mode == "publication" and bool(args.allow_qc_fail):
        raise ValueError("Publication mode forbids --allow-qc-fail.")

    preset_map = {
        "ortholog_small8_full": ("ortholog", "ORTHOMAM_SMALL8_FULL"),
        "hiv_env_b_full": ("viral", "HIV_ENV_B_FULL"),
        "sars_2020_full": ("viral", "SARS_2020_FULL"),
        "ortholog_real_v12": ("ortholog", "ORTHOMAM_SMALL8_FULL"),
        "hiv_env_b_real": ("viral", "HIV_ENV_B_FULL"),
        "sars_cov2_real": ("viral", "SARS_2020_FULL"),
    }
    if str(args.preset) not in preset_map:
        raise ValueError(f"Unsupported realdata preset: {args.preset}")
    track, mapped_preset = preset_map[str(args.preset)]

    data_arg = Path(str(args.data)).resolve()
    dataset_json = data_arg / "dataset.json" if data_arg.is_dir() else data_arg
    if not dataset_json.exists():
        raise FileNotFoundError(
            f"dataset.json not found for --data input: {dataset_json}. "
            "Provide a dataset JSON path or a dataset directory containing dataset.json."
        )

    outdir = (
        Path(str(args.outdir)).resolve()
        if args.outdir is not None
        else (Path.cwd() / "results" / str(args.preset)).resolve()
    )
    write_plots = not (bool(args.no_plots) or bool(args.results_only))
    if mode == "publication" and not write_plots:
        raise ValueError("Publication mode requires figures/report; --no-plots/--results-only is not allowed.")
    if mode == "publication" and int(args.N) < 999:
        raise ValueError(f"Publication mode requires N >= 999; got N={int(args.N)}.")

    cfg = BenchmarkRunConfig(
        track=track,
        preset=mapped_preset,
        outdir=outdir,
        seed=int(args.seed),
        tail=str(args.tail),
        dataset_json=str(dataset_json),
        include_baselines=True,
        include_relax=bool(args.include_relax),
        allow_baseline_fail=bool(args.allow_baseline_fail),
        baseline_container=str(args.baseline_container),
        baseline_timeout_sec=int(args.baseline_timeout),
        write_plots=write_plots,
        require_real_data=True,
        allow_qc_fail=bool(args.allow_qc_fail),
        calibration_size_override=int(args.N),
        empirical_profile=str(args.preset),
        mode=mode,
        min_taxa=(None if args.min_taxa is None else int(args.min_taxa)),
        require_full_completeness=bool(args.require_full_completeness),
        foreground_species=(None if args.foreground_species is None else str(args.foreground_species)),
        jobs=int(args.jobs),
    )
    summary = run_benchmark_track(cfg)
    print("Empirical real-data benchmark pack created.")
    print(f"Path: {cfg.outdir}")
    print(f"track: {cfg.track}")
    print(f"preset: {args.preset}")
    for key, value in summary.items():
        print(f"{key}: {value}")
    if args.manifest:
        _write_json_file(
            args.manifest,
            {
                "schema_version": 1,
                "command": "benchmark_realdata",
                "preset": str(args.preset),
                "mapped_track": cfg.track,
                "mapped_preset": cfg.preset,
                "dataset_json": str(dataset_json),
                "outdir": str(cfg.outdir),
                "seed": cfg.seed,
                "tail": cfg.tail,
                "N": cfg.calibration_size_override,
                "allow_qc_fail": cfg.allow_qc_fail,
                "allow_baseline_fail": cfg.allow_baseline_fail,
                "write_plots": cfg.write_plots,
                "mode": cfg.mode,
                "min_taxa": cfg.min_taxa,
                "require_full_completeness": cfg.require_full_completeness,
                "foreground_species": cfg.foreground_species,
                "jobs": cfg.jobs,
                "summary": summary,
            },
        )
    return 0


def _cmd_benchmark_track_rebuild(args: argparse.Namespace) -> int:
    from .benchmark.orchestrator import rebuild_benchmark_pack

    outdir = Path(str(args.outdir)).resolve()
    rebuild_benchmark_pack(outdir, write_plots=not bool(args.no_plots))
    print(f"Rebuilt benchmark artifacts from raw files in {outdir}")
    return 0


def _cmd_benchmark_resume(args: argparse.Namespace) -> int:
    from .benchmark.orchestrator import resume_benchmark_track

    outdir = Path(str(args.outdir)).resolve()
    summary = resume_benchmark_track(outdir, write_plots=not bool(args.no_plots))
    print("Resumed benchmark run.")
    print(f"Path: {outdir}")
    for key, value in summary.items():
        print(f"{key}: {value}")
    if args.manifest:
        _write_json_file(
            args.manifest,
            {
                "schema_version": 1,
                "command": "benchmark_resume",
                "outdir": str(outdir),
                "write_plots": not bool(args.no_plots),
                "summary": summary,
            },
        )
    return 0


def _cmd_reproduce(args: argparse.Namespace) -> int:
    import os
    import time

    from .benchmark.orchestrator import BenchmarkRunConfig, run_benchmark_track, run_dataset_audit
    from .report_compare import run_compare_report

    mode = str(getattr(args, "mode", "pilot")).strip().lower()
    if mode == "publication" and bool(args.allow_baseline_fail):
        raise ValueError("Publication mode forbids --allow-baseline-fail.")
    if mode == "publication" and bool(args.fast):
        raise ValueError("Publication mode forbids --fast.")
    if mode == "publication" and bool(args.no_plots):
        raise ValueError("Publication mode requires figures/report; --no-plots is not allowed.")

    outdir = Path(str(args.outdir)).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    if bool(args.fast):
        os.environ["BABAPPA_BENCHMARK_FAST"] = "1"
    paper_mode = "paper_ready" if bool(args.paper_ready) else str(args.paper)

    runs: list[dict[str, Any]] = []
    if mode == "publication":
        paper_mode = "publication"
        if args.dataset_json_ortholog is None:
            raise ValueError("Publication reproduce requires --dataset-json-ortholog.")
        if args.dataset_json_viral_hiv is None:
            raise ValueError("Publication reproduce requires --dataset-json-viral-hiv.")
        if args.dataset_json_viral_sars is None:
            raise ValueError("Publication reproduce requires --dataset-json-viral-sars.")
        audit_summary = run_dataset_audit(
            dataset_json=str(args.dataset_json_ortholog),
            outdir=str(outdir / "ortholog_audit"),
            min_len_codons=300,
            min_taxa=6,
            require_full_completeness=False,
            max_genes=None,
        )
        if int(audit_summary.get("kept", 0)) < 300:
            raise ValueError(
                "Publication reproduce audit failed: ortholog keep count < 300 "
                f"(kept={int(audit_summary.get('kept', 0))})."
            )
        plan = [
            {
                "name": "ortholog_small8_full",
                "track": "ortholog",
                "preset": "ORTHOMAM_SMALL8_FULL",
                "include_baselines": True,
                "include_relax": False,
                "dataset_json": args.dataset_json_ortholog,
            },
            {
                "name": "viral_hiv_env_b_full",
                "track": "viral",
                "preset": "HIV_ENV_B_FULL",
                "include_baselines": True,
                "include_relax": bool(args.foreground_species is not None and str(args.foreground_species).strip()),
                "dataset_json": args.dataset_json_viral_hiv,
            },
            {
                "name": "viral_sars_2020_full",
                "track": "viral",
                "preset": "SARS_2020_FULL",
                "include_baselines": True,
                "include_relax": False,
                "dataset_json": args.dataset_json_viral_sars,
            },
        ]
    elif paper_mode == "paper_ready":
        plan = [
            {
                "name": "simulation_null_full",
                "track": "simulation",
                "preset": "simulation_null_full",
                "include_baselines": False,
                "include_relax": False,
                "dataset_json": None,
            },
            {
                "name": "simulation_power_full",
                "track": "simulation",
                "preset": "simulation_power_full",
                "include_baselines": True,
                "include_relax": True,
                "dataset_json": None,
            },
            {
                "name": "ortholog_small8_full",
                "track": "ortholog",
                "preset": "ORTHOMAM_SMALL8_FULL",
                "include_baselines": True,
                "include_relax": False,
                "dataset_json": args.dataset_json_ortholog,
            },
            {
                "name": "viral_hiv_env_b_full",
                "track": "viral",
                "preset": "HIV_ENV_B_FULL",
                "include_baselines": True,
                "include_relax": True,
                "dataset_json": args.dataset_json_viral_hiv,
            },
        ]
        if bool(args.include_sars):
            plan.append(
                {
                    "name": "viral_sars_2020_full",
                    "track": "viral",
                    "preset": "SARS_2020_FULL",
                    "include_baselines": True,
                    "include_relax": False,
                    "dataset_json": args.dataset_json_viral_sars,
                }
            )
    else:
        plan = [
            {
                "name": "simulation_null_paper",
                "track": "simulation",
                "preset": "simulation_null_paper",
                "include_baselines": False,
                "include_relax": False,
                "dataset_json": None,
            },
            {
                "name": "simulation_power_paper",
                "track": "simulation",
                "preset": "simulation_power_paper",
                "include_baselines": True,
                "include_relax": True,
                "dataset_json": None,
            },
            {
                "name": "ortholog_small8_paper",
                "track": "ortholog",
                "preset": "ORTHOMAM_SMALL8_PAPER",
                "include_baselines": True,
                "include_relax": False,
                "dataset_json": args.dataset_json_ortholog,
            },
            {
                "name": "viral_hiv_paper",
                "track": "viral",
                "preset": "HIV_ENV_B_PAPER",
                "include_baselines": True,
                "include_relax": True,
                "dataset_json": args.dataset_json_viral_hiv,
            },
        ]
        if bool(args.include_sars):
            plan.append(
                {
                    "name": "viral_sars_paper",
                    "track": "viral",
                    "preset": "SARS_2020_PAPER",
                    "include_baselines": True,
                    "include_relax": False,
                    "dataset_json": args.dataset_json_viral_sars,
                }
            )

    for item in plan:
        run_out = outdir / item["name"]
        started = time.perf_counter()
        entry: dict[str, Any] = {
            "name": item["name"],
            "track": item["track"],
            "preset": item["preset"],
            "outdir": str(run_out),
            "status": "FAIL",
            "summary": None,
            "error": None,
        }
        try:
            cfg = BenchmarkRunConfig(
                track=str(item["track"]),
                preset=str(item["preset"]),
                outdir=run_out,
                seed=int(args.seed),
                tail=str(args.tail),
                dataset_json=(None if item["dataset_json"] is None else str(item["dataset_json"])),
                include_baselines=bool(item["include_baselines"]),
                include_relax=bool(item["include_relax"]),
                allow_baseline_fail=bool(args.allow_baseline_fail),
                baseline_container=str(args.baseline_container),
                baseline_timeout_sec=int(args.baseline_timeout),
                write_plots=not bool(args.no_plots),
                require_real_data=bool(mode == "publication" and str(item["track"]) in {"ortholog", "viral"}),
                mode=mode,
                foreground_species=(None if args.foreground_species is None else str(args.foreground_species)),
                jobs=int(args.jobs),
            )
            summary = run_benchmark_track(cfg)
            entry["status"] = "OK"
            entry["summary"] = summary
        except Exception as exc:
            entry["error"] = str(exc)
        entry["runtime_sec"] = float(time.perf_counter() - started)
        runs.append(entry)

    fail_n = int(sum(1 for r in runs if r["status"] != "OK"))
    compare_manifest: dict[str, Any] | None = None
    if mode == "publication" and fail_n == 0:
        compare_manifest = run_compare_report(
            inputs=[
                str(outdir / "ortholog_small8_full"),
                str(outdir / "viral_hiv_env_b_full"),
                str(outdir / "viral_sars_2020_full"),
            ],
            outdir=str(outdir / "compare_report"),
        )
    payload = {
        "schema_version": 1,
        "command": "reproduce",
        "mode": mode,
        "paper": paper_mode,
        "seed": int(args.seed),
        "tail": args.tail,
        "allow_baseline_fail": bool(args.allow_baseline_fail),
        "write_plots": not bool(args.no_plots),
        "baseline_container": args.baseline_container,
        "baseline_timeout_sec": int(args.baseline_timeout),
        "fast": bool(args.fast),
        "runs": runs,
        "n_runs": len(runs),
        "n_fail": fail_n,
        "compare_manifest": compare_manifest,
    }
    _write_json_file(outdir / "reproduce_summary.json", payload)
    print(f"Reproduce summary: {outdir / 'reproduce_summary.json'}")
    if args.manifest:
        _write_json_file(args.manifest, payload)
    return 0 if fail_n == 0 else 1


def _cmd_report_compare(args: argparse.Namespace) -> int:
    if args.pack and args.audit:
        from .report_ortholog_compare import run_ortholog_compare_report

        manifest = run_ortholog_compare_report(
            pack=str(args.pack),
            audit=str(args.audit),
            outdir=str(args.outdir),
        )
        print(f"compare_summary_tsv: {Path(str(args.outdir)).resolve() / 'tables' / 'summary.tsv'}")
        print(f"compare_report_pdf: {Path(str(args.outdir)).resolve() / 'report' / 'comparison_report.pdf'}")
    else:
        from .report_compare import run_compare_report

        if not args.inputs:
            raise ValueError("Use either --inputs <PACK...> or --pack <PACK_DIR> --audit <AUDIT_DIR>.")
        manifest = run_compare_report(
            inputs=[str(x) for x in args.inputs],
            outdir=str(args.outdir),
        )
        print(f"compare_summary_tsv: {Path(str(args.outdir)).resolve() / 'tables' / 'summary.tsv'}")
        print(f"compare_report_pdf: {Path(str(args.outdir)).resolve() / 'report' / 'report.pdf'}")
    if args.manifest:
        _write_json_file(args.manifest, manifest)
    return 0


def _cmd_audit_ortholog(args: argparse.Namespace) -> int:
    from .audit_ortholog import run_ortholog_audit

    manifest = run_ortholog_audit(
        pack=str(args.pack),
        outdir=str(args.outdir),
        seed=int(args.seed),
        null_N=int(args.null_N),
        null_G=int(args.null_G),
        jobs=int(args.jobs),
    )
    print(f"audit_summary_tsv: {Path(str(args.outdir)).resolve() / 'tables' / 'null_size.tsv'}")
    print(f"audit_report_pdf: {Path(str(args.outdir)).resolve() / 'report' / 'audit_report.pdf'}")
    if args.manifest:
        _write_json_file(args.manifest, manifest)
    if bool(manifest.get("severe_inflation", False)):
        print("audit_status: FAIL (severe inflation detected)")
        return 2
    if bool(manifest.get("calibration_fail", False)):
        print("audit_status: FAIL (calibration size check failed)")
        return 2
    if bool(manifest.get("frozen_energy_invariant_fail", False)):
        print("audit_status: FAIL (frozen-energy invariant violation)")
        return 2
    if bool(manifest.get("testable_set_mismatch", False)):
        print("audit_status: FAIL (testable set mismatch)")
        return 2
    n_units = int(manifest.get("n_units", 0))
    n_success_babappa = int(manifest.get("n_success_babappa", 0))
    n_success_busted = int(manifest.get("n_success_busted", 0))
    if n_units > 0 and n_success_babappa != n_units:
        print(f"audit_status: FAIL (babappa success {n_success_babappa}/{n_units})")
        return 2
    if n_units > 0 and n_success_busted != n_units:
        print(f"audit_status: FAIL (busted success {n_success_busted}/{n_units})")
        return 2
    print("audit_status: PASS")
    return 0


def _cmd_validate_null(args: argparse.Namespace) -> int:
    from .validation_battery import run_validate_null

    manifest = run_validate_null(
        config_path=str(args.config),
        outdir=str(args.out),
        G=int(args.G),
        N=int(args.N),
    )
    print(f"results_tsv: {Path(str(args.out)).resolve() / 'raw' / 'results.tsv'}")
    print(f"report_pdf: {Path(str(args.out)).resolve() / 'report' / 'report.pdf'}")
    if args.manifest:
        _write_json_file(args.manifest, manifest)
    return 0


def _cmd_validate_freeze(args: argparse.Namespace) -> int:
    from .validation_battery import run_validate_freeze

    manifest = run_validate_freeze(
        config_path=str(args.config),
        outdir=str(args.out),
    )
    print(f"results_tsv: {Path(str(args.out)).resolve() / 'raw' / 'results.tsv'}")
    print(f"report_pdf: {Path(str(args.out)).resolve() / 'report' / 'report.pdf'}")
    if args.manifest:
        _write_json_file(args.manifest, manifest)
    return 0


def _cmd_pmin_audit(args: argparse.Namespace) -> int:
    from .validation_battery import run_pmin_audit

    manifest = run_pmin_audit(
        gene_set=str(args.gene_set),
        config_path=str(args.config),
        outdir=str(args.out),
        N=int(args.N),
    )
    print(f"results_tsv: {Path(str(args.out)).resolve() / 'raw' / 'results.tsv'}")
    print(f"report_pdf: {Path(str(args.out)).resolve() / 'report' / 'report.pdf'}")
    if args.manifest:
        _write_json_file(args.manifest, manifest)
    return 0


def _cmd_bias_audit(args: argparse.Namespace) -> int:
    from .validation_battery import run_bias_audit

    manifest = run_bias_audit(
        gene_set=str(args.gene_set),
        config_path=str(args.config),
        outdir=str(args.out),
        N=int(args.N),
    )
    print(f"results_tsv: {Path(str(args.out)).resolve() / 'raw' / 'results.tsv'}")
    print(f"report_pdf: {Path(str(args.out)).resolve() / 'report' / 'report.pdf'}")
    if args.manifest:
        _write_json_file(args.manifest, manifest)
    return 0


def _cmd_sensitivity(args: argparse.Namespace) -> int:
    from .validation_battery import run_sensitivity

    manifest = run_sensitivity(
        gene_set=str(args.gene_set),
        grid_path=str(args.grid),
        config_path=str(args.config),
        outdir=str(args.out),
    )
    print(f"results_tsv: {Path(str(args.out)).resolve() / 'raw' / 'results.tsv'}")
    print(f"report_pdf: {Path(str(args.out)).resolve() / 'report' / 'report.pdf'}")
    if args.manifest:
        _write_json_file(args.manifest, manifest)
    return 0


def _cmd_benchmark_legacy(args: argparse.Namespace) -> int:
    from .baseline import run_baseline_doctor
    from .benchmark_pack import BenchmarkPackConfig, rebuild_from_raw, run_benchmark_pack

    if not args.results_pack:
        raise ValueError(
            "Legacy benchmark mode requires --results-pack DIR. "
            "Or use: babappa benchmark run --track ... --preset ... --outdir ..."
        )
    pack_dir = Path(args.results_pack).resolve()
    if args.rebuild_only:
        rebuild_from_raw(pack_dir, write_figures=not bool(args.no_plots))
        print(f"Rebuilt figures/tables from raw files in {pack_dir}")
        return 0

    L_grid = _parse_int_list(args.L_grid)
    taxa_grid = _parse_int_list(args.taxa_grid)
    N_grid = _parse_int_list(args.N_grid)
    n_null_genes = int(args.n_null_genes)
    n_alt_genes = int(args.n_alt_genes)
    run_null = True
    run_power = True
    include_baselines = bool(args.include_baselines)
    include_relax = bool(args.include_relax)
    allow_baseline_fail = bool(args.allow_baseline_fail)
    power_families = tuple(
        x.strip() for x in str(args.power_families).split(",") if x.strip()
    )
    if not power_families:
        power_families = (
            "A_distributed_weak_selection",
            "B_constraint_shift",
            "C_episodic_branch_site_surrogate",
            "D_misspecification_stress",
        )
    family_a_pi: tuple[float, ...] = (0.05, 0.1, 0.2)
    family_a_omega: tuple[float, ...] = (1.1, 1.2, 1.5)

    if args.preset == "null_smoke":
        L_grid = [300, 600]
        taxa_grid = [8]
        N_grid = [199, 999]
        n_null_genes = 1000
        n_alt_genes = 0
        run_null = True
        run_power = False
        include_baselines = False
    elif args.preset == "power_distributed_smoke":
        L_grid = [600]
        taxa_grid = [8]
        N_grid = [199]
        n_null_genes = 0
        n_alt_genes = 500
        run_null = False
        run_power = True
        include_baselines = True
        power_families = ("A_distributed_weak_selection",)
        family_a_pi = (0.1, 0.2)
        family_a_omega = (1.2, 1.5)

    if n_null_genes <= 0:
        run_null = False
    if n_alt_genes <= 0:
        run_power = False

    baseline_methods = tuple(
        x.strip().lower() for x in str(args.baseline_methods).split(",") if x.strip()
    )
    if not baseline_methods:
        baseline_methods = ("busted",)
    if include_baselines:
        methods = list(baseline_methods)
        if include_relax and "relax" not in methods:
            methods.append("relax")
        doctor_report = run_baseline_doctor(
            methods=tuple(methods),
            timeout_sec=5,
            work_dir=pack_dir / "raw" / "baseline_doctor",
            pull_images=False,
        )
        (pack_dir / "logs").mkdir(parents=True, exist_ok=True)
        (pack_dir / "logs" / "baseline_doctor_report.txt").write_text(
            doctor_report.render() + "\n",
            encoding="utf-8",
        )
        if doctor_report.has_failures and not allow_baseline_fail:
            raise ValueError(
                "Baseline doctor failed. Run `babappa baseline doctor` and fix backends, "
                "or re-run with --allow-baseline-fail."
            )

    config = BenchmarkPackConfig(
        outdir=pack_dir,
        seed=int(args.seed),
        tail=args.tail,
        L_grid=L_grid,
        taxa_grid=taxa_grid,
        N_grid=N_grid,
        n_null_genes=n_null_genes,
        n_alt_genes=n_alt_genes,
        training_replicates=int(args.training_replicates),
        training_length_nt=int(args.training_length),
        kappa=float(args.kappa),
        run_null=run_null,
        run_power=run_power,
        include_baselines=include_baselines,
        include_relax=include_relax,
        write_figures=not bool(args.no_plots),
        baseline_methods=baseline_methods,
        power_families=power_families,
        family_a_pi=family_a_pi,
        family_a_omega=family_a_omega,
    )
    summary = run_benchmark_pack(config)
    print("Benchmark results_pack created.")
    print(f"Path: {pack_dir}")
    for k, v in summary.items():
        print(f"{k}: {v}")

    if args.manifest:
        manifest = _base_manifest(args, "benchmark", int(args.seed))
        manifest.update(
            {
                "grid": {
                    "L_grid": config.L_grid,
                    "taxa_grid": config.taxa_grid,
                    "N_grid": config.N_grid,
                    "n_null_genes": config.n_null_genes,
                    "n_alt_genes": config.n_alt_genes,
                },
                "pack_dir": str(pack_dir),
                "tail": args.tail,
                "preset": args.preset,
                "run_null": run_null,
                "run_power": run_power,
                "include_baselines": include_baselines,
                "write_plots": not bool(args.no_plots),
                "allow_baseline_fail": allow_baseline_fail,
                "baseline_methods": list(baseline_methods),
            }
        )
        _write_validated_manifest(args.manifest, manifest, "benchmark")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    args._argv = list(argv if argv is not None else sys.argv[1:])
    try:
        if args.command == "doctor":
            return _cmd_doctor(args)
        if args.command == "train":
            return _cmd_train(args)
        if args.command == "analyze":
            return _cmd_analyze(args)
        if args.command == "batch":
            return _cmd_batch(args)
        if args.command == "fdr":
            return _cmd_fdr(args)
        if args.command == "explain":
            return _cmd_explain(args)
        if args.command == "baseline":
            if args.baseline_command == "run":
                return _cmd_baseline_run(args)
            if args.baseline_command == "doctor":
                return _cmd_baseline_doctor(args)
            raise ValueError(f"Unknown baseline subcommand: {args.baseline_command}")
        if args.command == "dataset":
            if args.dataset_command == "cache":
                return _cmd_dataset_cache(args)
            if args.dataset_command == "clear-cache":
                return _cmd_dataset_clear_cache(args)
            if args.dataset_command == "audit":
                return _cmd_dataset_audit(args)
            if args.dataset_command == "fetch":
                return _cmd_dataset_fetch(args)
            if args.dataset_command == "import":
                return _cmd_dataset_import(args)
            raise ValueError(f"Unknown dataset subcommand: {args.dataset_command}")
        if args.command == "validate-null":
            return _cmd_validate_null(args)
        if args.command == "validate-freeze":
            return _cmd_validate_freeze(args)
        if args.command == "pmin-audit":
            return _cmd_pmin_audit(args)
        if args.command == "bias-audit":
            return _cmd_bias_audit(args)
        if args.command == "sensitivity":
            return _cmd_sensitivity(args)
        if args.command == "report":
            if args.report_command == "compare":
                return _cmd_report_compare(args)
            raise ValueError(f"Unknown report subcommand: {args.report_command}")
        if args.command == "audit":
            if args.audit_command == "ortholog":
                return _cmd_audit_ortholog(args)
            raise ValueError(f"Unknown audit subcommand: {args.audit_command}")
        if args.command == "reproduce":
            return _cmd_reproduce(args)
        if args.command == "benchmark":
            if getattr(args, "benchmark_command", None) == "run":
                return _cmd_benchmark_track_run(args)
            if getattr(args, "benchmark_command", None) == "rebuild":
                return _cmd_benchmark_track_rebuild(args)
            if getattr(args, "benchmark_command", None) == "resume":
                return _cmd_benchmark_resume(args)
            if getattr(args, "benchmark_command", None) == "realdata":
                return _cmd_benchmark_realdata(args)
            return _cmd_benchmark_legacy(args)
    except Exception as exc:  # pragma: no cover
        parser.exit(status=2, message=f"error: {exc}\n")
    parser.exit(status=2, message="error: unknown command\n")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
