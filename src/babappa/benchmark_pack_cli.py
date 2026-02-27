from __future__ import annotations

import argparse
from pathlib import Path

from .benchmark_pack import BenchmarkPackConfig, rebuild_from_raw, run_benchmark_pack


def _parse_int_list(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="babappa.benchmark_pack_cli")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run")
    p_run.add_argument("--outdir", required=True)
    p_run.add_argument("--seed", type=int, default=123)
    p_run.add_argument("--tail", choices=["right", "left", "two-sided"], default="right")
    p_run.add_argument("--L-grid", default="150,300,600,1200")
    p_run.add_argument("--taxa-grid", default="8,16")
    p_run.add_argument("--N-grid", default="199,999")
    p_run.add_argument("--n-null-genes", type=int, default=200)
    p_run.add_argument("--n-alt-genes", type=int, default=120)
    p_run.add_argument("--training-replicates", type=int, default=200)
    p_run.add_argument("--training-length", type=int, default=1200)
    p_run.add_argument("--kappa", type=float, default=2.0)

    p_rebuild = sub.add_parser("rebuild")
    p_rebuild.add_argument("--pack-dir", required=True)

    args = parser.parse_args(argv)
    if args.cmd == "rebuild":
        rebuild_from_raw(args.pack_dir)
        return 0

    config = BenchmarkPackConfig(
        outdir=Path(args.outdir),
        seed=args.seed,
        tail=args.tail,
        L_grid=_parse_int_list(args.L_grid),
        taxa_grid=_parse_int_list(args.taxa_grid),
        N_grid=_parse_int_list(args.N_grid),
        n_null_genes=args.n_null_genes,
        n_alt_genes=args.n_alt_genes,
        training_replicates=args.training_replicates,
        training_length_nt=args.training_length,
        kappa=args.kappa,
    )
    run_benchmark_pack(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
