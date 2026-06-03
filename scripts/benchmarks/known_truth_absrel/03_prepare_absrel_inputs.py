#!/usr/bin/env python
"""Prepare aBSREL input folders from the known-truth benchmark manifest."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import read_config, read_tsv, resolve_outdir, write_tsv


def _mark_foreground(tree: str, foreground: str) -> str:
    pattern = re.compile(rf"(?<![A-Za-z0-9_.-])({re.escape(foreground)})(?=[:),;])")
    return pattern.sub(r"\1{Foreground}", tree, count=1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--benchmark-dir", default=None)
    args = parser.parse_args()

    config = read_config(args.config)
    benchmark_dir = resolve_outdir(config, args.benchmark_dir)
    rows = read_tsv(benchmark_dir / "manifest.tsv")
    if not rows:
        raise SystemExit(f"missing manifest: {benchmark_dir / 'manifest.tsv'}")
    inputs = benchmark_dir / "absrel_inputs"
    json_dir = benchmark_dir / "absrel_json"
    inputs.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)
    out_rows: List[Dict[str, str]] = []
    for row in rows:
        family_id = row["family_id"]
        family_dir = inputs / family_id
        family_dir.mkdir(parents=True, exist_ok=True)
        alignment = family_dir / "alignment.fasta"
        tree_out = family_dir / "tree.nwk"
        alignment.write_text((benchmark_dir / row["codon_fasta"]).read_text(encoding="utf-8"), encoding="utf-8")
        tree_text = (benchmark_dir / row["tree"]).read_text(encoding="utf-8")
        tree_out.write_text(_mark_foreground(tree_text, row["foreground"]), encoding="utf-8")
        out_rows.append(
            {
                "family_id": family_id,
                "alignment": str(alignment),
                "tree": str(tree_out),
                "branches": "Foreground",
                "output_json": str(json_dir / f"{family_id}.absrel.json"),
            }
        )
    write_tsv(benchmark_dir / "absrel_input_manifest.tsv", out_rows, ["family_id", "alignment", "tree", "branches", "output_json"])
    print(f"Prepared {len(out_rows)} aBSREL input families")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
