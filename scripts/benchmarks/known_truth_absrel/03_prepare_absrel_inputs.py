#!/usr/bin/env python
"""Prepare aBSREL input folders from the known-truth benchmark manifest."""

from __future__ import annotations

import argparse
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import config_jobs, read_config, read_tsv, resolve_outdir, write_json, write_tsv


def _mark_foreground(tree: str, foreground: str) -> str:
    pattern = re.compile(rf"(?<![A-Za-z0-9_.-])({re.escape(foreground)})(?=[:),;])")
    return pattern.sub(r"\1{Foreground}", tree, count=1)


def _prepare_one(row: Dict[str, str], *, benchmark_dir: Path, inputs: Path, json_dir: Path, force: bool) -> Dict[str, str]:
    family_id = row["family_id"]
    family_dir = inputs / family_id
    family_dir.mkdir(parents=True, exist_ok=True)
    alignment = family_dir / "alignment.fasta"
    tree_out = family_dir / "tree.nwk"
    status_path = family_dir / "prepare_status.json"
    if not force and alignment.exists() and tree_out.exists() and alignment.stat().st_size > 0 and tree_out.stat().st_size > 0:
        write_json(status_path, {"family_id": family_id, "status": "skipped_completed"})
    else:
        alignment.write_text((benchmark_dir / row["codon_fasta"]).read_text(encoding="utf-8"), encoding="utf-8")
        tree_text = (benchmark_dir / row["tree"]).read_text(encoding="utf-8")
        tree_out.write_text(_mark_foreground(tree_text, row["foreground"]), encoding="utf-8")
        write_json(status_path, {"family_id": family_id, "status": "prepared"})
    return {
        "family_id": family_id,
        "alignment": str(alignment),
        "tree": str(tree_out),
        "branches": "Foreground",
        "output_json": str(json_dir / f"{family_id}.absrel.json"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--benchmark-dir", default=None)
    parser.add_argument("--jobs", type=int, default=None)
    args = parser.parse_args()

    config = read_config(args.config)
    benchmark_dir = resolve_outdir(config, args.benchmark_dir)
    jobs = max(1, int(args.jobs or config_jobs(config, "prepare", 1)))
    force = os.environ.get("BABAPPA_FORCE", "").lower() in {"1", "true", "yes", "y"}
    rows = read_tsv(benchmark_dir / "manifest.tsv")
    if not rows:
        raise SystemExit(f"missing manifest: {benchmark_dir / 'manifest.tsv'}")
    inputs = benchmark_dir / "absrel_inputs"
    json_dir = benchmark_dir / "absrel_json"
    inputs.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)
    out_rows: List[Dict[str, str]] = []
    print(f"Preparing aBSREL input jobs={jobs} families={len(rows)} force={force}")
    completed = 0
    with ThreadPoolExecutor(max_workers=jobs) as executor:
        futures = [executor.submit(_prepare_one, row, benchmark_dir=benchmark_dir, inputs=inputs, json_dir=json_dir, force=force) for row in rows]
        for future in as_completed(futures):
            out_rows.append(future.result())
            completed += 1
            if completed == 1 or completed % 25 == 0 or completed == len(rows):
                print(f"Prepare progress {completed}/{len(rows)} families")
    out_rows = sorted(out_rows, key=lambda row: row["family_id"])
    write_tsv(benchmark_dir / "absrel_input_manifest.tsv", out_rows, ["family_id", "alignment", "tree", "branches", "output_json"])
    print(f"Prepared {len(out_rows)} aBSREL input families")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
