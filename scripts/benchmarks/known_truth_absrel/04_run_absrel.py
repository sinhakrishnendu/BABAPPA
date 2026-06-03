#!/usr/bin/env python
"""Run or parse HyPhy aBSREL outputs for the known-truth benchmark."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import read_config, read_tsv, repo_root, resolve_outdir, run_command, write_tsv


def _official_positive_count(payload: Dict[str, Any]) -> int | None:
    try:
        value = payload["test results"]["positive test results"]
    except (KeyError, TypeError):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_one(row: Dict[str, str]) -> Dict[str, str]:
    json_path = Path(row["output_json"])
    if not json_path.exists():
        return {"family_id": row["family_id"], "status": "pending_not_run", "positive_count": "NA", "call": "NA", "p_value": "NA", "notes": "aBSREL JSON absent"}
    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {"family_id": row["family_id"], "status": "parse_failed", "positive_count": "NA", "call": "NA", "p_value": "NA", "notes": str(exc)}
    positive_count = _official_positive_count(payload)
    if positive_count is None:
        return {"family_id": row["family_id"], "status": "warning_missing_official_field", "positive_count": "NA", "call": "NA", "p_value": "NA", "notes": "missing test results -> positive test results"}
    return {"family_id": row["family_id"], "status": "ok", "positive_count": str(positive_count), "call": str(int(positive_count > 0)), "p_value": "NA", "notes": "official family-level aBSREL field"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--benchmark-dir", default=None)
    parser.add_argument("--parse-only", action="store_true")
    parser.add_argument("--continue-on-failure", action="store_true")
    args = parser.parse_args()

    config = read_config(args.config)
    benchmark_dir = resolve_outdir(config, args.benchmark_dir)
    manifest = read_tsv(benchmark_dir / "absrel_input_manifest.tsv")
    if not manifest:
        raise SystemExit("missing absrel_input_manifest.tsv; run 03_prepare_absrel_inputs.py first")
    failures: List[Dict[str, str]] = []
    if not args.parse_only:
        if shutil.which("hyphy") is None:
            failures.append({"family_id": "ALL", "status": "tool_missing", "error": "hyphy not found on PATH"})
        else:
            for row in manifest:
                cmd = ["hyphy", "absrel", "--alignment", row["alignment"], "--tree", row["tree"], "--branches", row["branches"], "--output", row["output_json"]]
                completed = run_command(cmd, cwd=repo_root())
                log_dir = benchmark_dir / "absrel_logs" / row["family_id"]
                log_dir.mkdir(parents=True, exist_ok=True)
                (log_dir / "stdout.txt").write_text(completed.stdout, encoding="utf-8")
                (log_dir / "stderr.txt").write_text(completed.stderr, encoding="utf-8")
                if completed.returncode != 0:
                    failures.append({"family_id": row["family_id"], "status": "failed", "error": completed.stderr.strip()[-500:]})
                    if not args.continue_on_failure:
                        break
    rows = [_parse_one(row) for row in manifest]
    write_tsv(benchmark_dir / "absrel_results.tsv", rows, ["family_id", "status", "positive_count", "call", "p_value", "notes"])
    write_tsv(benchmark_dir / "absrel_failures.tsv", failures, ["family_id", "status", "error"])
    print(f"Wrote aBSREL benchmark results: {benchmark_dir / 'absrel_results.tsv'}")
    return 0 if args.continue_on_failure or not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
