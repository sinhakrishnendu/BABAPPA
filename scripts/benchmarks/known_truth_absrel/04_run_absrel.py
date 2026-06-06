#!/usr/bin/env python
"""Run or parse HyPhy aBSREL outputs for the known-truth benchmark."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import config_jobs, read_config, read_tsv, repo_root, resolve_outdir, run_command, write_json, write_tsv


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


def _absrel_thread_env() -> Dict[str, str]:
    return {
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    }


def _run_one_absrel(row: Dict[str, str], *, benchmark_dir: Path, force: bool) -> Dict[str, str] | None:
    family_id = row["family_id"]
    log_dir = benchmark_dir / "absrel_logs" / family_id
    log_dir.mkdir(parents=True, exist_ok=True)
    status_path = log_dir / "absrel_family_status.json"
    if not force:
        parsed = _parse_one(row)
        if parsed["status"] == "ok":
            write_json(status_path, {"family_id": family_id, "status": "skipped_completed", "reason": "valid aBSREL JSON already present"})
            return None
        if Path(row["output_json"]).exists():
            write_json(status_path, {"family_id": family_id, "status": "rerun_partial_or_invalid", "reason": parsed["notes"]})

    cmd = ["hyphy", "absrel", "--alignment", row["alignment"], "--tree", row["tree"], "--branches", row["branches"], "--output", row["output_json"]]
    completed = run_command(cmd, cwd=repo_root(), env=_absrel_thread_env())
    (log_dir / "stdout.txt").write_text(completed.stdout, encoding="utf-8")
    (log_dir / "stderr.txt").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        error = " ".join((completed.stderr.strip() or completed.stdout.strip()).split())[-500:]
        write_json(status_path, {"family_id": family_id, "status": "failed", "returncode": completed.returncode, "error": error})
        return {"family_id": family_id, "status": "failed", "error": error}
    parsed = _parse_one(row)
    write_json(status_path, {"family_id": family_id, "status": parsed["status"], "returncode": completed.returncode, "notes": parsed.get("notes", "")})
    if parsed["status"] != "ok":
        return {"family_id": family_id, "status": parsed["status"], "error": parsed.get("notes", "")}
    return None


def _summarize(rows: List[Dict[str, str]], failures: List[Dict[str, str]], attempted: int) -> Dict[str, Any]:
    completed = sum(1 for row in rows if row["status"] == "ok")
    failed_ids = {row["family_id"] for row in failures if row.get("family_id") not in {"", "ALL"}}
    pending = sum(1 for row in rows if row["status"] == "pending_not_run")
    return {
        "status": "ok" if not failures else "warning",
        "attempted": attempted,
        "completed": completed,
        "failed": len(failed_ids),
        "pending": pending,
        "tool_missing": any(row.get("status") == "tool_missing" for row in failures),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--benchmark-dir", default=None)
    parser.add_argument("--parse-only", action="store_true")
    parser.add_argument("--continue-on-failure", action="store_true")
    parser.add_argument("--jobs", type=int, default=None)
    args = parser.parse_args()

    config = read_config(args.config)
    benchmark_dir = resolve_outdir(config, args.benchmark_dir)
    jobs = max(1, int(args.jobs or config_jobs(config, "absrel", 1)))
    if not args.continue_on_failure:
        jobs = 1
    force = os.environ.get("BABAPPA_FORCE", "").lower() in {"1", "true", "yes", "y"}
    manifest = read_tsv(benchmark_dir / "absrel_input_manifest.tsv")
    if not manifest:
        raise SystemExit("missing absrel_input_manifest.tsv; run 03_prepare_absrel_inputs.py first")
    failures: List[Dict[str, str]] = []
    attempted = len(manifest)
    if not args.parse_only:
        if shutil.which("hyphy") is None:
            failures.append({"family_id": "ALL", "status": "tool_missing", "error": "hyphy not found on PATH"})
        else:
            print(f"Running aBSREL family jobs={jobs} families={len(manifest)} force={force}")
            print("aBSREL inner thread caps: OMP/OPENBLAS/MKL/VECLIB/NUMEXPR=1")
            completed_count = 0
            with ThreadPoolExecutor(max_workers=jobs) as executor:
                future_to_family = {executor.submit(_run_one_absrel, row, benchmark_dir=benchmark_dir, force=force): row["family_id"] for row in manifest}
                for future in as_completed(future_to_family):
                    completed_count += 1
                    family_id = future_to_family[future]
                    try:
                        failure = future.result()
                    except Exception as exc:
                        failure = {"family_id": family_id, "status": "failed", "error": str(exc)[-500:]}
                    if failure is not None:
                        failures.append(failure)
                    if completed_count == 1 or completed_count % 25 == 0 or completed_count == len(manifest):
                        print(f"aBSREL progress {completed_count}/{len(manifest)} families")
    rows = [_parse_one(row) for row in manifest]
    rows = sorted(rows, key=lambda row: row["family_id"])
    failures = sorted(failures, key=lambda row: row.get("family_id", ""))
    summary = _summarize(rows, failures, attempted)
    write_tsv(benchmark_dir / "absrel_results.tsv", rows, ["family_id", "status", "positive_count", "call", "p_value", "notes"])
    write_tsv(benchmark_dir / "absrel_failures.tsv", failures, ["family_id", "status", "error"])
    write_json(benchmark_dir / "absrel_run_summary.json", summary)
    (benchmark_dir / "absrel_run_summary.md").write_text(
        "\n".join(
            [
                "# aBSREL Smoke/Pilot Run Summary",
                "",
                f"- attempted: `{summary['attempted']}`",
                f"- completed: `{summary['completed']}`",
                f"- failed: `{summary['failed']}`",
                f"- pending: `{summary['pending']}`",
                f"- tool missing: `{summary['tool_missing']}`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"Wrote aBSREL benchmark results: {benchmark_dir / 'absrel_results.tsv'}")
    print(f"attempted={summary['attempted']} completed={summary['completed']} failed={summary['failed']} pending={summary['pending']}")
    return 0 if args.continue_on_failure or not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
