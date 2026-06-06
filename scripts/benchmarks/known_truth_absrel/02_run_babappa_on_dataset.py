#!/usr/bin/env python
"""Run BABAPPA direct prediction on a known-truth benchmark dataset."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import config_jobs, deterministic_score, read_config, read_tsv, repo_root, resolve_outdir, run_command, safe_float, write_json, write_tsv


RESULT_FIELDS = [
    "family_id",
    "method",
    "truth_class",
    "truth_positive",
    "expected_applicability",
    "status",
    "status_class",
    "score",
    "call",
    "result_class",
    "diagnostic_only",
    "applicability",
    "failure_reason",
]
FAILURE_FIELDS = ["family_id", "status", "returncode", "error"]
VALID_RESULT_CLASSES = {"diagnostic_positive", "diagnostic_negative", "diagnostic_only", "no_call", "failed"}


def _read_gene_summary(path: Path) -> Dict[str, str]:
    rows = read_tsv(path)
    return rows[0] if rows else {}


def _truth_positive(row: Dict[str, str]) -> int:
    return int(row.get("truth_class") in {"positive", "ood_positive"})


def _normalize_result_class(raw: str, diagnostic_only: bool, call: int, status: str) -> str:
    if status != "ok":
        return "failed"
    if diagnostic_only:
        return "diagnostic_only"
    lowered = (raw or "").lower()
    if "positive" in lowered and "negative" not in lowered:
        return "diagnostic_positive"
    if call == 1:
        return "diagnostic_positive"
    return "diagnostic_negative"


def _status_class(status: str, result_class: str, call: int | str) -> str:
    if status != "ok":
        return "method_failed"
    if result_class == "diagnostic_only":
        return "diagnostic_only"
    if result_class == "no_call":
        return "no_call"
    if str(call) == "1":
        return "method_positive"
    return "method_negative"


def _mock_row(row: Dict[str, str], seed: int) -> Dict[str, Any]:
    truth_positive = bool(_truth_positive(row))
    ood = row["expected_applicability"] == "out_of_domain"
    score = deterministic_score(seed, row["family_id"], truth_positive, ood)
    call = int(score >= 0.5 and not ood)
    result_class = "diagnostic_positive" if call else ("diagnostic_only" if ood else "diagnostic_negative")
    return {
        "family_id": row["family_id"],
        "method": "BABAPPA",
        "truth_class": row["truth_class"],
        "truth_positive": int(truth_positive),
        "expected_applicability": row["expected_applicability"],
        "status": "ok",
        "status_class": _status_class("ok", result_class, call),
        "score": f"{score:.6f}",
        "call": call,
        "result_class": result_class,
        "diagnostic_only": str(ood),
        "applicability": row["expected_applicability"],
        "failure_reason": "",
    }


def _failure_row(row: Dict[str, str], reason: str, status: str = "failed") -> Dict[str, Any]:
    return {
        "family_id": row["family_id"],
        "method": "BABAPPA",
        "truth_class": row.get("truth_class", ""),
        "truth_positive": _truth_positive(row),
        "expected_applicability": row.get("expected_applicability", ""),
        "status": status,
        "status_class": "method_failed",
        "score": "NA",
        "call": "NA",
        "result_class": "failed",
        "diagnostic_only": "NA",
        "applicability": "",
        "failure_reason": reason,
    }


def _family_status_path(family_outdir: Path) -> Path:
    return family_outdir / "benchmark_family_status.json"


def _write_family_status(family_outdir: Path, payload: Dict[str, Any]) -> None:
    payload = dict(payload)
    payload.setdefault("family_outdir", str(family_outdir))
    write_json(_family_status_path(family_outdir), payload)


def _check_preflight(model_package: str, smoke_surrogate: bool) -> List[str]:
    if smoke_surrogate:
        return []
    failures: List[str] = []
    package = Path(model_package)
    if not package.is_absolute():
        package = repo_root() / package
    if not package.exists():
        failures.append(f"deployable model package missing: {package}")
    if importlib.util.find_spec("torch") is None:
        failures.append("PyTorch/torch is not importable; real BABAPPA scoring cannot run")
    return failures


def _extract_real_score(family_outdir: Path, summary: Dict[str, str]) -> tuple[float | None, str]:
    raw = summary.get("max_gene_support")
    if raw in {"", None, "NA"}:
        return None, "gene_summary.tsv missing max_gene_support"
    score = safe_float(raw, None)  # type: ignore[arg-type]
    if score is None:
        return None, f"max_gene_support is not numeric: {raw}"
    score_rows = read_tsv(family_outdir / "scores" / "empirical_branch_site_scores.tsv")
    if not score_rows:
        return None, "empirical_branch_site_scores.tsv missing or empty"
    non_numeric = [row.get("prob_positive", "") for row in score_rows if safe_float(row.get("prob_positive"), None) is None]  # type: ignore[arg-type]
    if non_numeric:
        return None, "non-numeric prob_positive values present"
    return float(score), ""


def _result_row_from_family_output(row: Dict[str, str], family_outdir: Path) -> tuple[Dict[str, Any] | None, str]:
    summary_path = family_outdir / "gene_summary.tsv"
    if not summary_path.exists():
        return None, "gene_summary.tsv missing"
    summary = _read_gene_summary(summary_path)
    score, score_error = _extract_real_score(family_outdir, summary)
    if score is None:
        return None, score_error
    diagnostic_only = str(summary.get("diagnostic_only", "")).lower() == "true"
    raw_class = summary.get("babappa_native_result_class") or summary.get("result_class") or ""
    n_called = int(safe_float(summary.get("n_called_positive"), 0.0) or 0)
    call = int((n_called > 0 or "positive" in raw_class.lower()) and not diagnostic_only)
    result_class = _normalize_result_class(raw_class, diagnostic_only, call, "ok")
    return (
        {
            "family_id": row["family_id"],
            "method": "BABAPPA",
            "truth_class": row["truth_class"],
            "truth_positive": _truth_positive(row),
            "expected_applicability": row["expected_applicability"],
            "status": "ok",
            "status_class": _status_class("ok", result_class, call),
            "score": f"{score:.12g}",
            "call": call,
            "result_class": result_class,
            "diagnostic_only": str(diagnostic_only),
            "applicability": summary.get("applicability_status", ""),
            "failure_reason": "",
        },
        "",
    )


def _run_family(
    row: Dict[str, str],
    *,
    benchmark_dir: Path,
    results_dir: Path,
    model_package: str,
    device: str,
    null_replicates: str,
    seed: int,
    smoke_surrogate: bool,
    force: bool,
) -> tuple[Dict[str, Any], Dict[str, Any] | None]:
    family_id = row["family_id"]
    family_outdir = results_dir / family_id
    if smoke_surrogate:
        result = _mock_row(row, seed)
        _write_family_status(family_outdir, {"family_id": family_id, "status": "ok", "mode": "smoke_surrogate"})
        return result, None

    if not force:
        existing, existing_error = _result_row_from_family_output(row, family_outdir)
        if existing is not None:
            _write_family_status(family_outdir, {"family_id": family_id, "status": "skipped_completed", "reason": "valid output already present"})
            return existing, None
        if family_outdir.exists():
            _write_family_status(family_outdir, {"family_id": family_id, "status": "rerun_partial", "reason": existing_error})

    cmd = [
        "babappa",
        "predict-branch-sites",
        "--msa",
        str(benchmark_dir / row["codon_fasta"]),
        "--tree",
        str(benchmark_dir / row["tree"]),
        "--foreground",
        row["foreground"],
        "--outdir",
        str(family_outdir),
        "--model-package",
        model_package,
        "--device",
        device,
        "--allow-missing-start-codon",
        "--null-replicates",
        str(null_replicates),
    ]
    completed = run_command(cmd, cwd=repo_root())
    (family_outdir / "command_stdout.txt").parent.mkdir(parents=True, exist_ok=True)
    (family_outdir / "command_stdout.txt").write_text(completed.stdout, encoding="utf-8")
    (family_outdir / "command_stderr.txt").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        error_text = " ".join((completed.stderr.strip() or completed.stdout.strip()).split())[-500:]
        failure = {"family_id": family_id, "status": "failed", "returncode": completed.returncode, "error": error_text}
        _write_family_status(family_outdir, {"family_id": family_id, "status": "failed", "returncode": completed.returncode, "error": error_text})
        return _failure_row(row, error_text), failure

    result, score_error = _result_row_from_family_output(row, family_outdir)
    if result is None:
        failure = {"family_id": family_id, "status": "failed", "returncode": "missing_score", "error": score_error}
        _write_family_status(family_outdir, {"family_id": family_id, "status": "failed", "returncode": "missing_score", "error": score_error})
        return _failure_row(row, score_error), failure

    _write_family_status(family_outdir, {"family_id": family_id, "status": "ok", "returncode": completed.returncode})
    return result, None


def _audit_results(rows: Sequence[Dict[str, Any]], outdir: Path, allow_constant_scores: bool) -> Dict[str, Any]:
    malformed = []
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    scores: List[float] = []
    for index, row in enumerate(rows, start=1):
        missing = [field for field in RESULT_FIELDS if field not in row]
        if missing:
            malformed.append({"row": index, "reason": "missing columns: " + ",".join(missing)})
            continue
        if row.get("status") == "ok":
            score = safe_float(row.get("score"), None)  # type: ignore[arg-type]
            if score is None:
                malformed.append({"row": index, "family_id": row.get("family_id", ""), "reason": "ok row has non-numeric score"})
            else:
                scores.append(float(score))
            if str(row.get("call")) not in {"0", "1"}:
                malformed.append({"row": index, "family_id": row.get("family_id", ""), "reason": "ok row call is not 0/1"})
        if row.get("result_class") not in VALID_RESULT_CLASSES:
            malformed.append({"row": index, "family_id": row.get("family_id", ""), "reason": "invalid result_class"})
    unique_scores = sorted(set(scores))
    all_zero = bool(scores) and all(score == 0.0 for score in scores)
    constant = len(unique_scores) <= 1 if scores else False
    status = "pass"
    reasons: List[str] = []
    if malformed:
        status = "fail"
        reasons.append("schema_invalid")
    if not ok_rows:
        status = "fail"
        reasons.append("no_successful_babappa_rows")
    if not scores:
        status = "fail"
        reasons.append("scores_missing")
    if constant and not allow_constant_scores:
        status = "fail"
        reasons.append("scores_constant")
    if all_zero and not allow_constant_scores:
        status = "fail"
        reasons.append("scores_all_zero")
    payload = {
        "status": status,
        "reasons": reasons,
        "n_rows": len(rows),
        "n_ok_rows": len(ok_rows),
        "malformed_row_count": len(malformed),
        "malformed_rows": malformed,
        "score_count": len(scores),
        "score_min": min(scores) if scores else None,
        "score_median": statistics.median(scores) if scores else None,
        "score_max": max(scores) if scores else None,
        "score_unique_count": len(unique_scores),
        "score_constant": constant,
        "scores_all_zero": all_zero,
        "allow_constant_scores": allow_constant_scores,
    }
    write_json(outdir / "babappa_score_audit.json", payload)
    (outdir / "babappa_score_audit.md").write_text(_render_audit_md(payload), encoding="utf-8")
    return payload


def _render_audit_md(payload: Dict[str, Any]) -> str:
    if payload["status"] == "pass":
        closing = "BABAPPA score audit passed. Scores are nonconstant and nonzero."
    else:
        closing = "Smoke must not be scaled to pilot while this audit fails."
    return "\n".join(
        [
            "# BABAPPA Score Audit",
            "",
            f"- status: `{payload['status']}`",
            f"- reasons: `{','.join(payload['reasons'])}`",
            f"- rows: `{payload['n_rows']}`",
            f"- ok rows: `{payload['n_ok_rows']}`",
            f"- malformed rows: `{payload['malformed_row_count']}`",
            f"- score count: `{payload['score_count']}`",
            f"- score min: `{payload['score_min']}`",
            f"- score median: `{payload['score_median']}`",
            f"- score max: `{payload['score_max']}`",
            f"- unique score count: `{payload['score_unique_count']}`",
            f"- score constant: `{payload['score_constant']}`",
            f"- scores all zero: `{payload['scores_all_zero']}`",
            "",
            closing,
            "",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--benchmark-dir", default=None)
    parser.add_argument("--model-package", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--continue-on-failure", action="store_true")
    parser.add_argument("--smoke-surrogate", action="store_true", help="Test-only surrogate backend; public benchmark scripts do not use this.")
    parser.add_argument("--mock-for-tests", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--allow-constant-scores", action="store_true")
    parser.add_argument("--jobs", type=int, default=None, help="Number of independent family jobs to run in parallel.")
    args = parser.parse_args()

    config = read_config(args.config)
    benchmark_dir = resolve_outdir(config, args.benchmark_dir)
    manifest = read_tsv(benchmark_dir / "manifest.tsv")
    if not manifest:
        raise SystemExit(f"missing or empty manifest: {benchmark_dir / 'manifest.tsv'}")
    model_package = args.model_package or config.get("model_package", "deployable_model_conservative_branch_site_100k_mps")
    device = args.device or config.get("device", "auto")
    null_replicates = config.get("babappa_null_replicates", "0")
    seed = int(config.get("seed", "42"))
    smoke_surrogate = bool(args.smoke_surrogate or args.mock_for_tests)
    jobs = max(1, int(args.jobs or config_jobs(config, "babappa", 1)))
    if not args.continue_on_failure:
        jobs = 1
    force = os.environ.get("BABAPPA_FORCE", "").lower() in {"1", "true", "yes", "y"}
    results_dir = benchmark_dir / "babappa_scores"
    results_dir.mkdir(parents=True, exist_ok=True)
    result_rows: List[Dict[str, Any]] = []
    failure_rows: List[Dict[str, Any]] = []

    preflight_failures = _check_preflight(model_package, smoke_surrogate)
    if preflight_failures:
        for row in manifest:
            reason = "; ".join(preflight_failures)
            result_rows.append(_failure_row(row, reason))
            failure_rows.append({"family_id": row["family_id"], "status": "failed", "returncode": "preflight", "error": reason})
        write_tsv(benchmark_dir / "babappa_results.tsv", result_rows, RESULT_FIELDS)
        write_tsv(benchmark_dir / "babappa_failures.tsv", failure_rows, FAILURE_FIELDS)
        audit = _audit_results(result_rows, benchmark_dir, args.allow_constant_scores)
        write_json(benchmark_dir / "babappa_run_manifest.json", {"status": "fail", "families": len(result_rows), "failures": len(failure_rows), "audit": audit})
        print("BABAPPA real scoring preflight failed: " + "; ".join(preflight_failures))
        return 1

    print(f"Running BABAPPA family jobs={jobs} families={len(manifest)} force={force}")
    completed_count = 0
    with ThreadPoolExecutor(max_workers=jobs) as executor:
        future_to_family = {
            executor.submit(
                _run_family,
                row,
                benchmark_dir=benchmark_dir,
                results_dir=results_dir,
                model_package=model_package,
                device=device,
                null_replicates=str(null_replicates),
                seed=seed,
                smoke_surrogate=smoke_surrogate,
                force=force,
            ): row["family_id"]
            for row in manifest
        }
        for future in as_completed(future_to_family):
            family_id = future_to_family[future]
            completed_count += 1
            try:
                result, failure = future.result()
            except Exception as exc:  # defensive guard so one family cannot corrupt final TSV writing
                source = next(row for row in manifest if row["family_id"] == family_id)
                error = str(exc)[-500:]
                result = _failure_row(source, error)
                failure = {"family_id": family_id, "status": "failed", "returncode": "exception", "error": error}
            result_rows.append(result)
            if failure is not None:
                failure_rows.append(failure)
            if completed_count == 1 or completed_count % 25 == 0 or completed_count == len(manifest):
                print(f"BABAPPA progress {completed_count}/{len(manifest)} families")

    result_rows = sorted(result_rows, key=lambda row: str(row.get("family_id", "")))
    failure_rows = sorted(failure_rows, key=lambda row: str(row.get("family_id", "")))
    write_tsv(benchmark_dir / "babappa_results.tsv", result_rows, RESULT_FIELDS)
    write_tsv(benchmark_dir / "babappa_failures.tsv", failure_rows, FAILURE_FIELDS)
    audit = _audit_results(result_rows, benchmark_dir, args.allow_constant_scores)
    run_status = "ok" if audit["status"] == "pass" and not failure_rows else "fail"
    write_json(benchmark_dir / "babappa_run_manifest.json", {"status": run_status, "families": len(result_rows), "failures": len(failure_rows), "audit": audit})
    print(f"Wrote BABAPPA benchmark results: {benchmark_dir / 'babappa_results.tsv'}")
    print(f"BABAPPA score audit status={audit['status']} reasons={','.join(audit['reasons'])}")
    if audit["status"] != "pass":
        return 1
    return 0 if not failure_rows or args.continue_on_failure else 1


if __name__ == "__main__":
    raise SystemExit(main())
