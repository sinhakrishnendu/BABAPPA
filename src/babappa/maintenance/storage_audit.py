"""Storage audit and dry-run cleanup planning for BABAPPA workspaces."""

from __future__ import annotations

import csv
import json
import os
import shlex
from dataclasses import dataclass
from datetime import datetime
from fnmatch import fnmatch
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


@dataclass(frozen=True)
class StorageAuditConfig:
    root: str = "."
    outdir: str = "storage_cleanup_audit"
    target_size_gb: float = 10.0


KEEP_PREFIXES = (
    ".git",
    "src",
    "tests",
    "docs",
    "examples",
    "Manuscript",
    "manuscript",
    "deployable_model_conservative_branch_site_100k_mps",
    "explicit_branch_truth_100k_mps_cross_tier_summary",
    "branch_truth_status_audit_explicit_branch_truth_100k_mps",
    "deployable_model_package_plan_100k_mps",
    "real_empirical_pilot/evidence_packs/WRKY_candidate_02_close",
    "real_empirical_pilot/summary",
    "real_empirical_pilot/reference_results",
    "real_empirical_pilot/comparison",
    "real_empirical_pilot/audits",
    "real_empirical_pilot/prefilter",
    "real_empirical_pilot/ood_summary",
    "real_empirical_pilot/target_taxa_recommendations",
    "real_empirical_pilot/readiness",
    "real_empirical_pilot/foreground_candidates",
)

KEEP_FILES = {
    "pyproject.toml",
    "README.md",
    "LICENSE",
    ".gitignore",
    "CITATION.cff",
    "environment.yml",
    "requirements.txt",
    "requirements-dev.txt",
    "setup.py",
    "setup.cfg",
    "explicit_branch_truth_100k_mps_final_validation_report.md",
    "explicit_branch_truth_100k_mps_final_validation_report.json",
    "explicit_branch_truth_100k_mps_final_validation_report.tsv",
    "docs/POST_100K_EMPIRICAL_TRANSITION_PLAN.md",
    "docs/DEPLOYABLE_MODEL_PACKAGE.md",
    "docs/SIMULATION_MATCHED_EMPIRICAL_CALIBRATION.md",
    "real_empirical_pilot/REAL_INPUT_GUIDE.md",
}

KEEP_PATTERNS = (
    "real_empirical_pilot/input/cds/*.fasta",
    "real_empirical_pilot/input/trees/*.treefile",
    "real_empirical_pilot/input/msas/*.protein.fasta",
    "real_empirical_pilot/input/msas/*.protein.aln.fasta",
    "real_empirical_pilot/input/msas/*.codon.aln.fasta",
    "real_empirical_pilot/reference_runs/WRKY_candidate_02_close/**/*.md",
    "real_empirical_pilot/reference_runs/WRKY_candidate_02_close/**/*.json",
    "real_empirical_pilot/reference_runs/WRKY_candidate_02_close/**/*.tsv",
    "real_empirical_pilot/reference_runs/WRKY_candidate_02_close/**/*.ctl",
    "real_empirical_pilot/reference_runs/WRKY_candidate_02_close/**/*.sh",
    "real_empirical_pilot/reference_runs/WRKY_candidate_02_close/**/*.nwk",
)

CACHE_PATTERNS = (
    "__pycache__",
    "*/__pycache__/*",
    ".pytest_cache",
    ".pytest_cache/*",
    ".mypy_cache",
    ".mypy_cache/*",
    ".ruff_cache",
    ".ruff_cache/*",
    "*.pyc",
    "*.pyo",
    ".DS_Store",
    "*.log",
    "*.tmp",
    "*.partial",
    ".stage_complete_*",
    ".stage_partial_*",
    "logs",
    "logs/*",
    "tmp",
    "tmp/*",
    "temp",
    "temp/*",
)

REMOVE_PREFIXES = (
    "saturation_panel_",
    "sim_explicit_branch_truth_",
    "align_explicit_branch_truth_",
    "site_map_explicit_branch_truth_",
    "method_policy_explicit_branch_truth_",
    "tensors_explicit_branch_truth_",
    "dataset_explicit_branch_truth_",
    "branch_site_oracle_explicit_branch_truth_",
    "branch_site_dataset_explicit_branch_truth_",
    "branch_site_leakage_explicit_branch_truth_",
    "branch_site_baseline_explicit_branch_truth_",
    "branch_site_neural_explicit_branch_truth_",
    "branch_site_calibration_explicit_branch_truth_",
    "branch_aggregation_explicit_branch_truth_",
    "branch_aggregation_controls_explicit_branch_truth_",
    "branch_site_run_summary_explicit_branch_truth_",
    "external_aligner_validation",
    "fast_external_10k",
)

REMOVE_PATTERNS = (
    "*_CONTAMINATED_*",
    "*/sim",
    "*/sim/*",
    "*/branch_site_truth.tsv",
    "real_empirical_pilot/input/raw_downloads",
    "real_empirical_pilot/input/raw_downloads/*",
    "real_empirical_pilot/acquisition_plans/*/downloads",
    "real_empirical_pilot/acquisition_plans/*/downloads/*",
    "real_empirical_pilot/acquisition_plans/*/genomes",
    "real_empirical_pilot/acquisition_plans/*/genomes/*",
    "real_empirical_pilot/acquisition_plans/*/blastdb",
    "real_empirical_pilot/acquisition_plans/*/blastdb/*",
    "real_empirical_pilot/acquisition_plans/*/hits",
    "real_empirical_pilot/acquisition_plans/*/hits/*",
    "real_empirical_pilot/calibration_runs",
    "real_empirical_pilot/calibration_runs/*",
    "*.zip",
    "*.gz",
    "*.tar",
    "*.tar.gz",
    "*.tgz",
    "*.fa.gz",
    "*.fasta.gz",
    "*.faa.gz",
    "*.fna.gz",
)

ARCHIVE_PATTERNS = (
    "*_100k*",
    "*_10k*",
    "*tensor*.npz",
    "*.npz",
    "*.parquet",
    "*.h5",
    "*.h5ad",
    "*.arrow",
    "*.pkl",
    "*.pickle",
)


def audit_storage(config: StorageAuditConfig) -> Dict[str, Any]:
    root = Path(config.root).expanduser().resolve()
    outdir = Path(config.outdir)
    if not outdir.is_absolute():
        outdir = root / outdir
    outdir.mkdir(parents=True, exist_ok=True)

    entries = _build_inventory(root, outdir)
    for entry in entries:
        entry.update(_classify(entry["path"], entry["type"]))
        entry["size_human"] = _human_size(entry["size_bytes"])

    keep_rows = [row for row in entries if row["recommendation"] == "keep"]
    remove_rows = [row for row in entries if row["recommendation"] == "remove"]
    archive_rows = [row for row in entries if row["recommendation"] == "archive"]

    total_bytes = _entry_size(entries, ".")
    removable_bytes = sum(row["size_bytes"] for row in _top_level_actions(remove_rows + archive_rows))
    expected_bytes = max(total_bytes - removable_bytes, 0)
    top_entries = sorted(entries, key=lambda row: row["size_bytes"], reverse=True)[:100]
    large_dirs = [
        row for row in entries if row["type"] == "dir" and row["size_bytes"] >= 1_000_000_000
    ]
    large_files = [
        row for row in entries if row["type"] == "file" and row["size_bytes"] >= 100_000_000
    ]

    summary = {
        "root": str(root),
        "outdir": str(outdir),
        "target_size_gb": config.target_size_gb,
        "total_size_bytes": total_bytes,
        "total_size_human": _human_size(total_bytes),
        "estimated_archive_or_remove_bytes": removable_bytes,
        "estimated_archive_or_remove_human": _human_size(removable_bytes),
        "expected_after_quarantine_bytes": expected_bytes,
        "expected_after_quarantine_human": _human_size(expected_bytes),
        "n_entries": len(entries),
        "n_keep": len(keep_rows),
        "n_remove": len(remove_rows),
        "n_archive": len(archive_rows),
        "n_inspect": sum(1 for row in entries if row["recommendation"] == "inspect"),
        "top_space_users": [
            {
                "path": row["path"],
                "type": row["type"],
                "size_human": row["size_human"],
                "recommendation": row["recommendation"],
                "reason": row["reason"],
            }
            for row in top_entries[:10]
        ],
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }

    _write_tsv(outdir / "storage_inventory.tsv", entries)
    (outdir / "storage_inventory.json").write_text(json.dumps(entries, indent=2), encoding="utf-8")
    _write_tsv(outdir / "keep_list.tsv", keep_rows)
    _write_tsv(outdir / "remove_candidates.tsv", remove_rows)
    _write_tsv(outdir / "archive_candidates.tsv", archive_rows)
    _write_top_text(outdir / "du_top_100.txt", top_entries)
    _write_tsv(outdir / "large_dirs_over_1gb.tsv", large_dirs)
    _write_tsv(outdir / "large_files_over_100mb.tsv", large_files)
    (outdir / "storage_summary.md").write_text(_render_summary_md(summary), encoding="utf-8")
    (outdir / "cleanup_dry_run.md").write_text(
        _render_dry_run_md(summary, remove_rows, archive_rows), encoding="utf-8"
    )
    _write_quarantine_script(outdir / "quarantine_large_reproducible_outputs.sh")
    _write_delete_script(outdir / "delete_quarantine_after_review.sh")
    _write_archive_script(outdir / "archive_key_reports.sh")
    _write_validate_script(outdir / "validate_after_cleanup.sh")
    for script in (
        "quarantine_large_reproducible_outputs.sh",
        "delete_quarantine_after_review.sh",
        "archive_key_reports.sh",
        "validate_after_cleanup.sh",
    ):
        try:
            (outdir / script).chmod(0o755)
        except OSError:
            pass
    return summary


def _build_inventory(root: Path, outdir: Path) -> List[Dict[str, Any]]:
    sizes: Dict[Path, int] = {}
    kinds: Dict[Path, str] = {root: "dir"}
    excluded = {outdir.resolve()}
    for current, dirnames, filenames in os.walk(root, topdown=True, followlinks=False):
        current_path = Path(current).resolve()
        dirnames[:] = [
            name
            for name in dirnames
            if (current_path / name).resolve() not in excluded
            and not name.startswith("BABAPPA_STORAGE_QUARANTINE_")
        ]
        current_size = 0
        for filename in filenames:
            file_path = current_path / filename
            try:
                stat = file_path.lstat()
            except OSError:
                continue
            size = 0 if file_path.is_symlink() else stat.st_size
            sizes[file_path] = size
            kinds[file_path] = "file"
            current_size += size
        sizes[current_path] = sizes.get(current_path, 0) + current_size
        kinds[current_path] = "dir"

    for path in sorted([p for p, kind in kinds.items() if kind == "dir"], key=lambda p: len(p.parts), reverse=True):
        if path == root:
            continue
        parent = path.parent
        sizes[parent] = sizes.get(parent, 0) + sizes.get(path, 0)

    entries: List[Dict[str, Any]] = []
    for path, size in sizes.items():
        rel = "." if path == root else path.relative_to(root).as_posix()
        entries.append({"path": rel, "type": kinds.get(path, "file"), "size_bytes": int(size)})
    return sorted(entries, key=lambda row: (row["path"] != ".", row["path"]))


def _classify(path: str, kind: str) -> Dict[str, Any]:
    if path == ".":
        return _row("workspace_total", "inspect", "workspace root", False, True)
    if path.startswith(".git"):
        return _row("git", "keep", "Git repository metadata is protected", False, True)
    if _matches(path, CACHE_PATTERNS):
        return _row("logs_caches_temp", "remove", "cache/log/temp artifact", True, False)
    if path in KEEP_FILES or _has_prefix(path, KEEP_PREFIXES) or _matches(path, KEEP_PATTERNS):
        return _row("protected_artifact", "keep", "protected source, report, package, or evidence artifact", False, True)
    if _has_any_name_prefix(path, REMOVE_PREFIXES) or _matches(path, REMOVE_PATTERNS):
        return _row("large_reproducible_output", "remove", "reproducible generated output or raw download", True, False)
    if _matches(path, ARCHIVE_PATTERNS):
        return _row("generated_large_artifact", "archive", "large generated artifact pattern", True, False)
    if kind == "dir" and path.endswith("_plan"):
        return _row("plan_directory", "keep", "lightweight plan/script directory", False, True)
    if kind == "file" and path.endswith((".md", ".json", ".tsv", ".txt", ".sh", ".ctl", ".nwk", ".treefile", ".fasta")):
        return _row("lightweight_metadata", "inspect", "lightweight file not covered by protected rules", False, False)
    return _row("unclassified", "inspect", "not matched by storage cleanup rules", False, False)


def _row(category: str, recommendation: str, reason: str, reproducible: bool, important: bool) -> Dict[str, Any]:
    return {
        "category": category,
        "recommendation": recommendation,
        "reason": reason,
        "reproducible": "yes" if reproducible else "no",
        "important": "yes" if important else "no",
    }


def _matches(path: str, patterns: Iterable[str]) -> bool:
    return any(fnmatch(path, pattern) for pattern in patterns)


def _has_prefix(path: str, prefixes: Iterable[str]) -> bool:
    return any(path == prefix or path.startswith(prefix + "/") for prefix in prefixes)


def _has_any_name_prefix(path: str, prefixes: Iterable[str]) -> bool:
    name = path.split("/", 1)[0]
    return any(name.startswith(prefix) for prefix in prefixes)


def _entry_size(entries: List[Dict[str, Any]], path: str) -> int:
    for row in entries:
        if row["path"] == path:
            return int(row["size_bytes"])
    return 0


def _top_level_actions(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    selected_paths: List[str] = []
    for row in sorted(rows, key=lambda r: r["path"].count("/")):
        path = row["path"]
        if path == "." or any(path.startswith(parent + "/") for parent in selected_paths):
            continue
        selected.append(row)
        selected_paths.append(path)
    return selected


def _write_tsv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fieldnames = [
        "path",
        "type",
        "size_bytes",
        "size_human",
        "category",
        "recommendation",
        "reason",
        "reproducible",
        "important",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_top_text(path: Path, rows: List[Dict[str, Any]]) -> None:
    lines = ["# Approximate du -ah top 100 equivalent", "size\tpath\ttype\trecommendation\treason"]
    for row in rows:
        lines.append(
            f"{row['size_human']}\t{row['path']}\t{row['type']}\t{row['recommendation']}\t{row['reason']}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _render_summary_md(summary: Dict[str, Any]) -> str:
    lines = [
        "# BABAPPA Storage Audit Summary",
        "",
        f"- Root: `{summary['root']}`",
        f"- Total size: `{summary['total_size_human']}`",
        f"- Target size: `{summary['target_size_gb']} GB`",
        f"- Estimated archive/remove candidates: `{summary['estimated_archive_or_remove_human']}`",
        f"- Expected size after quarantine: `{summary['expected_after_quarantine_human']}`",
        f"- Entries: `{summary['n_entries']}`",
        f"- Keep: `{summary['n_keep']}`",
        f"- Archive: `{summary['n_archive']}`",
        f"- Remove: `{summary['n_remove']}`",
        f"- Inspect: `{summary['n_inspect']}`",
        "",
        "## Top Space Users",
        "",
        "| path | size | recommendation | reason |",
        "|---|---:|---|---|",
    ]
    for row in summary["top_space_users"]:
        lines.append(
            f"| `{row['path']}` | {row['size_human']} | {row['recommendation']} | {row['reason']} |"
        )
    return "\n".join(lines) + "\n"


def _render_dry_run_md(summary: Dict[str, Any], remove_rows: List[Dict[str, Any]], archive_rows: List[Dict[str, Any]]) -> str:
    action_rows = _top_level_actions(remove_rows + archive_rows)
    lines = [
        "# BABAPPA Storage Cleanup Dry Run",
        "",
        "No files were moved or deleted by this audit.",
        "",
        f"- Current project size: `{summary['total_size_human']}`",
        f"- Estimated movable archive/remove size: `{summary['estimated_archive_or_remove_human']}`",
        f"- Expected size after quarantine: `{summary['expected_after_quarantine_human']}`",
        "",
        "## Proposed Top-Level Quarantine Moves",
        "",
        "| path | size | recommendation | reason |",
        "|---|---:|---|---|",
    ]
    for row in sorted(action_rows, key=lambda r: r["size_bytes"], reverse=True)[:100]:
        lines.append(
            f"| `{row['path']}` | {row['size_human']} | {row['recommendation']} | {row['reason']} |"
        )
    lines.extend(
        [
            "",
            "Review `remove_candidates.tsv` and `archive_candidates.tsv` before running the quarantine script.",
            "The generated quarantine script uses `mv` only; permanent deletion is a separate manual-confirmation script.",
        ]
    )
    return "\n".join(lines) + "\n"


def _write_quarantine_script(path: Path) -> None:
    text = """#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/Documents/GitHub/BABAPPA"

echo "MANUAL EXECUTION SCRIPT -- REVIEW BEFORE EXECUTION"
STAMP="$(date +%Y%m%d_%H%M%S)"
QUARANTINE="${BABAPPA_QUARANTINE_DIR:-$HOME/BABAPPA_STORAGE_QUARANTINE_${STAMP}}"
LOG="storage_cleanup_audit/quarantine_move_log.tsv"
mkdir -p "$QUARANTINE" "storage_cleanup_audit"
printf "path\\tdestination\\tstatus\\n" > "$LOG"
du -sh . | tee storage_cleanup_audit/size_before_quarantine.txt

move_candidate() {
  local rel="$1"
  [ -n "$rel" ] || return 0
  [ "$rel" != "." ] || return 0
  case "$rel" in
    .git|.git/*|src|src/*|tests|tests/*|docs|docs/*|examples|examples/*|Manuscript|Manuscript/*|manuscript|manuscript/*|deployable_model_conservative_branch_site_100k_mps|deployable_model_conservative_branch_site_100k_mps/*)
      printf "%s\\t%s\\tprotected_skip\\n" "$rel" "" >> "$LOG"
      return 0
      ;;
  esac
  if [ ! -e "$rel" ]; then
    printf "%s\\t%s\\tmissing_skip\\n" "$rel" "" >> "$LOG"
    return 0
  fi
  local dest="$QUARANTINE/$rel"
  mkdir -p "$(dirname "$dest")"
  mv "$rel" "$dest"
  printf "%s\\t%s\\tmoved\\n" "$rel" "$dest" >> "$LOG"
}

for table in storage_cleanup_audit/remove_candidates.tsv storage_cleanup_audit/archive_candidates.tsv; do
  [ -f "$table" ] || continue
  tail -n +2 "$table" | while IFS=$'\\t' read -r rel _rest; do
    move_candidate "$rel"
  done
done

du -sh . | tee storage_cleanup_audit/size_after_quarantine.txt
du -sh "$QUARANTINE" | tee storage_cleanup_audit/quarantine_size.txt
echo "Quarantine folder: $QUARANTINE"
echo "Move log: $LOG"
"""
    path.write_text(text, encoding="utf-8")


def _write_delete_script(path: Path) -> None:
    text = """#!/usr/bin/env bash
set -euo pipefail

echo "DANGER -- MANUAL EXECUTION SCRIPT AFTER MANUAL REVIEW"
CONFIRM_DELETE="${CONFIRM_DELETE:-NO}"
QUARANTINE="${1:-}"
if [ "$CONFIRM_DELETE" != "YES" ]; then
  echo "Refusing to delete. Re-run with CONFIRM_DELETE=YES and pass the quarantine folder path."
  exit 1
fi
if [ -z "$QUARANTINE" ] || [ ! -d "$QUARANTINE" ]; then
  echo "Usage: CONFIRM_DELETE=YES $0 /path/to/BABAPPA_STORAGE_QUARANTINE_YYYYMMDD_HHMMSS"
  exit 1
fi
case "$QUARANTINE" in
  "$HOME"/BABAPPA_STORAGE_QUARANTINE_*) ;;
  *)
    echo "Refusing to delete unexpected path: $QUARANTINE"
    exit 1
    ;;
esac
rm -rf "$QUARANTINE"
echo "Deleted reviewed quarantine folder: $QUARANTINE"
"""
    path.write_text(text, encoding="utf-8")


def _write_archive_script(path: Path) -> None:
    text = """#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/Documents/GitHub/BABAPPA"

echo "MANUAL EXECUTION SCRIPT -- creates a compact reports/manifests archive"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="BABAPPA_KEY_REPORTS_AND_MANIFESTS_${STAMP}.tar.gz"
tar -czf "$OUT" \\
  README.md docs examples \\
  deployable_model_conservative_branch_site_100k_mps/model_manifest.json \\
  deployable_model_conservative_branch_site_100k_mps/model_card.md \\
  deployable_model_conservative_branch_site_100k_mps/feature_schema.json \\
  deployable_model_conservative_branch_site_100k_mps/calibration_schema.json \\
  deployable_model_conservative_branch_site_100k_mps/training_envelope.json \\
  deployable_model_conservative_branch_site_100k_mps/checksums.sha256 \\
  explicit_branch_truth_100k_mps_final_validation_report.md \\
  explicit_branch_truth_100k_mps_final_validation_report.json \\
  explicit_branch_truth_100k_mps_final_validation_report.tsv \\
  explicit_branch_truth_100k_mps_cross_tier_summary \\
  branch_truth_status_audit_explicit_branch_truth_100k_mps \\
  real_empirical_pilot/evidence_packs/WRKY_candidate_02_close \\
  real_empirical_pilot/summary \\
  real_empirical_pilot/reference_results \\
  storage_cleanup_audit
shasum -a 256 "$OUT" > "${OUT}.sha256"
echo "Archive: $OUT"
"""
    path.write_text(text, encoding="utf-8")


def _write_validate_script(path: Path) -> None:
    text = """#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/Documents/GitHub/BABAPPA"

echo "MANUAL EXECUTION SCRIPT -- lightweight validation after quarantine"
python -m pip install -e ".[dev]"
python -m pytest -q
babappa validate-deployable-model-package --package-dir deployable_model_conservative_branch_site_100k_mps
if command -v babappa >/dev/null 2>&1 && [ -d real_empirical_pilot/evidence_packs/WRKY_candidate_02_close ]; then
  babappa validate-empirical-evidence-pack --evidence-pack real_empirical_pilot/evidence_packs/WRKY_candidate_02_close || echo "WARNING: evidence-pack validation reported a problem"
else
  echo "WARNING: evidence-pack validator or WRKY evidence pack unavailable"
fi
du -sh .
git status --short
"""
    path.write_text(text, encoding="utf-8")


def _human_size(size_bytes: int) -> str:
    value = float(size_bytes)
    for unit in ("B", "K", "M", "G", "T"):
        if value < 1024.0 or unit == "T":
            if unit == "B":
                return f"{int(value)}B"
            return f"{value:.1f}{unit}"
        value /= 1024.0

