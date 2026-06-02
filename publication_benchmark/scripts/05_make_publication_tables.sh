#!/usr/bin/env bash
set -euo pipefail

echo "MANUAL EXECUTION SCRIPT - REVIEW BEFORE RUNNING"

MANIFEST="${1:-publication_benchmark/panel_template.tsv}"
OUTROOT="${2:-publication_benchmark/results}"

cd "$(dirname "$0")/../.."
mkdir -p "$OUTROOT/tables"

python - "$MANIFEST" "$OUTROOT" <<'PY'
import csv
import json
import sys
from pathlib import Path

manifest = Path(sys.argv[1])
outroot = Path(sys.argv[2])
rows = list(csv.DictReader(manifest.open(), delimiter="\t"))
summary_path = outroot / "tables" / "publication_benchmark_summary.tsv"
fields = [
    "panel_id",
    "gene_family",
    "expected_category",
    "babappa_status",
    "applicability_status",
    "babappa_native_result_class",
    "babappa_native_evidence_class",
    "p_babappa_called_rows",
    "p_babappa_max_gene_support",
    "codeml_status",
    "hyphy_status",
    "notes",
]
with summary_path.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
    writer.writeheader()
    for row in rows:
        panel_id = row["panel_id"]
        manifest_json = outroot / "babappa" / panel_id / "prediction_manifest.json"
        payload = json.loads(manifest_json.read_text()) if manifest_json.exists() else {}
        gene_summary = payload.get("summary") or {}
        writer.writerow({
            "panel_id": panel_id,
            "gene_family": row.get("gene_family", ""),
            "expected_category": row.get("expected_category", ""),
            "babappa_status": payload.get("status", "missing"),
            "applicability_status": payload.get("applicability_status", ""),
            "babappa_native_result_class": gene_summary.get("babappa_native_result_class", ""),
            "babappa_native_evidence_class": gene_summary.get("babappa_native_evidence_class", ""),
            "p_babappa_called_rows": gene_summary.get("p_babappa_called_rows", ""),
            "p_babappa_max_gene_support": gene_summary.get("p_babappa_max_gene_support", ""),
            "codeml_status": "see_reference_results",
            "hyphy_status": "see_reference_results",
            "notes": row.get("notes", ""),
        })
print(summary_path)
PY
