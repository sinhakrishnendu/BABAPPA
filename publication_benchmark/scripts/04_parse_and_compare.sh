#!/usr/bin/env bash
set -euo pipefail

echo "MANUAL EXECUTION SCRIPT - REVIEW BEFORE RUNNING"

MANIFEST="${1:-publication_benchmark/panel_template.tsv}"
OUTROOT="${2:-publication_benchmark/results}"

cd "$(dirname "$0")/../.."
mkdir -p "$OUTROOT/parsed" "$OUTROOT/reference_results" "$OUTROOT/comparison"
COMBINED="$OUTROOT/reference_results/reference_results.tsv"
printf "panel_id\ttool\ttest_name\tp_value\tq_value\tselected_branch\tselected_sites\tresult_class\tnotes\n" > "$COMBINED"

tail -n +2 "$MANIFEST" | while IFS=$'\t' read -r panel_id gene_family cds_msa tree_file foreground expected_category notes; do
  if [[ -z "${panel_id}" || "${panel_id}" == \#* ]]; then
    continue
  fi
  babappa parse-codeml-reference \
    --codeml-dir "$OUTROOT/reference_runs/$panel_id/codeml" \
    --outdir "$OUTROOT/parsed/$panel_id/codeml_parsed"
  babappa parse-hyphy-reference \
    --hyphy-dir "$OUTROOT/reference_runs/$panel_id/hyphy" \
    --outdir "$OUTROOT/parsed/$panel_id/hyphy_parsed"
  babappa build-reference-results-table \
    --panel-id "$panel_id" \
    --codeml-parsed "$OUTROOT/parsed/$panel_id/codeml_parsed" \
    --hyphy-parsed "$OUTROOT/parsed/$panel_id/hyphy_parsed" \
    --outdir "$OUTROOT/reference_results/$panel_id"
  tail -n +2 "$OUTROOT/reference_results/$panel_id/reference_results.tsv" >> "$COMBINED"
done

python - "$MANIFEST" "$OUTROOT" <<'PY'
import csv
import json
import sys
from pathlib import Path

manifest = Path(sys.argv[1])
outroot = Path(sys.argv[2])
rows = list(csv.DictReader(manifest.open(), delimiter="\t"))
summary_path = outroot / "babappa" / "panel_run_summary.tsv"
fields = [
    "panel_id",
    "status",
    "qc_status",
    "applicability_status",
    "scoring_status",
    "diagnostic_only",
    "result_class",
    "max_gene_support",
    "called_branch_site_rows",
    "babappa_native_result_class",
    "babappa_native_evidence_class",
]
with summary_path.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
    writer.writeheader()
    for row in rows:
        panel_id = row["panel_id"]
        path = outroot / "babappa" / panel_id / "prediction_manifest.json"
        payload = json.loads(path.read_text()) if path.exists() else {}
        gene = payload.get("summary") or {}
        writer.writerow({
            "panel_id": panel_id,
            "status": payload.get("status", "missing"),
            "qc_status": payload.get("validation_status", ""),
            "applicability_status": payload.get("applicability_status", ""),
            "scoring_status": (payload.get("scoring") or {}).get("status", payload.get("status", "")),
            "diagnostic_only": gene.get("diagnostic_only", ""),
            "result_class": gene.get("result_class", ""),
            "max_gene_support": gene.get("max_gene_support", ""),
            "called_branch_site_rows": gene.get("n_called_positive", ""),
            "babappa_native_result_class": gene.get("babappa_native_result_class", ""),
            "babappa_native_evidence_class": gene.get("babappa_native_evidence_class", ""),
        })
manifest_payload = {
    "status": "ok",
    "families_processed": len(rows),
    "claim_boundary": "Publication benchmark only. BABAPPA-native evidence is compared with codeml/HyPhy as external comparators, not ground truth.",
}
(outroot / "babappa" / "panel_run_manifest.json").write_text(json.dumps(manifest_payload, indent=2) + "\n")
PY

babappa compare-empirical-reference-results \
  --babappa-panel-run "$OUTROOT/babappa" \
  --reference-results "$COMBINED" \
  --outdir "$OUTROOT/comparison"
