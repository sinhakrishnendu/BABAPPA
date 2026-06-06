#!/usr/bin/env bash
set -euo pipefail
echo "Long-run transfer-test script: review inputs before running."
PANEL="${1:-empirical_transfer_panel_template.tsv}"
OUTDIR="${2:-empirical_transfer_results}"
mkdir -p "$OUTDIR"
python - <<'PY' "$PANEL" "$OUTDIR" "auto" "1000"
import csv, subprocess, sys
panel, outdir, device, null_reps = sys.argv[1:]
with open(panel, newline="", encoding="utf-8") as handle:
    for row in csv.DictReader(handle, delimiter="\t"):
        if row.get("msa_path", "TODO") == "TODO" or row.get("tree_path", "TODO") == "TODO":
            print(f"skip {row.get('panel_id')}: missing MSA/tree")
            continue
        cmd = [
            "babappa", "predict-branch-sites",
            "--msa", row["msa_path"],
            "--tree", row["tree_path"],
            "--foreground", row.get("foreground", "leaves"),
            "--outdir", f"{outdir}/{row['panel_id']}",
            "--device", device,
            "--null-replicates", str(null_reps),
        ]
        print(" ".join(cmd))
        subprocess.run(cmd, check=True)
PY
