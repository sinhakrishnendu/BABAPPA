#!/usr/bin/env bash
set -euo pipefail
echo "Long-run benchmark script: review inputs before running."
PANEL="${1:-known_positive_control_panel_template.tsv}"
OUTDIR="${2:-known_positive_control_results}"
INCLUDE_OPTIONAL="${INCLUDE_OPTIONAL:-0}"
DEVICE="${BABAPPA_DEVICE:-auto}"
NULL_REPS="${BABAPPA_NULL_REPLICATES:-1000}"
mkdir -p "$OUTDIR"
python - <<'PY' "$PANEL" "$OUTDIR" "$DEVICE" "$NULL_REPS" "$INCLUDE_OPTIONAL"
import csv, subprocess, sys
panel, outdir, device, null_reps, include_optional = sys.argv[1:]
with open(panel, newline="", encoding="utf-8") as handle:
    for row in csv.DictReader(handle, delimiter="\t"):
        control_type = row.get("control_type", "")
        if row.get("msa_path", "TODO") == "TODO" or row.get("tree_path", "TODO") == "TODO":
            print(f"skip {row.get('panel_id')}: missing MSA/tree")
            continue
        if "not_runnable" in control_type or "pending" in control_type:
            print(f"skip {row.get('panel_id')}: {control_type}")
            continue
        if "optional" in control_type and include_optional not in {"1", "true", "yes", "y"}:
            print(f"skip {row.get('panel_id')}: optional; set INCLUDE_OPTIONAL=1 to run")
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
