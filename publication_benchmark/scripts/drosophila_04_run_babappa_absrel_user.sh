#!/usr/bin/env bash
set -euo pipefail

echo "MANUAL EXECUTION SCRIPT - REVIEW BEFORE RUNNING"

PANEL="${1:-publication_benchmark/drosophila_absrel_benchmark/drosophila_babappa_absrel_panel.tsv}"
OUTROOT="${2:-publication_benchmark/drosophila_absrel_benchmark/results}"
CONFIG="${CONFIG:-publication_benchmark/benchmark_config.env}"

cd "$(dirname "$0")/../.."

if [ ! -f "$PANEL" ]; then
  echo "ERROR: missing panel: $PANEL" >&2
  exit 2
fi

if [ -f "$CONFIG" ]; then
  # shellcheck disable=SC1090
  source "$CONFIG"
fi

BABAPPA_MODEL_PACKAGE="${BABAPPA_MODEL_PACKAGE:-deployable_model_conservative_branch_site_100k_mps}"
BABAPPA_DEVICE="${BABAPPA_DEVICE:-auto}"
BABAPPA_NULL_REPLICATES="${BABAPPA_NULL_REPLICATES:-100}"
BABAPPA_NULL_SEED="${BABAPPA_NULL_SEED:-20260530}"
HYPHY_ABSREL_BRANCHES="${HYPHY_ABSREL_BRANCHES:-Leaves}"
FORCE="${FORCE:-0}"

case "$(printf '%s' "$HYPHY_ABSREL_BRANCHES" | tr '[:upper:]' '[:lower:]')" in
  leaves) HYPHY_ABSREL_BRANCHES="Leaves" ;;
  all) HYPHY_ABSREL_BRANCHES="All" ;;
  internal) HYPHY_ABSREL_BRANCHES="Internal" ;;
  unlabeled|unlabeled_branches|"unlabeled branches") HYPHY_ABSREL_BRANCHES="Unlabeled branches" ;;
esac

mkdir -p "$OUTROOT/babappa" "$OUTROOT/hyphy_absrel" "$OUTROOT/logs"

tail -n +2 "$PANEL" | while IFS=$'\t' read -r panel_id gene_family cds_msa tree_file foreground expected_category notes; do
  if [[ -z "${panel_id}" || "${panel_id}" == \#* ]]; then
    continue
  fi

  echo "=== ${panel_id}: BABAPPA direct prediction ==="
  if [ "$FORCE" != "1" ] && [ -f "$OUTROOT/babappa/$panel_id/prediction_manifest.json" ]; then
    echo "Skipping BABAPPA for ${panel_id}; prediction_manifest.json exists. Set FORCE=1 to rerun."
  else
    babappa predict-branch-sites \
      --msa "$cds_msa" \
      --tree "$tree_file" \
      --foreground "$foreground" \
      --model-package "$BABAPPA_MODEL_PACKAGE" \
      --outdir "$OUTROOT/babappa/$panel_id" \
      --device "$BABAPPA_DEVICE" \
      --null-replicates "$BABAPPA_NULL_REPLICATES" \
      --null-seed "$BABAPPA_NULL_SEED" \
      2>&1 | tee "$OUTROOT/logs/${panel_id}.babappa.log"
  fi

  echo "=== ${panel_id}: HyPhy aBSREL branches=${HYPHY_ABSREL_BRANCHES} ==="
  HYPHY_BIN="${HYPHY_BIN:-}"
  if [ -z "$HYPHY_BIN" ]; then
    if command -v HYPHYMP >/dev/null 2>&1; then
      HYPHY_BIN="HYPHYMP"
    elif command -v hyphy >/dev/null 2>&1; then
      HYPHY_BIN="hyphy"
    fi
  fi
  if [ -z "$HYPHY_BIN" ]; then
    echo "ERROR: neither HYPHYMP nor hyphy is on PATH. Activate/install HyPhy in molevo." >&2
    exit 127
  fi
  family_hyphy="$OUTROOT/hyphy_absrel/$panel_id"
  mkdir -p "$family_hyphy"
  cp "$cds_msa" "$family_hyphy/alignment.fasta"
  cp "$tree_file" "$family_hyphy/tree.nwk"
  if [ "$FORCE" != "1" ] && [ -f "$family_hyphy/absrel.json" ]; then
    echo "Skipping HyPhy aBSREL for ${panel_id}; absrel.json exists. Set FORCE=1 to rerun."
  else
    (
      cd "$family_hyphy"
      "$HYPHY_BIN" absrel \
        --alignment alignment.fasta \
        --tree tree.nwk \
        --branches "$HYPHY_ABSREL_BRANCHES" \
        --output absrel.json
    ) 2>&1 | tee "$OUTROOT/logs/${panel_id}.hyphy_absrel.log"
  fi
done

echo "Benchmark execution completed. Summarize with:"
echo "python publication_benchmark/scripts/drosophila_05_summarize_babappa_absrel.py --panel $PANEL --results-root $OUTROOT --outdir $OUTROOT/summary"
