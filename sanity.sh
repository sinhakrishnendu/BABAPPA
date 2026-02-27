#!/usr/bin/env bash
# ==============================================================================
# BABAPPA independent sanity + audit runner (current CLI compatible)
# ==============================================================================

set -euo pipefail
IFS=$'\n\t'

ts() { date '+%F %T'; }
log() { echo "[$(ts)] $*"; }
die() { echo "[$(ts)] ERROR: $*" >&2; exit 2; }

# -----------------------------
# 0) Config
# -----------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$SCRIPT_DIR}"
WORKSPACE="${WORKSPACE:-$REPO_ROOT}"
RESULTS_ROOT="${RESULTS_ROOT:-$REPO_ROOT/results_independent}"
RESULT_PACK="${RESULT_PACK:-}"
SEED="${SEED:-12345}"
ORTHO_REAL_PRESET="${ORTHO_REAL_PRESET:-ortholog_real_v12}"
RUN_MODE="${RUN_MODE:-publication}"
ALLOW_QC_FAIL="${ALLOW_QC_FAIL:-false}"

# Null/audit settings.
G_NULL="${G_NULL:-300}"   # number of observed null replicates in matched-null audit
N_CAL="${N_CAL:-999}"     # calibration N per context
JOBS="${JOBS:-0}"         # 0=auto

# Optional HIV env import+benchmark.
RUN_HIV="${RUN_HIV:-false}"
HIV_DIR="${HIV_DIR:-$REPO_ROOT/data/manual/hiv_env_b}"
HIV_ENV="${HIV_ENV:-$HIV_DIR/hiv-db-env.fasta}"
HIV_PROVENANCE="${HIV_PROVENANCE:-$HIV_DIR/provenance.json}"
HIV_ALIGNMENT_ID="${HIV_ALIGNMENT_ID:-LANL-ENV-B-MANUAL}"

cd "$REPO_ROOT" || die "Cannot cd to REPO_ROOT=$REPO_ROOT"

if [[ -z "${BABAPPA_BIN:-}" ]]; then
  if [[ -x "$REPO_ROOT/.venv/bin/babappa" ]]; then
    BABAPPA_BIN="$REPO_ROOT/.venv/bin/babappa"
  elif command -v babappa >/dev/null 2>&1; then
    BABAPPA_BIN="$(command -v babappa)"
  else
    die "Could not find babappa executable. Set BABAPPA_BIN or install into .venv."
  fi
fi

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
    PYTHON_BIN="$REPO_ROOT/.venv/bin/python"
  else
    PYTHON_BIN="python3"
  fi
fi

STAMP="$(date '+%Y%m%d_%H%M%S')"
RUN_DIR="$RESULTS_ROOT/run_$STAMP"
LOG_DIR="$RUN_DIR/logs"
mkdir -p "$RUN_DIR" "$LOG_DIR"

log "Repo root: $REPO_ROOT"
log "Run dir:   $RUN_DIR"
log "BABAPPA:   $BABAPPA_BIN"

run_cmd() {
  local name="$1"; shift
  local logfile="$LOG_DIR/${name}.log"
  log "RUN $name"
  log "CMD $*"
  set +e
  ( set -x; "$@" ) 2>&1 | tee "$logfile"
  local rc=${PIPESTATUS[0]}
  set -e
  if [[ "$rc" -ne 0 ]]; then
    die "$name failed with exit=$rc (see $logfile)"
  fi
  log "DONE $name"
}

# -----------------------------
# 1) Resolve GENESET_DIR robustly
# -----------------------------
if [[ -z "${GENESET_DIR:-}" && -n "$RESULT_PACK" ]]; then
  pack_lc="$(echo "$RESULT_PACK" | tr '[:upper:]' '[:lower:]')"
  if [[ "$pack_lc" == *"304"* ]]; then
    GENESET_DIR="$WORKSPACE/data/orthomam_v12_real_304/genes"
  elif [[ "$pack_lc" == *"qc"* ]]; then
    GENESET_DIR="$WORKSPACE/data/orthomam_v12_real_qc/genes"
  fi
fi

if [[ -z "${GENESET_DIR:-}" ]]; then
  CANDIDATES=(
    "$WORKSPACE/data/orthomam_v12_real_304/genes"
    "$WORKSPACE/data/orthomam_v12_real_qc/genes"
    "$WORKSPACE/data/orthomam_v12_real/genes"
    "$WORKSPACE/data/orthomam_v12_real_final/genes"
  )
  for d in "${CANDIDATES[@]}"; do
    if [[ -d "$d" ]]; then
      GENESET_DIR="$d"
      break
    fi
  done
fi

if [[ -z "${GENESET_DIR:-}" || ! -d "${GENESET_DIR:-}" ]]; then
  echo "ERROR: GENESET_DIR not found: ${GENESET_DIR:-<unset>}" >&2
  echo "HINT: export GENESET_DIR=/full/path/to/.../genes" >&2
  echo "CANDIDATES:" >&2
  find "$WORKSPACE/data" -maxdepth 3 -type d -name genes 2>/dev/null | sed 's/^/  - /' >&2 || true
  exit 2
fi

DATASET_ROOT="$(dirname "$GENESET_DIR")"
DATASET_JSON="$DATASET_ROOT/dataset.json"
NEUTRAL_JSON="$DATASET_ROOT/neutral_model.json"
TREE_NWK="$DATASET_ROOT/tree.nwk"
GENE_COUNT="$(find "$GENESET_DIR" -maxdepth 1 -type f \( -name '*.fna' -o -name '*.fasta' -o -name '*.fa' \) | wc -l | tr -d ' ')"

if [[ "$ORTHO_REAL_PRESET" == "ortholog_real_v12" && "$GENE_COUNT" -lt 300 && -d "$WORKSPACE/data/orthomam_v12_real_304/genes" ]]; then
  GENESET_DIR="$WORKSPACE/data/orthomam_v12_real_304/genes"
  DATASET_ROOT="$(dirname "$GENESET_DIR")"
  DATASET_JSON="$DATASET_ROOT/dataset.json"
  NEUTRAL_JSON="$DATASET_ROOT/neutral_model.json"
  TREE_NWK="$DATASET_ROOT/tree.nwk"
  GENE_COUNT="$(find "$GENESET_DIR" -maxdepth 1 -type f \( -name '*.fna' -o -name '*.fasta' -o -name '*.fa' \) | wc -l | tr -d ' ')"
fi

log "Using GENESET_DIR: $GENESET_DIR"
log "Dataset root:      $DATASET_ROOT"
log "Gene files:        $GENE_COUNT"
[[ "$GENE_COUNT" -gt 0 ]] || die "No FASTA/FNA files found in $GENESET_DIR"
[[ -f "$DATASET_JSON" ]] || die "dataset.json missing at $DATASET_JSON"
[[ -f "$NEUTRAL_JSON" ]] || log "WARN: neutral_model.json missing at $NEUTRAL_JSON"
[[ -f "$TREE_NWK" ]] || log "WARN: tree.nwk missing at $TREE_NWK"

# -----------------------------
# 2) Run ortholog benchmark + audit + compare
# -----------------------------
PACK_OUT="$RUN_DIR/00_benchmark_pack"
AUDIT_OUT="$RUN_DIR/01_audit"
COMPARE_OUT="$RUN_DIR/02_compare"
mkdir -p "$PACK_OUT" "$AUDIT_OUT" "$COMPARE_OUT"

if [[ "$ALLOW_QC_FAIL" == "true" ]]; then
  run_cmd "00_benchmark_realdata" \
    "$BABAPPA_BIN" benchmark realdata \
      --preset "$ORTHO_REAL_PRESET" \
      --data "$DATASET_ROOT" \
      --outdir "$PACK_OUT" \
      --N "$N_CAL" \
      --seed "$SEED" \
      --mode "$RUN_MODE" \
      --jobs "$JOBS" \
      --allow-qc-fail
else
  run_cmd "00_benchmark_realdata" \
    "$BABAPPA_BIN" benchmark realdata \
      --preset "$ORTHO_REAL_PRESET" \
      --data "$DATASET_ROOT" \
      --outdir "$PACK_OUT" \
      --N "$N_CAL" \
      --seed "$SEED" \
      --mode "$RUN_MODE" \
      --jobs "$JOBS"
fi

run_cmd "01_audit_ortholog" \
  "$BABAPPA_BIN" audit ortholog \
    --pack "$PACK_OUT" \
    --outdir "$AUDIT_OUT" \
    --seed "$SEED" \
    --null_N "$N_CAL" \
    --null_G "$G_NULL" \
    --jobs "$JOBS"

run_cmd "02_compare_report" \
  "$BABAPPA_BIN" report compare \
    --pack "$PACK_OUT" \
    --audit "$AUDIT_OUT" \
    --outdir "$COMPARE_OUT"

# Publication-scale required artifacts + strict pass/fail gate.
REQUIRED_FILES=(
  "$PACK_OUT/raw/babappa_results.tsv"
  "$PACK_OUT/raw/baseline_all.tsv"
  "$PACK_OUT/raw/drop_audit.tsv"
  "$PACK_OUT/tables/ortholog_results.tsv"
  "$PACK_OUT/tables/testable_set_diff.tsv"
  "$PACK_OUT/manifests/provenance_freeze.json"
  "$PACK_OUT/report/report.pdf"
  "$AUDIT_OUT/manifests/audit_manifest.json"
  "$AUDIT_OUT/tables/null_uniformity.tsv"
  "$AUDIT_OUT/tables/null_size.tsv"
)
for f in "${REQUIRED_FILES[@]}"; do
  [[ -f "$f" ]] || die "missing required output: $f"
done

STRICT_CAUSE="$("$PYTHON_BIN" - "$AUDIT_OUT/manifests/audit_manifest.json" <<'PY'
import json
import sys
from pathlib import Path

p = Path(sys.argv[1])
m = json.loads(p.read_text(encoding="utf-8"))
checks = [
    ("severe_inflation", bool(m.get("severe_inflation", False)) is False),
    ("frozen_energy_invariant_fail", bool(m.get("frozen_energy_invariant_fail", False)) is False),
    ("testable_set_mismatch", bool(m.get("testable_set_mismatch", False)) is False),
]
n_units = int(m.get("n_units", 0))
n_success_babappa = int(m.get("n_success_babappa", -1))
n_success_busted = int(m.get("n_success_busted", -1))
checks.append(("n_success_babappa", n_units > 0 and n_success_babappa == n_units))
checks.append(("n_success_busted", n_units > 0 and n_success_busted == n_units))
failed = [name for name, ok in checks if not ok]
if failed:
    print("FAIL:" + ",".join(failed))
else:
    print("OK")
PY
)"
if [[ "$STRICT_CAUSE" != "OK" ]]; then
  die "$STRICT_CAUSE"
fi

log "AUDIT GATE PASS: severe_inflation=false frozen_energy_invariant_fail=false testable_set_mismatch=false success counts match n_units."
log "Summary manifest: $AUDIT_OUT/manifests/audit_manifest.json"
log "Diff table:       $PACK_OUT/tables/testable_set_diff.tsv"

# -----------------------------
# 3) Optional HIV env benchmark from local manual alignment
# -----------------------------
if [[ "$RUN_HIV" == "true" ]]; then
  if [[ -f "$HIV_ENV" ]]; then
    HIV_DATASET_OUT="$RUN_DIR/03_hiv_env_dataset"
    HIV_BENCH_OUT="$RUN_DIR/04_hiv_env_benchmark"
    mkdir -p "$HIV_DATASET_OUT" "$HIV_BENCH_OUT"

    run_cmd "03_hiv_dataset_import" \
      "$BABAPPA_BIN" dataset import hiv \
        --alignment "$HIV_ENV" \
        --outdir "$HIV_DATASET_OUT" \
        --alignment-id "$HIV_ALIGNMENT_ID" \
        --provenance "$HIV_PROVENANCE" \
        --gene env \
        --subtype B \
        --recombination-policy subtype_filter

    run_cmd "04_hiv_realdata" \
      "$BABAPPA_BIN" benchmark realdata \
        --preset hiv_env_b_real \
        --data "$HIV_DATASET_OUT" \
        --outdir "$HIV_BENCH_OUT" \
        --N "$N_CAL" \
        --seed "$SEED" \
        --results-only \
        --allow-qc-fail \
        --allow-baseline-fail
  else
    log "RUN_HIV=true but HIV alignment missing: $HIV_ENV"
  fi
fi

log "============================================================"
log "ALL DONE."
log "Run folder: $RUN_DIR"
log "Logs:       $LOG_DIR"
log "Pack:       $PACK_OUT"
log "Audit:      $AUDIT_OUT"
log "Compare:    $COMPARE_OUT"
if [[ "$RUN_HIV" == "true" ]]; then
  log "HIV env benchmark requested (see run folders above)."
fi
log "============================================================"
