###############################################################################
# BABAPPA LONG RUN: INDEPENDENT MONITOR + INTEGRITY CHECKS
# Run from any terminal. No background work; all commands are immediate.
###############################################################################

RUN_DIR="/Users/krishnendu/Desktop/babappa/results_independent/run_20260227_091719"
PACK_DIR="$RUN_DIR/00_benchmark_pack"
AUDIT_DIR="$RUN_DIR/01_audit"

echo "RUN_DIR = $RUN_DIR"
echo "PACK_DIR = $PACK_DIR"
echo "AUDIT_DIR = $AUDIT_DIR"
echo "----"

###############################################################################
# 1) Confirm the run is alive and see CPU + children
###############################################################################
# Show sanity.sh / babappa processes
ps aux | egrep -i "sanity\.sh|babappa|hyphy|busted" | egrep -v egrep

echo "----"
# If you want a live view (press q to quit):
# top -o cpu

###############################################################################
# 2) Check that the run is making progress (file growth)
###############################################################################
# List newest files in the run directory (should change as work proceeds)
echo "Newest files in run dir:"
find "$RUN_DIR" -type f -maxdepth 4 -print0 2>/dev/null | xargs -0 ls -lt | head -n 25

echo "----"
# Check for partial outputs (these often appear before the final TSV)
echo "Raw folder snapshot:"
ls -lah "$PACK_DIR/raw" 2>/dev/null || echo "raw/ not created yet"

###############################################################################
# 3) Check whether workers are writing per-gene shards (common pattern)
###############################################################################
# Search for intermediate per-gene outputs (adjust patterns if needed)
echo "Searching for intermediate outputs (shards/logs)..."
find "$PACK_DIR" -type f \( -name "*part*" -o -name "*shard*" -o -name "*.log" -o -name "*worker*" \) 2>/dev/null | head -n 50

###############################################################################
# 4) Tail the most relevant logs (if present)
###############################################################################
echo "----"
echo "Tailing logs (if available):"
for f in \
  "$RUN_DIR/logs/00_benchmark_realdata.log" \
  "$RUN_DIR/logs/00_sanity.log" \
  "$RUN_DIR/logs/01_validate_null.log" \
  "$RUN_DIR/logs/02_bias_audit.log" \
  "$RUN_DIR/logs/03_hiv_benchmark.log" \
  "$RUN_DIR/00_benchmark_realdata.log" \
  "$RUN_DIR/sanity.log" \
  "$PACK_DIR/logs/benchmark.log" \
; do
  if [[ -f "$f" ]]; then
    echo "---- tail: $f"
    tail -n 40 "$f"
  fi
done

###############################################################################
# 5) Integrity: verify the "shared candidate set" and "testable set" artifacts
###############################################################################
echo "----"
echo "Looking for candidate/testable set artifacts:"
find "$RUN_DIR" -type f \( -name "*candidates*.tsv" -o -name "*testable*diff*.tsv" -o -name "*drop_audit*.tsv" \) 2>/dev/null | sed 's/^/  - /'

###############################################################################
# 6) Once finished: quick post-run validation commands (safe to run anytime)
###############################################################################
# These will only succeed after the files exist; until then they print messages.

echo "----"
BABAPPA_TSV="$PACK_DIR/raw/babappa_results.tsv"
BASELINE_TSV="$PACK_DIR/raw/baseline_all.tsv"
ORTHO_TSV="$PACK_DIR/tables/ortholog_results.tsv"
AUDIT_JSON="$AUDIT_DIR/manifests/audit_manifest.json"

if [[ -f "$BABAPPA_TSV" ]]; then
  echo "[OK] Found $BABAPPA_TSV"
  echo "Top lines:"
  head -n 3 "$BABAPPA_TSV"
  echo "Row count:"
  (tail -n +2 "$BABAPPA_TSV" | wc -l) | awk '{print "  n_rows=" $1}'
else
  echo "[WAIT] Not yet created: $BABAPPA_TSV"
fi

if [[ -f "$BASELINE_TSV" ]]; then
  echo "[OK] Found $BASELINE_TSV"
  head -n 3 "$BASELINE_TSV"
else
  echo "[WAIT] Not yet created: $BASELINE_TSV"
fi

if [[ -f "$ORTHO_TSV" ]]; then
  echo "[OK] Found $ORTHO_TSV"
  head -n 5 "$ORTHO_TSV"
else
  echo "[WAIT] Not yet created: $ORTHO_TSV"
fi

if [[ -f "$AUDIT_JSON" ]]; then
  echo "[OK] Found $AUDIT_JSON"
  # Print the key flags if jq is available; otherwise use grep
  if command -v jq >/dev/null 2>&1; then
    jq '{severe_inflation, ks_uniform_pvalue, size_hat_0_05, frozen_energy_invariant_fail, testable_set_mismatch, testable_set_size, n_units, n_success_babappa, n_success_busted}' "$AUDIT_JSON"
  else
    egrep -n "severe_inflation|ks_uniform|size_hat|frozen_energy|testable_set|n_units|n_success" "$AUDIT_JSON" || true
  fi
else
  echo "[WAIT] Not yet created: $AUDIT_JSON"
fi

###############################################################################
# 7) Fail-fast stall detector (manual)
###############################################################################
# If you suspect a stall:
#   - Check CPU still active
#   - Check newest file timestamp in RUN_DIR
# If newest file hasn't changed for a long time AND CPU is near 0, the run likely stalled.
###############################################################################
echo "----"
echo "Newest file timestamp (run dir):"
find "$RUN_DIR" -type f -print0 2>/dev/null | xargs -0 stat -f "%m %N" 2>/dev/null | sort -nr | head -n 5