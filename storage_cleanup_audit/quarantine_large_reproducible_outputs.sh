#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/Documents/GitHub/BABAPPA"

echo "USER-RUN ONLY -- REVIEW BEFORE EXECUTION"
STAMP="$(date +%Y%m%d_%H%M%S)"
QUARANTINE="${BABAPPA_QUARANTINE_DIR:-$HOME/BABAPPA_STORAGE_QUARANTINE_${STAMP}}"
LOG="storage_cleanup_audit/quarantine_move_log.tsv"
mkdir -p "$QUARANTINE" "storage_cleanup_audit"
printf "path\tdestination\tstatus\n" > "$LOG"
du -sh . | tee storage_cleanup_audit/size_before_quarantine.txt

move_candidate() {
  local rel="$1"
  [ -n "$rel" ] || return 0
  [ "$rel" != "." ] || return 0
  case "$rel" in
    .git|.git/*|src|src/*|tests|tests/*|docs|docs/*|examples|examples/*|Manuscript|Manuscript/*|manuscript|manuscript/*|deployable_model_conservative_branch_site_100k_mps|deployable_model_conservative_branch_site_100k_mps/*)
      printf "%s\t%s\tprotected_skip\n" "$rel" "" >> "$LOG"
      return 0
      ;;
  esac
  if [ ! -e "$rel" ]; then
    printf "%s\t%s\tmissing_skip\n" "$rel" "" >> "$LOG"
    return 0
  fi
  local dest="$QUARANTINE/$rel"
  mkdir -p "$(dirname "$dest")"
  mv "$rel" "$dest"
  printf "%s\t%s\tmoved\n" "$rel" "$dest" >> "$LOG"
}

for table in storage_cleanup_audit/remove_candidates.tsv storage_cleanup_audit/archive_candidates.tsv; do
  [ -f "$table" ] || continue
  tail -n +2 "$table" | while IFS=$'\t' read -r rel _rest; do
    move_candidate "$rel"
  done
done

du -sh . | tee storage_cleanup_audit/size_after_quarantine.txt
du -sh "$QUARANTINE" | tee storage_cleanup_audit/quarantine_size.txt
echo "Quarantine folder: $QUARANTINE"
echo "Move log: $LOG"
