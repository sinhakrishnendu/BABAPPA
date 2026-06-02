#!/usr/bin/env bash
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
