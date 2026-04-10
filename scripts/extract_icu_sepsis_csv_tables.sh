#!/usr/bin/env bash
# Extract optional legacy ICU-Sepsis CSV tarball into paths.json → icu_sepsis_csv_tables_dir
# (same tree as unified MIMIC rebuild output).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE}/.." && pwd)"
# Default: legacy tarball + published dir (matches paths.json)
ARCHIVE="${1:-${REPO}/data_mngmt/legacy/icu-sepsis-csv-tables-legacy.tar.gz}"
DEST="${2:-${REPO}/data_mngmt/generated/icu_sepsis_csv_tables}"
if [[ ! -f "${ARCHIVE}" ]]; then
  echo "Missing archive: ${ARCHIVE}" >&2
  echo "Usage: $0 [path/to/archive.tar.gz] [destination_dir]" >&2
  exit 1
fi
mkdir -p "${DEST}"
tar -xzf "${ARCHIVE}" -C "${DEST}"
echo "Extracted to ${DEST} (set paths.json icu_sepsis_csv_tables_dir to this folder)."
