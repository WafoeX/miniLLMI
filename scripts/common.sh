#!/usr/bin/env bash
# Sourced by Stage 0 scripts; all paths and execution are rooted at the repository.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
BUILD_DIR="${BUILD_DIR:-$ROOT/build}"
[[ "$BUILD_DIR" = /* ]] || BUILD_DIR="$ROOT/$BUILD_DIR"
PYTHON="${PYTHON:-python3}"
export LC_ALL=C
need() { command -v "$1" >/dev/null 2>&1 || { echo "Required command not found: $1" >&2; exit 1; }; }
new_id() { printf '%s-%s' "$(date -u +%Y%m%dT%H%M%SZ)" "$$"; }
require_clean_source() {
    "$PYTHON" "$ROOT/tools/provenance.py" | "$PYTHON" -c '
import json,sys
p=json.load(sys.stdin)
if p["source_dirty"] or p["commit"] == "UNCOMMITTED":
    sys.exit("Commit all source changes before a server build/official benchmark (results and generated baseline report are excluded).")
print("Source commit: " + p["commit"])
'
}
