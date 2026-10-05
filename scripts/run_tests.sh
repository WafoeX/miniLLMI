#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
need ctest
LOG="${TEST_LOG:-$BUILD_DIR/stage0_build/ctest.log}"
mkdir -p "$(dirname "$LOG")"
# Server build contains GPU tests; CPU-only local build intentionally does not.
ctest --test-dir "$BUILD_DIR" --output-on-failure --no-tests=error 2>&1 | tee "$LOG"
