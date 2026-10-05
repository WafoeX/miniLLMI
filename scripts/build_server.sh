#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
need cmake; need git; need nvcc; need "$PYTHON"
require_clean_source
mkdir -p "$BUILD_DIR/stage0_build"
mkdir -p "$ROOT/results/environment/builds"
LOG_DIR="$ROOT/results/environment/builds/$(new_id)"
mkdir "$LOG_DIR"
trap 'rc=$?; printf "build_exit_status=%s\n" "$rc" > "$LOG_DIR/status.txt"' EXIT
"$PYTHON" tools/provenance.py > "$LOG_DIR/source.json"
{
    printf 'CONFIGURE:'
    printf ' %q' cmake -S "$ROOT" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release -DENABLE_CUDA=ON "$@"
    printf '\n'
} > "$LOG_DIR/commands.txt"
# Defaults to querying the visible GPU (native), never a hardcoded GPU architecture.
cmake -S "$ROOT" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release -DENABLE_CUDA=ON "$@" 2>&1 | tee "$LOG_DIR/configure.log"
printf 'BUILD: cmake --build %q --parallel %q --verbose\n' "$BUILD_DIR" "${JOBS:-4}" >> "$LOG_DIR/commands.txt"
cmake --build "$BUILD_DIR" --parallel "${JOBS:-4}" --verbose 2>&1 | tee "$LOG_DIR/build.log"
cp "$BUILD_DIR/CMakeCache.txt" "$LOG_DIR/CMakeCache.txt"
cp "$BUILD_DIR/compile_commands.json" "$LOG_DIR/compile_commands.json"
"$BUILD_DIR/bench_gemm" --build-info > "$LOG_DIR/build_info.json"
"$PYTHON" - "$LOG_DIR/source.json" "$LOG_DIR/build_info.json" <<'PY'
import json,sys
source,build=(json.load(open(p,encoding="utf-8")) for p in sys.argv[1:])
if build["build_type"] != "Release" or any(source[k] != build[k] for k in ("commit","source_digest","source_dirty")):
    sys.exit("Build provenance mismatch, concurrent source edit, or non-Release build")
PY
# Copy a complete build evidence snapshot into the ignored build directory for the runner.
cp "$LOG_DIR/commands.txt" "$LOG_DIR/configure.log" "$LOG_DIR/build.log" "$LOG_DIR/source.json" \
    "$LOG_DIR/build_info.json" "$LOG_DIR/CMakeCache.txt" "$LOG_DIR/compile_commands.json" "$BUILD_DIR/stage0_build/"
printf '%s\n' "$LOG_DIR" > "$BUILD_DIR/stage0_build/original_log_dir.txt"
"$ROOT/scripts/collect_environment.sh" "$LOG_DIR/environment"
echo "Release build complete. Logs: $LOG_DIR"
