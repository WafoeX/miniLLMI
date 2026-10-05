#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
need "$PYTHON"; need git; need nvidia-smi
require_clean_source
[[ -x "$BUILD_DIR/bench_gemm" ]] || { echo "Run scripts/build_server.sh first" >&2; exit 1; }
# Reserved output/provenance/experiment options cannot be overridden.
for arg in "$@"; do
    case "$arg" in --csv|--raw-dir|--run-id|--experiment|--kernel|--build-info|--help)
        echo "Reserved runner option: $arg" >&2; exit 1;; esac
done
mkdir -p "$ROOT/results/gemm/raw"
LOCK="$ROOT/results/.stage0.lock"
if ! mkdir "$LOCK" 2>/dev/null; then echo "Another Stage 0 runner holds $LOCK (do not run CSV writers concurrently)" >&2; exit 1; fi
RUN_DIR=""
cleanup() {
    local rc=$?
    if [[ -n "$RUN_DIR" ]]; then printf 'runner_exit_status=%s\n' "$rc" > "$RUN_DIR/runner_status.txt"; fi
    rmdir "$LOCK"
}
trap cleanup EXIT
REPEATS="${REPEATS:-1}"
[[ "$REPEATS" =~ ^[1-9][0-9]*$ ]] || { echo "REPEATS must be positive" >&2; exit 1; }
for (( rep=1; rep<=REPEATS; rep++ )); do
    ID="$(new_id)-$rep"
    # Do not let an ID collision overwrite the previous run's status in the EXIT trap.
    RUN_DIR=""
    NEW_DIR="$ROOT/results/gemm/raw/$ID"
    mkdir "$NEW_DIR"
    RUN_DIR="$NEW_DIR"
    "$PYTHON" tools/provenance.py > "$RUN_DIR/current_source.json"
    "$BUILD_DIR/bench_gemm" --build-info > "$RUN_DIR/current_binary.json"
    "$PYTHON" - "$RUN_DIR/current_source.json" "$RUN_DIR/current_binary.json" <<'PY'
import json,sys
source,build=(json.load(open(p,encoding="utf-8")) for p in sys.argv[1:])
if source["source_dirty"] or build["build_type"] != "Release" or any(source[k] != build[k] for k in ("commit","source_digest","source_dirty")):
    sys.exit("Stale binary/dirty code/non-Release build: rebuild after git pull/commit")
PY
    [[ -f "$BUILD_DIR/stage0_build/build_info.json" ]] || { echo "Missing build evidence: use build_server.sh" >&2; exit 1; }
    cmp "$BUILD_DIR/stage0_build/build_info.json" "$RUN_DIR/current_binary.json" || { echo "Build logs do not match binary: use build_server.sh again" >&2; exit 1; }
    cp -R "$BUILD_DIR/stage0_build" "$RUN_DIR/build"
    "$ROOT/scripts/collect_environment.sh" "$RUN_DIR/environment" 2>&1 | tee "$RUN_DIR/environment.log"
    TEST_LOG="$RUN_DIR/ctest.log" "$ROOT/scripts/run_tests.sh"
    nvidia-smi -q > "$RUN_DIR/nvidia_smi_before.txt"
    printf '%q ' "$BUILD_DIR/bench_gemm" --kernel all --warmup 10 --iterations 50 "$@" > "$RUN_DIR/command.txt"
    printf '\n' >> "$RUN_DIR/command.txt"
    "$BUILD_DIR/bench_gemm" --kernel all --warmup 10 --iterations 50 "$@" \
        --experiment baseline --run-id "$ID" --raw-dir "$RUN_DIR" \
        --csv "$ROOT/results/gemm/baseline.csv" 2>&1 | tee "$RUN_DIR/benchmark.log"
    nvidia-smi -q > "$RUN_DIR/nvidia_smi_after.txt"
    "$PYTHON" tools/provenance.py > "$RUN_DIR/source_after.json"
    cmp "$RUN_DIR/current_source.json" "$RUN_DIR/source_after.json" || { echo "Source changed during experiment" >&2; exit 1; }
    "$PYTHON" tools/analyze_results.py --run-id "$ID" \
        --csv "$ROOT/results/gemm/baseline.csv" --raw-root "$ROOT/results/gemm/raw" \
        --output "$RUN_DIR/baseline.md"
    cp "$RUN_DIR/baseline.md" "$ROOT/docs/baseline.md"
    printf 'runner_exit_status=0\n' > "$RUN_DIR/runner_status.txt"
    echo "Stage 0 run saved: $RUN_DIR"
done
