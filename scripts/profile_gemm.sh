#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
TOOL="${1:-nsys}"
KERNEL="${2:-naive}"
[[ "$TOOL" == nsys || "$TOOL" == ncu ]] || { echo "usage: profile_gemm.sh nsys|ncu naive|cublas" >&2; exit 1; }
[[ "$KERNEL" == naive || "$KERNEL" == cublas ]] || { echo "invalid kernel" >&2; exit 1; }
need "$TOOL"; need "$PYTHON"
require_clean_source
[[ -x "$BUILD_DIR/bench_gemm" ]] || { echo "Build first" >&2; exit 1; }
mkdir -p "$ROOT/results/profiling/$TOOL"
LOCK="$ROOT/results/.stage0.lock"
mkdir "$LOCK" || { echo "Another experiment holds $LOCK" >&2; exit 1; }
OUT=""
trap 'rc=$?; if [[ -n "$OUT" ]]; then printf "profile_exit_status=%s\n" "$rc" > "$OUT/status.txt"; fi; rmdir "$LOCK"' EXIT
ID="$(new_id)-$KERNEL"
NEW_DIR="$ROOT/results/profiling/$TOOL/$ID"
mkdir "$NEW_DIR"
OUT="$NEW_DIR"
"$PYTHON" tools/provenance.py > "$OUT/source.json"
"$BUILD_DIR/bench_gemm" --build-info > "$OUT/binary.json"
"$PYTHON" - "$OUT/source.json" "$OUT/binary.json" <<'PY'
import json,sys
s,b=(json.load(open(p,encoding="utf-8")) for p in sys.argv[1:])
if b["build_type"] != "Release" or any(s[k] != b[k] for k in ("commit","source_digest","source_dirty")):
    sys.exit("Stale binary or non-Release build")
PY
"$ROOT/scripts/collect_environment.sh" "$OUT/environment"
ARGS=("$BUILD_DIR/bench_gemm" --kernel "$KERNEL" --m 4096 --n 4096 --k 4096 \
    --warmup 10 --iterations 50 --experiment profiling --run-id "$ID" \
    --raw-dir "$OUT/raw" --csv "$OUT/profile_timing.csv")
if [[ "$TOOL" == nsys ]]; then
    CMD=(nsys profile '--trace=cuda,cublas,osrt' --output "$OUT/trace" "${ARGS[@]}")
else
    # Initial correctness launch + 10 warmups -> profile the first measured naive launch.
    # cuBLAS internal kernel names vary; capture one launch without guessing its name.
    FILTER=()
    [[ "$KERNEL" != naive ]] || FILTER=(--kernel-name 'regex:sgemm_v0_naive')
    CMD=(ncu --set full "${FILTER[@]}" --launch-skip 11 --launch-count 1 --export "$OUT/metrics" "${ARGS[@]}")
fi
printf '%q ' "${CMD[@]}" > "$OUT/command.txt"; printf '\n' >> "$OUT/command.txt"
"${CMD[@]}" 2>&1 | tee "$OUT/profile.log"
if [[ "$TOOL" == nsys ]]; then REPORT="$OUT/trace.nsys-rep"; else REPORT="$OUT/metrics.ncu-rep"; fi
[[ -s "$REPORT" ]] || { echo "Profiler did not produce a nonempty report" >&2; exit 1; }
echo "Profile saved: $OUT. Profiler timings MUST NOT be compared with unprofiled baseline."
