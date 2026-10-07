#!/usr/bin/env bash
# Capture one registered CUDA GEMM kernel with Nsight Systems or Nsight Compute.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

usage() {
    cat >&2 <<'EOF'
usage: profile_gemm.sh nsys|ncu v0|v1|cublas [--artifact-location LOCATION]

Compatibility aliases: naive=v0, tiled=v1.  The selected kernel is run alone at
4096^3 with the profiling experiment label.  Profile timing is never benchmark
evidence.
EOF
    exit 2
}

[[ $# -ge 2 ]] || usage
TOOL="$1"
REQUESTED_KERNEL="$2"
shift 2
ARTIFACT_LOCATION="${PROFILE_ARTIFACT_LOCATION:-}"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --artifact-location)
            [[ $# -ge 2 ]] || usage
            ARTIFACT_LOCATION="$2"
            shift 2
            ;;
        *) usage ;;
    esac
done

case "$TOOL" in nsys|ncu) ;; *) usage ;; esac
case "$REQUESTED_KERNEL" in
    v0|naive|sgemm_v0_naive) KERNEL="sgemm_v0_naive" ;;
    v1|tiled|sgemm_v1_tiled) KERNEL="sgemm_v1_tiled" ;;
    cublas) KERNEL="cublas" ;;
    *) echo "invalid registered kernel: $REQUESTED_KERNEL" >&2; usage ;;
esac

need "$TOOL"; need "$PYTHON"
require_clean_source
[[ -x "$BUILD_DIR/bench_gemm" ]] || { echo "Build first: $BUILD_DIR/bench_gemm" >&2; exit 1; }
OUTPUT_ROOT="${PROFILE_OUTPUT_ROOT:-$ROOT/results/profiling}"
[[ "$OUTPUT_ROOT" = /* ]] || OUTPUT_ROOT="$ROOT/$OUTPUT_ROOT"
mkdir -p "$OUTPUT_ROOT/$TOOL"
LOCK="$ROOT/results/.profiling.lock"
mkdir "$LOCK" || { echo "Another profiling experiment holds $LOCK" >&2; exit 1; }
OUT=""
trap 'rc=$?; if [[ -n "$OUT" ]]; then printf "profile_exit_status=%s\n" "$rc" > "$OUT/status.txt"; fi; rmdir "$LOCK"' EXIT
ID="$(new_id)-${KERNEL}"
OUT="$OUTPUT_ROOT/$TOOL/$ID"
mkdir "$OUT"

"$PYTHON" tools/provenance.py > "$OUT/source.json"
"$BUILD_DIR/bench_gemm" --build-info > "$OUT/binary.json"
"$PYTHON" - "$OUT/source.json" "$OUT/binary.json" <<'PY'
import json, sys
source, binary = (json.load(open(path, encoding="utf-8")) for path in sys.argv[1:])
if binary["build_type"] != "Release" or any(source[key] != binary[key] for key in ("commit", "source_digest", "source_dirty")):
    sys.exit("Stale binary or non-Release build")
PY
"$ROOT/scripts/collect_environment.sh" "$OUT/environment"
ARGS=("$BUILD_DIR/bench_gemm" --kernel "$KERNEL" --m 4096 --n 4096 --k 4096 \
    --warmup 10 --iterations 50 --experiment profiling --run-id "$ID" \
    --raw-dir "$OUT/raw" --csv "$OUT/profile_timing.csv")
if [[ -n "${PROFILE_REFERENCE_CACHE_DIR:-}" ]]; then
    ARGS+=(--reference-cache-dir "$PROFILE_REFERENCE_CACHE_DIR")
fi
if [[ "$TOOL" == nsys ]]; then
    CMD=(nsys profile '--trace=cuda,cublas,osrt' --output "$OUT/trace" "${ARGS[@]}")
else
    # Initial correctness launch + 10 warmups precede the first measured launch.
    CMD=(ncu --set full --kernel-name "regex:${KERNEL}" --launch-skip 11 --launch-count 1 \
        --export "$OUT/metrics" "${ARGS[@]}")
fi
printf '%q ' "${CMD[@]}" > "$OUT/command.txt"; printf '\n' >> "$OUT/command.txt"
"${CMD[@]}" 2>&1 | tee "$OUT/profile.log"

if [[ "$TOOL" == nsys ]]; then
    REPORT="$OUT/trace.nsys-rep"
    nsys stats --report cuda_gpu_kern_sum,cuda_api_sum "$REPORT" > "$OUT/stats.txt"
    TEXT_EXPORT="$OUT/stats.txt"
else
    REPORT="$OUT/metrics.ncu-rep"
    ncu --import "$REPORT" --page details --csv > "$OUT/metrics.csv"
    ncu --query-metrics > "$OUT/available_metrics.txt"
    TEXT_EXPORT="$OUT/metrics.csv"
fi
[[ -s "$REPORT" ]] || { echo "Profiler did not produce a nonempty report" >&2; exit 1; }
[[ -s "$TEXT_EXPORT" ]] || { echo "Profiler did not produce a nonempty text export" >&2; exit 1; }
printf '%s\n' "${ARTIFACT_LOCATION:-unarchived; retain this report before cleaning the host}" > "$OUT/artifact_location.txt"
"$PYTHON" - "$OUT" "$TOOL" "$KERNEL" "$REPORT" "$ARTIFACT_LOCATION" <<'PY'
import hashlib, json, sys
from pathlib import Path
out = Path(sys.argv[1])
tool, kernel = sys.argv[2:4]
report = Path(sys.argv[4])
location = sys.argv[5]
def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
manifest = {
    "schema_version": 1,
    "stage": "stage10",
    "tool": tool,
    "kernel": kernel,
    "shape": [4096, 4096, 4096],
    "warmup": 10,
    "iterations": 50,
    "source": json.loads((out / "source.json").read_text()),
    "binary": json.loads((out / "binary.json").read_text()),
    "command": (out / "command.txt").read_text(),
    "report": Path(report).name,
    "report_sha256": sha256(report),
    "artifact_location": location or None,
    "timing_scope": "profiling only; never formal benchmark evidence",
}
(out / "profile_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
PY
printf 'Profile saved: %s\n' "$OUT"
