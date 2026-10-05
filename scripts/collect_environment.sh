#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
need "$PYTHON"; need git; need cmake
OUT="${1:-$ROOT/results/environment/$(new_id)}"
[[ "$OUT" = /* ]] || OUT="$ROOT/$OUT"
mkdir -p "$OUT"
# Refuse accidental replacement of a previous capture, including failed captures.
if [[ -e "$OUT/system_info.txt" ]]; then echo "Environment capture already exists: $OUT" >&2; exit 1; fi
capture() {
    local file="$1"; shift
    local rc=0
    { printf 'COMMAND:'; printf ' %q' "$@"; printf '\n'; "$@"; } > "$OUT/$file" 2>&1 || rc=$?
    if (( rc != 0 )); then printf '\nEXIT_STATUS=%s\n' "$rc" >> "$OUT/$file"; return "$rc"; fi
}
{
    date -u +%Y-%m-%dT%H:%M:%SZ
    uname -a
    printf 'CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES:-<unset>}"
    printf 'PATH=%s\n' "$PATH"
    printf '\nCPU information:\n'
    if command -v lscpu >/dev/null 2>&1; then lscpu; else sysctl -n machdep.cpu.brand_string || true; fi
    printf '\nCompiler:\n'; "${CXX:-c++}" --version
    printf '\nCMake:\n'; cmake --version
    printf '\nPython:\n'; "$PYTHON" --version
    printf '\nGit:\n'; git --version
    printf '\nNsight Systems:\n'; if command -v nsys >/dev/null 2>&1; then nsys --version; else echo unavailable; fi
    printf '\nNsight Compute:\n'; if command -v ncu >/dev/null 2>&1; then ncu --version; else echo unavailable; fi
} > "$OUT/system_info.txt" 2>&1
capture git_commit.txt git rev-parse HEAD
capture git_status.txt git status --porcelain=v1 --untracked-files=all
"$PYTHON" tools/provenance.py > "$OUT/source.json"
rc=0
capture nvidia_smi.txt nvidia-smi || rc=1
capture nvcc_version.txt nvcc --version || rc=1
capture gcc_version.txt gcc --version || rc=1
capture cmake_version.txt cmake --version || rc=1
if [[ -x "$BUILD_DIR/gpu_info" ]]; then
    capture gpu_info.txt "$BUILD_DIR/gpu_info" || rc=1
else
    printf 'gpu_info executable missing: %s/gpu_info\n' "$BUILD_DIR" > "$OUT/gpu_info.txt"
    rc=1
fi
# CUDA 13 removes memoryClockRate from cudaDeviceProp. Use actual driver clock telemetry,
# preserving UUIDs to disambiguate device indices / CUDA_VISIBLE_DEVICES remapping.
capture gpu_clocks.txt nvidia-smi --query-gpu=uuid,clocks.max.memory,clocks.current.memory,clocks.current.sm,temperature.gpu,pstate --format=csv || rc=1
if [[ -f "$OUT/gpu_clocks.txt" ]]; then
    printf '\nDriver clock telemetry (MHz, match by UUID):\n' >> "$OUT/gpu_info.txt"
    while IFS= read -r line; do printf '%s\n' "$line"; done < "$OUT/gpu_clocks.txt" >> "$OUT/gpu_info.txt"
fi
# Canonical filenames are latest snapshots; immutable copies stay in OUT.
mkdir -p "$ROOT/results/environment"
for name in gpu_info.txt nvidia_smi.txt nvcc_version.txt system_info.txt gcc_version.txt cmake_version.txt git_commit.txt gpu_clocks.txt; do
    if [[ "$OUT" != "$ROOT/results/environment" && -f "$OUT/$name" ]]; then cp "$OUT/$name" "$ROOT/results/environment/$name"; fi
done
printf '%s\n' "$OUT" > "$ROOT/results/environment/latest_capture.txt"
printf 'environment_exit_status=%s\n' "$rc" > "$OUT/capture_status.txt"
echo "Environment saved: $OUT (status=$rc)"
exit "$rc"
