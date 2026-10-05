#!/usr/bin/env bash
set -euo pipefail
# Stage 0 exposes only GEMM. Do not pretend allocator/KV/quant benchmarks exist.
exec "$(dirname "${BASH_SOURCE[0]}")/run_gemm_benchmark.sh" "$@"
