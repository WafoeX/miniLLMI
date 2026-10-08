#!/usr/bin/env bash
# Execute the frozen Stage 18 RC matrix. Run only from a clean checkout at its RC tag.
set -euo pipefail

usage() {
  cat <<'EOF'
usage: scripts/run_stage18_evaluation.sh [--tag v1.0-rc4] [--run-id ID] [--jobs N] [--profile-artifact-location URI]

Runs the required CPU/T4 functional matrix and final performance/memory suite,
then generates a report solely by validating the captured raw artifacts. It
must be run on a single visible Tesla T4 from a clean RC checkout.
EOF
}

TAG=v1.0-rc4
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-stage18"
JOBS=2
PROFILE_ARTIFACT_LOCATION=""
while (($#)); do
  case "$1" in
    --tag) TAG=${2:?missing tag}; shift 2 ;;
    --run-id) RUN_ID=${2:?missing run ID}; shift 2 ;;
    --jobs) JOBS=${2:?missing job count}; shift 2 ;;
    --profile-artifact-location) PROFILE_ARTIFACT_LOCATION=${2:?missing artifact location}; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
  esac
done
[[ "$JOBS" =~ ^[1-9][0-9]*$ ]] || { echo '--jobs must be positive' >&2; exit 2; }

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
[[ "$(git describe --exact-match --tags HEAD 2>/dev/null || true)" == "$TAG" ]] || {
  echo "HEAD must be exactly tag $TAG" >&2; exit 1;
}
[[ -z "$(git status --porcelain)" ]] || { echo 'RC checkout must be clean before capture' >&2; exit 1; }
SOURCE_COMMIT=$(git rev-parse HEAD)
SOURCE_DIGEST=$(python3 tools/provenance.py --root "$ROOT" | python3 -c 'import json,sys; print(json.load(sys.stdin)["source_digest"])')
RESULT_DIR="results/release/${TAG}/${RUN_ID}"
[[ ! -e "$RESULT_DIR" ]] || { echo "result directory exists: $RESULT_DIR" >&2; exit 1; }
mkdir -p "$RESULT_DIR/models"
printf '%s\n' "$SOURCE_COMMIT" > "$RESULT_DIR/tested_commit.txt"
printf '%s\n' "$SOURCE_DIGEST" > "$RESULT_DIR/source_digest.txt"
printf '%s\n' "$TAG" > "$RESULT_DIR/release_tag.txt"
printf 'running\n' > "$RESULT_DIR/status.txt"

die() { printf 'failed\n' > "$RESULT_DIR/status.txt"; echo "stage18: $*" >&2; exit 1; }
trap 'status=$?; if (( status != 0 )); then printf "failed\n" > "$RESULT_DIR/status.txt"; printf "stage18 capture retained: %s\n" "$RESULT_DIR" >&2; fi' EXIT
run() {
  local label=$1; shift
  printf '+ %q ' "$@" | tee "$RESULT_DIR/${label}.command" >/dev/null
  printf '\n' >> "$RESULT_DIR/${label}.command"
  "$@" >"$RESULT_DIR/${label}.log" 2>&1 || die "$label failed; preserved $RESULT_DIR/${label}.log"
}
new_dir() {
  local base=$1 marker=$2 result
  result=$(find "$base" -mindepth 1 -maxdepth 1 -type d -newer "$marker" -print | sort)
  [[ $(printf '%s\n' "$result" | sed '/^$/d' | wc -l | tr -d ' ') == 1 ]] || die "expected one new result under $base"
  printf '%s\n' "$result"
}

printf 'T4 final evaluation: source=%s digest=%s result=%s\n' "$SOURCE_COMMIT" "$SOURCE_DIGEST" "$RESULT_DIR"
run nvidia-smi nvidia-smi
run nvcc-version nvcc --version
run cmake-version cmake --version

# C2: fresh test and production trees, then the matrix's CPU/mixed functional paths.
run configure-test cmake -S "$ROOT" -B "$RESULT_DIR/build-test" -G 'Unix Makefiles' \
  -DCMAKE_BUILD_TYPE=Release -DENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75 -DBUILD_TESTING=ON
run build-test cmake --build "$RESULT_DIR/build-test" --parallel "$JOBS"
run ctest-full ctest --test-dir "$RESULT_DIR/build-test" --output-on-failure
run ctest-gpu ctest --test-dir "$RESULT_DIR/build-test" --output-on-failure -L gpu
run ctest-release-focused ctest --test-dir "$RESULT_DIR/build-test" --output-on-failure \
  -R '^(cli_tools|generation|inference_benchmark_tools|quant_benchmark_tools|cuda_decoder|cuda_quantized_decoder|cuda_kv_decoder)$'
run configure-production cmake -S "$ROOT" -B "$RESULT_DIR/build-production" -G 'Unix Makefiles' \
  -DCMAKE_BUILD_TYPE=Release -DENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75 -DBUILD_TESTING=OFF
run build-production cmake --build "$RESULT_DIR/build-production" --parallel "$JOBS"

WEIGHTS=tests/fixtures/operators-v1/tiny-weights-v1.bin
FLOAT_MODEL="$RESULT_DIR/models/tiny-v1.mllm"
INT8_MODEL="$RESULT_DIR/models/tiny-v2-int8.mllm"
run convert-float "$RESULT_DIR/build-production/convert_tiny_model" "$WEIGHTS" "$FLOAT_MODEL"
run convert-int8 "$RESULT_DIR/build-production/convert_tiny_model" "$WEIGHTS" "$INT8_MODEL" --int8
run model-sha256 sha256sum "$FLOAT_MODEL" "$INT8_MODEL"

# C3: all controls are frozen by their stage runners; each runner self-validates raw data.
MARKER="$RESULT_DIR/.stage5-before"; touch "$MARKER"
run stage5-runner python3 tools/run_planner_benchmark.py --results-root "$RESULT_DIR/planner" --jobs "$JOBS"
STAGE5=$(new_dir "$RESULT_DIR/planner" "$MARKER")
mv "$STAGE5" "$RESULT_DIR/stage5"
STAGE5="$RESULT_DIR/stage5"
MARKER="$RESULT_DIR/.stage7-before"; touch "$MARKER"
run stage7-runner python3 tools/run_cpu_parallel_benchmark.py
STAGE7=$(new_dir results/cpu/parallel "$MARKER")
mv "$STAGE7" "$RESULT_DIR/stage7"
STAGE7="$RESULT_DIR/stage7"
MARKER="$RESULT_DIR/.stage9-before"; touch "$MARKER"
run stage9-runner python3 tools/run_cuda_gemm_stage9.py
STAGE9=$(new_dir results/gemm/stage9 "$MARKER")
mv "$STAGE9" "$RESULT_DIR/stage9"
STAGE9="$RESULT_DIR/stage9"
MARKER="$RESULT_DIR/.stage10-before"; touch "$MARKER"
if [[ -n "$PROFILE_ARTIFACT_LOCATION" ]]; then
  run stage10-runner python3 tools/run_cuda_gemm_stage10.py --artifact-location "$PROFILE_ARTIFACT_LOCATION"
else
  run stage10-runner python3 tools/run_cuda_gemm_stage10.py
fi
STAGE10=$(new_dir results/profiling/stage10 "$MARKER")
mv "$STAGE10" "$RESULT_DIR/stage10"
STAGE10="$RESULT_DIR/stage10"
MARKER="$RESULT_DIR/.stage11-before"; touch "$MARKER"
run stage11-runner python3 tools/run_scheduler_benchmark.py
STAGE11=$(new_dir results/scheduler/stage11-c4 "$MARKER")
mv "$STAGE11" "$RESULT_DIR/stage11"
STAGE11="$RESULT_DIR/stage11"

run kv-cpu "$RESULT_DIR/build-production/bench_kv_cache" --weights "$WEIGHTS"
cp "$RESULT_DIR/kv-cpu.log" "$RESULT_DIR/kv_cache.jsonl"
run kv-mixed "$RESULT_DIR/build-production/bench_kv_cache" --weights "$WEIGHTS" --mode mixed
cat "$RESULT_DIR/kv-mixed.log" >> "$RESULT_DIR/kv_cache.jsonl"
run quant-cpu "$RESULT_DIR/build-production/bench_quantization" --float-model "$FLOAT_MODEL" --int8-model "$INT8_MODEL" --backend cpu --cache on
cp "$RESULT_DIR/quant-cpu.log" "$RESULT_DIR/quant_cpu_cache_on.json"
run quant-mixed "$RESULT_DIR/build-production/bench_quantization" --float-model "$FLOAT_MODEL" --int8-model "$INT8_MODEL" --backend mixed --cache on
cp "$RESULT_DIR/quant-mixed.log" "$RESULT_DIR/quant_mixed_cache_on.json"

for item in float_cpu_cache_off:'--backend cpu --quant float --cache off' \
            float_cpu_cache_on:'--backend cpu --quant float --cache on' \
            int8_cpu_cache_off:'--backend cpu --quant int8 --cache off' \
            int8_cpu_cache_on:'--backend cpu --quant int8 --cache on' \
            float_mixed_cache_on:'--backend mixed --quant float --cache on' \
            int8_mixed_cache_on:'--backend mixed --quant int8 --cache on'; do
  name=${item%%:*}; flags=${item#*:}
  # shellcheck disable=SC2086  # Flags are fixed literals declared immediately above.
  run "inference-${name}" "$RESULT_DIR/build-production/bench_inference" --model "$([[ $name == int8_* ]] && printf '%s' "$INT8_MODEL" || printf '%s' "$FLOAT_MODEL")" $flags
  cp "$RESULT_DIR/inference-${name}.log" "$RESULT_DIR/inference_${name}.jsonl"
done

run evidence-report python3 tools/verify_stage18_evidence.py --tag "$TAG" --commit "$SOURCE_COMMIT" --source-digest "$SOURCE_DIGEST" \
  --stage5 "$STAGE5" --stage7 "$STAGE7" --stage9 "$STAGE9" --stage10 "$STAGE10" --stage11 "$STAGE11" \
  --kv "$RESULT_DIR/kv_cache.jsonl" --quant-cpu "$RESULT_DIR/quant_cpu_cache_on.json" --quant-mixed "$RESULT_DIR/quant_mixed_cache_on.json" \
  --inference "$RESULT_DIR/inference_float_cpu_cache_off.jsonl" "$RESULT_DIR/inference_float_cpu_cache_on.jsonl" \
  "$RESULT_DIR/inference_int8_cpu_cache_off.jsonl" "$RESULT_DIR/inference_int8_cpu_cache_on.jsonl" \
  "$RESULT_DIR/inference_float_mixed_cache_on.jsonl" "$RESULT_DIR/inference_int8_mixed_cache_on.jsonl" \
  --output "$RESULT_DIR/stage18_report.md"
cat > "$RESULT_DIR/evidence_paths.env" <<EOF
STAGE5=$STAGE5
STAGE7=$STAGE7
STAGE9=$STAGE9
STAGE10=$STAGE10
STAGE11=$STAGE11
EOF
printf 'passed\n' > "$RESULT_DIR/status.txt"
find "$RESULT_DIR" -type f ! -name artifact_sha256.txt -print0 | sort -z | xargs -0 sha256sum > "$RESULT_DIR/artifact_sha256.txt"
printf 'Stage 18 evidence complete: %s\n' "$RESULT_DIR"
