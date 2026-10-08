# Stage 14 T4 reproduction

Stage 14 source is locally CPU-tested, but its required context-512 paired
benefit gate and mixed CUDA timing require the T4 server. Keep the checkout
clean. Preserve and push failed captures; do not edit their timing rows.

```bash
set -euo pipefail
REPO=miniLLMI
SOURCE_BRANCH=feat/kv-cache-stage14
rm -rf "$REPO"
git clone --branch "$SOURCE_BRANCH" --single-branch ssh://git@ssh.github.com:443/WafoeX/miniLLMI.git "$REPO"
cd "$REPO"
git status --short                    # must print nothing
SOURCE_COMMIT=$(git rev-parse HEAD)
printf 'source_commit=%s\n' "$SOURCE_COMMIT"
nvidia-smi
nvcc --version

# Separate testing and production Release trees; no fast-math flags.
cmake -S . -B build-stage14-test -DCMAKE_BUILD_TYPE=Release -DENABLE_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=75 -DBUILD_TESTING=ON
cmake --build build-stage14-test --parallel 2
ctest --test-dir build-stage14-test --output-on-failure
ctest --test-dir build-stage14-test --output-on-failure -L gpu
ctest --test-dir build-stage14-test --output-on-failure -L kv

cmake -S . -B build-stage14-prod -DCMAKE_BUILD_TYPE=Release -DENABLE_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=75 -DBUILD_TESTING=OFF
cmake --build build-stage14-prod --parallel 2

# This is the predeclared CPU paired sweep: 3 runs × 10 raw samples, contexts
# 128/256/512, fixed 32-token teacher-forced continuation. Prefill is reported
# separately; decode includes replan/dispatch/KV writes/transfers.
RUN_ID=$(date -u +%Y%m%dT%H%M%S)-$RANDOM
RESULT_DIR="results/kv_cache/stage14-c4/$RUN_ID"
mkdir -p "$RESULT_DIR"
set +e
build-stage14-prod/bench_kv_cache --weights tests/fixtures/operators-v1/tiny-weights-v1.bin \
  | tee "$RESULT_DIR/kv_cache.jsonl"
CPU_STATUS=${PIPESTATUS[0]}
# Mixed timing is diagnostic only (no speed gate), but records the actual
# scheduler/copy path at context 512 with 3 warmups and 10 raw samples.
build-stage14-prod/bench_kv_cache --weights tests/fixtures/operators-v1/tiny-weights-v1.bin --mode mixed \
  | tee "$RESULT_DIR/kv_cache_mixed.jsonl"
MIXED_STATUS=${PIPESTATUS[0]}
BENCH_STATUS=$(( CPU_STATUS || MIXED_STATUS ))
set -e
python3 - "$RESULT_DIR/kv_cache.jsonl" <<'PY' | tee "$RESULT_DIR/summary.txt"
import json, statistics, sys
rows = [json.loads(line) for line in open(sys.argv[1]) if line.strip()]
assert len(rows) == 18
for context in (128, 256, 512):
    ratios = []
    for run in range(3):
        base = next(r for r in rows if r['context']==context and r['run']==run and r['variant']=='full_prefix')
        cached = next(r for r in rows if r['context']==context and r['run']==run and r['variant']=='kv_cache')
        assert base['warmups']==3 and base['samples']==cached['samples']==10
        assert len(base['decode_ms_per_token_samples']) == len(cached['decode_ms_per_token_samples']) == 10
        ratios.append(base['decode_ms_per_token_median'] / cached['decode_ms_per_token_median'])
    print(f'context={context} paired_ratios={ratios} median={statistics.median(ratios)}')
print('context-512 gate requires every ratio > 1 and median >= 1.05')
PY

# Commit raw results even if the benchmark fails; keep the result separate from
# the tested source commit. Report the branch/commit/path back here.
git switch -c "test/stage14-t4-${RUN_ID}"
git add "$RESULT_DIR"
git commit -m "test(kv): record Stage 14 T4 capture"
git push -u origin HEAD
printf 'result_commit=%s\nresult_branch=%s\nresult_dir=%s\n' \
  "$(git rev-parse HEAD)" "$(git branch --show-current)" "$RESULT_DIR"
exit "$BENCH_STATUS"
```

Send back the source commit, result commit/branch, result directory, full CTest
counts and the context-512 paired ratios. `cuda_kv_decoder` validates mixed scheduler/copy correctness; the production
mixed benchmark records its separate raw timing/copy evidence without making a
mixed speedup claim.
