# Stage 17 / Stage 16-C4 T4 reproduction and result push

This procedure validates the remaining GPU-facing Stage 17 and Stage 16-C4
work. It records measurements; it does **not** require an INT8 throughput win.
Run every command in one shell cell or re-declare every variable in a new cell.
Keep the tested source checkout clean, and commit results separately as a direct
child of that source.

## Clone, identify source, and build

```bash
set -euo pipefail
REPO=/content/miniLLMI
SOURCE_BRANCH=feat/int8-stage16
SOURCE_COMMIT=0c258548177eac8d8ae042d0fba402dbbdedcd65
rm -rf "$REPO"
# The server SSH key is used over port 443 for networks that block port 22.
git clone --branch "$SOURCE_BRANCH" --single-branch \
  ssh://git@ssh.github.com:443/WafoeX/miniLLMI.git "$REPO"
cd "$REPO"
git fetch origin "$SOURCE_BRANCH"
git cat-file -e "${SOURCE_COMMIT}^{commit}"
git checkout --detach "$SOURCE_COMMIT"
test -z "$(git status --porcelain)"
test "$(git rev-parse HEAD)" = "$SOURCE_COMMIT"
printf 'source_commit=%s\n' "$SOURCE_COMMIT"
nvidia-smi
nvcc --version

cmake -S "$REPO" -B "$REPO/build-stage17-test" -G 'Unix Makefiles' \
  -DCMAKE_BUILD_TYPE=Release -DENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75 \
  -DBUILD_TESTING=ON
cmake --build "$REPO/build-stage17-test" --parallel 2
ctest --test-dir "$REPO/build-stage17-test" --output-on-failure
ctest --test-dir "$REPO/build-stage17-test" --output-on-failure -L gpu
ctest --test-dir "$REPO/build-stage17-test" --output-on-failure \
  -R '^(cli_tools|generation|inference_benchmark_tools|quant_benchmark_tools|cuda_quantized_decoder)$'

cmake -S "$REPO" -B "$REPO/build-stage17-prod" -G 'Unix Makefiles' \
  -DCMAKE_BUILD_TYPE=Release -DENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75 \
  -DBUILD_TESTING=OFF
cmake --build "$REPO/build-stage17-prod" --parallel 2
```

All test commands must pass before accepting a capture. In particular, the GPU
label validates normal CUDA graph paths; `cuda_quantized_decoder` validates the
prepared INT8/scales/dequantized model through the scheduler; and `generation`
validates float and INT8 CPU generation determinism/cache semantics.

## Capture raw C3/C4 data

The next cell creates canonical V1/V2 models only from the frozen committed
weights, executes three independent paired teacher-forced runs per row (each
with 3 warmups and 10 raw samples), and preserves output without editing it.
Cache off/on is compared within the same dtype/device; float/INT8 is compared
within the same cache/device; CPU/mixed is only a separately labelled device
comparison.

```bash
set -euo pipefail
REPO=/content/miniLLMI
SOURCE_COMMIT=0c258548177eac8d8ae042d0fba402dbbdedcd65
cd "$REPO"
test "$(git rev-parse HEAD)" = "$SOURCE_COMMIT"
test -z "$(git status --porcelain)"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-colab-stage17"
RESULT_DIR="results/inference/stage17-c3/${RUN_ID}"
mkdir -p "$RESULT_DIR/models"
printf '%s\n' "$SOURCE_COMMIT" > "$RESULT_DIR/tested_commit.txt"
printf '\n' > "$RESULT_DIR/source_status.txt"
nvidia-smi > "$RESULT_DIR/nvidia-smi.txt"
nvcc --version > "$RESULT_DIR/nvcc-version.txt"

FLOAT_MODEL="$RESULT_DIR/models/tiny-v1.mllm"
INT8_MODEL="$RESULT_DIR/models/tiny-v2-int8.mllm"
WEIGHTS=tests/fixtures/operators-v1/tiny-weights-v1.bin
build-stage17-prod/convert_tiny_model "$WEIGHTS" "$FLOAT_MODEL" \
  | tee "$RESULT_DIR/convert_float.log"
build-stage17-prod/convert_tiny_model "$WEIGHTS" "$INT8_MODEL" --int8 \
  | tee "$RESULT_DIR/convert_int8.log"
sha256sum "$FLOAT_MODEL" "$INT8_MODEL" > "$RESULT_DIR/model_sha256.txt"

# C3: raw, labelled inference measurements. Do not combine these comparisons.
build-stage17-prod/bench_inference --model "$FLOAT_MODEL" --backend cpu --quant float --cache off \
  | tee "$RESULT_DIR/inference_float_cpu_cache_off.jsonl"
build-stage17-prod/bench_inference --model "$FLOAT_MODEL" --backend cpu --quant float --cache on \
  | tee "$RESULT_DIR/inference_float_cpu_cache_on.jsonl"
build-stage17-prod/bench_inference --model "$INT8_MODEL" --backend cpu --quant int8 --cache off \
  | tee "$RESULT_DIR/inference_int8_cpu_cache_off.jsonl"
build-stage17-prod/bench_inference --model "$INT8_MODEL" --backend cpu --quant int8 --cache on \
  | tee "$RESULT_DIR/inference_int8_cpu_cache_on.jsonl"
build-stage17-prod/bench_inference --model "$FLOAT_MODEL" --backend mixed --quant float --cache on \
  | tee "$RESULT_DIR/inference_float_mixed_cache_on.jsonl"
build-stage17-prod/bench_inference --model "$INT8_MODEL" --backend mixed --quant int8 --cache on \
  | tee "$RESULT_DIR/inference_int8_mixed_cache_on.jsonl"

# C4: same teacher-forced IDs; emits compression, file/resident/dequant
# accounting, logit MAE/max error and each dtype's own greedy token smoke.
build-stage17-prod/bench_quantization --float-model "$FLOAT_MODEL" --int8-model "$INT8_MODEL" \
  --backend cpu --cache on | tee "$RESULT_DIR/quant_cpu_cache_on.json"
build-stage17-prod/bench_quantization --float-model "$FLOAT_MODEL" --int8-model "$INT8_MODEL" \
  --backend mixed --cache on | tee "$RESULT_DIR/quant_mixed_cache_on.json"
```

Before committing, inspect every JSONL/JSON record. Every `teacher_forced`
record must have `source_dirty:false`, `warmups:3`, `samples:10`, three runs
`0..2`, ten positive raw values per timing field, and `correctness:"passed"`.
Both C4 records must have `correctness:"passed"`, eligible ratio `<=0.35`, and
finite error fields. Smaller files do not mean lower resident memory: report
the recorded `int8_resident_parameter_bytes` and
`int8_prepare_dequant_bytes` alongside payload bytes and tok/s.

## Preserve and push evidence

This result commit must have exactly `SOURCE_COMMIT` as its parent and change
only the unique result directory. If any command failed, preserve the logs on
a result branch too, state the failure, and do not call the stage accepted.

```bash
set -euo pipefail
REPO=/content/miniLLMI
SOURCE_COMMIT=0c258548177eac8d8ae042d0fba402dbbdedcd65
cd "$REPO"
test "$(git rev-parse HEAD)" = "$SOURCE_COMMIT"
# Reuse the RUN_ID printed above, or discover its one directory explicitly.
RUN_ID="$(basename "$(find results/inference/stage17-c3 -mindepth 1 -maxdepth 1 -type d | sort | tail -n1)")"
RESULT_DIR="results/inference/stage17-c3/${RUN_ID}"
test -f "$RESULT_DIR/tested_commit.txt"
test "$(cat "$RESULT_DIR/tested_commit.txt")" = "$SOURCE_COMMIT"
# The result directory is intentionally untracked now. Reject any tracked
# modification or untracked path outside it, rather than requiring a clean tree.
git diff --quiet
git diff --cached --quiet
while IFS= read -r status; do
  path="${status:3}"
  case "$path" in "$RESULT_DIR"/*) ;; *) printf 'unexpected worktree path: %s\n' "$status" >&2; exit 1;; esac
done < <(git status --porcelain --untracked-files=all)
git switch -c "results/stage17-stage16-c4-${RUN_ID}" "$SOURCE_COMMIT"
git add "$RESULT_DIR"
git diff --cached --name-only | while IFS= read -r path; do
  case "$path" in "$RESULT_DIR"/*) ;; *) printf 'unexpected staged path: %s\n' "$path" >&2; exit 1;; esac
done
git commit -m 'test(inference): record Stage 17 and Stage 16 C4 T4 capture'
git show --stat --oneline HEAD
git push -u ssh://git@ssh.github.com:443/WafoeX/miniLLMI.git HEAD
printf 'result_commit=%s\nresult_branch=%s\nresult_dir=%s\n' \
  "$(git rev-parse HEAD)" "$(git branch --show-current)" "$RESULT_DIR"
```

Send back the source commit, result commit/branch, result directory, full and
GPU CTest counts, and the JSON/JSONL records. Do not amend the source commit
or merge the result branch before provenance review.
