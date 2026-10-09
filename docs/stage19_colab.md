# Stage 19 final reproducibility checklist

**Status:** this guide is the required T4 hand-off for Stage 19 C4. The
documentation commits can be locally link-checked, but Stage 19 is not accepted
until a clean T4 checkout has returned the result branch described below. This
is a verification pass, not permission to retune or change runtime code.

The build/test capture proves the Stage 19 documentation source still builds and
runs the CUDA matrix. The report replay proves the already accepted immutable
RC4 evidence is reproducible from its raw artifacts. They are deliberately
separate source identities: documentation does not retag or alter `v1.0-rc4`.

## A. Fresh Stage 19 source checkout and T4 test capture

Use the SSH remote already configured on the server. `BRANCH` is the pushed
Stage 19 documentation branch; do not replace it with `main` or a result branch.

```bash
set -euo pipefail
REPO=/content/miniLLMI-stage19
BRANCH=feat/documentation-stage19
rm -rf "$REPO"
git clone ssh://git@ssh.github.com:443/WafoeX/miniLLMI.git "$REPO"
cd "$REPO"
git fetch origin "$BRANCH"
git switch --detach "origin/$BRANCH"
git status --short
# Must print nothing. Record this immutable documentation source before results exist.
SOURCE_COMMIT=$(git rev-parse HEAD)
SOURCE_DIGEST=$(python3 tools/provenance.py --root . | python3 -c 'import json,sys; print(json.load(sys.stdin)["source_digest"])')
printf 'commit=%s\ndigest=%s\n' "$SOURCE_COMMIT" "$SOURCE_DIGEST"
nvidia-smi
nvcc --version
```

The source needs one visible Tesla T4 (CC 7.5), a supported CUDA toolkit, and
cuBLAS. Create a unique result directory only after recording clean source
identity. The commands preserve logs on failure because `tee` is used with
`pipefail`; do not delete a failed directory.

```bash
set -euo pipefail
cd /content/miniLLMI-stage19
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-colab-stage19-c4"
RESULT_DIR="results/documentation/stage19-c4/${RUN_ID}"
BUILD_DIR="/content/build-stage19-${RUN_ID}"
mkdir -p "$RESULT_DIR"
printf '%s\n' "$SOURCE_COMMIT" > "$RESULT_DIR/tested_commit.txt"
printf '%s\n' "$SOURCE_DIGEST" > "$RESULT_DIR/source_digest.txt"
git status --porcelain > "$RESULT_DIR/source_status_before_results.txt"
nvidia-smi -q > "$RESULT_DIR/nvidia-smi.txt"
nvcc --version > "$RESULT_DIR/nvcc-version.txt"
python3 tools/check_documentation.py |& tee "$RESULT_DIR/documentation-check.log"
cmake -S . -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release -DENABLE_CUDA=ON -DBUILD_TESTING=ON \
  |& tee "$RESULT_DIR/configure.log"
cmake --build "$BUILD_DIR" --parallel 2 \
  |& tee "$RESULT_DIR/build.log"
ctest --test-dir "$BUILD_DIR" --output-on-failure --no-tests=error \
  |& tee "$RESULT_DIR/ctest-full.log"
ctest --test-dir "$BUILD_DIR" -L gpu --output-on-failure --no-tests=error \
  |& tee "$RESULT_DIR/ctest-gpu.log"
ctest --test-dir "$BUILD_DIR" -R 'cuda_(decoder|kv_decoder|quantized_decoder)|cli_tools|generation|quant_benchmark_tools' \
  --output-on-failure --no-tests=error |& tee "$RESULT_DIR/ctest-focused.log"
printf 'passed\n' > "$RESULT_DIR/status.txt"
find "$RESULT_DIR" -type f ! -name artifact_sha256.txt -print0 | sort -z \
  | xargs -0 sha256sum > "$RESULT_DIR/artifact_sha256.txt"
printf 'result_dir=%s\n' "$RESULT_DIR"
```

For a successful RC4-equivalent environment, the historical release matrix is
61 full, 8 GPU and 7 focused tests. Record the actual totals from these logs;
do not edit them to match historical totals. A failure stays a failed capture.

## B. Commit only the Stage 19 test result

The result commit must be a direct child of the tested documentation source and
must change only its unique result directory. It is not merged into the feature
branch and it does not modify the RC4 tag.

```bash
set -euo pipefail
cd /content/miniLLMI-stage19
RESULT_DIR="$(find results/documentation/stage19-c4 -mindepth 1 -maxdepth 1 -type d | sort | tail -n1)"
test -n "$RESULT_DIR"
test "$(cat "$RESULT_DIR/tested_commit.txt")" = "$SOURCE_COMMIT"
test "$(cat "$RESULT_DIR/status.txt")" = passed
test -z "$(cat "$RESULT_DIR/source_status_before_results.txt")"
git switch -c "results/stage19-c4-$(basename "$RESULT_DIR")" "$SOURCE_COMMIT"
git add "$RESULT_DIR"
git diff --cached --name-only | while IFS= read -r path; do
  case "$path" in "$RESULT_DIR"/*) ;; *) echo "unexpected staged path: $path" >&2; exit 1;; esac
done
git commit -m "test(docs): record Stage 19 T4 reproducibility capture"
git push -u origin HEAD
printf 'source_commit=%s\nresult_commit=%s\nresult_branch=%s\nresult_dir=%s\n' \
  "$SOURCE_COMMIT" "$(git rev-parse HEAD)" "$(git branch --show-current)" "$RESULT_DIR"
```

If configure/build/CTest fails, do **not** write `passed`, do **not** rerun in
place, and do not present the capture as acceptance. Commit the logs on a
similarly constrained `results/stage19-c4-...` branch with `status.txt` set to
`failed`, then return the branch, commit, directory and failing log.

## C. Replay the accepted RC4 generated report

This check needs the complete RC4 result branch and the checksum-addressed
companion archive described in [Stage 18](stage18_report.md). The six excluded
Nsight files are required for complete profiler-package verification even though
the raw-data report verifier reads the retained text/JSON artifacts.

```bash
set -euo pipefail
cd /content/miniLLMI-stage19
git fetch origin results/stage18-20261009T010708Z-colab-stage18
EVIDENCE=/content/miniLLMI-rc4-evidence
rm -rf "$EVIDENCE"
git worktree add --detach "$EVIDENCE" origin/results/stage18-20261009T010708Z-colab-stage18
cd "$EVIDENCE"
test "$(git rev-parse HEAD^)" = ec116ce078eb7601e7cccc27a03f0c005eac9c7e
RESULT_DIR=results/release/v1.0-rc4/20261009T010708Z-colab-stage18
source "$RESULT_DIR/evidence_paths.env"
SOURCE_DIGEST=$(cat "$RESULT_DIR/source_digest.txt")
python3 tools/verify_stage18_evidence.py --tag v1.0-rc4 \
  --commit ec116ce078eb7601e7cccc27a03f0c005eac9c7e --source-digest "$SOURCE_DIGEST" \
  --stage5 "$STAGE5" --stage7 "$STAGE7" --stage9 "$STAGE9" --stage10 "$STAGE10" --stage11 "$STAGE11" \
  --kv "$RESULT_DIR/kv_cache.jsonl" --quant-cpu "$RESULT_DIR/quant_cpu_cache_on.json" \
  --quant-mixed "$RESULT_DIR/quant_mixed_cache_on.json" \
  --inference "$RESULT_DIR/inference_float_cpu_cache_off.jsonl" "$RESULT_DIR/inference_float_cpu_cache_on.jsonl" \
  "$RESULT_DIR/inference_int8_cpu_cache_off.jsonl" "$RESULT_DIR/inference_int8_cpu_cache_on.jsonl" \
  "$RESULT_DIR/inference_float_mixed_cache_on.jsonl" "$RESULT_DIR/inference_int8_mixed_cache_on.jsonl" \
  --output /tmp/stage18-replayed-report.md
cmp /tmp/stage18-replayed-report.md "$RESULT_DIR/stage18_report.md"
cmp /tmp/stage18-replayed-report.json "$RESULT_DIR/stage18_report.json"
sha256sum /content/stage18-v1.0-rc4-20261009T010708Z-colab-stage18-evidence.tar.gz
# Must equal de91f2190964dbc46e0478e40642e050a5a24ea912aaf90a3ef183e2946e7185.
```

Return the four identifiers printed in section B, actual full/GPU/focused test
totals, and the successful `cmp`/archive checksum output. I will independently
inspect the result branch and then update the Stage 19 acceptance record.
