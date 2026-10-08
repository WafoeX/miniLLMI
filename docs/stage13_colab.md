# Stage 13 T4 integration reproduction

Stage 13 C4 requires a clean T4 run because local CPU acceptance cannot validate
CUDA compilation, placement or transfers. The capture is correctness, memory
and diagnostic latency evidence—not a speedup or language-quality claim.

The preserved first attempt (`710e3df`, run
`20261008T022848623044Z-3926`) passed 46/47 tests and exposed a noncontiguous V
alias transfer; it must not be deleted or relabelled. The source branch now
contains the explicit contiguous V projection boundary and a metadata-only
scheduler regression. The second preserved attempt (`d97d107`, run
`20261008T024029021362Z-9005`) reached mixed execution and exposed an incomplete
test-side copy-count expectation; the test now derives exact COPY/MATERIALIZE
counts and bytes from the rewritten graph. The third capture (`44ca52f`, run
`20261008T025048601322Z-13737`) passes 47/47 full, 6/6 GPU and 6/6 decoder plus
all functional/memory checks, but retained medians without the declared ten raw
latency samples. The latest runner records both raw arrays and independently
replays conventional medians. The final accepted capture (`1b46111`, run
`20261008T072123049978Z-1761`) uses source `441b6ec`, passes 47/47 full, 6/6 GPU
and 6/6 decoder, and retains all raw samples. This procedure remains the
reproduction path; start from a fresh source branch, not a result branch.

## Clone, test, capture and push

The server already has an SSH key, so use the SSH remote directly. Replace only
`SOURCE_BRANCH` if the handoff names a different pushed source branch.

```bash
set -euo pipefail
REPO=miniLLMI
SOURCE_BRANCH=feat/decoder-stage13
rm -rf "$REPO"
git clone --branch "$SOURCE_BRANCH" --single-branch git@github.com:WafoeX/miniLLMI.git "$REPO"
cd "$REPO"
git status --short                 # must print nothing
SOURCE_COMMIT=$(git rev-parse HEAD)
printf 'source_commit=%s\n' "$SOURCE_COMMIT"
nvidia-smi
nvcc --version

# Preserve a failed capture too; the runner always finalizes its manifest/logs.
set +e
python3 tools/run_decoder_integration.py
RUN_STATUS=$?
set -e
RESULT_DIR=$(find results/decoder/stage13-c4 -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)
test -n "$RESULT_DIR"
printf 'result_dir=%s\n' "$RESULT_DIR"
python3 - "$RESULT_DIR/manifest.json" <<'PY'
import json, sys
value = json.load(open(sys.argv[1]))
print("result_status=" + value["status"])
print("tested_commit=" + value["source_before"]["commit"])
print("source_digest=" + value["source_before"]["source_digest"])
if "metrics" in value:
    print("cpu_execute_median_ms=" + str(value["metrics"]["cpu"]["execute_median_ms"]))
    print("mixed_execute_median_ms=" + str(value["metrics"]["mixed"]["execute_median_ms"]))
    print("mixed_copies=" + str(value["metrics"]["mixed"]["copies"]))
    print("mixed_copy_bytes=" + str(value["metrics"]["mixed"]["copy_bytes"]))
    for mode in ("cpu", "mixed"):
        print(mode + "_execute_raw_samples=" + str(len(value["metrics"][mode]["execute_samples_ms"])))
        print(mode + "_shape_change_raw_samples=" + str(len(value["metrics"][mode]["shape_change_end_to_end_samples_ms"])))
PY

RESULT_STATUS=$(python3 - "$RESULT_DIR/manifest.json" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["status"])
PY
)
RESULT_RUN=$(basename "$RESULT_DIR")
RESULT_BRANCH="test/stage13-t4-${RESULT_STATUS}-${RESULT_RUN}"
git switch -c "$RESULT_BRANCH"
git add "$RESULT_DIR"
git commit -m "test(model): record Stage 13 T4 ${RESULT_STATUS} integration"
git push -u origin HEAD
printf 'result_commit=%s\n' "$(git rev-parse HEAD)"
printf 'result_branch=%s\n' "$(git branch --show-current)"
exit "$RUN_STATUS"
```

The runner itself performs:

- T4/CC 7.5 enforcement and environment capture;
- fresh Release `BUILD_TESTING=ON` and separate `BUILD_TESTING=OFF` SM75 builds;
- full CTest, then explicit `gpu` and `decoder` label suites;
- production `bench_decoder` CPU and mixed runs;
- source-before/source-after identity checks, fixture hash checks, structured
  metric validation and artifact SHA-256 recording.

Do not edit a failed result into a pass, amend the tested source commit, reuse a
previous build directory or delete slow/failing logs. Push the result branch and
send back the full source commit, result commit, result branch and result path.
