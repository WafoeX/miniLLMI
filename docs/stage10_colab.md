# Stage 10 T4 / Colab acceptance

This is the mandatory GPU profiling gate for Stage 10. It produces one V0/V1 Nsight Systems comparison and one V0/V1 Nsight Compute comparison at 4096³. It does **not** rerun or replace the accepted Stage 9 benchmark evidence, and no profiler duration is a formal performance metric.

Run this only after the Stage 10 source commit has been pushed. Use a fresh clean checkout, build, and result directory.

## 1. Select and verify the runtime

In Colab, select **Runtime → Change runtime type → T4 GPU**, then run:

```bash
%%bash
set -euo pipefail
nvidia-smi --query-gpu=name,uuid,compute_cap --format=csv,noheader
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
case "$GPU_NAME" in
  *T4*) ;;
  *) echo "Stage 10 requires a T4; actual GPU: $GPU_NAME" >&2; exit 1 ;;
esac
nvcc --version
nsys --version
ncu --version
cmake --version
python3 --version
```

If `nsys` or `ncu` is unavailable, or NCU reports that performance counters are not permitted, retain its error log and report the failed/inconclusive environment. Do not label it an accepted profile.

## 2. Clone exactly the pushed branch and commit

Replace `BRANCH` and `SOURCE_COMMIT` with the values supplied with the implementation. The checkout must be clean before the runner starts.

```bash
%%bash
set -euo pipefail
cd /content
rm -rf miniLLMI
# Authenticate first if the repository is private.
git clone https://github.com/WafoeX/miniLLMI.git miniLLMI
cd miniLLMI
git checkout BRANCH
git pull --ff-only
git rev-parse HEAD
test "$(git rev-parse HEAD)" = "SOURCE_COMMIT"
git status --short
test -z "$(git status --porcelain)"
```

## 3. Capture all required T4 profiles

Mount Google Drive or choose another durable external storage location first. Its path is recorded in each profile manifest; the runner does not silently delete or upload artifacts.

```bash
%%bash
set -euo pipefail
cd /content/miniLLMI
ARTIFACT_LOCATION='Google Drive: /content/drive/MyDrive/miniLLMI-stage10/<run-id>.tar.gz'
python3 tools/run_cuda_gemm_stage10.py --artifact-location "$ARTIFACT_LOCATION"
```

The command makes a clean Release CUDA build for SM75, runs the two GPU CTests, then captures `nsys` and `ncu` for V0 and V1. It may take time because it performs a full 4096³ FP64 oracle for the first profile; the run-local cache is reused only by the other profiler captures, while each capture performs its own full initial/final output check.

## 4. Verify, archive, and return evidence

Use the printed run directory, or locate the newest one. This verification only reads the immutable capture; it does not rerun a benchmark or change the manifest.

```bash
%%bash
set -euo pipefail
cd /content/miniLLMI
RUN="$(find results/profiling/stage10 -mindepth 1 -maxdepth 1 -type d -print | sort | tail -1)"
printf 'RUN=%s\n' "$RUN"
python3 - "$RUN" <<'PY'
import hashlib, json, sys
from pathlib import Path
run = Path(sys.argv[1])
manifest = json.loads((run / 'manifest.json').read_text())
assert manifest['status'] == 'passed', manifest
assert manifest['source'] == manifest['source_after'], manifest
assert len(manifest['profiles']) == 4, manifest['profiles']
for name, expected in manifest['artifact_sha256'].items():
    actual = hashlib.sha256((run / name).read_bytes()).hexdigest()
    assert actual == expected, (name, expected, actual)
for profile in manifest['profiles']:
    path = run / profile['directory'] / 'profile_manifest.json'
    detail = json.loads(path.read_text())
    assert detail['report_sha256'] == profile['report_sha256'], profile
print(json.dumps({'status': manifest['status'], 'profiles': manifest['profiles']}, indent=2))
PY
find "$RUN/profiles" -type f \( -name stats.txt -o -name metrics.csv -o -name profile_manifest.json \) -print
```

Archive **the complete run directory**, including ignored `.nsys-rep` and `.ncu-rep` files, before the Colab runtime ends. Substitute the real target in this example; it must match the location recorded in the command above.

```bash
%%bash
set -euo pipefail
cd /content/miniLLMI
RUN="$(find results/profiling/stage10 -mindepth 1 -maxdepth 1 -type d -print | sort | tail -1)"
tar -C "$(dirname "$RUN")" -czf /content/drive/MyDrive/miniLLMI-stage10/"$(basename "$RUN")".tar.gz "$(basename "$RUN")"
sha256sum /content/drive/MyDrive/miniLLMI-stage10/"$(basename "$RUN")".tar.gz
```

Return the run directory, `manifest.json`, four `profile_manifest.json` files, `stats.txt`, `metrics.csv`, the archive SHA-256, and any failure log. We will then add evidence cards containing only metric names/units actually exported by this T4 environment and update Stage 10’s final acceptance status.
