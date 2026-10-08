# Stage 16 C1–C3 T4 replay and result push

This procedure validates the **prepare-dequant correctness path**, not a
throughput claim. Do not run a quant benchmark or claim Stage 16-C4: that
Change requires Stage 17-C2.

The code source to test is:

```bash
SOURCE_COMMIT=29b2d58ec087dddc9cde922c2a948233def9c07e
```

## Clone and clean CUDA build

```bash
sudo apt-get update -qq
sudo apt-get install -y -qq cmake build-essential

# Keep all variables in this same %%bash cell; Colab does not persist them.
set -euo pipefail
# SSH-over-443 works on networks that block GitHub port 22.
REPO=/content/miniLLMI
SOURCE_COMMIT=29b2d58ec087dddc9cde922c2a948233def9c07e
git clone ssh://git@ssh.github.com:443/WafoeX/miniLLMI.git "$REPO"
cd "$REPO"
git fetch origin feat/int8-stage16
git cat-file -e "${SOURCE_COMMIT}^{commit}"
git checkout --detach "$SOURCE_COMMIT"
test -z "$(git status --porcelain)"
test "$(git rev-parse HEAD)" = "$SOURCE_COMMIT"
nvidia-smi
nvcc --version

cmake -S "$REPO" -B "$REPO/build-stage16-t4" -G 'Unix Makefiles' \
  -DCMAKE_BUILD_TYPE=Release -DENABLE_CUDA=ON -DBUILD_TESTING=ON
cmake --build "$REPO/build-stage16-t4" -j"$(nproc)"
```

## Required tests

```bash
# Define and verify these in every independent %%bash cell.
set -euo pipefail
REPO=/content/miniLLMI
SOURCE_COMMIT=29b2d58ec087dddc9cde922c2a948233def9c07e
cd "$REPO"
test "$(git rev-parse HEAD)" = "$SOURCE_COMMIT"
ctest --test-dir "$REPO/build-stage16-t4" --output-on-failure \
  -R '^(quantization|model_file|model_loader|cuda_quantized_decoder)$'
ctest --test-dir "$REPO/build-stage16-t4" --output-on-failure -L gpu
```

The first command must pass all four named tests. In particular,
`cuda_quantized_decoder` constructs a V2 artifact from the frozen weights,
loads persistent INT8/scales plus FP32 `prepare_dequant` workspace, schedules
all 21 learned projections to CUDA, verifies explicit copies and zero
execute-time intermediate allocations, then checks every CPU-returned logit
against `abs(error) <= 1e-2 + 1e-2*abs(FP32 reference)`.

## Preserve and push evidence

Run this only after both test commands pass. It creates a branch whose result
commit is a direct child of the tested source and changes only one unique result
directory.

```bash
# This is one self-contained cell: it fails before committing on a bad source
# or failed test, and its result commit is a direct child of SOURCE_COMMIT.
set -euo pipefail
REPO=/content/miniLLMI
SOURCE_COMMIT=29b2d58ec087dddc9cde922c2a948233def9c07e
cd "$REPO"
git fetch origin feat/int8-stage16
git cat-file -e "${SOURCE_COMMIT}^{commit}"
git checkout --detach "$SOURCE_COMMIT"
test -z "$(git status --porcelain)"
test "$(git rev-parse HEAD)" = "$SOURCE_COMMIT"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-colab-stage16"
git switch -c "results/stage16-c3-${RUN_ID}" "$SOURCE_COMMIT"
RESULT_DIR="results/quant/stage16-c3/${RUN_ID}"
mkdir -p "$RESULT_DIR"

printf '%s\n' "$SOURCE_COMMIT" > "$RESULT_DIR/tested_commit.txt"
printf '\n' > "$RESULT_DIR/source_status.txt"
nvidia-smi > "$RESULT_DIR/nvidia-smi.txt"
nvcc --version > "$RESULT_DIR/nvcc-version.txt"
cmake -S "$REPO" -B "$REPO/build-stage16-t4" -G 'Unix Makefiles' \
  -DCMAKE_BUILD_TYPE=Release -DENABLE_CUDA=ON -DBUILD_TESTING=ON \
  > "$RESULT_DIR/configure.log" 2>&1
cmake --build "$REPO/build-stage16-t4" -j"$(nproc)" > "$RESULT_DIR/build.log" 2>&1
ctest --test-dir "$REPO/build-stage16-t4" --output-on-failure \
  -R '^(quantization|model_file|model_loader|cuda_quantized_decoder)$' \
  > "$RESULT_DIR/stage16_ctest.log" 2>&1
ctest --test-dir "$REPO/build-stage16-t4" --output-on-failure -L gpu \
  > "$RESULT_DIR/gpu_ctest.log" 2>&1

# Inspect both logs; they must show success before committing.
git status --short
git add "$RESULT_DIR"
git commit -m 'test(quant): record Stage 16 server acceptance'
git show --stat --oneline HEAD
git push -u origin HEAD
```

Send the pushed branch name or commit hash back here. The result commit should
contain only `results/quant/stage16-c3/<RUN_ID>/`; do not amend the code source
or modify historical result data.
