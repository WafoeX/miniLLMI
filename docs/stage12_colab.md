# Stage 12 T4 conformance reproduction

Stage 12 requires a real CUDA run only for C5: projections execute on CUDA while
RMSNorm/Softmax and the other implemented transformer primitives execute on CPU
through scheduler-inserted copies. This is a functional conformance test, **not**
a performance claim.

## Clean Colab procedure

```bash
set -euo pipefail
REPO=miniLLMI
BRANCH=feat/transformer-ops-stage12
rm -rf "$REPO"
git clone --branch "$BRANCH" --single-branch git@github.com:WafoeX/miniLLMI.git "$REPO"
cd "$REPO"
git status --short                 # must print nothing
TESTED_COMMIT=$(git rev-parse HEAD)
printf 'tested_commit=%s\n' "$TESTED_COMMIT"
nvidia-smi

cmake -S . -B build-stage12-t4 -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON -DENABLE_CUDA=ON
cmake --build build-stage12-t4 --parallel
ctest --test-dir build-stage12-t4 --output-on-failure -L gpu
ctest --test-dir build-stage12-t4 --output-on-failure -L transformer
```

The required C5 evidence is a zero exit status for `cuda_transformer_ops`,
including its `copies=3` result. The full GPU and transformer label output must
be retained; `cuda_transformer_ops` verifies CUDA MATMUL plus explicit H2D/D2H
scheduler transfers and CPU Softmax/RMSNorm execution.

If all commands pass, capture the provenance and complete test output without
editing it, commit it on a new result branch, then push it:

```bash
mkdir -p results/transformer/stage12-c5
{
  date -u +%Y-%m-%dT%H:%M:%SZ
  printf 'tested_commit=%s\n' "$TESTED_COMMIT"
  nvidia-smi
  ctest --test-dir build-stage12-t4 --output-on-failure -L gpu
  ctest --test-dir build-stage12-t4 --output-on-failure -L transformer
} | tee results/transformer/stage12-c5/colab-t4.log

git switch -c bench/stage12-t4-conformance
git add results/transformer/stage12-c5/colab-t4.log
git commit -m "test(transformer): record Stage 12 T4 conformance"
git push -u origin HEAD
```

Do not amend the tested source commit or overwrite a prior failed log.
