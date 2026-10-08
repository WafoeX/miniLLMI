# Stage 15 clone, test, and result-push procedure

Stage 15's required C1–C3 gate is CPU functional correctness. A GPU is **not**
required and no benchmark/profiling command should be treated as Stage 15
acceptance. CUDA may be enabled separately for broader regression testing, but
it is not evidence for this stage.

## Clone and select the exact branch

The server's SSH key is sufficient for the normal GitHub SSH URL:

```bash
git clone git@github.com:WafoeX/miniLLMI.git
cd miniLLMI
git fetch origin feat/tokenizer-loader-stage15
git switch --track -c feat/tokenizer-loader-stage15 origin/feat/tokenizer-loader-stage15
git rev-parse HEAD
git status --short
```

If port 22 is blocked, use GitHub SSH-over-443 instead:

```bash
GIT_SSH_COMMAND='ssh -o HostName=ssh.github.com -p 443' \
  git clone git@github.com:WafoeX/miniLLMI.git
```

Record the printed commit in your feedback. Acceptance must run on a clean tree
(the repository's unrelated, untracked `AGENTS.md` does not alter tracked source
identity if your clone contains one).

## Clean CPU acceptance build

```bash
cmake -S . -B build-stage15-release \
  -DCMAKE_BUILD_TYPE=Release -DENABLE_CUDA=OFF -DBUILD_TESTING=ON
cmake --build build-stage15-release --parallel "$(nproc)"
ctest --test-dir build-stage15-release --output-on-failure
ctest --test-dir build-stage15-release --output-on-failure -L 'model_file|tokenizer'
sha256sum tests/fixtures/model-v1/tiny-model-v1.mllm
```

Expected focused tests are `model_file`, `model_loader`, and `tokenizer`. The
committed artifact digest is:

```text
f91f89491735d324193295bb489f16ff6301186d957e04029c7a1402be729ee8
```

Optional sanitizer check (CPU only):

```bash
cmake -S . -B build-stage15-sanitize \
  -DCMAKE_BUILD_TYPE=Debug -DENABLE_CUDA=OFF -DENABLE_SANITIZERS=ON -DBUILD_TESTING=ON
cmake --build build-stage15-sanitize --parallel "$(nproc)"
ctest --test-dir build-stage15-sanitize --output-on-failure -L 'model_file|tokenizer'
```

## Commit and push your server result

Keep the source commit intact; append logs/results in a unique run directory.
Replace `RUN_ID` with a UTC timestamp plus your job ID.

```bash
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-colab-stage15"
mkdir -p "results/loader/stage15-c3/${RUN_ID}"
git rev-parse HEAD > "results/loader/stage15-c3/${RUN_ID}/tested_commit.txt"
git status --short > "results/loader/stage15-c3/${RUN_ID}/source_status.txt"
ctest --test-dir build-stage15-release --output-on-failure -L 'model_file|tokenizer' \
  |& tee "results/loader/stage15-c3/${RUN_ID}/stage15_ctest.log"
sha256sum tests/fixtures/model-v1/tiny-model-v1.mllm \
  > "results/loader/stage15-c3/${RUN_ID}/artifact.sha256"

git add "results/loader/stage15-c3/${RUN_ID}"
git commit -m "test(loader): record Stage 15 server acceptance"
git push origin HEAD:refs/heads/feat/tokenizer-loader-stage15
# Port-443 fallback:
# GIT_SSH_COMMAND='ssh -o HostName=ssh.github.com -p 443' \
#   git push origin HEAD:refs/heads/feat/tokenizer-loader-stage15
```

Send the resulting commit SHA and the focused/full CTest counts. Preserve a
failure log and push it too; do not replace or hand-edit a failed run.
