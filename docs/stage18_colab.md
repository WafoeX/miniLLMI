# Stage 18 T4 reproduction, report generation, and result push

Run this only after `v1.0-rc2` has been pushed. It is the complete C2/C3/C4
procedure: a fresh T4 build/test matrix, all required final measurements, and a
report generated from the captured raw files. Do not edit the checkout. Failed
captures are evidence too: commit and push them, then report the failure instead
of calling the RC accepted.

## Clone the immutable source tag

The server SSH key can use normal GitHub SSH. The `ssh.github.com:443` form
below also works when port 22 is blocked.

```bash
set -euo pipefail
REPO=/content/miniLLMI-stage18
TAG=v1.0-rc2
rm -rf "$REPO"
git clone ssh://git@ssh.github.com:443/WafoeX/miniLLMI.git "$REPO"
cd "$REPO"
git fetch --tags origin
git checkout --detach "$TAG"
test "$(git describe --exact-match --tags HEAD)" = "$TAG"
test -z "$(git status --porcelain)"
SOURCE_COMMIT=$(git rev-parse HEAD)
SOURCE_DIGEST=$(python3 tools/provenance.py --root . | python3 -c 'import json,sys; print(json.load(sys.stdin)["source_digest"])')
printf 'tag=%s\ncommit=%s\ndigest=%s\n' "$TAG" "$SOURCE_COMMIT" "$SOURCE_DIGEST"
nvidia-smi
nvcc --version
```

The required host is one visible Tesla T4 (compute capability 7.5). The runner
checks that requirement in the Stage 9/10/11 portions. If Nsight binary outputs
must live in external storage, choose a stable location string and supply it as
`--profile-artifact-location`; reports/hashes are retained in Git either way.

## Execute the frozen RC matrix

This can take a long T4 session: it includes the full CUDA CTest matrix, 4
CUDA GEMM sizes × three pairs, two Nsight tools × V0/V1, CPU/KV inference,
quantization, and six raw inference configurations. It intentionally performs
no parameter search.

```bash
set -euo pipefail
cd /content/miniLLMI-stage18
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-colab-stage18"
./scripts/run_stage18_evaluation.sh --tag v1.0-rc2 --run-id "$RUN_ID" --jobs 2 \
  |& tee "/content/stage18-${RUN_ID}.console.log"
RESULT_DIR="results/release/v1.0-rc2/${RUN_ID}"
test -f "$RESULT_DIR/stage18_report.md"
test "$(cat "$RESULT_DIR/status.txt")" = passed
sed -n '1,160p' "$RESULT_DIR/stage18_report.md"
```

The runner runs correctness before measurements and preserves every command/log
in `RESULT_DIR`. Its generated report is valid only when all five required
benefit gates pass: planner memory, CPU GEMM, CUDA GEMM, KV context-512, and
INT8 payload-with-scales. It makes no scheduler/mixed/INT8-throughput speedup
claim and explicitly records optional skips.

## Commit and push only the result directory

The evidence branch must be a **direct child** of the detached RC source, and
its one commit must change only this unique result directory. Do not merge it
into the feature branch or move `v1.0-rc2`.

```bash
set -euo pipefail
cd /content/miniLLMI-stage18
TAG=v1.0-rc2
SOURCE_COMMIT=$(git rev-parse "$TAG^{commit}")
RUN_ID="$(basename "$(find results/release/v1.0-rc2 -mindepth 1 -maxdepth 1 -type d | sort | tail -n1)")"
RESULT_DIR="results/release/v1.0-rc2/${RUN_ID}"
test "$(git rev-parse HEAD)" = "$SOURCE_COMMIT"
test "$(cat "$RESULT_DIR/tested_commit.txt")" = "$SOURCE_COMMIT"
test "$(cat "$RESULT_DIR/status.txt")" = passed
git diff --quiet
git diff --cached --quiet
while IFS= read -r status; do
  path="${status:3}"
  case "$path" in "$RESULT_DIR"/*) ;; *) printf 'unexpected worktree path: %s\n' "$status" >&2; exit 1;; esac
done < <(git status --porcelain --untracked-files=all)
BRANCH="results/stage18-${RUN_ID}"
git switch -c "$BRANCH" "$SOURCE_COMMIT"
git add "$RESULT_DIR"
git diff --cached --name-only | while IFS= read -r path; do
  case "$path" in "$RESULT_DIR"/*) ;; *) printf 'unexpected staged path: %s\n' "$path" >&2; exit 1;; esac
done
git commit -m 'bench(release): record Stage 18 RC evaluation'
git show --stat --oneline HEAD
git push -u origin HEAD
printf 'source_tag=%s\nsource_commit=%s\nresult_commit=%s\nresult_branch=%s\nresult_dir=%s\n' \
  "$TAG" "$SOURCE_COMMIT" "$(git rev-parse HEAD)" "$BRANCH" "$RESULT_DIR"
```

Send back the five printed identifiers, full/GPU/focused CTest totals, and the
complete generated `stage18_report.md` / `stage18_report.json`. I will verify
that the result commit is a direct child of the tag, that only its result path
changed, and replay the raw-data validators before marking Stage 18 accepted.
