# Stage 7 acceptance report

**Status: complete — required S7-C1/C3/C5 pass; C2 static per-call threads and C4 work stealing are `skipped_optional`.** This is a local CPU-only gate, not a CUDA/GPU result.

- Tested code: `86d83c5ecdeeaa3fa9debd9266022c5fa81896b6`, digest `98e912bb358d9cd79691e62eb78943eb4e29b3cdeb4606e727713ced9509a258`.
- Evidence: [`results/cpu/parallel/20261006T125218081250Z-95780`](../results/cpu/parallel/20261006T125218081250Z-95780/), with fresh Release test/production logs, 72 raw CSVs/correctness records, command exits and SHA-256 manifest coverage.
- C1 adds separately selectable `cpu-ikj-fp32-c1`; C3 adds the persistent explicit FIFO pool integrated through `CpuBackend`. v0 is untouched.
- Three alternating paired v0/C1 geometric-mean speedups are **9.241503×, 8.866688×, 8.791276×**; their median is **8.866688×**. Every retained run wins, exceeding the required 1.05× gate. This is descriptive same-host evidence, not a cross-host or GPU claim.
- FIFO scaling is reported at 1/2/4/8 workers (8 is the recorded available-thread cap). No pool scaling-efficiency threshold is required. Smaller 128-sized 8-worker rows regress relative to fewer workers and are retained in raw data; no automatic per-size fallback is selected.

Focused Debug and ASan+UBSan backend/pool tests passed locally during implementation. The formal run uses a clean detached worktree and Release tests. macOS arm64 does not certify Linux/LSan/CUDA/TSAN behavior. See [the CPU parallel contract](cpu_parallel.md).
