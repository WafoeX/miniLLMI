# CPU backend / frozen Stage 6 baseline

## Scope and local acceptance (predeclared)

Stage 6 is **CPU-only**. Local named-host Release measurements are explicitly allowed; no server or GPU is required for this gate. CUDA-enabled build/storage/dispatch, CPU optimization, pool/SIMD hand code, scheduler and transformer numerical implementations are deferred. This does not certify Linux, T4, CUDA or end-to-end model performance.

## One execution path

`model → graph/operators → (future scheduler) → Backend → Tensor/shared Storage`.

- [`Backend`](../include/runtime/backend.hpp) is an internal synchronous interface: capability, buffer allocation, explicit copy, metadata-only prepare and execute with borrowed caller workspace. No CUDA-specific stream or model method. Capability is coarse; prepare checks actual schema, shape, dtype, device, layout and binding. Unsupported code/device is an error, never an implicit copy/fallback.
- [`CpuBackend`](../include/runtime/cpu_backend.hpp) uses only existing S1 Tensor/Storage and S2 inference/copy/core semantics. Backend buffer allocation calls `Tensor::allocate_cpu`; it is not a new allocator. Graph output allocation stays with the existing dynamic/arena/planned providers. Execute does not allocate or replace output backing, nor retain caller workspace.
- [`execute_graph`](../src/runtime/graph_executor.cpp) calls `Backend::execute` for every node, including metadata aliases. It includes no CPU kernel/reference header. Last-use release, output pinning, counts, traces, cleanup and external-write failure semantics remain S3–S5. `execute_planned` accepts the same optional backend. No separate graph executor or tensor path.
- `CpuBackend::prepare` is metadata-only and returns zero workspace bytes; execute revalidates and performs finite-input checks. Empty scratch is required; nonempty scratch is rejected. Zero Tensor backing allocations **does not mean zero C++ heap allocations** (descriptor/inference/container metadata can allocate).
- COPY/MATERIALIZE accept explicit strided FP32/INT32 bindings and share S1 `copy_cpu`. Arithmetic requires contiguous FP32 and rejects overlapping input/output spans. Input NaN/Inf fails before writes; result overflow returns `NonFinite`, potentially after partial writes. Failed graph results expose no outputs; already completed writes to external state are not rolled back.

## Stable default and immutable v0

`default_cpu_backend()` and `CpuBackend{}` retain **S2 FP64-accumulating MATMUL** behind the new interface. The default graph allocation policy remains dynamic. Explicit `CpuBackend(CpuMatmul::ScalarFP32V0)` selects `cpu-ijk-fp32-v0` for both direct and graph execution; no automatic selection or performance promise.

[`src/runtime/cpu_scalar.cpp`](../src/runtime/cpu_scalar.cpp) is the frozen S6 v0: row-major FP32 input/output, single-thread `i → j → k`, FP32 `float sum`, alpha=1/beta=0, no pool, tiling or hand SIMD. **Do not rewrite/optimize v0 in S7**: add a separate selectable candidate, keep shared validation and timing boundary, and rerun v0 paired on the same host/flags. [`src/runtime/cpu_dispatch.hpp`](../src/runtime/cpu_dispatch.hpp) shares S2 binding/overlap/value/copy semantics; only MATMUL reduction precision changes. Stage 0 `cpu_reference` and `sgemm_v0_naive` remain unchanged. The FP64 oracle is never timed as a CPU baseline.

RMSNorm, Softmax, RoPE, Embedding, SwiGLU and Attention already have S2 descriptors/attrs/inference. CPU capability/prepare/execute return **`Unsupported`** with `<OP> CPU primitive reserved for Stage 12`. No model-specific hook and no early transformer kernel. Invalid requested device still returns `DeviceMismatch` rather than a host fallback.

## Frozen CPU experiment v1

[`bench/bench_cpu.cpp`](../bench/bench_cpu.cpp), [`tools/run_cpu_benchmark.py`](../tools/run_cpu_benchmark.py), [`tools/analyze_cpu.py`](../tools/analyze_cpu.py):

- FP32 row-major, seed=42, unmodified Stage 0 deterministic input generator/hash, alpha=1/beta=0, threads=1, atol=rtol=1e-3, actual compiler/flags/build/host archived.
- Required square GEMM sizes **128/256/512/1024**. No 2048/4096 timings in this stage. Eight small/empty/non-square/boundary self-tests and 240 randomized long-double-oracle trials supplement fixture/error tests. Stage 0 oracle requires positive shapes; empty and K=0 correctness use the independently known empty/zero result without changing Stage 0.
- Release only, `BUILD_TESTING=OFF` for timings, 3 warmup batches, 10 samples per case. GEMM batch=1, graph batch=20. `steady_clock` monotonic wall time, no CSV writes in timed interval.
- GEMM timer includes the **whole backend call** (metadata/shape/binding/finite checks + scalar loop), status handling and a volatile first-output read. Allocation, generation, FP64 oracle, correctness scans, data copies, CSV and prepare are excluded. The final actual timed GEMM output is fully compared; initial output is NaN-filled.
- Graph workload is fixed `[16,16] MATMUL → ADD(result,result) → MUL(result,result)`, same seed/input/hash/backend, explicit dynamic versus S5 reuse provider. Timer includes executor/provider/backend validation, counters, **full oracle comparison for every execution**, volatile output read and output destruction. Graph/arena prepare and oracle generation are excluded. This is not pure kernel latency. Dynamic has 3 execute-time backing allocations; prepared reuse has zero, without altering logical operations.
- Three independent GEMM baseline runs (sizes ascending / descending / ascending), not an oracle-versus-v0 speedup comparison. There is no S6 optimization candidate. Graph policy pairs alternate D/R, R/D, D/R. Preserve all slow/failing runs. No affinity/exclusive-core/process isolation guarantee; variance/thermal effects remain possible. Do not pool sizes or claim statistical significance.
- GEMM GFLOPS = `2*M*N*K / (median_ms*1e6)`. Report each independent run; no GFLOPS for graph. Graph ratios are per-pair dynamic/reuse median ratios, then their median. No Stage 6 latency improvement gate; S7's ≥1.05× paired geometric-mean gate is **not attempted** here.
- Ordinary compiler auto-vectorization/FMA behavior is allowed and **not artificially suppressed**. Actual production compile commands and compiler vectorization remarks for the v0 source with those same flags are archived. Diagnostic recompilation is separate and untimed. No `-ffast-math`/`-Ofast`. Thread count=1 means program threads, not physical-core affinity.

Each run has a unique directory and fresh test/production builds. Runner rejects dirty/stale/non-Release/test-hook identities, validates initial/final correctness, captures all command exits/logs, source before/after, binary SHA256, CMakeCache/compile_commands, compiler remarks and `nm` production symbols. Analyzer checks schema/sample coverage, controls/order/inputs/checksums, all artifact hashes and correctness before recomputing statistics/GFLOPS. Moved archives replay against original recorded absolute argv but read raw bytes from their new archive directory. Different derived artifacts are never silently overwritten. Failed runs retain raw files/logs/manifests; mock timings exist only in temporary unit-test directories.

```bash
python3 tools/validate_cpu.py            # clean Release/Debug/ASan+UBSan/production
python3 tools/run_cpu_benchmark.py       # clean fresh Release; local CPU-only
python3 tools/analyze_cpu.py results/cpu/baseline/<actual-run-id>
# Development correctness only, never formal timing:
python3 tools/validate_cpu.py --allow-dirty
```

Formal status/evidence: [Stage 6 report](stage6_report.md). GPU acceptance is not needed for Stage 6; notify the user before later CUDA gates.
