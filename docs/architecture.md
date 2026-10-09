# Runtime architecture and operation

This document describes the implementation retained by the accepted Stage 18
release candidate, [`v1.0-rc4`](stage18_report.md). It is an ownership and
execution map, not a performance claim. Exact acceptance numbers and the
source/result provenance are in the linked stage reports.

## Layering

```text
Tokenizer / model-file loader / CLI
              |
       model::Decoder + KVCache
              |
Graph nodes, tensor descriptors, operator descriptors
              |
Scheduler: capability placement + explicit COPY insertion
              |
GraphExecutor / PlannedExecutor + MemoryPlanner
       |                                  |
 CPU backend / FIFO pool             CUDA backend / Stage-0 GEMM registry
       \                                  /
          Tensor metadata + shared Storage
```

The source follows this dependency direction: model → graph/operators →
scheduler → backends → tensor/storage. `model` constructs ordinary graph nodes;
it neither calls CUDA kernels nor owns a second allocator, tensor type, or
matmul route. The public headers are grouped under `include/model/` and
`include/runtime/`; CMake builds the `runtime` and `model` libraries before the
CLI and benchmark executables.

| Layer | Primary code | Contract and tests |
|---|---|---|
| Storage and views | `include/runtime/{storage,tensor,layout}.hpp`, `src/runtime/{storage,tensor,layout}.cpp` | [Tensor contract](tensor.md); `tensor`, `tensor_properties` |
| Operators and validated DAG | `operator.hpp`, `shape_inference.hpp`, `graph.hpp`, `graph_executor.hpp` | [Operator](operators.md) and [graph](graph.md) contracts; graph/operator CTests |
| Allocation and planning | `allocation_provider.hpp`, `arena.hpp`, `memory_planner.hpp`, `planned_executor.hpp` | [Planner contract](planner.md); `memory_planner`, `planned_executor` |
| Execution backends | `cpu_backend.hpp`, `cuda_backend.hpp`, `src/runtime/cpu_*`, `backend/cuda/` | [CPU](cpu_backend.md), [CUDA](cuda_backend.md), and [SGEMM](cuda_gemm.md) contracts |
| Placement | `scheduler.hpp`, `src/runtime/scheduler.cpp`, `copy.hpp` | [Scheduler contract](scheduler.md); scheduler/CUDA scheduler CTests |
| Decoder and files | `include/model/{decoder,kv_cache,model_file,quantization,tokenizer,generation}.hpp` | [Decoder](decoder.md), [KV](kv_cache.md), [model file](model_file.md), [tokenizer](tokenizer.md) |

## Ownership and lifetime

`Tensor` owns shape, strides, dtype, offset, and a shared `Storage` handle; it
does **not** own a separate data buffer. Views share that storage and change
metadata/offset only. `Storage` owns exactly one host or CUDA backing buffer;
CPU↔CUDA movement is an explicit graph `COPY` operation. A non-contiguous view
must be materialized visibly before a backend that requires contiguous input.

A frozen `Graph` owns topology and tensor descriptors. At execute time, the
ordinary executor uses dynamic last-use reclamation. The opt-in prepared path
has `MemoryPlanner` compute deterministic slots and `PlannedExecutor` allocate
the backing arena at prepare time; warmed execution allocates no intermediate
backing buffer. Outputs, external inputs, aliases, persistent K/V state, and
copy buffers have separately defined lifetimes and are never silently counted
as reusable intermediates. The required planner gate and its slower tiny-graph
latency result are documented in [Stage 5](stage5_report.md).

`KVCache` owns persistent FP32 K/V storage and commits its active range only
after all layer writes complete. Quantized V2 loading intentionally keeps the
INT8/scales **and** a persistent FP32 dequantized workspace; artifact
compression is therefore not a resident-memory-reduction claim. See
[Stage 14](stage14_report.md) and [Stage 16](stage16_report.md).

## Execution path

1. `model_file` validates a versioned, bounded little-endian model container;
   `tokenizer` maps bytes plus BOS/EOS deterministically.
2. `Decoder` composes embedding, RMSNorm, RoPE, causal attention, SwiGLU and
   output projections as standard graph operators. Prefill/decode graph writes
   to `KVCache` are graph-visible copies.
3. `Scheduler` checks backend capabilities, assigns supported learned
   projections to CUDA when requested, and inserts/deduplicates explicit copy
   nodes. Unsupported primitives stay on CPU; there is no hidden model-level
   CUDA fallback.
4. The executor either performs dynamic allocation or uses a previously
   validated prepared plan. Each backend owns its own dispatch and validates
   device/dtype/layout bindings.
5. `llm_cli` calls `generate_greedy`; it exposes backend, quantization, and
   cache mode and records deterministic configuration. It does not claim
   sampling or language quality.

The CPU reference remains FP64 accumulation where the corresponding contract
requires it. Stable defaults retain dynamic allocation and CUDA V0; the Stage
9 V1 kernel is explicit/selectable rather than an automatic replacement. The
frozen Stage 0 `sgemm_v0_naive` is preserved.

## Operational runbooks

### Local CPU correctness

A local machine needs CMake, a C++17 compiler, Python 3 and Git. CUDA is
intentionally disabled below; passing this suite does not validate CUDA.

```bash
cmake -S . -B build-local -DCMAKE_BUILD_TYPE=Release -DENABLE_CUDA=OFF -DBUILD_TESTING=ON
cmake --build build-local --parallel 4
ctest --test-dir build-local --output-on-failure --no-tests=error
python3 tools/check_documentation.py
```

### T4 release validation

CUDA-backed claims require a visible Tesla T4, CUDA Toolkit and cuBLAS. The
complete frozen Stage 18 measurement/report procedure is
[stage18_colab.md](stage18_colab.md). The final Stage 19 fresh-checkout audit,
which reuses the accepted Stage 18 artifact rather than tuning again, is
[stage19_colab.md](stage19_colab.md).

### Evidence discipline

Every formal measurement is captured under `results/` with raw samples,
commands, environment, source identity and hashes. Code source and result
commits are separate. Failed or slower runs stay in the evidence tree; a
correctness pass is not rewritten as a performance pass. The accepted RC4
capture has an external archive for six Git-size-excluded Nsight files; its
SHA-256 and retrieval condition are in [Stage 18](stage18_report.md).

## Supported scope and explicit limits

The project demonstrates a batch-1, bias-free, deterministic tiny decoder
runtime. It does not claim pretrained-model compatibility, text quality,
batching service support, multi-GPU execution, Flash/paged attention, training,
INT8 GEMM acceleration, scheduler acceleration, or lower INT8 resident memory.
Optional skips are listed in [Stage 18](stage18_report.md); they are not hidden
as completed work.
