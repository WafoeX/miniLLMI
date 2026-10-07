# Scheduler contract (Stage 11)

## C1 — Deterministic capability policy

`Scheduler` borrows the existing CPU backend and optionally one CUDA backend.
`place` is metadata-only: a requested registered backend wins only when its
operator/dtype capability and input layout are supported. Current CUDA compute
coverage is contiguous FP32 rank-2 MATMUL plus COPY; no CUDA ADD/MUL/transformer
primitive is claimed. Input shape/attribute semantics are validated by Graph
inference, not redefined by placement. Noncontiguous CUDA bindings need explicit
MATERIALIZE; placement does not insert materialization or change kernels.

Default unsupported/missing CUDA requests become explicit CPU placements if CPU
supports the operation. `PlacementFallback::Error` rejects them instead. A
primitive unsupported by both backends is an error, not a hidden fallback.
There is no cost model, autotuning, multi-GPU, stream overlap, or default-kernel
change. C1 does not allocate, copy, rewrite, or execute a graph.

Local C1 acceptance: CPU Release build and 32/32 CTests, including repeated
placement, unavailable/unsupported CUDA, strict errors, dtype/layout rejection,
and no early Stage 12 support. CUDA tests here use a metadata-only capability
double, never GPU execution evidence. Subsequent mixed execution needs T4.

## C2 — Rewrite, copies and lifetime planning

`Scheduler::rewrite` requires a frozen logical graph. External bindings stay on
actual devices; compute nodes use their hint (or declared output device) through
C1. Metadata aliases stay with their source Storage, without claiming CUDA
view kernels; contradictory alias hints are errors. An explicit COPY destination
is never silently relocated. Unsupported transformer primitives still error.

An allocating, **one-input COPY** uses `CopyAttrs::destination` and has
`OutputKind::NewTensor`. Existing two-input COPY has no destination attribute,
retains its exact persistent-state write contract and `OutputKind::Write`.
Legacy serialization/fixtures are unchanged. Automatic transfer nodes use the
allocating form, are traceable in `inserted_copies`, and are deduplicated by the
exact logical source tensor version and destination device, not by storage root.
Named outputs retain their declared boundary device through explicit copies.
Noncontiguous cross-device copies are rejected: this stage never hides a
materialization kernel. D2D means distinct allocations on the same registered
CUDA device; multi-GPU/peer copying is out of scope.

The rewrite re-freezes the graph, including topology, shape/layout, aliases,
overlapping state accesses and version checks. Node IDs are remapped in original
topological order, but state correctness still requires data dependencies, not
numeric node order. COPY writes keep their explicit external state destinations.
Each logical write-output ID is a new version; cached transfer values are never
reused across writes or across executions. Backends/copies are synchronous,
so reads finish before dependency-ordered mutation and consumers follow copy
completion. Invalid/stale state graphs must fail before scheduling.

`plan_memory` recomputes closed liveness intervals including transfer outputs,
views and named outputs. Slot offsets/capacities are **per device**; aggregate
capacity is their sum. Reuse is allowed only within one device and disjoint
lifetimes. `ScheduledAllocationProvider` allocates one shared backing Storage
per device during prepare, using Backend::allocate and the original Tensor.
It validates all bindings and backend preparation before external writes. Current
CPU/CUDA workspace requirements are zero; a future nonzero requirement is an
explicit error, never omitted storage. CPU-only S5 plans cannot validate the
rewritten mixed graph. CPU/CUDA homogeneous providers keep their restrictions.
Returned outputs or any escaped alias prevent reuse until released.

Call the **existing** executor explicitly:

```cpp
Scheduler scheduler(cpu_backend, &cuda_backend);
auto scheduled = scheduler.rewrite(logical_graph);
// check scheduled.ok() before dereferencing
ScheduledAllocationProvider prepared(*scheduled.graph, scheduler);
auto result = execute_graph(*scheduled.graph, &trace, &prepared, nullptr, &scheduler);
```

No alternate graph/tensor runtime or execution-time hidden transfer path exists.
All copies are descriptor-visible, counters record actual nonempty/non-self copy
calls and bytes. `BackendCPU`/`BackendCUDA` trace events record actual dispatched
calls, including the CUDA copy backend for D2H. Metadata aliases and empty/self
copies do not add dispatches. `backend_switches` counts transitions between
consecutive executed backend devices (not placement/segment-count proxies).
CUDA adapter MATMUL accepts the Runtime's zero dimensions: M/N=0 is an
empty-output no-op; K=0 with nonempty output is explicitly zero-filled on its
own CUDA stream and synchronized. Positive-size Stage 0/v1/cuBLAS dispatch is
unchanged. Backend dispatch counters describe calls, not CUDA kernel launches;
an empty compute call can dispatch without launching a kernel. Legacy
unscheduled traces and default CPU dynamic/CUDA v0 paths are preserved.
The prepared path does not claim zero C++ metadata heap allocations.

## Validation and limits

CPU tests cover deterministic rewrite, deduplication, per-device plan validation,
CPU-only fallback execution, INT32/empty COPY, version invalidation and stale
state rejection, repeated external mutation, and output lifetime/busy contexts.
CUDA metadata doubles cannot allocate or execute. T4 tests separately cover
manual vs automatic CPU→CUDA→CPU values/counts, rectangular and empty shapes,
GPU v0 and explicit v1, actual traces, repeated state writes, D2D and aliases.
They retain full CPU reference/output vectors and both execution traces when
invoked with a new artifact directory.

See [C2 Colab procedure](stage11_colab.md). C2 T4 acceptance is pending; C4 is
not started until it passes. C3 segmentation is `skipped_optional`: no tracing
has yet justified a grouping optimization; no reduction in actual switches or
latency is claimed. No Stage 11 completion or model/scheduler speedup claim.
