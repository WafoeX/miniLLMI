# Stage 3 — Graph / sequential CPU executor contract

Stage 3 is a correctness-only CPU runtime foundation, not a scheduler, arena,
planner, optimized backend or model executor. Stage 0 GEMM is untouched. All
execution uses the existing `runtime::Tensor`, shared `Storage`, S2 operators
and CPU reference kernels. There is no second tensor/allocator path.

Stage 4 adds an optional third `AllocationProvider*` argument to execute_graph;
the null-provider behavior below remains the stable S3 dynamic baseline. See
[arena/provider contracts](arena.md) for opt-in arena spans, output-context
pinning, expanded counters and BlockAllocate/BlockFree trace events.

Stage 6 adds an optional fourth `const Backend*` argument. Default execution
retains S2 FP64 reference math behind the common Backend interface; explicit
CPU FP32 v0 selection uses the same graph/provider path. See [CPU backend
contracts](cpu_backend.md). The lifetime/trace/default semantics below remain
unchanged; no automatic scheduling or hidden copy is introduced.

## Ownership, construction and freeze

`include/runtime/graph.hpp` owns value records in ID-keyed maps, not raw node
pointers. Tensor IDs and node IDs are separate uint32 namespaces; UINT32_MAX is
reserved. Caller-chosen IDs need not be contiguous. Named inputs own persistent
Tensor handles; intermediate declarations own **no backing buffer**. Several
input names may share one Storage, with consistent state declarations. Different
Storage wrappers covering intersecting allocations are conservatively rejected:
use one shared Storage and checked Tensor views instead.

- `add_input(id, name, Tensor, persistent_state=false)` declares an initialized,
  externally owned root, including weights and future state/cache bindings.
- `add_tensor(id, Shape, dtype=FP32, device=CPU)` declares a logical intermediate
  or output, without allocation. Stride/base/offset are inferred, not guessed.
- `add_node(id, OpDesc)` requires known input/output IDs and one producer per
  non-root output. Descriptors may be inserted before their producers; IDs and
  consumer lists never point into reallocatable node objects. Repeated operands
  are one consumer edge. Roots cannot be overwritten as logical IDs.
- `add_output(name, id)` pins the tensor. Multiple names may pin the same ID.
- Builder misuse throws `invalid_argument`; mutation after freeze throws
  `logic_error`. Read access is const. Graph value copies own independent topology
  and share only external Storage bindings. Do not concurrently mutate bindings
  or execute stateful copies of the same graph.

`freeze()` validates all declared records, including disconnected/dead branches:
missing roots/producers, cycle, schema, inference, declared output shape/dtype/
device and mutation hazards. A min-ready-node-ID Kahn traversal yields a stable
order independent of insertion order. There is no scheduling, backend selection,
allocation or value read. Diagnostics identify a node/tensor or conflicting
state access. Freeze is transactional; failure installs no inferred layouts or
order, and the graph remains editable. Success is idempotent and locks topology.
Empty graphs, disconnected valid roots, unused inputs, no outputs and input-only
outputs are allowed. Output shapes are explicit; dynamic shapes require a new
prepare/freeze, not mutable runtime metadata.

### Shared pure metadata inference

`Layout` (`layout.hpp/.cpp`) is a declarative shape/stride/dtype/device/capacity/
byte-offset/root-key record. It has **no buffer pointer, ownership, data access or
allocator**. The conservative non-overlap, range, empty-slice, stride and view
checks were extracted from Stage 1 Tensor, which now delegates those same checks.
Tensor's public fields/API, shared Storage ownership, typed access, allocation
and CPU copy path are unchanged. `infer_layout` is the single S2 semantic
implementation; `infer_operator` adapts real Tensor inputs and binds Alias/Write
contracts back to those same Storage objects. It additionally retains the real
address-overlap check for distinct wrapped Storage. Existing S1/S2 tests remain
mandatory regressions. This avoids fake pointers or allocating prototype buffers
while validating a graph. Root keys are equality tokens, never host addresses in
trace records.

## Explicit state writes and aliases

COPY is `(source_id, destination_version_id) -> new_version_id`. Only a root
explicitly declared `persistent_state=true`, or its inferred view/version, can
be a destination. The output references the destination's same Storage and
checked span. Neither COPY nor a metadata alias allocates a new buffer.
RESHAPE/VIEW/NARROW/SLICE/TRANSPOSE/PERMUTE inherit the physical base and logical
write lineage. Physical base links are **not** producer dependency edges.

For intersecting conservative address spans, writes/reads must be explicitly
ordered by DAG dependencies; numeric node order is not a write dependency.
Reads after a write must consume that new logical version (or a descendant
alias), not an old root even when another operand happens to order the nodes.
Reads explicitly completed before a write are valid. Unordered overlapping
writes, unordered reads, stale ordered reads and pinned stale state outputs are
rejected. Disjoint checked slices may be written independently. Destination
operands of later COPY nodes also follow the new-version rule. Metadata aliases
alone do not read values; their eventual consumers/output spans are checked.
Conservative span checks may reject exotic layouts with disjoint elements but
intersecting envelopes; there is no unrestricted mutable-tensor escape hatch.

A failed execution discards **all** returned outputs; prior completed external
state writes are not rolled back. Kernels retain S2's partial-result-overflow
policy. There is no snapshot/transactional cache implementation in S3. Holding
state outputs across another execution observes shared mutable Storage, not a
historical snapshot; explicitly MATERIALIZE a snapshot if needed. This protocol
is foundational metadata for future KV work, not a KV policy/implementation.

## Sequential CPU execution and lifetime baseline

`execute_graph(const Graph&, ExecutionTrace* = nullptr)` requires successful
freeze and CPU tensor devices, including pass-through outputs. No hidden host
fallback, transfer, materialization or backend-hint dispatch is permitted.
Every node executes in frozen order, including dead/disconnected valid branches.

- Alias nodes bind checked metadata from S2 inference without invoking a kernel.
- NewTensor nodes allocate a canonical CPU Tensor **inside execute**, one
  nonzero backing buffer per node. Allocation of graph outputs is inside the
  same boundary. Empty tensors own no nonzero backing allocation.
- COPY/MATERIALIZE and ADD/MUL/MATMUL dispatch through Backend with
  caller-provided outputs. The default CPU backend runs existing S2 reference
  semantics; default MATMUL remains FP64-accumulate, **not** the explicit S6
  FP32 performance baseline. Alias nodes likewise validate through Backend but
  invoke no numerical kernel.
- Transformer descriptors can freeze, but reference execution returns
  `Unsupported`; there are no Transformer kernels until S12.
- Release each unpinned TensorId immediately after its last unique consuming
  node. Release unconsumed intermediates immediately after production. Shared
  Storage extends physical base lifetime through every alias; named outputs pin
  both their handle and base. External bindings remain owned by the Graph.
- Return `ExecutionResult` with Status, optional failed_node, named Tensor
  outputs and counts. Stop on the first error, return no valid outputs, and free
  all dynamic-owned buffers. Backing/metadata `bad_alloc` maps to
  `ResourceExhausted`; ordinary catastrophic inability to allocate even a Status
  diagnostic may still propagate a standard exception.

No executor instance caches intermediate buffers. Repeated calls allocate fresh
intermediates; input values may change in place, but metadata may not. Planned
execution in S5 must preserve these exact last-use, alias-base, output-pinning,
error and output-allocation boundary semantics; it must not compare against a
retain-until-graph-end dynamic baseline.

### Always-on count vocabulary (not timings)

`ExecutionCounts` measures this call's nonzero **dynamic-owned backing buffers**:
allocations, frees inside execution, cumulative allocated_bytes, current live_bytes,
peak_live_bytes, nodes_completed and successful data-moving copy events/bytes.
New graph-output buffers are included. External inputs/weights/persistent state
and metadata/container heap allocations are excluded, explicitly; these counters
are **not total resident process memory**. A pinned output's eventual destructor
is outside execution, so its free is not part of the returned count. Arithmetic
is not a copy. Empty and exact-self COPY have zero movement; a state version can
still be produced. A successful MATERIALIZE is explicit movement.

The tests compare counters with actual Storage backing hooks, and preserve the
fixed ADD->MATMUL and diamond counts in verbose raw test logs. No latency,
throughput, optimization benefit or CPU/GPU performance claim is made.

## Bounded execution trace

`graph_trace.hpp/.cpp` is the common diagnostic event interface for later
allocator/scheduler work. Passing nullptr disables event construction/storage.
`ExecutionTrace(limit=1024)` keeps the first N events and reports `dropped()`;
limit 0 stores none. Each execution resets it. Full traces/diagnostic allocation
failures drop events without changing execution or backing policy. Rank-bounded
metadata and an explicit event cap keep memory bounded. A trace owns only layout
snapshots/IDs, never Tensor/Storage, and cannot extend buffer lifetimes.

Events: NodeBegin; Tensor input/output metadata; Allocate; Alias; StateWrite;
Copy; NodeEnd; Release (logical handle); Free (dynamic physical base); Output;
Failure. Metadata includes shape/stride/dtype/device/base key/offset/capacity;
Allocate/Free/Copy include bytes; Failure includes StatusCode. The optional node
ID associates releases with the completed/failed node; output export has no
node. Free is emitted only when the final dynamic base handle disappears.
Output events describe transfer to the returned result, not execution-time frees.
The trace doesn't observe caller destruction after return or roll back state.

Golden diamond tests check the entire event sequence, inferred metadata, backing
counters, truncation/reset, disabled equivalence, alias-base delayed free, explicit
copy/self-copy and first-error cleanup. Trace is diagnostics only: future
benchmarks must disable it or declare its bounded configuration; no event here
contains a timestamp or supports a timing claim.

## Commands and scope

```bash
cmake -S . -B build-local -DCMAKE_BUILD_TYPE=Release -DENABLE_CUDA=OFF
cmake --build build-local --parallel 4
cmake --build build-local --target check_graph
# Clean committed source, fresh Release/Debug/ASan+UBSan/production evidence:
python3 tools/validate_graph.py
# Uncommitted development only (labelled source_dirty=true):
# python3 tools/validate_graph.py --allow-dirty
```

The fixture harness reuses the S2 SHA/provenance verifier and committed offline
fixtures; 8 core numeric cases run through frozen graphs, 9 Transformer cases
explicitly test Unsupported execution, not numerical kernels. CPU tests need no
GPU. Linux/LSan and CUDA-enabled/T4 integration are separate untested environments
here; S8+ CUDA gates remain mandatory when reached. No Stage 4+ implementation
is included. Acceptance evidence is recorded separately in `stage3_report.md`.
