# Stage 3 — Computation graph and sequential executor

**Goal:** turn descriptors into a validated DAG, without allocator/scheduler policy. **Prerequisite:** S2-C5. **Gate:** CPU graph tests execute core ops in topological order with clear diagnostics.

## S3-C1 — Graph ownership model
- **Goal/type:** Define graph-owned tensor records, node IDs, producer/consumer links, named inputs/outputs. *Functional.*
- **Depends/files:** S1–S2; `graph.hpp/.cpp`, tests.
- **Scope/not:** `add_tensor`, `add_node`, IDs plus alias/base/range records; no execution, allocation or backend selection. External state destinations (future KV) are explicit bindings with ordered write nodes, not unrestricted mutable tensors. A write yields a new logical state/version ID referencing the same persistent Storage; reads depend on that ID, preserving single-producer DAG validation. Unordered overlapping writes are rejected.
- **Verify/baseline/metrics:** duplicate IDs, unknown tensors, producer uniqueness; N/A.
- **Accept/commit/risk:** graph has one owner of topology and no raw dangling node pointers. `feat(graph): add tensor-node graph model`. Risk: conflating runtime storage with graph tensor identity.

## S3-C2 — Validation and topology
- **Goal/type:** Validate arity/inference, missing producer/input, cycle detection; compute stable topological order. *Correctness.*
- **Depends/files:** C1; graph validator/tests.
- **Scope/not:** no mutation after freeze and no scheduling.
- **Verify/baseline/metrics:** diamond, disconnected allowed roots, cycles, invalid output; N/A.
- **Accept/commit/risk:** same graph yields deterministic order; invalid graph cannot execute. `feat(graph): validate DAG and topological order`. Risk: distinguish graph input from missing producer.

## S3-C3 — Sequential CPU executor
- **Goal/type:** Execute frozen topology through S2 CPU reference registry. *Functional + Integration.*
- **Depends/files:** C2/S2-C3; `graph_executor.cpp`, tests.
- **Scope/not:** allocate output backing buffer per node (intentional baseline), release dead intermediates after last use, pin graph outputs, and extend base lifetimes for aliases. Dynamic and planned executors use identical lifetime/output semantics; no arena/CUDA. Keep output allocation outside the timed execute boundary only if both paths do so.
- **Verify/baseline/metrics:** add→matmul graph, fixture comparison, error propagation; baseline is per-node allocation, count allocations.
- **Accept/commit/risk:** graph output matches direct reference and executor stops on first error. `feat(graph): add sequential CPU executor`. Risk: hidden implicit copies.

## S3-C4 — Graph execution trace
- **Goal/type:** Emit test/debug trace of node order, tensor metadata, allocation/copy events. *Integration.*
- **Depends/files:** C3; trace interface/docs/tests.
- **Scope/not:** diagnostics only; no timing claim.
- **Verify/baseline/metrics:** expected trace for diamond graph; N/A.
- **Accept/commit/risk:** later allocator/scheduler benchmarks can count events from one interface. `feat(graph): add execution trace hooks`. Risk: tracing must be disabled or bounded in benchmarks.
