# Stage 1 — Tensor and Storage

**Goal:** freeze one backend-neutral tensor contract before any operator or backend implementation. **Prerequisite:** Stage 0. **Produces:** `include/runtime/{dtype,device,shape,storage,tensor}.hpp`, `src/runtime/{storage,tensor}.cpp`, `tests/test_tensor.cpp` (paths are proposed). **Stage gate:** CPU-only build/tests pass; CUDA declarations compile conditionally but no CUDA allocation is required yet.

## S1-C1 — Scalar and metadata vocabulary
- **Goal / type:** Define `DType`, `DeviceType`, `Device`, bounded rank `Shape`, `Stride`, checked element-count/byte-size helpers. *Functional + Correctness.*
- **Depends / files:** Stage 0; `dtype.hpp`, `device.hpp`, `shape.hpp`, tests. Define explicit overflow and negative-dimension errors.
- **Scope / not scope:** Row-major semantics, rank 0–MAX_RANK; define FP32 and INT32 token-ID metadata/access (INT8 payload arrives in S16). Scalar rank 0 has numel=1; any zero dimension has numel=0; nonnegative strides only initially. No allocation, broadcasting, symbolic shapes or FP16. Specify signed dimension checks and overflow-safe byte/span arithmetic.
- **Verify / baseline / metrics:** CPU unit cases for scalar, rank limit, overflow, invalid device; baseline N/A; record no performance claim.
- **Accept / commit / risk:** Exact byte counts and deterministic diagnostic failures. `feat(tensor): add dtype device shape metadata`. Risk: ambiguous empty/scalar semantics; decide and document before C2.

## S1-C2 — Shared storage and buffer ownership
- **Goal / type:** Implement reference-counted `Storage` with immutable device, byte capacity, data pointer, custom deleter; CPU allocation is the first implementation. *Functional + Memory + Correctness.*
- **Depends / files:** C1; `storage.hpp/.cpp`, allocation-test helpers.
- **Scope / not scope:** Tensor metadata references `shared_ptr<Storage>`; storage outlives views. No arena, pooling, CUDA `cudaMalloc`, or implicit host/device copies.
- **Verify / baseline / metrics:** Lifetime, alias, zero-byte policy, allocation failure injection, ASan/UBSan. Baseline N/A; expose test-only allocation counter.
- **Accept / commit / risk:** Last reference frees exactly once, no dangling data pointer/double free. `feat(tensor): add shared CPU storage`. Risk: custom deleter must remain valid across shared ownership.

## S1-C3 — Contiguous tensor construction and checked access
- **Goal / type:** Build tensors from storage + metadata and factories for CPU-contiguous tensors; validate offset/capacity/stride address range. *Functional + Correctness.*
- **Depends / files:** C1–C2; `tensor.hpp/.cpp`, tests.
- **Scope / not scope:** Typed access is explicit and dtype-checked; implement `numel`, `nbytes`, `is_contiguous`, `data_offset`. No implicit reallocation or operator kernels.
- **Verify / baseline / metrics:** 1D–4D addressing, empty policy, bounds and dtype mismatch; baseline N/A.
- **Accept / commit / risk:** Invalid metadata cannot construct a tensor; contiguous layout address mapping is tested. `feat(tensor): add contiguous tensor construction`. Risk: integer overflow in offset+stride bounds.

## S1-C4 — Zero-copy reshape and view
- **Goal / type:** Implement reshape when contiguous and `view(shape, stride, offset)` when range-valid; both share `Storage`. *Functional + Correctness + Memory.*
- **Depends / files:** C3; tensor API/tests.
- **Scope / not scope:** Metadata-only reshape/view and checked positive-step narrow/slice, needed for per-head attention and active KV ranges; aliases retain base storage. Offset is documented in bytes and strides in elements. Reject impossible reshape rather than copy; zero strides/writable overlapping views are unsupported initially. No broadcasting/advanced indexing.
- **Verify / baseline / metrics:** Mutation through source/view aliases, storage use-count/lifetime, zero allocator delta; baseline is C3 contiguous tensor and metric is allocations=0.
- **Accept / commit / risk:** View destruction never releases live source storage; reshape cannot change numel. `feat(tensor): add zero-copy reshape and views`. Risk: views over empty tensors need an explicit offset rule.

## S1-C5 — Transpose, contiguous materialization, copy contract
- **Goal / type:** Implement dimension permutation as stride-only transpose and a CPU `contiguous()` materialization/copy primitive. *Functional + Correctness + Memory.*
- **Depends / files:** C4; `tensor.cpp`, proposed `ops/copy.cpp`, tests.
- **Scope / not scope:** Transpose is zero-copy; explicit CPU contiguous materialization handles strided data. This initial copy helper is reused behind the backend COPY/materialization contract in S6, not kept as a second runtime dispatch path. Device-specific copy behavior arrives in S8; CUDA `contiguous()` cannot silently copy on the host.
- **Verify / baseline / metrics:** 2D/3D transpose values and strides; alias test; materialized output independent; bytes copied and allocation count. Baseline is zero-copy view (0 allocation) versus intentional contiguous copy.
- **Accept / commit / risk:** Non-contiguous tensors never falsely report contiguous; copy handles overlapping storage safely or rejects it explicitly. `feat(tensor): add transpose and contiguous copy`. Risk: overlap semantics must be specified.

## S1-C6 — Tensor contract freeze and developer evidence
- **Goal / type:** Add API invariants documentation, randomized metadata/property tests, and CPU sanitizer target. *Correctness + Integration.*
- **Depends / files:** C1–C5; `docs/tensor.md`, tests, CMake.
- **Scope / not scope:** Freeze public names/ownership rules for S2; no CUDA implementation or graph API.
- **Verify / baseline / metrics:** Debug/Release CPU tests, ASan/UBSan, property-generated valid views compared to reference indexing. Baseline N/A.
- **Accept / commit / risk:** One documented tensor representation is used by all later work; no known sanitizer failure. `test(tensor): lock tensor storage invariants`. Risk: premature API freeze; additions must preserve core ownership semantics.
