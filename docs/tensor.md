# Tensor/storage contract (Stage 1)

Namespace: `runtime`. The single representation is backend-neutral metadata referencing shared `Storage`; later operators/backends must reuse it.

## Metadata (S1-C1)

- `DType::{FP32, INT32}` have 4-byte elements. Unsupported enum values fail.
- `Device(DeviceType::{CPU, CUDA}, index)` requires CPU index 0 or CUDA index ≥0. CUDA identity is metadata only and does not assert that a GPU exists; headers work without the CUDA toolkit.
- `Shape` and `Stride` hold immutable-to-callers, signed 64-bit nonnegative entries, rank 0–`MAX_RANK` (8). Axis access is checked. Strides are in **elements**; storage/tensor offsets and capacities are in **bytes**.
- Rank-0 is a scalar: numel=1. Any zero dimension is empty: numel=0, independent of other dimensions. Empty tensors have no addressable elements.
- Canonical row-major strides multiply `max(dim, 1)` from the right, so empty layouts never introduce broadcast/zero strides. Unrepresentable canonical strides still fail even when numel=0.
- `nbytes` is logical payload; `storage_span_bytes` is the bounding address span including holes. All counts, spans and offset additions use checked arithmetic.
- Invalid values/ranks use `std::invalid_argument`, invalid axes use `std::out_of_range`, unrepresentable arithmetic uses `std::overflow_error`. Allocation failures use `std::bad_alloc` once storage is implemented. Diagnostics are nonempty; tests check exception categories rather than exact wording.

No allocation/performance claim is made by metadata tests. INT8, negative strides, broadcasting and symbolic dimensions are outside Stage 1.

## Storage (S1-C2)

- `Storage::allocate_cpu(bytes)` owns one buffer from `operator new`, aligned for FP32/INT32. Zero capacity means a null pointer and no backing-buffer allocation. Storage/control-block metadata may still allocate.
- `Storage::wrap(device, bytes, pointer, deleter)` requires pointer null iff bytes=0, and a nonempty deleter. Invalid arguments do not transfer ownership. After validation, ownership transfers even if metadata/control-block allocation fails. The deleter is retained by value and called once on last release, including zero-byte wrapped storage; it **must not throw**.
- Device, capacity and pointer are immutable. Storage is noncopyable/nonmovable; `shared_ptr<Storage>` is the ownership handle. Views keep it alive. Wrapped CPU pointers must have adequate alignment and live typed objects for their later dtype; caller-supplied capacity/device identity is trusted.
- CUDA storage can be described by a wrapped externally owned device pointer, but no CUDA allocation/copy is implemented here. CUDA identity never permits CPU dereference.
- `RUNTIME_TESTING` exposes cumulative nonzero CPU backing-buffer allocation/free/live counters and one-shot failure injection; these are absent in `BUILD_TESTING=OFF` builds. No timing or throughput claim is made.

## Tensor construction and access (S1-C3)

- `Tensor(storage, dtype, shape, stride, offset_bytes=0)` validates nonnull storage, dtype alignment, offset divisible by element size, rank agreement, positive strides, overflow and bounding span ≤capacity. Empty offsets may be exactly one-past capacity, but never beyond it; `data<T>()` returns null for every empty tensor.
- Writable overlapping layouts and zero strides are rejected. Non-overlap is conservatively proven by sorting non-singleton axes by stride: each next stride must be at least the bounding span of faster axes. Exotic disjoint layouts that cannot pass this sufficient proof are intentionally unsupported.
- `Tensor::allocate_cpu(shape, dtype=FP32)` creates canonical, contiguous storage and value-initializes typed C++17 elements to zero. Scalar construction allocates one element; empty construction allocates no backing buffer. Factories validate sizes before allocation.
- `shape()`, `stride()`, `dtype()`, `device()`, `storage()`, `data_offset()`, `numel()`, `nbytes()`, `is_contiguous()` are the public metadata vocabulary. Singleton axes do not constrain nonempty contiguity. Empty contiguity requires canonical strides; if canonical strides are unrepresentable, `is_contiguous()` returns false and explicit canonical construction/materialization fails with overflow.
- `data<float/int32_t>()` points to the first logical element, **not** a promise that the whole logical tensor is flat/contiguous. `at<T>(indices)` handles strides with checked index rank and bounds. Both reject dtype mismatch and CUDA host access (including empty tensors); const access returns const elements. No unchecked operator kernels are implemented.
- Copying a Tensor copies metadata and shares storage. Moved-from objects are only valid for destruction/reassignment. Callers own synchronization for concurrent mutation; shared reference counting is not a data-race policy.

## Metadata-only transforms (S1-C4)

- `reshape(shape)` requires contiguous source and equal numel, including scalar/empty cases. It retains storage and byte offset. It never copies on failure.
- `view(shape, stride, offset_bytes=0)` adds a **relative byte offset** to the current tensor's storage offset with checked arithmetic. The new layout is validated against the shared storage capacity, not the source logical extent; this low-level API may expose other initialized regions in the same storage. Use checked slices for logical subranges.
- `narrow(axis, start, length)` and `slice(axis, start, length, step=1)` use nonnegative start/length; length is an element count, not an end index. Step is strictly positive. Axis and final logical source index are checked before construction. Result strides/offset calculations are overflow-checked.
- Empty slices (including start=axis size and length=0) retain the source byte offset instead of manufacturing a possibly invalid strided one-past address. Explicit empty views may use an aligned offset up to capacity.
- These operations allocate **zero backing buffers** (metadata vectors/control blocks are not claimed allocation-free), share mutable elements and keep storage alive independently of the source Tensor's lifetime. Metadata transforms also work for CUDA identity without touching device memory.

## Permutation and copy (S1-C5)

- `permute(axes)` requires a complete unique axis permutation; `transpose(first, second)` swaps two valid axes. They share storage/offset and change only shape/stride; scalar `permute({})` is identity.
- `contiguous()` is explicitly CPU-only. Already-contiguous inputs return a shared alias; non-contiguous inputs produce independent, canonical CPU storage (one nonzero backing allocation, or zero for an empty result) populated in logical row-major order. CUDA calls always fail, even for an already-contiguous tensor; no hidden host fallback.
- `copy_cpu(source, destination)` is one CPU **leaf primitive**, not a parallel operator/dispatch stack. S6 must reuse it behind COPY/materialization execution. It accepts identical shapes/dtypes on CPU, including positive strided destinations; returns logical bytes copied, and never allocates backing buffers.
- Exact same-storage/offset/stride self-copy is a no-op (returns 0). Empty copy returns 0. All other intersecting bounding **address** spans are rejected before mutation, even across different wrappers of one pointer. Conservative rejection includes disjoint logical elements with intersecting spans; caller can materialize to independent storage explicitly. Disjoint subranges of one storage are accepted. No memmove/inplace-copy claim is made.

```cpp
#include "runtime/tensor.hpp"
auto x = runtime::Tensor::allocate_cpu({2, 3});
x.at<float>({1, 2}) = 7.0f;
auto t = x.transpose(0, 1);           // alias, strides {1, 3}
auto independent = t.contiguous();   // explicit CPU materialization
```

Raw pointers and custom wrapped storage remain a trusted low-level interface; checked metadata cannot prevent caller writes through raw pointers or incorrect external lifetime/capacity declarations.

## Contract freeze and verification (S1-C6)

The public names, byte/element units, aliasing, ownership, empty/scalar semantics, conservative overlap rejection and explicit CPU copy boundary above are frozen for S2. Additions must preserve this one representation; no operator or CUDA runtime path has been introduced.

`tests/test_tensor.cpp` covers deterministic metadata/error cases, 1D–4D/scalar/empty access, custom deleters/failure injection, view survival/refcounts/mutations, zero buffer-allocation deltas, 2D/3D transposes, strided destinations, copy overlap/dtype/device errors and full buffer release. `tests/test_tensor_properties.cpp` executes 600 fixed-seed (`0x51a7`) FP32/INT32 cases, ranks 0–8, with independent reference indexing, padded/gapped layouts, relative offsets, permutation, positive slices, materialization and padding-preserving copies. Failure output retains the seed/trial index. `tests/test_tensor_validation.py` tests evidence-runner failure handling using mock subprocesses in temporary directories; mock outputs are never stored as real results.

```bash
cmake -S . -B build-tensor-sanitizer -DCMAKE_BUILD_TYPE=Debug \
  -DENABLE_CUDA=OFF -DENABLE_SANITIZERS=ON
cmake --build build-tensor-sanitizer --target check_tensor_sanitizers --parallel 4
# Full, fresh Release/Debug/sanitizer tests + BUILD_TESTING=OFF production build:
python3 tools/validate_tensor.py
```

Sanitizer instrumentation is CPU-only, GCC/Clang, ASan+UBSan with undefined-behavior recovery disabled. The evidence runner records sanitizer environment options and source before/after; source changes invalidate a run. macOS disables unavailable LeakSanitizer explicitly; ownership counters and ASan/UBSan still run. On Linux LSan is enabled by default. Raw logs/snapshots and failures live in unique `results/tensor/local/<run_id>/` directories; no timing or performance metrics are generated.
