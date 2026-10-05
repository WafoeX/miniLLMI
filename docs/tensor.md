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
- `shape()`, `stride()`, `dtype()`, `device()`, `storage()`, `data_offset()`, `numel()`, `nbytes()`, `is_contiguous()` are the public metadata vocabulary. Singleton axes do not constrain nonempty contiguity. Empty contiguity requires canonical strides.
- `data<float/int32_t>()` points to the first logical element, **not** a promise that the whole logical tensor is flat/contiguous. `at<T>(indices)` handles strides with checked index rank and bounds. Both reject dtype mismatch and CUDA host access (including empty tensors); const access returns const elements. No unchecked operator kernels are implemented.
- Copying a Tensor copies metadata and shares storage. Moved-from objects are only valid for destruction/reassignment. Callers own synchronization for concurrent mutation; shared reference counting is not a data-race policy.
