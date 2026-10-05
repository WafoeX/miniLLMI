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
