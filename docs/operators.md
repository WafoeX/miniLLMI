# Operator contract — semantics version 1

Stage 2 is declarative operators and a CPU correctness reference, not a graph, scheduler or optimized backend. All operators reuse the Stage 1 `runtime::Tensor`/shared `Storage` representation. No CUDA dispatch is introduced.

## S2-C1: schema

`OpDesc(code, input_ids, output_ids, attrs={}, backend_hint=nullopt)` has immutable-to-callers fields, typed `OpAttrs`, and **no `execute()`, backend pointer, allocation or graph ownership**. `TensorId` is uint32; `INVALID_TENSOR_ID` is reserved. There is exactly one output. Inputs may repeat (e.g. x+x); the output must be a different logical ID, even if an alias or a future persistent-state write shares Storage. Graph ID resolution, producer uniqueness and write ordering belong to S3.

| OpCode | Inputs | Attributes |
|---|---|---|
| ADD, MUL, MATMUL | 2 | monostate |
| COPY | 2: source, predeclared destination | `CopyAttrs`: reject overlap except exact self-copy |
| MATERIALIZE | 1 | monostate |
| RESHAPE | 1 | `ReshapeAttrs{shape}` |
| VIEW | 1 | `ViewAttrs{shape,stride,offset_bytes}` |
| NARROW, SLICE | 1 | `SliceAttrs{axis,start,length,step}`; narrow step=1 |
| TRANSPOSE | 1 | `TransposeAttrs{first,second}` |
| PERMUTE | 1 | `PermuteAttrs{axes}` |

Shape/stride entries use Stage 1 bounded signed metadata. Slice length is a count, not an end index; step is positive. View byte offset is relative to the source Tensor offset. Complete permutations have unique axes, including scalar identity `{}`. Tensor-dependent ranges/layouts are validated at inference, not guessed from IDs.

`validate_schema` returns a structured `StatusCode` and nonempty error diagnostic; malformed descriptor construction throws `invalid_argument`. Semantic errors are not backend capability hints. Status categories distinguish arity/attribute/shape/dtype/layout/device/bounds/overflow/alias/unsupported/nonfinite errors. Memory allocation exceptions for C++ metadata are not disguised as semantic success.

`serialize()` produces canonical JSON in fixed field order, classic locale, with `version=1` and deterministic typed attributes. This is a diagnostic/fixture format, not a general deserializer or graph persistence API. Backend hint (`cpu:N`/`cuda:N`) is advisory and does not cause allocation, execution or implicit transfers. Later semantic extensions must be explicit, never silently introduce broadcasting.

No performance claim is made. Stage 2 acceptance is CPU-only; a GPU/Colab gate is not required by this task book.

## S2-C2: pure inference and bindings

`infer_operator(desc, TensorInputs)` receives existing Stage 1 tensors in descriptor input-ID order. It reads only checked metadata, never values, kernels or buffers; binding resolution is the future graph's responsibility. Success exposes one `OutputContract`; errors expose no output. Output contracts are declarative requirements, not another executable Tensor representation:

- `NewTensor`: caller/executor must supply contiguous storage of inferred dtype/shape/device. Canonical strides/byte sizes are checked before returning the contract, including zero axes and output overflow. Inference does **not** allocate that buffer.
- `Alias`: the optional alias is an actual Stage 1 Tensor sharing the input Storage. View/reshape/slice/transpose/permutation reuse Stage 1 validation and lifetime semantics.
- `Write`: COPY returns the predeclared destination Tensor alias, with a new logical output/state ID. Future graph write ordering is mandatory; no unrestricted cache mutation is introduced here.

`validate_output_binding` checks the caller's output metadata; aliases/writes must bind exactly the inferred Storage/offset/stride. Arithmetic (ADD/MUL/MATMUL) is FP32, same-device and contiguous, with no promotion, broadcasting or implicit materialization. ADD/MUL require exact equal shapes, including scalar/empty cases. MATMUL is rank-2 A[M,K]×B[K,N]→[M,N], including K=0 and empty outputs. INT32 is data-only at this stage (COPY/views/materialization), never arithmetic.

COPY's source/destination **ranges are explicit checked Tensor views**, not raw unchecked byte offsets. Both layouts may be strided, but shape/dtype must match. The shared Stage 1 `same_tensor_layout`/`memory_spans_overlap` helpers enforce the existing conservative span policy, including different wrappers over one address. Different Device identities are distinct address spaces. Explicit cross-device COPY may be *declared/inferred*; CPU reference execution still rejects it and CUDA copy implementation is deferred to S8.

MATERIALIZE is a graph-visible request for a **new independent canonical output**, even for a contiguous input. It is not a hidden call to `Tensor::contiguous()` (which keeps its documented already-contiguous alias behavior). Alias transforms themselves allocate zero backing buffers. Backends must either meet declared layout requirements or receive explicit MATERIALIZE/COPY nodes later.

## S2-C3: CPU correctness reference

`runtime::reference::execute(desc, inputs, caller_output)` validates inference/binding, CPU device identity and overlap before executing single-thread scalar ADD/MUL/MATMUL or the existing Stage 1 `copy_cpu` leaf (COPY/MATERIALIZE). It never allocates/replaces a backing buffer or performs hidden host transfers. Metadata-only ops bind the inferred alias and have no reference kernel. Transformer kernels are intentionally absent until S12.

ADD/MUL use FP32 values; MATMUL uses deterministic row/column/inner scalar loops with FP64 accumulation and FP32 output as a **correctness oracle**, not an optimized CPU path or the future S6 FP32 performance baseline. K=0 writes zeros, including nonempty M×N; empty outputs do not dereference null data.

Reference arithmetic rejects all overlapping input/output address spans; repeated read inputs remain legal. COPY follows its documented self-copy/range policy. MATERIALIZE rejects any intersecting source/output span. Contiguous requirements are explicit; callers must create an independent MATERIALIZE output rather than pass strided arithmetic inputs and expect a hidden copy.

All arithmetic input values are checked for NaN/Inf **before writes**. COPY/MATERIALIZE preserve arbitrary FP32 bit patterns, including nonfinite data. Nonfinite arithmetic results (e.g. finite-input overflow) return `NonFinite`; earlier result elements may already have been written. No failure output is valid, and executors must stop on first failure; no rollback/scratch-buffer allocation is promised. Metadata/device/layout/alias errors and nonfinite inputs leave output unchanged.

Tests include hand ADD/MUL vectors, exact rectangular MATMUL, zero-inner/scalar/empty semantics, INT32 strided COPY with padding, explicit output ownership/allocation assertions, error-before-write checks, overflow partial-write semantics and 100 seeded rectangular/zero matrix cases compared against independent long-double indexing.
