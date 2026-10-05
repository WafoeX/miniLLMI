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

## S2-C4: Transformer descriptors (no execution kernels)

All floating tensors are FP32, contiguous and same-device unless an explicit alias/copy/materialization node says otherwise. Inference is metadata-only; it cannot certify input values or initialized cache contents. The following value checks are mandatory for S12 execution, not claims of existing S2 kernels.

| Op | Inputs → output | Typed attributes / contract |
|---|---|---|
| RMSNORM | x[...,C], scale[C] → same shape as x, rank≥1, C>0 | `NormAttrs{epsilon}`; finite positive FP32-representable epsilon; last-axis `x / sqrt(mean(x²)+epsilon) * scale` |
| SOFTMAX | scores[Q,K] → [Q,K] | `SoftmaxAttrs{causal,query_position,key_position,max_positions}`; row-wise max-subtracted softmax, no batched/broadcast axes |
| ROPE | x[T,D] or x[T,H,D] → same shape | `RopeAttrs{position,base,max_positions}`; positive even D, H>0, finite FP32 base>1 |
| EMBEDDING | IDs[T] INT32, table[V,C] FP32 → [T,C] FP32 | monostate; V,C>0; runtime checks every ID in [0,V) before writes, never implicit cast or host access in inference |
| SWIGLU | gate and up, exact equal shapes → same shape | monostate; `SiLU(gate)*up`, stable sigmoid, no hidden broadcast |
| ATTENTION | Q[Q,H,D], K/V[K,H,D] → [Q,H,D] | `AttentionAttrs{heads,head_dim,causal,query_position,key_position,max_positions}`; H/D match, positive even D; **graph composition**, not mandatory opaque backend kernel |

Absolute position intervals are bounded by the explicit maximum; empty intervals may start at one-past maximum. Causal validity is `key_position+k <= query_position+q`, including decode Q=1 at absolute position p seeing keys 0..p. All-masked softmax rows are zero output; K=0 has no elements. Nonfinite **unmasked** scores are errors, masked scores are ignored. Softmax may not accidentally normalize masked entries or produce NaNs from an all-masked row.

RoPE uses **interleaved** adjacent pairs, not split-half rotation: for i=0..D/2-1, `theta=(position+t)/base^(2*i/D)`, `(x[2i],x[2i+1]) -> (x[2i]*cos(theta)-x[2i+1]*sin(theta), x[2i]*sin(theta)+x[2i+1]*cos(theta))`. Position is explicit during decode, never inferred from query length. Loader boundaries must translate any other weight/rotary convention rather than change this contract.

### Frozen tiny-decoder contract

`runtime::tiny_model` constants are version 1: batch=1, bias-free FP32 MHA, 2 layers, hidden=64, heads=4, head_dim=16, SwiGLU intermediate=128, vocab=258, max_seq=1088, RMSNorm epsilon=1e-5, RoPE base=10000/interleaved. Byte IDs 0..255, BOS=256, EOS=257. Every projection weight is row-major **W[in,out]**, including Q/K/V/O (64×64), gate/up (64×128), down (128×64), LM head (64×258); quantization output-channel axis is **1**. Embedding table is [258,64], norm scales are [64]. No bias, model execution, tokenizer, loader or language-quality claim is introduced.

### Required ordinary-2D lowering and checked state writes

Projection outputs reshape from [T,64] to token-major [T,4,16]. Each head uses an explicit NARROW/SLICE on axis 1, giving [T,1,16]. When not contiguous, MATERIALIZE to caller/planner storage **before** RESHAPE to [T,16]. K's [K,16] transpose is [16,K], again explicitly materialized for the contiguous MATMUL contract. Per-head QKᵀ yields [Q,K]; scaling by 1/sqrt(D) uses MUL with a **declared same-shaped scale tensor**, not undocumented scalar broadcasting. SOFTMAX uses the absolute offsets above, then ordinary MATMUL P[Q,K]×V[K,16] produces head output. Assemble heads with checked COPY writes into explicit output head ranges; no required batched matmul, hidden allocator, model-level memcpy or fused attention kernel.

Persistent K/V capacity is [1088,4,16]; active initialized prefix is an explicit narrow on axis 0. Append only into checked `[position,position+new_tokens)` destination views, never read an uninitialized suffix. COPY takes projected source and destination range as two inputs and yields a **new logical state/version ID** referring to destination Storage. S3/S14 must chain reads/writes through these IDs and reject unordered intersecting spans (including conservative head-span overlap); an old state ID may not be used to bypass a write dependency. Inference checks range/layout/type bounds but does not supply a cache policy or prove initialization. The tests validate this lowering using manual caller-provided Stage 1 tensors and existing core copies, not an implemented graph/model.
