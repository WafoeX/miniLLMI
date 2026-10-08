# Versioned tiny-model container — Stage 15 / Stage 16

`MLLMRTF\0` is a little-endian custom artifact for this project's bias-free
small decoder. It makes no GGUF, mmap, or external-model compatibility claim.

## Version 1 (Stage 15, retained)

V1 is FP32-only and remains byte-compatible with the committed
`tests/fixtures/model-v1/tiny-model-v1.mllm` fixture:

- 21 tensors / 461,056 payload bytes;
- SHA-256 `f91f89491735d324193295bb489f16ff6301186d957e04029c7a1402be729ee8`;
- generated from frozen weights with `convert_tiny_model`.

Its layout is: magic, `u32 version=1`, `u32 vocabulary_version=1`, `u32`
tensor count, nine `i64` configuration dimensions, two IEEE-754 `f64`
constants, one-byte bias plus seven zero reserved bytes, then one record per
tensor (`u32 name_bytes`, name, `u32 dtype=1`, `u32 rank`, `u64` absolute
payload offset, `u64` payload bytes, and rank `i64` dimensions), followed by
contiguous FP32 payloads.

## Version 2 (Stage 16 C2)

V2 preserves the header/configuration and canonical 21 parameter names. Every
base tensor record has the V1 fields followed by:

1. `u32 quantization_version` (`0` for retained FP32 tensors; `1` for
   Stage-16 INT8 tensors);
2. for `quantization_version=1`: `i64 output_axis` (must be `1`), `u32`
   scale dtype (must be FP32), `u32` scale rank, `u64` scale offset, `u64`
   scale bytes, and scale dimensions;
3. `u64` FNV-1a checksum over the value payload followed by its scale payload
   when present.

Only `layers.<n>.{q,k,v,o,gate,up,down}_proj` and `lm_head` are eligible. Their
layout stays `W[in,out]`; a contiguous FP32 scale vector `[out]` follows the
contiguous INT8 value payload. Embeddings and norm tensors remain FP32. Each
all-zero output column stores scale `1` and zero values.

`inspect_model_file` validates all metadata before allocating tensors: supported
version/vocabulary/configuration, canonical names/shapes, checked byte
arithmetic, exact contiguous payload coverage, allocation budget, INT8
eligibility/output axis, positive finite FP32 scales, and V2 checksums. V1
continues to reject non-FP32 payloads.

## Loader boundary and prepare dequantization

`load_model_file` returns the normal FP32 `ParameterTable`. For a V2 INT8
weight it also retains the normal runtime INT8/scales `Tensor` storage in
`LoadedModel::prepare_dequant_weights`, allocates an explicit persistent FP32
`prepare_dequant` tensor, and binds that tensor into the unchanged graph,
scheduler, and backend MATMUL path. This is **not** native INT8 GEMM and it
does not imply lower resident memory: the INT8 values, scales, and FP32
workspace coexist while the loaded model is live.

`convert_tiny_model <legacy-weights.bin> <model.mllm>` writes V1. Append
`--int8` to deterministically write V2 with all eligible weights quantized.
