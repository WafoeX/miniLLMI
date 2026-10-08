# Versioned tiny-model container — Stage 15 C1/C2

The v1 container is a little-endian, custom `MLLMRTF\0` model artifact. It is
only for the project's bias-free FP32 decoder; it makes no GGUF, mmap, or
external-model compatibility claim.

## Layout

All integers are unsigned little-endian except configuration dimensions, which
are checked signed 64-bit little-endian integers. Floating-point configuration
values are IEEE-754 binary64 bit patterns in little-endian order.

1. 8-byte magic: `MLLMRTF\0`.
2. `u32` format version (`1`).
3. `u32` vocabulary version (`1`, the byte vocabulary below).
4. `u32` tensor count.
5. Decoder configuration: nine `i64` dimensions (`batch`, `layers`, `hidden`,
   `heads`, `kv_heads`, `head_dim`, `ffn`, `vocab`, `max_seq`), two `f64`
   constants (`rms_epsilon`, `rope_base`), one byte `bias`, then seven zero
   reserved bytes.
6. One metadata record per tensor: `u32 name_bytes`, name bytes, `u32 dtype`,
   `u32 rank`, `u64 absolute_payload_offset`, `u64 payload_bytes`, then `rank`
   `i64` dimensions.
7. Concatenated tensor payloads.

V1 admits dtype tag `1` (contiguous CPU FP32) only. The writer emits canonical
`parameter_specs(config)` order and contiguous payloads. The committed test
artifact is `tests/fixtures/model-v1/tiny-model-v1.mllm`:

- 21 tensors / 461,056 payload bytes;
- SHA-256 `f91f89491735d324193295bb489f16ff6301186d957e04029c7a1402be729ee8`;
- generated deterministically from the frozen v1 test weights with
  `convert_tiny_model`.

## Loader boundary

`inspect_model_file` performs all validation before any runtime tensor
allocation. It validates magic/version/reserved bytes, finite/valid decoder
configuration, byte-vocabulary compatibility, tensor count/names/shapes/dtype,
rank and dimension limits, checked byte arithmetic, exact file bounds,
non-overlap, no payload gaps/trailing bytes, and a caller-configurable total
payload allocation budget (default 64 MiB). Duplicate, unknown, and missing
names are rejected.

`load_model_file` then creates normal CPU `runtime::Tensor` instances and
returns a `LoadedModel` containing the normal `ParameterTable`; no second
storage/runtime route exists. Tensor shared storage survives graph construction
and prepared execution after the `LoadedModel` object itself is destroyed.

`convert_tiny_model <legacy-weights.bin> <model.mllm>` is an offline fixture
converter for the frozen Stage 12/13 raw weight fixture. It rejects truncated or
trailing input. It is not an inference CLI.
