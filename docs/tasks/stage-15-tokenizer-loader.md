# Stage 15 — Tokenizer and model file

**Goal:** make tiny-model inference use a persistent, versioned artifact. **Prerequisite:** S13; KV optional. **Gate:** saved model and deterministic tokenizer round-trip into identical logits.

## S15-C1 — Versioned model-file format
- **Goal/type:** Define magic/version/config/tensor metadata(dtype,shape,offset)/payload and strict bounds checks. *Functional + Correctness.*
- **Depends/files:** S1/S13; `model_file.hpp/.cpp`, tests/docs.
- **Scope/not:** little-endian versioned custom format only; checked file size/rank/dimensions/offset+length/product and a configured maximum allocation budget. Reject overlapping payloads, duplicate/missing parameter names and incompatible model shapes; no GGUF/llama claim.
- **Verify/baseline/metrics:** malformed/truncated/overlap/version tests; N/A.
- **Accept/commit/risk:** loader never trusts file offsets unchecked. `feat(loader): define versioned model container`. Risk: format changes; version explicitly.

## S15-C2 — Model writer/loader and binding
- **Goal/type:** Offline writer fixture plus loader that creates runtime tensors and maps names/config. *Integration + Correctness.*
- **Depends/files:** C1; tools/loader/tests.
- **Scope/not:** no mmap required, no quantized payload yet.
- **Verify/baseline/metrics:** write→load checksum/logits equality; load time/bytes optional.
- **Accept/commit/risk:** model storage lifetime survives graph prepare. `feat(loader): load tiny decoder model tensors`. Risk: endian/alignment.

## S15-C3 — Simple deterministic tokenizer
- **Goal/type:** tokenizer interface with fixed vocabulary encode/decode, unknown and special token policy. *Functional + Correctness.*
- **Depends/files:** S13; `tokenizer/`, tests.
- **Scope/not:** initial byte tokenizer IDs 0..255 plus BOS=256/EOS=257, vocab=258 matching S13; encode bytes deterministically and document BOS/EOS insertion and decoding skip policy. Byte round-trip includes arbitrary UTF-8 bytes; invalid generated UTF-8 uses documented display replacement, never changes token IDs. Not BPE.
- **Verify/baseline/metrics:** byte encode/decode fixtures (ASCII, multibyte UTF-8, invalid bytes), special IDs and vocabulary/config mismatch; N/A.
- **Accept/commit/risk:** IDs are versioned with vocabulary artifact. `feat(tokenizer): add deterministic vocabulary tokenizer`. Risk: promising general-language quality.

## S15-C4 — Optional BPE compatibility spike
- **Goal/type:** investigate/implement BPE only after simple path works. *Functional + Experimental.*
- **Depends/files:** C3; tokenizer docs/tests.
- **Scope/not:** optional; no blocker for CLI.
- **Verify/baseline/metrics:** published small merge fixtures; baseline simple tokenizer function only.
- **Accept/commit/risk:** retain simple tokenizer fallback. `feat(tokenizer): add optional BPE tokenizer`. Risk: Unicode/pretokenization scope creep.
