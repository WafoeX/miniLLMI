# Interview evidence index

This is a claim-to-artifact map for discussions of the runtime. Each topic
states the design, a real trade-off, evidence, and a follow-up boundary. It is
not a list of generic answers and does not replace the acceptance reports.

## Cross-cutting rules

- The implementation keeps one path: model → graph/operators → scheduler →
  backends → tensor/storage. There is no model-owned CUDA or allocator path.
- Results are source-identified, raw, and retained even when slower or failed.
  A correctness result does not establish a performance benefit.
- The final required benefit gates were already met at RC4, so additional tuning
  stopped. The final capture is [Stage 18](../stage18_report.md); it reports
  only planner, CPU GEMM, CUDA GEMM, KV cache and INT8 payload gates.

## Evidence dossiers

| Area | Design and trade-off | Measurement / evidence | Follow-up boundary |
|---|---|---|---|
| Tensor and graph | `Tensor` is metadata over shared `Storage`; views alter metadata only. A frozen validated DAG prevents hidden topology changes. | [Tensor](../tensor.md), [operators](../operators.md), [graph](../graph.md), and their CTests; source map in [architecture](../architecture.md). | No alternate tensor representation or implicit backend copy path. |
| Allocator and planner | Dynamic last-use reclamation is the stable baseline; prepared slots trade prepare-time planning for zero intermediate execute allocations. | [Stage 5](../stage5_report.md): chain allocations 11→0 and prepared reuse capacity 512/3072, while retained tiny-graph latency ratios are slower. | Do not call this an overall latency win or zero C++-heap allocation. |
| CPU backend | The FP64-reference path is preserved; the explicitly selected FP32 CPU path and FIFO pool are separate from the baseline. | [Stage 7](../stage7_report.md) records three same-host paired geometric means and retained small-size regressions. RC4 recomputes the required CPU gate in [Stage 18](../stage18_report.md). | This is not a cross-host comparison or general model-throughput claim. |
| CUDA GEMM | Stage 0 V0 remains intact; V1 is registered and explicitly selectable through the common dispatch layer. | [Stage 9](../stage9_report.md) gives T4 correctness and paired V0/V1 samples; [Stage 10](../stage10_report.md) keeps a separate Nsight comparison. | V1 is not claimed to match cuBLAS; profiler timing is not benchmark timing. |
| Scheduler | Capability policy inserts visible CPU↔CUDA copies and leaves unsupported primitives on CPU. | [Stage 11](../stage11_report.md) records exact copy/dispatch counts and a retained slower automatic pair. | There is no scheduler speedup or automatic segmentation claim. |
| Transformer and decoder | RMSNorm/RoPE/causal softmax/SwiGLU and attention are composed from graph operators; CUDA handles supported projections only. | [Stage 12](../stage12_report.md) and [Stage 13](../stage13_report.md) cover frozen vectors/logits, mixed copies and zero planned execute allocations. | Random tiny weights validate runtime behavior, not language quality. |
| KV cache | Persistent K/V state is written through graph-visible copies; decode consumes only committed active ranges. | [Stage 14](../stage14_report.md) retains three paired context-512 runs and separate mixed diagnostic timing. | Claim only the defined CPU decode gate, not an end-to-end mixed speedup. |
| Model file and tokenizer | Strict, versioned model containers and byte tokens reject malformed metadata before allocation. | [Stage 15](../stage15_report.md), [model-file contract](../model_file.md), [tokenizer contract](../tokenizer.md). | No BPE or external-format compatibility claim. |
| INT8 weight-only | Per-output-channel INT8 + scales is loaded, then explicitly dequantized to a persistent FP32 workspace used by the ordinary MATMUL path. | [Stage 16](../stage16_report.md) and RC4 [Stage 18](../stage18_report.md) retain error, artifact, workspace and throughput records. | Compression is an artifact-payload result, not an INT8 GEMM, throughput, or resident-memory result. |
| CLI and release | Greedy CLI makes backend/quant/cache explicit; RC4 orchestrates a fixed matrix and regenerates its report from raw artifacts. | [Stage 17](../stage17_report.md), [Stage 18](../stage18_report.md), and [reproduction guide](../stage18_colab.md). | No sampling/text-quality claim; six excluded Nsight binaries require the documented companion archive. |

## Concrete failure narratives

These examples show how the record treats failure instead of retroactively
smoothing it away:

1. The prepared planner met its memory gate but was slower on the small
   predeclared graphs; dynamic remains default ([Stage 5](../stage5_report.md)).
2. An early Stage 9 T4 run retained its build/test data when its analyzer had a
   schema error; the corrected clean-source run is separately identified
   ([Stage 9](../stage9_report.md)).
3. Stage 13 preserved two T4 failures: a noncontiguous CUDA copy boundary and
   then an incomplete test-side copy-count expectation. The final capture added
   raw samples instead of pretending earlier medians were sufficient
   ([Stage 13](../stage13_report.md)).
4. A Stage 18 RC3 capture retained its raw data when a historical CTest-count
   assumption rejected a modern complete suite. RC4 reran the whole frozen
   matrix rather than combining source identities ([Stage 18](../stage18_report.md)).

## How to verify a statement

Start from a stage report, follow its source/result identifiers and raw result
path, then run the command in its Colab guide. For the final record, first use
`tools/verify_stage18_evidence.py` in the RC4 evidence checkout and verify the
companion archive checksum. The Stage 19 audit procedure is
[stage19_colab.md](../stage19_colab.md); it verifies links and a fresh
CPU/T4 build without inventing a new performance measurement.
