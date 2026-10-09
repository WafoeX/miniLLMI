# Stage 19 — documentation and interview evidence

**Status: C1–C3 implemented; C4 is awaiting the fresh T4 result described in
[stage19_colab.md](stage19_colab.md). Stage 19 is not accepted yet.** This
report deliberately makes that blocker visible instead of treating prior RC4
GPU evidence as a waiver for the final independent pass.

## C1 — architecture and operation docs

[architecture.md](architecture.md) now maps the actual source layers, ownership
and execution flow to contracts and tests. [README.md](../README.md) is updated
from its obsolete Stage-15 status to the accepted Stage 18 state, current local
CPU workflow, T4 hand-off, and explicit scope limits. The local
`tools/check_documentation.py` audit resolves repository-local Markdown links
from the root README and `docs/` tree.

There are no new performance values or behavior invented by this documentation:
the immutable RC4 release report remains the source of final claims and metrics.

## C2 — interview evidence index

[docs/interview/README.md](interview/README.md) has one concise dossier table
for tensor/graph, allocator, CPU, CUDA, scheduler, transformer/decoder, KV,
model-file/tokenizer, INT8 and CLI/release work. Each row names a concrete
design, trade-off, report/raw evidence and a follow-up boundary. It explicitly
retains the planner regression, Stage 9 analyzer failure, Stage 13 CUDA/test
failures and Stage 18 RC3 aggregation failure.

## C3 — resume claim matrix

[resume_evidence.md](resume_evidence.md) maps every proposed external bullet to
Change IDs, source commits, tests, raw/generated artifacts, exact supported
metric and disallowed overclaim. In particular, it separates component GEMM,
planner memory, KV decode and INT8 artifact compression from unmeasured overall
inference, quality, scheduler, INT8-throughput and resident-memory claims.

## C4 — required independent pass

The hand-off guide requires all of the following before acceptance:

1. clean SSH clone of the pushed Stage 19 source;
2. documentation-link audit plus a fresh Release CUDA build, full CTest,
   GPU-labelled CTest and focused CLI/decoder/KV/quant CTest capture on one T4;
3. data-only direct-child result branch containing command logs, source
   identity and SHA-256 artifact manifest; and
4. byte-identical regeneration of the accepted RC4 generated report from its
   result branch, plus verification of the external profiler archive checksum.

The T4 test result is not present in this source commit. When it is returned,
its branch/commit/run path and actual test totals will be audited against this
checklist. A failed capture remains evidence and leaves Stage 19 unaccepted;
no runtime or performance tuning is authorized by this documentation Change.
