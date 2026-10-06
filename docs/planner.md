# Stage 5 — CPU graph lifetime planner

Stage 5 is explicitly **CPU-only**: host slot planning and the existing CPU
reference executor. The frozen benchmark below is a CPU-only memory gate and
latency diagnostic; it may run on the named local host. No CUDA/server/model
performance claim follows. CUDA planning remains S8+, and T4 gates are unchanged.
S5-C4 inplace is `skipped_optional`: it is not needed for the memory gate.

## Analysis and plan

`analyze_lifetimes` requires a frozen Graph. Birth and last use are **closed**
node-order indices, not NodeIds. External birth is -1; named outputs last through
`order.size()`. First use and unique consumer counts are logical-tensor facts.
Physical-root death is the maximum death of all aliases, including strided views
and COPY state-write aliases. Persistent inputs/weights/state remain external.
Unused produced roots still live during their producer, then are released.

`plan_memory` places NewTensor roots (including produced outputs) in aligned
byte spans. CPU device and power-of-two alignment >= max_align_t are mandatory.
Deterministic aligned first fit scans the gaps between sorted occupied spans;
only roots whose death is **strictly before** the new birth may be reused.
Producer inputs and outputs coexist. Different shapes/dtypes may reuse bytes;
there is no inplace overwrite. Zero-byte roots reserve no capacity. Arithmetic
and alignment padding are overflow-checked. Capacity includes final trailing
alignment, with identical accounting for Reuse and prepare-only NoReuse.

`validate_memory_plan` independently checks exact roots, intervals, bytes,
devices, alignment, bounds, simultaneous non-overlap and derived accounting.
A full structural signature includes version, topology, attributes, shapes,
strides, offsets, external/state flags, bindings' metadata and I/O names. It
excludes data values and addresses, so compatible input contents/backings may
change. Changed graph/layout/output metadata requires explicit prepare/replan;
no collision-prone hash or dangling graph pointer is used.

## Explicit prepare and shared executor

`PlannedAllocationProvider(graph, policy)` prepares one existing Arena/shared
Storage and a fixed reservation with per-root binding metadata. `execute_planned`
is a wrapper around the **same** S3 executor, S2 inference/reference operators
and S1 Tensor/Storage. Validation runs before begin or any state writes; the
immutable private plan is never resized at execute. Outputs bind checked Tensor
metadata at their planned offsets; FP32/INT32 object lifetimes start explicitly
in reused storage. No dynamic intermediate/copy/workspace backing is needed for
the supported S2 core reference ops. Unsupported future kernels do not acquire
an untracked workspace or fallback. Metadata/container allocations still occur,
including executor/inference/result maps and signature/plan checking.
This is **not zero C++ heap allocation**, nor real-time execution.

Returned produced outputs and any copied Tensor/Storage alias pin the whole
context. The next execute returns InvalidArgument while those handles exist,
even for empty produced outputs. Destroy all returned handles to release them,
or make an explicit independent snapshot with Tensor::allocate_cpu + copy_cpu
(or MATERIALIZE in another context). contiguous() is not always a snapshot.
Another provider can execute while old results survive. Provider destruction
preserves output backing through the same shared Storage owner. External-only
outputs retain external Storage and do not pin an unrelated prepared arena.

Failures clear returned outputs and leases; completed external state writes
are not rolled back. Prepare allocation failure propagates; there is no malloc
fallback or execute-time resize. One-threaded protocol only. Direct provider
calls have the same correctly-bracketed-root-lifetime precondition as S4.
Default execute_graph without a provider remains the preserved dynamic baseline.
Backing versus logical span counters and trace follow the S4 definitions.

## Predeclared CPU-only benchmark, workload version 1

Before measurement, freeze these two workloads (same S4 graph mathematics):

- **Gate graph chain**: twelve alternating MUL/ADD nodes, FP32 [8,8], x[i]
  =0.125+i/16, scale=0.5, final output pinned. Independent scalar recurrence.
- **Diamond diagnostic**: ADD(x,x), MUL by scale, ADD(first,x), join, then
  MATMUL with identity [8,8]. Independent scalar oracle 4*x. Final output pinned.

Compare last-use **dynamic**, prepared **no_reuse**, and prepared **reuse**.
NoReuse and Reuse both align to 64 and include the same outputs and padding.
Dynamic keeps original malloc alignment (not an alignment-isolated latency
comparison). Identical external x/scale/identity bindings are accounted in all
policies. No model/decoder exists yet; that workload is explicitly unavailable.
No parameter search, optional inplace or latency gate.

Three independent runs, policy order dynamic/no_reuse/reuse, then reversed,
then original. Every process runs untimed oracle checks before and after timing,
3 warmups and 10 samples, fixed batch 20, steady_clock wall time. Execute timing
includes plan validation/begin, reference math, oracle/checksum and destruction
of returned outputs. Graph construction, preparation and CSV I/O are excluded.
Prepare is measured separately: constructor/plan validation/Storage allocation
and binding setup, 3 warmups and 10 samples, batch 1. Prepared-context destruction
is outside prepare timing; dynamic has no provider preparation (0, not a speedup
ratio). No exclusive-core/process-isolation guarantee; preserve slow runs.

Raw data distinguishes executor backing alloc/free calls, intermediate-only
backing allocations (named outputs excluded), logical requests/releases, peak
intermediate and produced payload, live output bytes, prepared capacity/backing,
external bytes, peak tensor/arena resident bytes and planned reuse/inplace counts.
Output bytes are already inside arena capacity, not added twice. Resident bytes
are a tensor/backing accounting metric, **not process RSS**, and exclude C++
metadata, aligned allocator overhead and common graph objects. Dynamic free
counts at executor return exclude final output destruction; an additional
output-destruction count reports that boundary separately.

Required gate: chain dynamic intermediate allocations >0 -> planned 0;
Reuse capacity <=80% of NoReuse capacity, both prepared identically. Do not
compare reserved arena bytes to dynamic peak as though it were a memory win.
Publish both workloads and latency regressions. Ratios are medians of three
paired median-latency ratios; no pooling workloads and no latency pass claim.

```bash
python3 tools/validate_planner.py
python3 tools/run_planner_benchmark.py
python3 tools/analyze_planner.py results/planner/cpu/<actual-run-id>
# Development correctness only:
# python3 tools/validate_planner.py --allow-dirty
```

Source/build/binary identity and raw samples are archived in unique run paths.
Generated artifacts are recomputable and cannot overwrite different existing
bytes. Actual acceptance results belong in stage5_report.md after measurement.
