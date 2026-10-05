# Stage 4 — CPU arena and allocation-policy contract

Scope is explicitly **CPU-only**: independently tested host arena, online
first-fit reuse, and a supplied allocation-provider seam in the S3 executor.
There is no graph lifetime planner, gallocr, inplace policy, CUDA pool or model.
The default executor remains the last-use dynamic allocation baseline. Arena
policy is opt-in; latency improvement is not a Stage 4 acceptance gate.

## Arena / Block / Storage

`Arena(capacity, alignment=64, policy=FirstFit)` owns one aligned CPU Storage.
`Storage::allocate_cpu_aligned` uses C++17 aligned new/delete with matching
alignment, the existing Storage lifetime/control-block guards and test hooks.
Existing `allocate_cpu` behavior is unchanged. Backing alignment is a power of
two at least `alignof(max_align_t)`; request alignment may be any power of two
from 1 through that backing alignment. No resize or malloc fallback is allowed.

`allocate(bytes, alignment=64)` returns an immutable-to-callers Block containing
byte offset, exact payload size, alignment and a monotonic ticket. Zero bytes
returns an arena-origin zero handle, offset/ticket 0, reserving no range. Zero
handles may be freed repeatedly while their originating arena exists. Capacity
0 owns no nonzero backing buffer. Invalid alignment throws invalid_argument;
checked size/offset or ticket overflow throws overflow_error; insufficient
contiguous space throws ArenaExhausted (a bad_alloc subtype). Failed allocation
preserves active ranges/payload accounting, but counts a request and failure.

`free(Block)` validates arena identity, ticket and immutable metadata before
mutation. Foreign/expired/default/double/stale handles fail explicitly. A freed
address reused by a new ticket never revives an old Block. One-threaded use only;
no compaction, hidden copies or best-fit policy. Sorted aligned first fit
preserves prefix/suffix gaps and coalesces adjacent free ranges on both sides.
Allocation reserves enough free-list vector capacity for insertion-before-merge,
so a valid free needs no heap allocation. A fully freed nonempty FirstFit arena
has exactly one free range. `BumpNoReuse` is retained as the C1 baseline; it
accounts frees but does not recycle addresses until an explicit rewind.

`rewind()` requires no live blocks and resets the cursor/high-water marker,
never cumulative counters or tickets. `validate()` independently checks range
bounds/alignment, active non-overlap, sorted/coalesced free ranges, complete
capacity partition and exact live/peak accounting. It is diagnostic and may
allocate metadata; it is not called inside measured execution.

Counters distinguish requests (including zero and failed attempts), successful
nonzero span allocations, free attempts, successful nonzero frees, allocation
failures, live block count, payload live/peak bytes and allocations below the
current address high-water mark. The latter `reused_allocations` includes
recovered alignment gaps, not an exact previously-written-byte intersection
metric. Padding is free-list space, not payload; capacity/backing reservation is
reported separately. Metadata/container heap activity is not backing allocation.
Raw Block callers must not free a range while using it. Storage snapshots keep
backing memory alive, not an independently reference-counted span lease.

## Executor provider and output lifetime

`execute_graph(graph, trace=nullptr, provider=nullptr)` uses the S3 control flow,
operators, Tensor and shared Storage. Null provider selects DynamicAllocationProvider,
whose allocate delegates to the existing Tensor::allocate_cpu. Supplied
ArenaAllocationProvider preallocates backing **before execute**, then allocates
spans online when nodes need NewTensor outputs. All NewTensor requests, including
outputs, pass through the provider. Aliases and state writes allocate neither
new Storage nor individual spans; no kernels/API bypass this seam.

Provider begin/end bracket a one-threaded execution session. The executor retains
producer inputs while allocating/executing their output, releases physical roots
only after the final unpinned alias handle's last use, and pins named outputs.
Arena-owned tensors all share **one arena-owning Storage**, never one Storage
with a deleter per view/span. FP32/INT32 object lifetimes are started explicitly
in reused spans; Tensor dtype/checked metadata/CPU access are still authoritative.
Graph-frozen alias range checks remain relative to each logical root's required
capacity. Direct caller-created views must stay within their initialized leased
root; the broader Storage capacity is not evidence that padding/other spans are
initialized tensor data.

Returned outputs/any copied Tensor aliases retain this same Storage owner.
Another begin on that provider is rejected as `InvalidArgument` while any such
Storage reference is outstanding, including Graph inputs bound from prior
outputs. Once all handles die, begin retires pinned block records, rewinds and
allows reuse. Holding even an empty output pins its context. Use another provider
or explicitly MATERIALIZE a snapshot to keep old outputs while executing again.
Destroying a provider does not destroy backing still owned by output Tensors;
final shared Storage destruction frees it exactly once. No per-view release
returns live memory to the free list. Provider protocol calls are for the
executor (or an equivalent correctly bracketed root-lifetime owner), not a way
to manually release an output view.

Capacity/fragmentation failure returns ResourceExhausted with no fallback;
failed execution clears all returned outputs and releases every leased root.
Existing reference failure semantics and non-rollback of completed external
state writes are unchanged. Provider busy failure does not modify old outputs.

### Counters and trace

S3 allocations/frees remain **nonzero backing calls inside execute**. Dynamic
has one per successful NewTensor request; arena has zero, plus one prepared
backing for nonzero capacity outside that boundary. New allocation_requests/
releases count successful nonzero physical-root allocations/releases, not zero
outputs, failed attempts or per-alias frees. ArenaCounts additionally records
failed allocation attempts. allocated_bytes is cumulative requested payload;
live_bytes/peak_live_bytes include produced outputs for both policies.
arena_capacity_bytes reports reservation separately. External inputs/weights/
state and metadata heap allocations are excluded from execute payload counters.
They are not process RSS or total resident memory.

Allocate/Free trace events retain their backing-call meaning. BlockAllocate/
BlockFree describe arena span requests/releases. Logical Release events remain
per TensorId. Traced layouts include actual arena offsets/capacity and logical
base IDs, no raw pointer or data. Traces retain no Storage and are bounded/null
by the existing S3 interface. Benchmarks disable traces. Existing dynamic golden
trace/order/output/lifetime tests remain regressions.

## Explicit CPU-only benchmark, workload version 1

`bench_allocator --self-test` is an untimed correctness check, permitted in
Debug/dirty/test builds. Real timing requires a clean Release binary with
BUILD_TESTING=OFF and matching source identity; runner checks its embedded probe
commit/digest/build/compiler/backend/testing flags against the current checkout.
Fresh configure/build logs, caches, binary SHA and raw samples are archived.

Frozen workloads (all FP32 graph math is unchanged S2 correctness reference):

- Synthetic: 8 repetitions of four live spans (64/128/192/256 bytes), free slots
  1/3, allocate 128/256 replacements, free all. Live-byte sentinels are checked.
- Chain: twelve alternating MUL/ADD nodes, tensors [8,8]; x[i]=0.125+i/16,
  scale=0.5; independent scalar recurrence validates every output.
- Diamond: ADD/MUL/ADD/join then MATMUL with identity [8,8]; independent expected
  output is 4*x. Inputs/external backing are separately accounted.

Both synthetic policies use alignment 64. Graph dynamic deliberately preserves
S3's default CPU malloc alignment; arena uses 64. This is a disclosed policy
comparison, **not an alignment-isolated speedup experiment**. Arena capacity is
fixed at 1024 bytes for every workload, not tuned from measured outcomes. Backing
preparation is excluded; the identical timed boundary includes the whole
execute/allocation workload, oracle/checksum and returned-output destruction.
On the arena graph path, retirement of previous no-longer-held output leases is
included in the next begin. No retained output is overwritten.

Three independent paired runs, order dynamic/arena then arena/dynamic then
dynamic/arena, 3 warmups, 10 raw samples, 20 calls per sample, steady_clock wall
time. Prepared graphs/input generation, arena construction and CSV I/O are
outside timing. No guaranteed CPU exclusivity/process isolation; report variance
and keep slower valid runs. The benchmark is a small host allocation diagnostic,
not a model/gallocr/LLM throughput result. There is no representative decoder yet.

Per-run raw CSV records call-normalized milliseconds, per-call successful span
requests/releases, execute backing calls, peak payload, output live payload,
high-water reuse count, prepared arena capacity/backings, external backing bytes
and peak tensor/arena backing bytes. The latter excludes allocator metadata,
alignment implementation overhead and process RSS; arena capacity already
contains outputs, which must not be added twice. Summary CSV attaches tested
source/host/compiler/pair/provenance/raw paths and recomputes statistics.

`analyze_allocator.py` validates all 180 samples and fixed counters/checksums,
source/binary/configuration/command/order identity. It derives 18 policy/workload/
pair rows, min/max/mean/median/stddev and three paired median ratios per workload;
reports the median of those three ratios, without pooling different workloads.
Generated summaries cannot overwrite differing artifacts. There is no ≥1.05x
latency gate for Stage 4; slower arena results remain valid diagnostics. Stage 5
planner/no-reuse-prepared-capacity gates are **not** satisfied/claimed by this work.

```bash
# Clean committed checkout; each command creates a unique result run.
python3 tools/validate_arena.py              # CPU correctness, four fresh builds
python3 tools/run_allocator_benchmark.py     # explicit CPU-only Release diagnostic
python3 tools/analyze_allocator.py results/allocator/cpu/<actual-run-id>
# Uncommitted development correctness only; never labelled performance:
# python3 tools/validate_arena.py --allow-dirty
```

The shared correctness runner includes Release/Debug/ASan+UBSan/production,
18,000 randomized operations, provider/value/alias/OOM/type tests, benchmark
workload oracles, negative artifact/runner tests and all existing CPU regressions.
GPU is not needed; Linux/LSan and CUDA-enabled/T4 environments remain separate
unverified targets here. Source commits and acceptance evidence are separate;
see the Stage 4 report for actual source identity/results.
