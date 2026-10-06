# Stage 5 acceptance — CPU graph lifetime planner

**Status:** required S5-C1/C2/C3/C5 implemented and accepted for the explicitly
CPU-only scope. S5-C4 is `skipped_optional` (inplace disabled, not needed).
Default execution remains S3 dynamic. No CUDA planner, model/decoder, scheduler,
zero-C++-heap, or latency speedup is claimed. Stop optimizing this memory stage;
Stage 6 is not started by this delivery.

## Tested source and evidence

Final tested source: `4473e3002b93374222974043c73190d904a833dc`;
SHA-256 code-tree digest:
`b82137fce9b1f29ea23aedbc8f552c878df7689d0c395c9d787c0b7a0acc7802`.
Both formal runners recorded clean source unchanged before/after. Testing used a
clean detached worktree, excluding the original untracked, untouched AGENTS.md.
Later report/results commits are separate from the tested source; they do not
change implementation. Recorded absolute build/command paths remain original;
archive replay reads relative raw paths without rewriting that provenance.

- [Final correctness manifest](../results/planner/local/20261006T024055095509Z-63246/manifest.json)
- [Final paired benchmark manifest](../results/planner/cpu/20261006T024144778921Z-65641/manifest.json)
- [Generated memory/latency summary](../results/planner/cpu/20261006T024144778921Z-65641/summary.md)
- [Full generated statistics/counters](../results/planner/cpu/20261006T024144778921Z-65641/planner.csv)
- [Machine-readable analysis](../results/planner/cpu/20261006T024144778921Z-65641/analysis.json)
- [Supplementary build/cache/hooks/replay audit](../results/planner/verification/audit.json)
- [Contracts/predeclaration](planner.md)

Host: `Wafoe-MacBook.local`, macOS 26.7 / arm64,
AppleClang 21.0.0.21000334, C++17, CPU-only. Release is used for measurements.
No server/GPU benchmark was required or performed for this explicitly host-only
Change. Future CUDA-backed gates still require the T4 server.

## Change-by-Change acceptance

| Change | Source commit | Acceptance and regression evidence |
|---|---|---|
| S5-C1 | `924ff53` | metadata-only birth/first/last-use/unique consumers, external/output flags; physical roots extended by aliases, chain/diamond/multiple outputs |
| S5-C2 | `6799471` | deterministic aligned gap first fit, closed intervals, no producer-input overwrite, byte-based cross-shape/dtype reuse, independent validity/accounting checker |
| S5-C3 | `281fde2` | explicit prepared provider through existing executor/Tensor/Storage/reference; fixed offsets, no execute backing allocation, shape/topology staleness rejection, output leases, snapshot, failure/state/zero behavior |
| S5-C4 | skipped_optional | no inplace; stable out-of-place path sufficient |
| S5-C5 | `91b8da2` + `4473e30` | frozen three-policy paired experiment, separate prepare timing, raw source/build/binary identity, negative analyzer/runner cases and relocated-byte-identical replay |

Fresh final Release, Debug and ASan+UBSan builds each pass **23/23 CTest**.
Sanitizer planner target passes **4/4**. Fresh production Release builds with
BUILD_TESTING=OFF pass the untimed workload oracle; production compile commands
and `nm` on both runtime archives contain no backing-allocation test hooks.
The benchmark's separate fresh test Release also passes 23/23 before measuring.
All six configure/build pairs have no compiler/CMake warnings or errors; all
12 archived CMakeCache/compile_commands snapshots match their actual builds.

Planner tests include chains, branches/diamonds, multiple outputs, alias-pinned
bases, non-contiguous transpose/materialization, malformed/out-of-bounds plans,
closed-node conflicts, varied sizes/alignments, empty/external-only graphs,
CPU-only rejection of CUDA metadata, prepare OOM, kernel failure cleanup with
non-rollback of completed state writes, held/copied output guards and provider
lifetime. Actual Storage hooks verify zero backing allocation over repeated
executes, even with fail-next-allocation armed. Reused FP32→INT32 bytes start
C++17 typed lifetimes explicitly. Six seeds × 100 randomized DAGs compare both
prepared policies against dynamic outputs (1,200 prepared executions per suite).
All existing Tensor/operator/fixture/graph/arena/Stage 0 CPU tests remain enabled.
The 17 new analyzer/runner tests use mock-only temporary data, never raw benchmark
artifacts. The shared validation runner now has 11 regression cases.

macOS ASan does not supply LSan: no Linux/LSan or CUDA-enabled build/run claim.
Changed source files were actively checked; a sibling script import is an editor
fallback false positive, verified by actual CLI imports, tests and both formal
runs, not silenced with inline ignores. Raw generated Markdown/EOF whitespace
advisories are retained; metrics/logs are not hand-edited for formatting.

## Required memory gate and honest latency result

The chain was predeclared **before** measurement, not chosen afterwards.
Both prepared policies use alignment 64, identical padding accounting, graph,
external inputs and pinned final outputs. Dynamic uses last-use reclamation.

| Workload / metric | Dynamic | Prepared no-reuse | Prepared reuse |
|---|---:|---:|---:|
| chain execute intermediate backing allocations | 11 | 0 | 0 |
| chain all execute backing allocations / frees at return | 12 / 11 | 0 / 0 | 0 / 0 |
| chain capacity bytes | not prepared | 3072 | 512 |
| chain peak intermediate / produced payload bytes | 512 / 512 | 512 / 512 | 512 / 512 |
| chain external / peak resident backing bytes | 512 / 1024 | 512 / 3584 | 512 / 1024 |
| diamond execute intermediate backing allocations | 4 | 0 | 0 |
| diamond all execute backing allocations / frees at return | 5 / 4 | 0 / 0 | 0 / 0 |
| diamond capacity bytes | not prepared | 1280 | 768 |
| diamond peak intermediate / produced payload bytes | 768 / 768 | 768 / 768 | 768 / 768 |
| diamond external / peak resident backing bytes | 768 / 1536 | 768 / 2048 | 768 / 1536 |

Gate: chain **11→0**, capacity ratio **512/3072 = 1/6** (83.33% reduction,
well below required 80% ratio). Diamond capacity ratio 0.6 (40% reduction),
reported separately. Reuse counts chain 10, diamond 2; inplace 0. Logical
requests/releases are unchanged (12/11 and 5/4). Prepared policies allocate one
backing before execute. Output payload is 256 bytes and already inside capacity;
dynamic frees the last output on returned-result destruction, an additional
backing free separately recorded. Resident bytes are tensor/backing accounting,
not RSS; metadata and aligned allocator overhead are excluded. The reusable
arena **does not beat** dynamic peak-live payload: it equals it on these aligned
graphs. The capacity win is only versus the prepared no-reuse baseline.

Final three dynamic/reuse paired median-latency ratios:

- chain: 0.908989, 0.579266, 0.561675; median **0.579266×**.
- diamond: 0.719880, 0.621505, 0.537079; median **0.621505×**.

Thus planned execution is **slower** on both tiny graphs, in all retained final
pairs. No latency gate applies; the required memory gate passes. Full structural
plan checking/metadata work is included in execute, not moved outside timing to
inflate a speedup. No optimization was attempted to conceal this regression;
dynamic stays default. Contention/variance are uncontrolled and disclosed.
Raw per-policy prepare medians (constructor only), execute samples and min/max/
mean/stddev are in planner.csv/analysis.json. There are 18 execute CSVs and 18
prepare CSVs, 180 samples in each category. Three warmups, execute batch 20,
prepare batch 1, alternating policy order; slow samples are never removed.
No decoder graph is available until S13; no tiny-model/LLM throughput claim.

## Preserved failure and earlier-source evidence

[Development logs](../results/planner/development/README.md) preserve the first
C5 failure: the shared runner's old test expected Stage 5 to be unsupported.
The test now verifies Stage 5 and rejects Stage 6; no semantic error was hidden.
An initial formal run on `91b8da2` passed CPU tests and the memory gate:
[correctness](../results/planner/local/20261006T023831574039Z-59428/manifest.json),
[benchmark](../results/planner/cpu/20261006T023924277732Z-61772/manifest.json).
Its slower latency ratios (chain 0.530507× / diamond 0.624025×) remain archived,
not pooled with final-source results or discarded as invalid environment runs.
The original analyzer failed after that archive was copied to another checkout;
[failed](../results/planner/development/relocation/failed-analyze.log) and
[corrected](../results/planner/development/relocation/fixed-analyze.log) replay
logs are preserved. Fix `4473e30` validates original recorded command paths while
reading local relative raw files, adds a relocation regression, and reruns all
fresh correctness/build/benchmark steps on the final clean source. Both original
and relocated final analyses reproduce all three derived files byte-for-byte.

Stage 0 CUDA kernels/wrappers, original Tensor/Storage/copy/reference/topology,
fixtures and all historical result paths are unchanged. Only the documented
provider validation seam extends the common executor. Nothing starts S6–S19.

## Reproduce

```bash
# Clean checkout at the tested source (leave untracked local instructions alone).
python3 tools/validate_planner.py
python3 tools/run_planner_benchmark.py
# New result run each time; never overwrite existing raw data.
python3 tools/analyze_planner.py results/planner/cpu/20261006T024144778921Z-65641
```

Acceptance evidence is CPU-only and bound to the tested source above. Reports
and raw results are committed independently afterwards, retaining that identity.
