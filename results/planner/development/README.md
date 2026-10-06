# Stage 5 development logs (not acceptance/performance evidence)

These incremental dirty-build logs are preserved, including failures. They are
not clean-build performance data and are not the authoritative Stage 5 gate.
Early incremental runs did not capture a complete source digest before execution;
that provenance limitation is explicit and is not retroactively invented.

- c1: source implementation committed immediately afterwards as 924ff53; Release
  incremental build and all 20 tests passed.
- c2: source implementation committed immediately afterwards as 6799471; Release
  incremental build and all 20 tests passed.
- c3 / c3-final: integration work committed afterwards as 281fde2; Release and
  fresh Debug ASan/UBSan development builds passed all 21 tests.
- c5: dirty C5 work on 281fde2 failed tensor_validation's old unknown-stage test:
  it still expected Stage 5 to be unsupported. Raw failed 23-suite log retained.
  The test was extended for the new Stage 5 runner; unknown stage is now Stage 6.
  This was a validation-configuration regression, not a suppressed planner error.
- c5-final / c5-freeze: dirty source before/after snapshots are captured; all 23 tests pass.
- relocation: analyzer on source 91b8da2 failed to replay copied artifacts because
  it compared original absolute raw paths to the new checkout path. Failed log
  retained. Fix 4473e30 and a mock-only relocation regression permit identical
  reanalysis without rewriting argv or raw bytes. All fresh formal runs were then
  repeated on the final clean source; earlier successful-source data remains.

All logs retain the original output. Formal fresh clean-source CPU correctness
and independently paired Release measurements are in sibling local/ and cpu/
run directories, with exact source identity, commands, caches and raw samples.
