# Stage 6 development logs (not formal acceptance)

Dirty working-tree correctness only, local macOS CPU. No performance measurements, source-before digest or formal build identity was captured for these early iterative logs; do not retrospectively infer one from the later implementation commits. Formal clean-source identity/build/test/timing evidence is in separate unique `local/` and `baseline/` run directories.

- `c1/`: configure/build/CTest pass (24/24), before the S6-C1 implementation commit.
- `c2/`: build/CTest pass (26/26), before S6-C2 commit.
- `c3/`: build/CTest pass (27/27), before S6-C3 commit.
- `c4/python-tests.log`: actual mock-test failure caused by macOS `/var` versus canonical `/private/var` path comparison; fixed with canonical temporary results root. `python-tests-fixed.log` retains the successful rerun (20 cases).
- `c4/ctest.log`: actual benchmark self-test failure, Stage 0 oracle requires positive shapes. Preserve Stage 0; use independently known empty/zero result for zero-dimension tests.
- `c4/ctest-fixed.log`: next actual self-test failure, Stage 0 comparator requires nonempty output. Preserve it; explicitly validate empty output without invoking the positive-size comparator.
- Subsequent build/test logs retain successful reruns under separate filenames, not overwrites of failures.

All C4 mock timing data is generated only in temporary test directories and discarded there; these real logs are failure/success evidence of tool tests, not benchmark numbers. Development configure uses `cmake -S . -B build-stage6-dev -DCMAKE_BUILD_TYPE=Release -DENABLE_CUDA=OFF`; build uses `cmake --build build-stage6-dev --parallel 4`; tests use `ctest --test-dir build-stage6-dev --output-on-failure` and `python3 tests/test_cpu_tools.py`. Formal runners record exact argv, cwd, exits, environment and SHA identities automatically.
