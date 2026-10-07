# Preserved Stage 11 C2 development failures

These are CPU-only **dirty-development** test logs, not formal acceptance or
performance evidence. They ran in the original macOS checkout after C1 commit
`cdadc25`, while the C2 code/tests were still being edited. Pre-test source
digests were not captured; none is reconstructed retroactively.

- `c2-test.log`: 31/32 passed; the newly added dispatch trace events accidentally
  changed the legacy trace golden order. Fixed by emitting the new events only
  for explicit scheduled execution, preserving unscheduled trace format.
- `c2-fixed-tests.log`: 32/33 passed; runner protocol mocks intercepted the
  Python 3.14 `platform.platform()` subprocess (`uname -p`) and rejected its
  extra keywords. Fixed by explicitly mocking host/platform/machine in the
  protocol test, without changing production host capture.

Both fixes are in C2 source commit `79df7d277415249adf6113096017462776c67d65`.
The later fresh, clean-source Release/Debug/ASan+UBSan evidence is in
`results/scheduler/stage11-c2/20261007T074011978256Z-55180/` (each 33/33).
These preserved failures do not certify T4 or satisfy dependent C4 acceptance.
