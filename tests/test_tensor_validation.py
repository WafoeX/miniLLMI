#!/usr/bin/env python3
"""Evidence-runner failure paths; subprocesses are deliberately mocked."""
import contextlib
import importlib.util
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
spec = importlib.util.spec_from_file_location("validate_tensor", TOOLS / "validate_tensor.py")
if spec is None or spec.loader is None:
    raise RuntimeError("Cannot load tensor validation module")
validate_tensor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validate_tensor)


class ValidationTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "repo"
        self.root.mkdir()
        self.results = self.root / "results/tensor/local"
        self.source = {"commit": "test-commit", "source_digest": "test-digest", "source_dirty": False}

    def fake_run(self, argv, **kwargs):
        kwargs["stdout"].write("mock command output (not real build evidence)\n")
        if "-B" in argv:
            build = Path(argv[argv.index("-B") + 1])
            build.mkdir(parents=True)
            for name in ("CMakeCache.txt", "compile_commands.json"):
                (build / name).write_text("mock snapshot\n", encoding="utf-8")
        return subprocess.CompletedProcess(argv, 0)

    def call(self, source=None, runner=None, allow_dirty=False):
        with patch.object(validate_tensor, "inspect", return_value=source or self.source), \
                patch.object(validate_tensor, "run_process", side_effect=runner or self.fake_run), \
                contextlib.redirect_stdout(io.StringIO()):
            return validate_tensor.validate(self.root, self.results, 2, allow_dirty)

    def test_complete_manifest_and_snapshots(self):
        output = self.call()
        manifest = json.loads((output / "manifest.json").read_text())
        self.assertEqual(manifest["status"], "passed")
        self.assertEqual(manifest["source"], manifest["source_after"])
        self.assertEqual(len(manifest["commands"]), 14)
        self.assertTrue(all(command["exit_code"] == 0 for command in manifest["commands"]))
        self.assertEqual((output / "release-CMakeCache.txt").read_text(), "mock snapshot\n")
        self.assertTrue((output / "sanitizer-target.log").exists())

    def test_dirty_rejected_before_running(self):
        with self.assertRaisesRegex(RuntimeError, "Commit source first"):
            self.call(source={**self.source, "source_dirty": True})
        self.assertFalse(self.results.exists())

    def test_development_dirty_is_explicitly_labelled(self):
        output = self.call(source={**self.source, "source_dirty": True}, allow_dirty=True)
        self.assertTrue(json.loads((output / "manifest.json").read_text())["source"]["source_dirty"])

    def test_failure_preserves_log_and_exit_status(self):
        def fail(argv, **kwargs):
            if "--build" in argv:
                kwargs["stdout"].write("injected build failure\n")
                return subprocess.CompletedProcess(argv, 7)
            return self.fake_run(argv, **kwargs)
        with self.assertRaisesRegex(RuntimeError, "Command failed"):
            self.call(runner=fail)
        output, = self.results.iterdir()
        manifest = json.loads((output / "manifest.json").read_text())
        self.assertEqual(manifest["status"], "failed")
        self.assertEqual(manifest["commands"][-1]["exit_code"], 7)
        self.assertEqual((output / "release-build.log").read_text(), "injected build failure\n")
        self.assertFalse((output / "debug-configure.log").exists())

    def test_source_change_invalidates_success(self):
        with patch.object(validate_tensor, "inspect", side_effect=[self.source, {**self.source, "source_digest": "changed"}]), \
                patch.object(validate_tensor, "run_process", side_effect=self.fake_run), \
                contextlib.redirect_stdout(io.StringIO()), \
                self.assertRaisesRegex(RuntimeError, "Source provenance changed"):
            validate_tensor.validate(self.root, self.results, 2, False)
        output, = self.results.iterdir()
        self.assertEqual(json.loads((output / "manifest.json").read_text())["status"], "failed")

    def test_stage2_reuses_runner_with_operator_target(self):
        with patch.object(validate_tensor, "inspect", return_value=self.source), \
                patch.object(validate_tensor, "run_process", side_effect=self.fake_run), \
                contextlib.redirect_stdout(io.StringIO()):
            output = validate_tensor.validate(self.root, self.root / "results/operators/local", 2, False, stage=2)
        manifest = json.loads((output / "manifest.json").read_text())
        self.assertEqual(manifest["stage"], 2)
        self.assertEqual(manifest["status"], "passed")
        self.assertTrue(any("check_operator_sanitizers" in command["argv"] for command in manifest["commands"]))
        self.assertTrue(any("build-stage2-validation" in argument for command in manifest["commands"] for argument in command["argv"]))

    def test_stage3_reuses_runner_with_graph_target(self):
        with patch.object(validate_tensor, "inspect", return_value=self.source), \
                patch.object(validate_tensor, "run_process", side_effect=self.fake_run), \
                contextlib.redirect_stdout(io.StringIO()):
            output = validate_tensor.validate(self.root, self.root / "results/graph/local", 2, False, stage=3)
        manifest = json.loads((output / "manifest.json").read_text())
        self.assertEqual(manifest["stage"], 3)
        self.assertEqual(manifest["status"], "passed")
        self.assertTrue(any("check_graph_sanitizers" in command["argv"] for command in manifest["commands"]))
        self.assertTrue(any("build-stage3-validation" in argument for command in manifest["commands"] for argument in command["argv"]))
        self.assertEqual(validate_tensor.configuration(3)[0], "graph")

    def test_stage4_reuses_runner_with_arena_target(self):
        with patch.object(validate_tensor, "inspect", return_value=self.source), \
                patch.object(validate_tensor, "run_process", side_effect=self.fake_run), \
                contextlib.redirect_stdout(io.StringIO()):
            output = validate_tensor.validate(self.root, self.root / "results/allocator/local", 2, False, stage=4)
        manifest = json.loads((output / "manifest.json").read_text())
        self.assertEqual(manifest["stage"], 4)
        self.assertTrue(any("check_arena_sanitizers" in command["argv"] for command in manifest["commands"]))
        self.assertEqual(validate_tensor.configuration(4)[0], "allocator")

    def test_stage5_reuses_runner_with_planner_target(self):
        with patch.object(validate_tensor, "inspect", return_value=self.source), \
                patch.object(validate_tensor, "run_process", side_effect=self.fake_run), \
                contextlib.redirect_stdout(io.StringIO()):
            output = validate_tensor.validate(self.root, self.root / "results/planner/local", 2, False, stage=5)
        manifest = json.loads((output / "manifest.json").read_text())
        self.assertEqual(manifest["stage"], 5)
        self.assertTrue(any("check_planner_sanitizers" in command["argv"] for command in manifest["commands"]))
        self.assertEqual(validate_tensor.configuration(5)[0], "planner")

    def test_stage6_reuses_runner_with_backend_target(self):
        with patch.object(validate_tensor, "inspect", return_value=self.source), \
                patch.object(validate_tensor, "run_process", side_effect=self.fake_run), \
                contextlib.redirect_stdout(io.StringIO()):
            output = validate_tensor.validate(self.root, self.root / "results/cpu/local", 2, False, stage=6)
        manifest = json.loads((output / "manifest.json").read_text())
        self.assertEqual(manifest["stage"], 6)
        self.assertTrue(any("check_cpu_backend_sanitizers" in command["argv"] for command in manifest["commands"]))
        self.assertEqual(validate_tensor.configuration(6)[0], "cpu")

    def test_unknown_stage_rejected(self):
        with self.assertRaises(ValueError):
            validate_tensor.configuration(7)

    def test_nonpositive_jobs_rejected(self):
        with patch.object(sys, "argv", ["validate_tensor", "--jobs", "0"]), \
                contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            validate_tensor.main()
        self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
