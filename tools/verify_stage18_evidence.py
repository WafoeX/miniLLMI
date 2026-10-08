#!/usr/bin/env python3
"""Validate Stage 18 RC evidence and generate its final report from raw artifacts.

This tool deliberately consumes the existing stage-specific raw-evidence analyzers.
It never accepts historical evidence: every supplied artifact must name the one
release-candidate source identity given with --commit/--source-digest.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
sys.path.insert(0, str(TOOLS))


def load_module(name):
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CPU = load_module("analyze_cpu_parallel")
PLANNER = load_module("analyze_planner")
CUDA = load_module("analyze_cuda_gemm")
SCHEDULER = load_module("analyze_scheduler")


def document(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read JSON {path}: {error}") from error


def jsonl(path):
    try:
        rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read JSONL {path}: {error}") from error
    if not rows:
        raise ValueError(f"empty JSONL: {path}")
    return rows


def finite_positive(value, name):
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def source(value, expected, label):
    if not isinstance(value, dict):
        raise ValueError(f"{label} has no source identity")
    actual = {key: value.get(key) for key in ("commit", "source_digest", "source_dirty")}
    if actual != {**expected, "source_dirty": False}:
        raise ValueError(f"{label} source identity mismatch: {actual}")


def row_source(row, expected, label):
    source({"commit": row.get("commit"), "source_digest": row.get("source_digest"),
            "source_dirty": row.get("source_dirty")}, expected, label)


def verify_kv(path, expected):
    rows = jsonl(path)
    cpu = [row for row in rows if row.get("mode") == "cpu"]
    mixed = [row for row in rows if row.get("mode") == "mixed"]
    if len(cpu) != 18 or len(mixed) != 1:
        raise ValueError("KV evidence requires 18 CPU rows and one mixed row")
    indexed = {}
    for row in cpu:
        row_source(row, expected, "KV CPU")
        if (row.get("stage"), row.get("build_type"), row.get("testing"), row.get("warmups"), row.get("samples"),
                row.get("continuation_tokens"), row.get("correctness")) != ("stage14-c4", "Release", False, 3, 10, 32, "passed"):
            raise ValueError("KV CPU controls/correctness mismatch")
        context, run, variant = row.get("context"), row.get("run"), row.get("variant")
        if context not in (128, 256, 512) or run not in (0, 1, 2) or variant not in ("full_prefix", "kv_cache"):
            raise ValueError("KV coverage mismatch")
        samples = row.get("decode_ms_per_token_samples")
        if not isinstance(samples, list) or len(samples) != 10:
            raise ValueError("KV sample coverage mismatch")
        values = [finite_positive(value, "KV sample") for value in samples]
        median = statistics.median(values)
        recorded_median = row.get("decode_ms_per_token_median")
        if (not isinstance(recorded_median, (int, float)) or isinstance(recorded_median, bool) or
                not math.isfinite(recorded_median) or
                not math.isclose(recorded_median, median, rel_tol=1e-12, abs_tol=1e-12)):
            raise ValueError("KV median does not replay raw samples")
        key = (context, run, variant)
        if key in indexed:
            raise ValueError("duplicate KV record")
        indexed[key] = median
    ratios = []
    for run in range(3):
        ratio = indexed[(512, run, "full_prefix")] / indexed[(512, run, "kv_cache")]
        if ratio <= 1:
            raise ValueError("KV cache did not win every required context-512 pair")
        ratios.append(ratio)
    mixed_row = mixed[0]
    row_source(mixed_row, expected, "KV mixed")
    if (mixed_row.get("stage"), mixed_row.get("context"), mixed_row.get("warmups"), mixed_row.get("samples"),
            mixed_row.get("correctness")) != ("stage14-c4", 512, 3, 10, "passed"):
        raise ValueError("KV mixed controls/correctness mismatch")
    if len(mixed_row.get("decode_ms_per_token_samples", [])) != 10:
        raise ValueError("KV mixed samples missing")
    return {"context_512_ratios": ratios, "context_512_median_ratio": statistics.median(ratios)}


def verify_quant(path, expected, backend):
    row = document(path)
    row_source(row, expected, f"quant {backend}")
    if (row.get("schema_version"), row.get("stage"), row.get("backend"), row.get("cache"), row.get("correctness")) != (1, "stage16-c4", backend, True, "passed"):
        raise ValueError(f"quant {backend} controls/correctness mismatch")
    ratio = row.get("eligible_int8_ratio")
    if not isinstance(ratio, (int, float)) or not math.isfinite(ratio) or ratio <= 0 or ratio > .35:
        raise ValueError(f"quant {backend} compression gate failed")
    finite_positive(row.get("eligible_fp32_bytes"), "eligible_fp32_bytes")
    finite_positive(row.get("eligible_int8_scale_bytes"), "eligible_int8_scale_bytes")
    if row.get("int8_resident_parameter_bytes", 0) < row.get("float_resident_parameter_bytes", 0):
        raise ValueError("quant evidence unexpectedly claims an unimplemented resident-memory reduction")
    if row.get("int8_prepare_dequant_bytes", 0) <= 0:
        raise ValueError("quant dequant workspace accounting missing")
    if not all(isinstance(row.get(field), (int, float)) and math.isfinite(row[field]) and row[field] >= 0
               for field in ("logit_mae", "logit_max_abs_error")):
        raise ValueError("quant errors missing or nonfinite")
    return {"eligible_ratio": ratio, "mae": row["logit_mae"], "max_abs_error": row["logit_max_abs_error"],
            "int8_resident_bytes": row["int8_resident_parameter_bytes"]}


def verify_inference(path, expected, backend, quant, cache):
    rows = jsonl(path)
    teacher = [row for row in rows if row.get("variant") == "teacher_forced"]
    smoke = [row for row in rows if row.get("variant") == "greedy_smoke"]
    if len(teacher) != 3 or len(smoke) != 1:
        raise ValueError(f"{path}: expected three teacher-forced records and one smoke record")
    seen = set()
    for row in teacher:
        row_source(row, expected, "inference")
        if (row.get("stage"), row.get("backend"), row.get("quant"), row.get("cache"), row.get("warmups"),
                row.get("samples"), row.get("independent_runs"), row.get("correctness")) != ("stage17-c3", backend, quant, cache, 3, 10, 3, "passed"):
            raise ValueError(f"{path}: inference controls/correctness mismatch")
        run = row.get("run")
        if run not in (0, 1, 2) or run in seen:
            raise ValueError(f"{path}: inference run coverage mismatch")
        seen.add(run)
        for field in ("prefill_ms_samples", "first_token_ms_samples", "decode_ms_per_token_samples"):
            values = row.get(field)
            if not isinstance(values, list) or len(values) != 10:
                raise ValueError(f"{path}: missing {field}")
            for value in values:
                finite_positive(value, field)
        if row.get("execute_allocations") != 0:
            raise ValueError(f"{path}: execute allocation regression")
    return {"backend": backend, "quant": quant, "cache": cache, "records": 3}


def hash_tree(root):
    hashes = {}
    for path in sorted(Path(root).rglob("*")):
        if path.is_file():
            hashes[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def verify(args):
    expected = {"commit": args.commit, "source_digest": args.source_digest}
    cpu = CPU.analyze(args.stage7)
    source(cpu["source"], expected, "Stage 7")
    if not cpu["gate_passed"]:
        raise ValueError("Stage 7 benefit gate failed")
    planner = PLANNER.analyze(args.stage5)
    source(planner["source"], expected, "Stage 5")
    cuda = CUDA.analyze(args.stage9)
    source(cuda["source"], expected, "Stage 9")
    if not cuda["gate_passed"]:
        raise ValueError("Stage 9 benefit gate failed")
    scheduler = SCHEDULER.verify(args.stage11)
    source(scheduler["source"], expected, "Stage 11")
    profile_manifest = document(Path(args.stage10) / "manifest.json")
    source(profile_manifest.get("source"), expected, "Stage 10")
    if profile_manifest.get("status") != "passed" or {(p.get("tool"), p.get("kernel")) for p in profile_manifest.get("profiles", [])} != {
            ("nsys", "sgemm_v0_naive"), ("nsys", "sgemm_v1_tiled"), ("ncu", "sgemm_v0_naive"), ("ncu", "sgemm_v1_tiled")}:
        raise ValueError("Stage 10 profile coverage is incomplete")
    kv = verify_kv(args.kv, expected)
    if kv["context_512_median_ratio"] < 1.05:
        raise ValueError("KV context-512 benefit gate failed")
    quant = {"cpu": verify_quant(args.quant_cpu, expected, "cpu"), "mixed": verify_quant(args.quant_mixed, expected, "mixed")}
    configurations = [("cpu", "float", False), ("cpu", "float", True), ("cpu", "int8", False), ("cpu", "int8", True),
                      ("mixed", "float", True), ("mixed", "int8", True)]
    inference = [verify_inference(path, expected, *config) for path, config in zip(args.inference, configurations, strict=True)]
    return {"release_tag": args.tag, "source": {**expected, "source_dirty": False}, "gates": {
        "planner": planner["memory_gate"], "cpu_gemm": cpu["median_paired_geomean_speedup"],
        "cuda_gemm": cuda["median_paired_geomean_speedup"], "kv_cache": kv["context_512_median_ratio"],
        "int8_eligible_ratio": quant["mixed"]["eligible_ratio"]}, "scheduler": scheduler,
        "quantization": quant, "inference": inference, "optional_skips": ["S5-C4 inplace", "S7-C2 static per-call threads",
        "S7-C4 work stealing", "S9-C3..C6 extra variants", "S11-C3 segmentation", "S15-C4 BPE", "S16-C5 fused/INT4"],
        "artifact_sha256": {"stage5": hash_tree(args.stage5), "stage7": hash_tree(args.stage7), "stage9": hash_tree(args.stage9),
                            "stage10": hash_tree(args.stage10), "stage11": hash_tree(args.stage11), "kv": hashlib.sha256(Path(args.kv).read_bytes()).hexdigest(),
                            "quant_cpu": hashlib.sha256(Path(args.quant_cpu).read_bytes()).hexdigest(), "quant_mixed": hashlib.sha256(Path(args.quant_mixed).read_bytes()).hexdigest()}}


def render(result):
    gates = result["gates"]
    return "\n".join(["# Stage 18 — RC final evaluation (generated)", "", f"- Release tag: `{result['release_tag']}`", f"- Tested commit: `{result['source']['commit']}`", f"- Source digest: `{result['source']['source_digest']}`", "", "## Required benefit gates", "", "| Gate | Generated result |", "|---|---:|", f"| Planner execute allocation/capacity | {gates['planner']} |", f"| CPU GEMM paired geomean | {gates['cpu_gemm']:.6f}x |", f"| CUDA GEMM paired geomean | {gates['cuda_gemm']:.6f}x |", f"| KV cache context-512 paired median | {gates['kv_cache']:.6f}x |", f"| INT8 eligible payload including scales | {gates['int8_eligible_ratio']:.6%} |", "", "## Limits", "", "No scheduler, mixed-path, cache-outside-the-KV-gate, or INT8-throughput speedup is claimed. INT8 resident memory includes a persistent prepare-dequant workspace and is not claimed smaller.", "", "## Optional skips", "", *[f"- `{item}`" for item in result["optional_skips"]], ""])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--source-digest", required=True)
    parser.add_argument("--stage5", type=Path, required=True)
    parser.add_argument("--stage7", type=Path, required=True)
    parser.add_argument("--stage9", type=Path, required=True)
    parser.add_argument("--stage10", type=Path, required=True)
    parser.add_argument("--stage11", type=Path, required=True)
    parser.add_argument("--kv", type=Path, required=True)
    parser.add_argument("--quant-cpu", type=Path, required=True)
    parser.add_argument("--quant-mixed", type=Path, required=True)
    parser.add_argument("--inference", type=Path, nargs=6, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if len(args.commit) != 40 or len(args.source_digest) != 64:
        parser.error("--commit and --source-digest must be full hashes")
    try:
        result = verify(args)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.output.exists():
            raise ValueError(f"refusing to overwrite {args.output}")
        args.output.write_text(render(result), encoding="utf-8")
        args.output.with_suffix(".json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(args.output)
    except (OSError, ValueError, KeyError, TypeError, StopIteration) as error:
        parser.exit(1, f"verify_stage18_evidence: {error}\n")


if __name__ == "__main__":
    main()
