#!/usr/bin/env python3
"""Verify fixture SHA256/provenance/weight metadata before running the C++ harness."""
import argparse
import hashlib
import json
import math
import re
import struct
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL = {"version": 1, "batch": 1, "layers": 2, "hidden": 64, "heads": 4,
         "head_dim": 16, "ffn": 128, "vocab": 258, "max_seq": 1088,
         "bos": 256, "eos": 257, "rms_epsilon": 1e-5, "rope_base": 10000,
         "projection_output_channel_axis": 1, "rope_layout": "interleaved", "bias": False}
NUMERIC = {"ADD", "MUL", "MATMUL", "COPY", "MATERIALIZE", "RMSNORM", "SOFTMAX", "ROPE", "EMBEDDING", "SWIGLU"}


def expected_weights():
    result = {"token_embedding": [258, 64], "final_norm": [64], "lm_head": [64, 258]}
    for layer in range(2):
        for name in ("attn_norm", "ffn_norm"):
            result[f"layers.{layer}.{name}"] = [64]
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            result[f"layers.{layer}.{name}"] = [64, 64]
        for name in ("gate_proj", "up_proj"):
            result[f"layers.{layer}.{name}"] = [64, 128]
        result[f"layers.{layer}.down_proj"] = [128, 64]
    return result


def verify(root):
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest["version"] != 1 or manifest["generator_version"] != 1 or manifest["model_contract"] != MODEL or manifest["performance_fixture"]:
        raise ValueError("fixture version/model contract mismatch")
    if set(manifest["files"]) != {"operators-v1.txt", "tiny-weights-v1.bin", "tiny-weights-v1.json"}:
        raise ValueError("unexpected fixture files; paths must be fixed basenames")
    if manifest["generator"] != "tools/generate_operator_fixtures.py" or hashlib.sha256((ROOT / manifest["generator"]).read_bytes()).hexdigest() != manifest["generator_sha256"]:
        raise ValueError("fixture generator provenance mismatch")
    for name, identity in manifest["files"].items():
        data = (root / name).read_bytes()
        if len(data) != identity["bytes"] or hashlib.sha256(data).hexdigest() != identity["sha256"]:
            raise ValueError("fixture checksum/size mismatch: " + name)
    names = [case["name"] for case in manifest["cases"]]
    if not names or len(names) != len(set(names)):
        raise ValueError("fixture cases missing or duplicated")
    for case in manifest["cases"]:
        if not math.isfinite(case["atol"]) or not math.isfinite(case["rtol"]) or case["atol"] < 0 or case["rtol"] < 0:
            raise ValueError("invalid fixture tolerance")
        for tensor in case["inputs"] + case["output"]:
            if tensor["dtype"] not in ("FP32", "INT32") or len(tensor["shape"]) > 8 or any(not isinstance(dim, int) or dim < 0 for dim in tensor["shape"]):
                raise ValueError("invalid fixture tensor metadata")
    weights = json.loads((root / "tiny-weights-v1.json").read_text(encoding="utf-8"))
    blob = (root / "tiny-weights-v1.bin").read_bytes()
    if weights["version"] != 1 or weights["format"] != "fixture-only-raw-fp32-le" or weights["model_contract"] != MODEL or weights["seed"] != 0x5205:
        raise ValueError("weight fixture contract mismatch")
    if len(blob) != weights["total_bytes"] or hashlib.sha256(blob).hexdigest() != weights["sha256"]:
        raise ValueError("weight payload identity mismatch")
    expected = expected_weights()
    offset = 0
    for entry in weights["tensors"]:
        name = entry["name"]
        if name not in expected or entry["shape"] != expected.pop(name) or entry["dtype"] != "FP32":
            raise ValueError("weight tensor name/shape/dtype mismatch")
        size = 4 * math.prod(entry["shape"])
        if entry["offset_bytes"] != offset or entry["nbytes"] != size or offset + size > len(blob):
            raise ValueError("weight byte ranges must be exact, ordered and disjoint")
        data = blob[offset:offset + size]
        if hashlib.sha256(data).hexdigest() != entry["sha256"] or not all(math.isfinite(value[0]) for value in struct.iter_unpack("<f", data)):
            raise ValueError("weight tensor checksum/nonfinite payload")
        if (name.endswith("_proj") or name == "lm_head") and entry.get("output_channel_axis") != 1:
            raise ValueError("projection W[in,out] output-channel axis mismatch")
        if "norm" in name and any(value[0] != 1 for value in struct.iter_unpack("<f", data)):
            raise ValueError("norm fixture scale must be one")
        offset += size
    if expected or offset != len(blob):
        raise ValueError("weight tensors missing or orphan payload bytes")
    return manifest


def expected_attributes(case):
    values = case["attrs"]
    if case["op"] == "COPY":
        return {"overlap": "reject_except_exact_self"}
    fields = {
        "RMSNORM": ["epsilon"],
        "SOFTMAX": ["causal", "query_position", "key_position", "max_positions"],
        "ROPE": ["position", "base", "max_positions"],
        "ATTENTION": ["heads", "head_dim", "causal", "query_position", "key_position", "max_positions"],
    }.get(case["op"], [])
    if len(fields) != len(values):
        raise ValueError("fixture typed attribute count mismatch")
    result = {field: values[index] for index, field in enumerate(fields)}
    if "causal" in result:
        if result["causal"] not in (0, 1):
            raise ValueError("fixture causal flag must be 0/1")
        result["causal"] = result["causal"] == 1
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=ROOT / "tests/fixtures/operators-v1")
    args = parser.parse_args()
    try:
        manifest = verify(args.root)
        weights = json.loads((args.root / "tiny-weights-v1.json").read_text(encoding="utf-8"))
        print(f"Fixture checksum/provenance: PASS; tiny weights={len(weights['tensors'])} tensors, {weights['total_bytes']} bytes (not model execution)", flush=True)
        process = subprocess.run([str(args.binary), str(args.root / "operators-v1.txt")], text=True, capture_output=True)
        print(process.stdout, end="")
        print(process.stderr, end="", file=sys.stderr)
        if process.returncode:
            return process.returncode
        descriptors = [json.loads(line[len("descriptor "):]) for line in process.stdout.splitlines() if line.startswith("descriptor ")]
        if len(descriptors) != len(manifest["cases"]):
            raise ValueError("C++ reader/manifest case count mismatch")
        for index, desc in enumerate(descriptors):
            case = manifest["cases"][index]
            if desc["op"] != case["op"] or desc["version"] != 1 or desc["inputs"] != list(range(len(case["inputs"]))) or desc["outputs"] != [len(case["inputs"])] or desc["backend_hint"] is not None:
                raise ValueError("C++ canonical serialization/fixture metadata mismatch")
            if desc["attributes"] != expected_attributes(case):
                raise ValueError("C++ typed attribute serialization/manifest mismatch")
        match = re.search(r"Operator fixtures: PASS numeric=(\d+) transformer_metadata=(\d+)", process.stdout)
        numeric = sum(case["op"] in NUMERIC for case in manifest["cases"])
        if not match or int(match[1]) != numeric or int(match[2]) != len(manifest["cases"]) - numeric:
            raise ValueError("fixture execution coverage mismatch")
        return 0
    except (KeyError, ValueError, OSError) as error:
        print("run_operator_fixtures: " + str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
