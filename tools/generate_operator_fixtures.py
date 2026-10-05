#!/usr/bin/env python3
"""Offline stdlib-only conformance artifacts; never imported by the C++ runtime."""
import argparse
import hashlib
import json
import math
import struct
import sys
from pathlib import Path

VERSION = 1
MODEL = {"version": 1, "batch": 1, "layers": 2, "hidden": 64, "heads": 4,
         "head_dim": 16, "ffn": 128, "vocab": 258, "max_seq": 1088,
         "bos": 256, "eos": 257, "rms_epsilon": 1e-5, "rope_base": 10000,
         "projection_output_channel_axis": 1, "rope_layout": "interleaved", "bias": False}
SEED = 0x5205


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def tensor(shape, values, dtype="FP32", transpose=None):
    if len(shape) > 8 or any(not isinstance(dim, int) or dim < 0 for dim in shape):
        raise ValueError("invalid fixture shape")
    values = list(values)
    if dtype == "FP32":
        values = [f32(value) for value in values]
        if not all(math.isfinite(value) for value in values):
            raise ValueError("nonfinite fixture payload")
    elif dtype != "INT32" or any(not isinstance(value, int) or not -(2**31) <= value < 2**31 for value in values):
        raise ValueError("invalid fixture dtype/INT32 payload")
    if math.prod(shape) != len(values):
        raise ValueError("fixture payload length differs from shape")
    return {"dtype": dtype, "shape": shape, "values": values, "transpose": transpose}


def case(name, op, inputs, output, attrs=None):
    return {"name": name, "op": op, "inputs": inputs, "output": output, "attrs": attrs or [],
            "atol": 1e-6 if op in {"ADD", "MUL", "MATMUL", "COPY", "MATERIALIZE"} else 2e-5,
            "rtol": 1e-5 if op in {"ADD", "MUL", "MATMUL", "COPY", "MATERIALIZE"} else 2e-4}


def softmax_rows(values, rows, columns, causal=False, query=0, key=0):
    result = []
    for row in range(rows):
        valid = [column for column in range(columns) if not causal or key + column <= query + row]
        if not valid:
            result.extend([0.0] * columns)
            continue
        maximum = max(values[row * columns + column] for column in valid)
        exp = {column: math.exp(values[row * columns + column] - maximum) for column in valid}
        denominator = sum(exp.values())
        result.extend(f32(exp[column] / denominator) if column in exp else 0.0 for column in range(columns))
    return result


def rope(values, tokens, heads, dim, position, base=10000):
    result = list(values)
    for token in range(tokens):
        for head in range(heads):
            for pair in range(dim // 2):
                theta = (position + token) / base ** (2 * pair / dim)
                cosine, sine = math.cos(theta), math.sin(theta)
                offset = (token * heads + head) * dim + 2 * pair
                even, odd = values[offset:offset + 2]
                result[offset] = f32(even * cosine - odd * sine)
                result[offset + 1] = f32(even * sine + odd * cosine)
    return result


def attention(q, k, v, queries, keys, heads, dim, query_position, key_position):
    result = [0.0] * (queries * heads * dim)
    for head in range(heads):
        scores = []
        for row in range(queries):
            for column in range(keys):
                dot = sum(q[(row * heads + head) * dim + i] * k[(column * heads + head) * dim + i] for i in range(dim))
                scores.append(f32(f32(dot) * f32(1 / math.sqrt(dim))))
        probabilities = softmax_rows(scores, queries, keys, True, query_position, key_position)
        for row in range(queries):
            for i in range(dim):
                result[(row * heads + head) * dim + i] = f32(sum(probabilities[row * keys + column] * v[(column * heads + head) * dim + i] for column in range(keys)))
    return result


def cases():
    fixtures = [
        case("add_hand", "ADD", [tensor([4], [-2, -1, 0, 1]), tensor([4], [1, 2, 3, 4])], tensor([4], [-1, 1, 3, 5])),
        case("mul_scalar", "MUL", [tensor([], [3]), tensor([], [-2])], tensor([], [-6])),
        case("matmul_rectangular", "MATMUL", [tensor([2, 3], range(1, 7)), tensor([3, 4], range(1, 13))], tensor([2, 4], [38, 44, 50, 56, 83, 98, 113, 128])),
        case("matmul_zero_inner", "MATMUL", [tensor([2, 0], []), tensor([0, 3], [])], tensor([2, 3], [0] * 6)),
        case("copy_ids_transpose", "COPY", [tensor([2, 3], [0, 256, 257, 3, 4, 5], "INT32", [0, 1]), tensor([3, 2], [-1] * 6, "INT32")], tensor([3, 2], [0, 3, 256, 4, 257, 5], "INT32")),
        case("copy_fp32", "COPY", [tensor([3], [1.25, -2.5, 0]), tensor([3], [99] * 3)], tensor([3], [1.25, -2.5, 0])),
        case("materialize_transpose", "MATERIALIZE", [tensor([2, 3], range(6), transpose=[0, 1])], tensor([3, 2], [0, 3, 1, 4, 2, 5])),
        case("add_empty", "ADD", [tensor([0, 3], []), tensor([0, 3], [])], tensor([0, 3], [])),
    ]
    x = [1, -1, 2, -2, 0, 0, 0, 0]
    scale = [1, 2, 3, 4]
    norm = [f32(value * scale[i % 4] / math.sqrt(2.5 + 1e-5)) if i < 4 else 0 for i, value in enumerate(x)]
    fixtures.append(case("rmsnorm_hand_zero_row", "RMSNORM", [tensor([2, 4], x), tensor([4], scale)], tensor([2, 4], norm), [1e-5]))
    for name, values, rows, columns, causal, query, key in [
        ("softmax_stable", [1000, 1001, 999, -1000, -1001, -1002], 2, 3, False, 0, 0),
        ("softmax_decode_offset", [1000, 1001, 1002, 999], 1, 4, True, 2, 0),
        ("softmax_all_masked", [1, 2, 3, 4, 5, 6], 2, 3, True, 0, 2),
    ]:
        fixtures.append(case(name, "SOFTMAX", [tensor([rows, columns], values)], tensor([rows, columns], softmax_rows(values, rows, columns, causal, query, key)), [1 if causal else 0, query, key, 1088]))
    fixtures.append(case("rope_interleaved_hand", "ROPE", [tensor([1, 4], [1, 2, 3, 4])], tensor([1, 4], rope([1, 2, 3, 4], 1, 1, 4, 1)), [1, 10000, 1088]))
    q = [f32((i % 13 - 6) * 0.1) for i in range(2 * 4 * 16)]
    fixtures.append(case("rope_tiny_heads", "ROPE", [tensor([2, 4, 16], q)], tensor([2, 4, 16], rope(q, 2, 4, 16, 17)), [17, 10000, 1088]))
    fixtures.append(case("embedding_hand", "EMBEDDING", [tensor([3], [0, 3, 1], "INT32"), tensor([4, 3], range(1, 13))], tensor([3, 3], [1, 2, 3, 10, 11, 12, 4, 5, 6])))
    gate, up = [0, 1, -1, 2], [1, 2, 3, 4]
    swiglu = [f32(g * up[i] / (1 + math.exp(-g))) for i, g in enumerate(gate)]
    fixtures.append(case("swiglu_hand", "SWIGLU", [tensor([2, 2], gate), tensor([2, 2], up)], tensor([2, 2], swiglu)))
    q = [f32((i % 7 - 3) * 0.125) for i in range(4 * 16)]
    k = [f32((i % 11 - 5) * 0.1) for i in range(3 * 4 * 16)]
    v = [f32((i % 17 - 8) * 0.05) for i in range(3 * 4 * 16)]
    fixtures.append(case("attention_decode_tiny_geometry", "ATTENTION", [tensor([1, 4, 16], q), tensor([3, 4, 16], k), tensor([3, 4, 16], v)], tensor([1, 4, 16], attention(q, k, v, 1, 3, 4, 16, 2, 0)), [4, 16, 1, 2, 0, 1088]))
    return fixtures


def encode(fixtures):
    lines = ["MLRT_OP_FIXTURES 1"]
    for fixture in fixtures:
        lines.append(f"CASE {fixture['name']} {fixture['op']} {len(fixture['inputs'])} {fixture['atol']:.9g} {fixture['rtol']:.9g}")
        lines.append("ATTR " + " ".join(str(value) for value in fixture["attrs"]) if fixture["attrs"] else "ATTR NONE")
        for label, t in [("INPUT", item) for item in fixture["inputs"]] + [("OUTPUT", fixture["output"])]:
            lines.append(f"{label} {t['dtype']} {len(t['shape'])}" + "".join(f" {dim}" for dim in t["shape"]))
            lines.append(f"DATA {len(t['values'])}" + "".join(f" {value:.9g}" if t["dtype"] == "FP32" else f" {value}" for value in t["values"]))
            lines.append("LAYOUT TRANSPOSE " + " ".join(str(axis) for axis in t["transpose"]) if t["transpose"] is not None else "LAYOUT DENSE")
        lines.append("END")
    lines.append("END_FIXTURES")
    return ("\n".join(lines) + "\n").encode()


def weights():
    state = SEED
    blob = bytearray()
    entries = []

    def add(name, shape, norm=False, projection=False):
        nonlocal state
        data = bytearray()
        for _ in range(math.prod(shape)):
            if norm:
                value = 1.0
            else:
                state ^= (state << 13) & 0xFFFFFFFF
                state ^= state >> 17
                state ^= (state << 5) & 0xFFFFFFFF
                state &= 0xFFFFFFFF
                value = (state % 257 - 128) / 8192  # exact dyadic values, no cross-language RNG requirement
            data.extend(struct.pack("<f", value))
        entry = {"name": name, "dtype": "FP32", "shape": shape, "offset_bytes": len(blob),
                 "nbytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        if projection:
            entry["output_channel_axis"] = 1
        entries.append(entry)
        blob.extend(data)

    add("token_embedding", [258, 64])
    for layer in range(2):
        prefix = f"layers.{layer}."
        add(prefix + "attn_norm", [64], norm=True)
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            add(prefix + name, [64, 64], projection=True)
        add(prefix + "ffn_norm", [64], norm=True)
        for name in ("gate_proj", "up_proj"):
            add(prefix + name, [64, 128], projection=True)
        add(prefix + "down_proj", [128, 64], projection=True)
    add("final_norm", [64], norm=True)
    add("lm_head", [64, 258], projection=True)
    metadata = {"version": VERSION, "format": "fixture-only-raw-fp32-le", "model_contract": MODEL,
                "seed": SEED, "rng": "xorshift32; integer dyadic scale 1/8192", "tensors": entries,
                "total_bytes": len(blob), "sha256": hashlib.sha256(blob).hexdigest()}
    return bytes(blob), (json.dumps(metadata, sort_keys=True, indent=2) + "\n").encode()


def artifacts():
    fixtures = cases()
    blob, metadata = weights()
    files = {"operators-v1.txt": encode(fixtures), "tiny-weights-v1.bin": blob, "tiny-weights-v1.json": metadata}
    summary = []
    for fixture in fixtures:
        entry = {key: value for key, value in fixture.items() if key not in ("inputs", "output")}
        for group in ("inputs", "output"):
            items = fixture[group] if group == "inputs" else [fixture[group]]
            entry[group] = [{key: value for key, value in item.items() if key != "values"} for item in items]
        summary.append(entry)
    manifest = {"version": VERSION, "generator": "tools/generate_operator_fixtures.py", "generator_version": VERSION,
                "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "precision": "FP32 inputs/output; stdlib scalar FP64 oracle; little-endian weight payload",
                "performance_fixture": False, "model_contract": MODEL, "cases": summary,
                "files": {name: {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)} for name, data in files.items()}}
    files["manifest.json"] = (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode()
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "tests/fixtures/operators-v1")
    parser.add_argument("--check", action="store_true", help="Compare regenerated artifacts without rewriting stored evidence")
    args = parser.parse_args()
    files = artifacts()
    if args.check:
        mismatches = [name for name, data in files.items() if not (args.output / name).is_file() or (args.output / name).read_bytes() != data]
        if mismatches:
            print("fixture regeneration mismatch: " + ", ".join(mismatches), file=sys.stderr)
            return 1
        print("Offline fixture regeneration: PASS")
    else:
        args.output.mkdir(parents=True, exist_ok=True)
        for name, data in files.items():
            (args.output / name).write_bytes(data)
        print(f"Generated {len(cases())} conformance cases and {len(files['tiny-weights-v1.bin'])} weight bytes in {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
