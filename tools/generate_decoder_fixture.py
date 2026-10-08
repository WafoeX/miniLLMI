#!/usr/bin/env python3
"""Generate the independent Stage 13 decoder-block reference fixture.

The implementation deliberately uses scalar Python and explicit IEEE-754 FP32
rounding at every C++ FP32 write. It does not call the C++ runtime or NumPy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HIDDEN, HEADS, HEAD_DIM, FFN, VOCAB = 64, 4, 16, 128, 258
TOKENS = [256, 0, 1, 257]


def f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def shapes():
    result = [("token_embedding", VOCAB, HIDDEN)]
    for layer in range(2):
        prefix = f"layers.{layer}."
        result += [
            (prefix + "attn_norm", HIDDEN),
            (prefix + "q_proj", HIDDEN, HIDDEN),
            (prefix + "k_proj", HIDDEN, HIDDEN),
            (prefix + "v_proj", HIDDEN, HIDDEN),
            (prefix + "o_proj", HIDDEN, HIDDEN),
            (prefix + "ffn_norm", HIDDEN),
            (prefix + "gate_proj", HIDDEN, FFN),
            (prefix + "up_proj", HIDDEN, FFN),
            (prefix + "down_proj", FFN, HIDDEN),
        ]
    result += [("final_norm", HIDDEN), ("lm_head", HIDDEN, VOCAB)]
    return result


def load_weights(path: Path):
    data = path.read_bytes()
    offset, weights = 0, {}
    for item in shapes():
        name, dims = item[0], item[1:]
        count = math.prod(dims)
        size = count * 4
        if offset + size > len(data):
            raise ValueError("truncated tiny weight fixture")
        weights[name] = (dims, list(struct.unpack(f"<{count}f", data[offset:offset + size])))
        offset += size
    if offset != len(data):
        raise ValueError("tiny weight fixture has trailing bytes")
    return weights


def matmul(a, rows, inner, b, cols):
    out = [0.0] * (rows * cols)
    for row in range(rows):
        for col in range(cols):
            total = 0.0
            for k in range(inner):
                total += a[row * inner + k] * b[k * cols + col]
            out[row * cols + col] = f32(total)
    return out


def add(a, b):
    return [f32(x + y) for x, y in zip(a, b)]


def rmsnorm(x, rows, channels, scale):
    out = [0.0] * len(x)
    for row in range(rows):
        total = sum(x[row * channels + col] ** 2 for col in range(channels))
        inv = 1.0 / math.sqrt(total / channels + 1e-5)
        for col in range(channels):
            out[row * channels + col] = f32(x[row * channels + col] * inv * scale[col])
    return out


def rope(x, tokens, heads, dimension):
    out = [0.0] * len(x)
    for token in range(tokens):
        for head in range(heads):
            for pair in range(dimension // 2):
                theta = token / math.pow(10000.0, 2.0 * pair / dimension)
                base = (token * heads + head) * dimension + 2 * pair
                even, odd = x[base], x[base + 1]
                out[base] = f32(even * math.cos(theta) - odd * math.sin(theta))
                out[base + 1] = f32(even * math.sin(theta) + odd * math.cos(theta))
    return out


def softmax_causal(scores, size):
    out = [0.0] * len(scores)
    for query in range(size):
        visible = scores[query * size:query * size + query + 1]
        maximum = max(visible)
        values = [math.exp(f32(value - maximum)) for value in visible]
        total = sum(values)
        for key, value in enumerate(values):
            out[query * size + key] = f32(value / total)
    return out


def swiglu(gate, up):
    out = []
    for value, multiplier in zip(gate, up):
        x = value
        sigmoid = 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))
        out.append(f32(x * sigmoid * multiplier))
    return out


def block(hidden, layer, weights, sequence):
    prefix = f"layers.{layer}."
    normalized = rmsnorm(hidden, sequence, HIDDEN, weights[prefix + "attn_norm"][1])
    q = rope(matmul(normalized, sequence, HIDDEN, weights[prefix + "q_proj"][1], HIDDEN), sequence, HEADS, HEAD_DIM)
    k = rope(matmul(normalized, sequence, HIDDEN, weights[prefix + "k_proj"][1], HIDDEN), sequence, HEADS, HEAD_DIM)
    v = matmul(normalized, sequence, HIDDEN, weights[prefix + "v_proj"][1], HIDDEN)
    o_weight = weights[prefix + "o_proj"][1]
    projected = []
    scale = f32(1.0 / math.sqrt(HEAD_DIM))
    for head in range(HEADS):
        qh = [q[(token * HEADS + head) * HEAD_DIM + dim] for token in range(sequence) for dim in range(HEAD_DIM)]
        kh = [k[(token * HEADS + head) * HEAD_DIM + dim] for token in range(sequence) for dim in range(HEAD_DIM)]
        vh = [v[token * HIDDEN + head * HEAD_DIM + dim] for token in range(sequence) for dim in range(HEAD_DIM)]
        scores = matmul(qh, sequence, HEAD_DIM,
                        [kh[token * HEAD_DIM + dim] for dim in range(HEAD_DIM) for token in range(sequence)], sequence)
        scores = [f32(value * scale) for value in scores]
        probabilities = softmax_causal(scores, sequence)
        context = matmul(probabilities, sequence, sequence, vh, HEAD_DIM)
        o_slice = [o_weight[(head * HEAD_DIM + row) * HIDDEN + col]
                   for row in range(HEAD_DIM) for col in range(HIDDEN)]
        projected.append(matmul(context, sequence, HEAD_DIM, o_slice, HIDDEN))
    attention = projected[0]
    for head in projected[1:]:
        attention = add(attention, head)
    residual = add(hidden, attention)
    ffn_input = rmsnorm(residual, sequence, HIDDEN, weights[prefix + "ffn_norm"][1])
    gate = matmul(ffn_input, sequence, HIDDEN, weights[prefix + "gate_proj"][1], FFN)
    up = matmul(ffn_input, sequence, HIDDEN, weights[prefix + "up_proj"][1], FFN)
    activated = swiglu(gate, up)
    down = matmul(activated, sequence, FFN, weights[prefix + "down_proj"][1], HIDDEN)
    return add(residual, down)


def write_floats(path: Path, values):
    path.write_bytes(struct.pack(f"<{len(values)}f", *values))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=ROOT / "tests/fixtures/operators-v1/tiny-weights-v1.bin")
    parser.add_argument("--output", type=Path, default=ROOT / "tests/fixtures/decoder-v1")
    args = parser.parse_args()
    weights = load_weights(args.weights)
    hidden = [weights["token_embedding"][1][token * HIDDEN + channel]
              for token in TOKENS for channel in range(HIDDEN)]
    block0 = block(hidden, 0, weights, len(TOKENS))
    args.output.mkdir(parents=True, exist_ok=True)
    block_path = args.output / "block0-output.bin"
    write_floats(block_path, block0)
    metadata = {
        "schema_version": 1,
        "model_contract_version": 1,
        "generator": "scalar-python-explicit-fp32-v1",
        "source_weights": str(args.weights.relative_to(ROOT)),
        "source_weights_sha256": hashlib.sha256(args.weights.read_bytes()).hexdigest(),
        "token_ids": TOKENS,
        "block": 0,
        "shape": [len(TOKENS), HIDDEN],
        "atol": 2e-6,
        "rtol": 2e-5,
        "artifacts": {block_path.name: hashlib.sha256(block_path.read_bytes()).hexdigest()},
    }
    (args.output / "fixture.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
