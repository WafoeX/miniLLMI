#pragma once

#include <cstdint>

namespace runtime::tiny_model {
// Metadata contract only: no model, graph, tokenizer or loader implementation.
inline constexpr unsigned VERSION = 1;
inline constexpr std::int64_t BATCH = 1;
inline constexpr std::int64_t LAYERS = 2;
inline constexpr std::int64_t HIDDEN = 64;
inline constexpr std::int64_t HEADS = 4;
inline constexpr std::int64_t HEAD_DIM = 16;
inline constexpr std::int64_t FFN = 128;
inline constexpr std::int64_t VOCAB = 258;
inline constexpr std::int64_t MAX_SEQ = 1088;
inline constexpr std::int32_t BOS = 256;
inline constexpr std::int32_t EOS = 257;
inline constexpr double RMS_EPSILON = 1e-5;
inline constexpr double ROPE_BASE = 10000;
inline constexpr unsigned PROJECTION_OUTPUT_CHANNEL_AXIS = 1; // W[in,out]
static_assert(HIDDEN == HEADS * HEAD_DIM && HEAD_DIM % 2 == 0, "frozen MHA geometry");
} // namespace runtime::tiny_model
