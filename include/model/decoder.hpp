#pragma once

#include "model/config.hpp"
#include "runtime/graph.hpp"

#include <cstddef>

namespace model {

struct DecoderBuildOptions {
    // Logical tensors remain CPU-bound. A CUDA hint lets the Stage 11 scheduler
    // insert explicit transfers for learned rank-2 projections only.
    runtime::Device projection_device{};
};

struct GraphCursor {
    runtime::TensorId next_tensor = 0;
    runtime::NodeId next_node = 0;
};

struct BlockBuildResult {
    runtime::TensorId output = runtime::INVALID_TENSOR_ID;
    std::size_t first_node = 0;
    std::size_t node_count = 0;
};

// Adds one pre-norm decoder block to an editable graph. Every attention and MLP
// intermediate is an ordinary graph tensor. Attention is lowered per head; each
// context is multiplied by its row-slice of O and the projected heads are added,
// which is equivalent to concat(contexts)@O without introducing CONCAT.
BlockBuildResult build_decoder_block(runtime::Graph& graph,
                                     const DecoderConfig& config,
                                     const ParameterTable& parameters,
                                     std::int64_t layer,
                                     std::int64_t sequence_length,
                                     runtime::TensorId input,
                                     GraphCursor& cursor,
                                     DecoderBuildOptions options = {});

} // namespace model
