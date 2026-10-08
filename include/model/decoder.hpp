#pragma once

#include "model/config.hpp"
#include "model/kv_cache.hpp"
#include "runtime/graph.hpp"
#include "runtime/graph_executor.hpp"

#include <cstddef>
#include <cstdint>

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
    runtime::TensorId key = runtime::INVALID_TENSOR_ID;
    runtime::TensorId value = runtime::INVALID_TENSOR_ID;
    runtime::NodeId first_node = runtime::INVALID_NODE_ID;
    std::size_t node_count = 0;
};

// Adds one complete block to an editable graph. This is intentionally public
// so the Stage 13 block fixture can verify its graph-only lowering.
BlockBuildResult build_decoder_block(runtime::Graph& graph, const DecoderConfig& config,
                                     const ParameterTable& parameters, std::int64_t layer,
                                     std::int64_t sequence_length, runtime::TensorId input,
                                     GraphCursor& cursor, DecoderBuildOptions options = {});

struct DecoderGraph {
    runtime::Graph graph;
    std::int64_t sequence_length = 0;
    runtime::TensorId logits = runtime::INVALID_TENSOR_ID;
    std::size_t projection_nodes = 0;
};

DecoderGraph build_decoder_prefill(const DecoderConfig& config, const ParameterTable& parameters,
                                   runtime::Tensor token_ids, DecoderBuildOptions options = {});

// Cache writes are ordinary ordered COPY nodes to persistent cache views. The
// cache's valid length advances only through execute_cached_decoder after every
// layer succeeds; failure invalidates it rather than exposing partial state.
struct CachedDecoderGraph {
    runtime::Graph graph;
    std::int64_t sequence_length = 0;
    runtime::TensorId logits = runtime::INVALID_TENSOR_ID;
    std::size_t projection_nodes = 0;
    std::int64_t cache_position = 0;
    std::int64_t cache_append = 0;
};

CachedDecoderGraph build_decoder_prefill_cached(const DecoderConfig& config, const ParameterTable& parameters,
                                                runtime::Tensor token_ids, KVCache& cache,
                                                DecoderBuildOptions options = {});
CachedDecoderGraph build_decoder_decode(const DecoderConfig& config, const ParameterTable& parameters,
                                        runtime::Tensor token_id, KVCache& cache,
                                        DecoderBuildOptions options = {});
runtime::Status finalize_cached_decoder(const CachedDecoderGraph& decoder, KVCache& cache,
                                         const runtime::ExecutionResult& result);
runtime::ExecutionResult execute_cached_decoder(const CachedDecoderGraph& decoder, KVCache& cache,
                                                runtime::ExecutionTrace* trace = nullptr,
                                                runtime::AllocationProvider* provider = nullptr,
                                                const runtime::Backend* backend = nullptr,
                                                const runtime::Scheduler* scheduler = nullptr);
} // namespace model
