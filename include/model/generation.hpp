#pragma once

#include "model/model_file.hpp"
#include "runtime/graph_executor.hpp"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace model {

enum class GenerationBackend { CPU, Mixed };

struct GenerationOptions {
    GenerationBackend backend = GenerationBackend::CPU;
    bool use_cache = true;
    std::int64_t max_tokens = 16;
};

struct GenerationResult {
    std::vector<std::int32_t> prompt_tokens;
    std::vector<std::int32_t> generated_tokens;
    std::optional<runtime::Tensor> final_logits;
    runtime::ExecutionCounts prefill_counts;
    runtime::ExecutionCounts total_counts;
    std::size_t cache_persistent_bytes = 0;
};

// Runs batch-one deterministic greedy generation entirely through the decoder
// graph, scheduler and regular backend paths. It checks the full context budget
// before allocating or mutating a KV cache. V1 byte tokenization is BOS-on and
// EOS-off; generated EOS is retained then terminates the loop.
GenerationResult generate_greedy(const LoadedModel& model, const std::string& prompt,
                                 GenerationOptions options = {});

} // namespace model
