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

struct TeacherForcedResult {
    std::vector<std::int32_t> prompt_tokens;
    runtime::Tensor final_logits;
    runtime::ExecutionCounts prefill_counts;
    runtime::ExecutionCounts total_counts;
    std::size_t cache_persistent_bytes = 0;
    double prefill_ms = 0.0;
    double first_token_ms = 0.0;
    std::vector<double> decode_ms;
};

// Runs batch-one deterministic greedy generation entirely through the decoder
// graph, scheduler and regular backend paths. It checks the full context budget
// before allocating or mutating a KV cache. V1 byte tokenization is BOS-on and
// EOS-off; generated EOS is retained then terminates the loop.
GenerationResult generate_greedy(const LoadedModel& model, const std::string& prompt,
                                 GenerationOptions options = {});

// Executes a caller-supplied continuation rather than generated tokens. This
// gives cache/dtype/device comparisons identical inputs while retaining the
// ordinary graph/scheduler execution and all model-step timing boundaries.
TeacherForcedResult run_teacher_forced(const LoadedModel& model, const std::string& prompt,
                                       const std::vector<std::int32_t>& continuation,
                                       GenerationOptions options = {});

} // namespace model
