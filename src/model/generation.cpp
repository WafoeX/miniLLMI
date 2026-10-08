#include "model/generation.hpp"

#include "model/decoder.hpp"
#include "model/tokenizer.hpp"
#include "runtime/planned_executor.hpp"
#include "runtime/scheduler.hpp"
#ifdef MODEL_CUDA_BUILD
#include "runtime/cuda_backend.hpp"
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <optional>
#include <stdexcept>

namespace model {
namespace {

runtime::Tensor token_tensor(const std::vector<std::int32_t>& tokens) {
    auto tensor = runtime::Tensor::allocate_cpu({static_cast<std::int64_t>(tokens.size())}, runtime::DType::INT32);
    for (std::size_t index = 0; index < tokens.size(); ++index) tensor.data<std::int32_t>()[index] = tokens[index];
    return tensor;
}

void add_counts(runtime::ExecutionCounts& total, const runtime::ExecutionCounts& next) {
    total.nodes_completed += next.nodes_completed;
    total.allocations += next.allocations;
    total.frees += next.frees;
    total.allocation_requests += next.allocation_requests;
    total.releases += next.releases;
    total.arena_capacity_bytes = std::max(total.arena_capacity_bytes, next.arena_capacity_bytes);
    total.allocated_bytes += next.allocated_bytes;
    total.live_bytes = next.live_bytes;
    total.peak_live_bytes = std::max(total.peak_live_bytes, next.peak_live_bytes);
    total.copies += next.copies;
    total.copy_bytes += next.copy_bytes;
    total.backend_dispatches += next.backend_dispatches;
    total.backend_switches += next.backend_switches;
}

std::int32_t greedy_token(const runtime::Tensor& logits) {
    if (logits.dtype() != runtime::DType::FP32 || logits.shape().rank() != 2 || logits.shape()[0] <= 0 ||
        logits.shape()[1] <= 0)
        throw std::runtime_error("generation logits have an invalid shape or dtype");
    const auto width = logits.shape()[1];
    const auto* values = logits.data<float>() + (logits.shape()[0] - 1) * width;
    std::int64_t winner = 0;
    float maximum = values[0];
    if (!std::isfinite(maximum)) throw std::runtime_error("generation logits are nonfinite");
    for (std::int64_t index = 1; index < width; ++index) {
        if (!std::isfinite(values[index])) throw std::runtime_error("generation logits are nonfinite");
        if (values[index] > maximum) {
            maximum = values[index];
            winner = index;
        }
    }
    return static_cast<std::int32_t>(winner);
}

struct Step {
    runtime::Tensor logits;
    runtime::ExecutionCounts counts;
};

class Runner final {
public:
    Runner(const LoadedModel& model, GenerationOptions options)
        : model_(model), options_(options) {
        if (options_.max_tokens <= 0) throw std::invalid_argument("generation max_tokens must be positive");
#ifdef MODEL_CUDA_BUILD
        if (options_.backend == GenerationBackend::Mixed) {
            cuda_.emplace(0, runtime::CudaMatmul::Stage0Naive);
            scheduler_.emplace(runtime::default_cpu_backend(), &*cuda_);
        }
#else
        if (options_.backend == GenerationBackend::Mixed)
            throw std::runtime_error("mixed generation requires an ENABLE_CUDA=ON build");
#endif
    }

    Step prefill(const std::vector<std::int32_t>& tokens, KVCache* cache) const {
        if (cache) {
            const auto logical = build_decoder_prefill_cached(model_.config(), model_.parameters, token_tensor(tokens),
                                                              *cache, build_options());
            return execute_cached(logical, *cache);
        }
        const auto logical = build_decoder_prefill(model_.config(), model_.parameters, token_tensor(tokens), build_options());
        return execute(logical.graph);
    }

    Step decode(std::int32_t token, KVCache& cache) const {
        const auto logical = build_decoder_decode(model_.config(), model_.parameters, token_tensor({token}), cache,
                                                  build_options());
        return execute_cached(logical, cache);
    }

private:
    DecoderBuildOptions build_options() const {
#ifdef MODEL_CUDA_BUILD
        if (cuda_) return {cuda_->device()};
#endif
        return {};
    }

    Step execute(const runtime::Graph& logical) const {
        if (!scheduler_) {
            runtime::PlannedAllocationProvider plan(logical);
            const auto result = runtime::execute_graph(logical, nullptr, &plan);
            if (!result.ok()) throw std::runtime_error(result.status.message);
            return {result.outputs.at("logits"), result.counts};
        }
        const auto scheduled = scheduler_->rewrite(logical);
        if (!scheduled.ok()) throw std::runtime_error(scheduled.status.message);
        runtime::ScheduledAllocationProvider plan(*scheduled.graph, *scheduler_);
        const auto result = runtime::execute_graph(*scheduled.graph, nullptr, &plan, nullptr, &*scheduler_);
        if (!result.ok()) throw std::runtime_error(result.status.message);
        return {result.outputs.at("logits"), result.counts};
    }

    Step execute_cached(const CachedDecoderGraph& logical, KVCache& cache) const {
        if (!scheduler_) {
            runtime::PlannedAllocationProvider plan(logical.graph);
            const auto result = execute_cached_decoder(logical, cache, nullptr, &plan);
            if (!result.ok()) throw std::runtime_error(result.status.message);
            return {result.outputs.at("logits"), result.counts};
        }
        const auto scheduled = scheduler_->rewrite(logical.graph);
        if (!scheduled.ok()) throw std::runtime_error(scheduled.status.message);
        runtime::ScheduledAllocationProvider plan(*scheduled.graph, *scheduler_);
        const auto result = runtime::execute_graph(*scheduled.graph, nullptr, &plan, nullptr, &*scheduler_);
        const auto status = finalize_cached_decoder(logical, cache, result);
        if (!status.ok()) throw std::runtime_error(status.message);
        return {result.outputs.at("logits"), result.counts};
    }

    const LoadedModel& model_;
    GenerationOptions options_;
#ifdef MODEL_CUDA_BUILD
    std::optional<runtime::CudaBackend> cuda_;
#endif
    std::optional<runtime::Scheduler> scheduler_;
};

} // namespace

GenerationResult generate_greedy(const LoadedModel& model, const std::string& prompt, GenerationOptions options) {
    ByteTokenizer tokenizer(model.config(), model.vocabulary_version);
    GenerationResult generated;
    generated.prompt_tokens = tokenizer.encode({prompt.begin(), prompt.end()});
    if (generated.prompt_tokens.empty()) throw std::invalid_argument("generation prompt is empty after tokenization");
    if (generated.prompt_tokens.size() + static_cast<std::size_t>(options.max_tokens) >
        static_cast<std::size_t>(model.config().max_seq))
        throw std::out_of_range("prompt tokens plus requested generated tokens exceed model max_seq");

    Runner runner(model, options);
    std::vector<std::int32_t> full_tokens = generated.prompt_tokens;
    std::optional<KVCache> cache;
    if (options.use_cache) {
        cache.emplace(model.config());
        generated.cache_persistent_bytes = cache->persistent_bytes();
    }

    auto step = runner.prefill(full_tokens, cache ? &*cache : nullptr);
    generated.prefill_counts = step.counts;
    add_counts(generated.total_counts, step.counts);
    for (std::int64_t iteration = 0; iteration < options.max_tokens; ++iteration) {
        const auto token = greedy_token(step.logits);
        generated.generated_tokens.push_back(token);
        generated.final_logits = step.logits;
        if (token == ByteTokenizer::EOS_ID) break;
        full_tokens.push_back(token);
        step = cache ? runner.decode(token, *cache) : runner.prefill(full_tokens, nullptr);
        add_counts(generated.total_counts, step.counts);
    }
    return generated;
}

TeacherForcedResult run_teacher_forced(const LoadedModel& model, const std::string& prompt,
                                       const std::vector<std::int32_t>& continuation, GenerationOptions options) {
    ByteTokenizer tokenizer(model.config(), model.vocabulary_version);
    auto prompt_tokens = tokenizer.encode({prompt.begin(), prompt.end()});
    if (prompt_tokens.empty()) throw std::invalid_argument("teacher-forced prompt is empty after tokenization");
    if (prompt_tokens.size() + continuation.size() > static_cast<std::size_t>(model.config().max_seq))
        throw std::out_of_range("prompt tokens plus teacher-forced continuation exceed model max_seq");

    Runner runner(model, options);
    std::vector<std::int32_t> full_tokens = prompt_tokens;
    std::optional<KVCache> cache;
    std::size_t cache_bytes = 0;
    if (options.use_cache) {
        cache.emplace(model.config());
        cache_bytes = cache->persistent_bytes();
    }

    const auto prefill_begin = std::chrono::steady_clock::now();
    auto step = runner.prefill(full_tokens, cache ? &*cache : nullptr);
    const auto prefill_end = std::chrono::steady_clock::now();
    (void)greedy_token(step.logits); // First-token latency includes the greedy logits reduction.
    const auto first_token_end = std::chrono::steady_clock::now();
    const auto milliseconds = [](auto begin, auto end) {
        return std::chrono::duration<double, std::milli>(end - begin).count();
    };

    TeacherForcedResult result{std::move(prompt_tokens), step.logits, step.counts, step.counts,
                                cache_bytes, 0.0, 0.0, {}};
    result.prefill_ms = milliseconds(prefill_begin, prefill_end);
    result.first_token_ms = milliseconds(prefill_begin, first_token_end);
    result.decode_ms.reserve(continuation.size());
    for (const auto token : continuation) {
        const auto begin = std::chrono::steady_clock::now();
        full_tokens.push_back(token);
        step = cache ? runner.decode(token, *cache) : runner.prefill(full_tokens, nullptr);
        const auto end = std::chrono::steady_clock::now();
        result.decode_ms.push_back(milliseconds(begin, end));
        result.final_logits = step.logits;
        add_counts(result.total_counts, step.counts);
    }
    return result;
}

} // namespace model
