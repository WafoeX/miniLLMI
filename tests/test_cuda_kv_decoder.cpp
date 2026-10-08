#include "decoder_fixture.hpp"
#include "model/decoder.hpp"
#include "runtime/cuda_backend.hpp"
#include "runtime/scheduler.hpp"

#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
using namespace runtime;
using namespace model;
void require(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
void success(const Status& status) { if (!status.ok()) throw std::runtime_error(status.message); }

Tensor last_row(const Tensor& tensor) { return tensor.narrow(0, tensor.shape()[0] - 1, 1).reshape({1, tensor.shape()[1]}); }
ExecutionResult run_scheduled(const CachedDecoderGraph& logical, KVCache& cache, Scheduler& scheduler) {
    const auto rewritten = scheduler.rewrite(logical.graph);
    success(rewritten.status);
    ScheduledAllocationProvider plan(*rewritten.graph, scheduler);
    auto result = execute_graph(*rewritten.graph, nullptr, &plan, nullptr, &scheduler);
    success(finalize_cached_decoder(logical, cache, result));
    return result;
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::invalid_argument("expected weights fixture");
        const auto config = DecoderConfig::tiny();
        const ParameterTable parameters(config, decoder_fixture::load_parameters(argv[1], config));
        CudaBackend cuda(0, CudaMatmul::Stage0Naive);
        Scheduler scheduler(default_cpu_backend(), &cuda);
        KVCache cache(config);
        std::vector<std::int32_t> tokens{256, 0, 1, 257};
        const auto options = DecoderBuildOptions{cuda.device()};
        const auto prefill = build_decoder_prefill_cached(config, parameters, decoder_fixture::token_tensor(tokens), cache, options);
        const auto prefill_result = run_scheduled(prefill, cache, scheduler);
        require(cache.valid_length() == 4 && prefill_result.counts.copies > 0, "mixed cached prefill cache/copy accounting");
        const auto decode = build_decoder_decode(config, parameters, decoder_fixture::token_tensor({2}), cache, options);
        const auto mixed = run_scheduled(decode, cache, scheduler);
        tokens.push_back(2);
        const auto full = build_decoder_prefill(config, parameters, decoder_fixture::token_tensor(tokens));
        const auto reference = execute_graph(full.graph);
        success(reference.status);
        decoder_fixture::compare(mixed.outputs.at("logits"), last_row(reference.outputs.at("logits")), 2e-4, 2e-3);
        require(cache.valid_length() == 5 && mixed.counts.copies > 0 && mixed.counts.copy_bytes > 0,
                "mixed cached decode commits with explicit copies");
        std::cout << "Stage 14 mixed cached decode: PASS copies=" << mixed.counts.copies << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "test_cuda_kv_decoder: " << error.what() << '\n';
        return 1;
    }
}
