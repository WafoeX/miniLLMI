#include "decoder_fixture.hpp"
#include "model/decoder.hpp"
#include "runtime/planned_executor.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
using namespace runtime;
using namespace model;
using decoder_fixture::require;

void success(const Status& status) { if (!status.ok()) throw std::runtime_error(status.message); }

Tensor last_row(const Tensor& logits) {
    const auto vocab = logits.shape()[1];
    return logits.narrow(0, logits.shape()[0] - 1, 1).reshape({1, vocab});
}

Tensor full_prefix_logits(const DecoderConfig& config, const ParameterTable& parameters,
                          const std::vector<std::int32_t>& tokens) {
    const auto full = build_decoder_prefill(config, parameters, decoder_fixture::token_tensor(tokens));
    const auto result = execute_graph(full.graph);
    success(result.status);
    return last_row(result.outputs.at("logits"));
}

void compare_cache_values(const KVCache& cache, const DecoderConfig& config) {
    for (std::int64_t layer = 0; layer < config.layers; ++layer) {
        const auto key = cache.key_range(layer, cache.valid_length());
        const auto value = cache.value_range(layer, cache.valid_length());
        require(key.is_contiguous() && value.is_contiguous(), "KV active ranges are contiguous graph bindings");
        for (std::size_t index = 0; index < key.numel(); ++index)
            require(std::isfinite(key.data<float>()[index]) && std::isfinite(value.data<float>()[index]),
                    "cache write produced finite K/V values");
    }
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::invalid_argument("expected weights fixture");
        const auto config = DecoderConfig::tiny();
        const ParameterTable parameters(config, decoder_fixture::load_parameters(argv[1], config));
        std::vector<std::int32_t> prefix{256, 0, 1, 257};
        KVCache cache(config);

        const auto cached_prefill = build_decoder_prefill_cached(config, parameters, decoder_fixture::token_tensor(prefix), cache);
        require(cached_prefill.cache_position == 0 && cached_prefill.cache_append == 4 &&
                    cached_prefill.graph.order().size() > 185, "cached prefill contains graph-visible cache writes");
        PlannedAllocationProvider prefill_plan(cached_prefill.graph);
        const auto prefill = execute_cached_decoder(cached_prefill, cache, nullptr, &prefill_plan);
        success(prefill.status);
        require(cache.valid_length() == static_cast<std::int64_t>(prefix.size()) && prefill.counts.allocations == 0,
                "successful prefill commits all cache layers without execute allocations");
        const auto prefill_reference = build_decoder_prefill(config, parameters, decoder_fixture::token_tensor(prefix));
        const auto prefill_full = execute_graph(prefill_reference.graph);
        success(prefill_full.status);
        decoder_fixture::compare(prefill.outputs.at("logits"), prefill_full.outputs.at("logits"));
        // An independently executed full prefix is the semantic oracle; every
        // cached continuation step below compares against it.
        compare_cache_values(cache, config);

        const std::vector<std::int32_t> continuation{2, 3, 4, 5};
        for (const auto token : continuation) {
            const auto cached_decode = build_decoder_decode(config, parameters, decoder_fixture::token_tensor({token}), cache);
            require(cached_decode.cache_position == static_cast<std::int64_t>(prefix.size()) &&
                        cached_decode.cache_append == 1, "decode appends exactly at the active position");
            PlannedAllocationProvider decode_plan(cached_decode.graph);
            const auto decoded = execute_cached_decoder(cached_decode, cache, nullptr, &decode_plan);
            success(decoded.status);
            prefix.push_back(token);
            decoder_fixture::compare(decoded.outputs.at("logits"), full_prefix_logits(config, parameters, prefix));
            require(cache.valid_length() == static_cast<std::int64_t>(prefix.size()) && decoded.counts.allocations == 0,
                    "decode commits one token with prepared intermediate storage");
            compare_cache_values(cache, config);
        }

        cache.reset();
        require(cache.valid_length() == 0, "reset invalidates cached prefix");
        const auto invalid = build_decoder_prefill_cached(config, parameters, decoder_fixture::token_tensor({static_cast<std::int32_t>(config.vocab)}), cache);
        const auto failed = execute_cached_decoder(invalid, cache);
        require(!failed.ok() && failed.status.code == StatusCode::OutOfRange && cache.valid_length() == 0,
                "failed prefill invalidates cache rather than publishing partial state");

        cache.commit_append(0, config.max_seq);
        try {
            (void)build_decoder_decode(config, parameters, decoder_fixture::token_tensor({0}), cache);
            throw std::runtime_error("decode accepted a full cache");
        } catch (const std::out_of_range&) {}
        std::cout << "Stage 14 C2/C3 cached prefill/decode: PASS steps=" << continuation.size()
                  << " cache_bytes=" << cache.persistent_bytes() << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "test_kv_decoder: " << error.what() << '\n';
        return 1;
    }
}
