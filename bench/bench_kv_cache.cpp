#include "model/decoder.hpp"
#include "runtime/planned_executor.hpp"
#include "stage0/build_info.hpp"
#include "stage0/common.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace runtime;
using namespace model;
using Clock = std::chrono::steady_clock;
constexpr int Warmups = 3, Samples = 10;
constexpr std::int64_t ContinuationTokens = 32;
void require(bool value, const char* message) { if (!value) throw std::runtime_error(message); }

std::map<std::string, Tensor> load_parameters(const std::string& path, const DecoderConfig& config) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open tiny weights");
    std::map<std::string, Tensor> tensors;
    for (const auto& spec : parameter_specs(config)) {
        auto tensor = Tensor::allocate_cpu(spec.shape);
        input.read(reinterpret_cast<char*>(tensor.data<float>()), static_cast<std::streamsize>(tensor.nbytes()));
        if (!input) throw std::runtime_error("truncated tiny weights");
        tensors.emplace(spec.name, std::move(tensor));
    }
    require(input.peek() == std::ifstream::traits_type::eof(), "trailing tiny weights");
    return tensors;
}
Tensor ids(const std::vector<std::int32_t>& values) {
    auto result = Tensor::allocate_cpu({static_cast<std::int64_t>(values.size())}, DType::INT32);
    std::copy(values.begin(), values.end(), result.data<std::int32_t>());
    return result;
}
std::vector<std::int32_t> prompt(std::int64_t context) {
    std::vector<std::int32_t> values(static_cast<std::size_t>(context));
    values[0] = 256;
    for (std::int64_t i = 1; i < context; ++i) values[static_cast<std::size_t>(i)] = static_cast<std::int32_t>((i * 37) % 256);
    return values;
}
const std::vector<std::int32_t>& continuation() {
    static const std::vector<std::int32_t> values = {1, 7, 13, 29, 31, 47, 53, 61, 71, 73, 89, 97, 101, 107, 127, 131,
                                                       137, 149, 157, 163, 173, 179, 181, 191, 193, 197, 211, 223, 227, 229, 233, 239};
    return values;
}
double ms(Clock::time_point begin, Clock::time_point end) { return std::chrono::duration<double, std::milli>(end - begin).count(); }
double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    return values.size() % 2 ? values[values.size() / 2] : (values[values.size() / 2 - 1] + values[values.size() / 2]) / 2.0;
}
void json_array(const std::vector<double>& values) {
    std::cout << '[' << std::setprecision(17);
    for (std::size_t i = 0; i < values.size(); ++i) { if (i) std::cout << ','; std::cout << values[i]; }
    std::cout << ']';
}
std::vector<float> logits(const ExecutionResult& result) {
    require(result.ok(), result.status.message.c_str());
    const auto& output = result.outputs.at("logits");
    const auto width = static_cast<std::size_t>(output.shape()[1]);
    return {output.data<float>() + output.numel() - width, output.data<float>() + output.numel()};
}
void compare(const std::vector<float>& actual, const std::vector<float>& expected) {
    require(actual.size() == expected.size(), "cached/full logits shape mismatch");
    for (std::size_t i = 0; i < actual.size(); ++i)
        require(std::isfinite(actual[i]) && std::abs(static_cast<double>(actual[i]) - expected[i]) <=
                2e-6 + 2e-5 * std::abs(expected[i]), "cached/full logits mismatch");
}

std::vector<float> full_once(const DecoderConfig& config, const ParameterTable& parameters,
                             const std::vector<std::int32_t>& tokens) {
    const auto graph = build_decoder_prefill(config, parameters, ids(tokens));
    PlannedAllocationProvider plan(graph.graph);
    return logits(execute_planned(graph.graph, plan));
}
std::vector<float> full_continuation(const DecoderConfig& config, const ParameterTable& parameters,
                                     std::vector<std::int32_t> tokens, const std::vector<std::int32_t>& suffix) {
    std::vector<float> output;
    for (const auto token : suffix) {
        tokens.push_back(token);
        output = full_once(config, parameters, tokens);
    }
    return output;
}

struct CachedRun { std::vector<float> output; double prefill_ms = 0; std::size_t persistent_bytes = 0; std::size_t transient_bytes = 0; };
CachedRun cached_once(const DecoderConfig& config, const ParameterTable& parameters,
                      const std::vector<std::int32_t>& initial, const std::vector<std::int32_t>& tokens,
                      bool measure_decode, double* decode_ms = nullptr) {
    KVCache cache(config);
    const auto prefill_graph = build_decoder_prefill_cached(config, parameters, ids(initial), cache);
    const auto prefill_begin = Clock::now();
    PlannedAllocationProvider prefill_plan(prefill_graph.graph);
    auto prefill = execute_cached_decoder(prefill_graph, cache, nullptr, &prefill_plan);
    const auto prefill_end = Clock::now();
    require(prefill.ok(), prefill.status.message.c_str());
    CachedRun run; run.prefill_ms = ms(prefill_begin, prefill_end); run.persistent_bytes = cache.persistent_bytes();
    auto begin = Clock::now();
    for (const auto token : tokens) {
        const auto graph = build_decoder_decode(config, parameters, ids({token}), cache);
        PlannedAllocationProvider plan(graph.graph);
        auto result = execute_cached_decoder(graph, cache, nullptr, &plan);
        run.output = logits(result);
        run.transient_bytes = std::max(run.transient_bytes, plan.capacity());
    }
    const auto end = Clock::now();
    if (measure_decode && decode_ms) *decode_ms = ms(begin, end) / static_cast<double>(tokens.size());
    return run;
}

void validate_correctness(const DecoderConfig& config, const ParameterTable& parameters, std::int64_t context) {
    auto tokens = prompt(context);
    KVCache cache(config);
    const auto prefill = build_decoder_prefill_cached(config, parameters, ids(tokens), cache);
    PlannedAllocationProvider prefill_plan(prefill.graph);
    require(execute_cached_decoder(prefill, cache, nullptr, &prefill_plan).ok(), "cached prefill failed");
    for (const auto token : continuation()) {
        const auto decode = build_decoder_decode(config, parameters, ids({token}), cache);
        PlannedAllocationProvider plan(decode.graph);
        const auto cached = logits(execute_cached_decoder(decode, cache, nullptr, &plan));
        tokens.push_back(token);
        compare(cached, full_once(config, parameters, tokens));
    }
}

void record(const char* variant, std::int64_t context, int run, const std::vector<double>& samples,
            const CachedRun* cached, bool baseline_first) {
    std::cout << "{\"schema_version\":1,\"stage\":\"stage14-c4\",\"mode\":\"cpu\",\"commit\":"
              << stage0::json_quote(stage0::kCommit) << ",\"source_digest\":" << stage0::json_quote(stage0::kSourceDigest)
              << ",\"source_dirty\":" << (stage0::kSourceDirty ? "true" : "false")
              << ",\"build_type\":" << stage0::json_quote(stage0::kBuildType)
              << ",\"testing\":"
#ifdef RUNTIME_TESTING
              << "true";
#else
              << "false";
#endif
    std::cout << ",\"context\":" << context << ",\"run\":" << run << ",\"variant\":" << stage0::json_quote(variant)
              << ",\"baseline_first\":" << (baseline_first ? "true" : "false") << ",\"warmups\":" << Warmups
              << ",\"samples\":" << Samples << ",\"continuation_tokens\":" << ContinuationTokens
              << ",\"decode_ms_per_token_samples\":";
    json_array(samples);
    std::cout << ",\"decode_ms_per_token_median\":" << median(samples)
              << ",\"tokens_per_second\":" << 1000.0 / median(samples)
              << ",\"correctness\":\"passed\",\"cache_persistent_bytes\":" << (cached ? cached->persistent_bytes : 0)
              << ",\"transient_capacity_bytes\":" << (cached ? cached->transient_bytes : 0);
    if (cached) std::cout << ",\"prefill_ms\":" << cached->prefill_ms;
    std::cout << "}\n";
}

void benchmark(const DecoderConfig& config, const ParameterTable& parameters) {
    for (const auto context : {std::int64_t{128}, std::int64_t{256}, std::int64_t{512}}) {
        validate_correctness(config, parameters, context);
        for (int run = 0; run < 3; ++run) {
            const bool baseline_first = run % 2 == 0;
            std::vector<double> baseline, cached; CachedRun cached_metadata;
            const auto initial = prompt(context);
            const auto warm = [&] {
                (void)full_continuation(config, parameters, initial, continuation());
                (void)cached_once(config, parameters, initial, continuation(), false);
            };
            for (int i = 0; i < Warmups; ++i) warm();
            for (int sample = 0; sample < Samples; ++sample) {
                const auto baseline_one = [&] { const auto b = Clock::now(); (void)full_continuation(config, parameters, initial, continuation()); return ms(b, Clock::now()) / ContinuationTokens; };
                const auto cached_one = [&] { double value = 0; auto meta = cached_once(config, parameters, initial, continuation(), true, &value); cached_metadata = std::move(meta); return value; };
                if (baseline_first) { baseline.push_back(baseline_one()); cached.push_back(cached_one()); }
                else { cached.push_back(cached_one()); baseline.push_back(baseline_one()); }
            }
            record("full_prefix", context, run, baseline, nullptr, baseline_first);
            record("kv_cache", context, run, cached, &cached_metadata, baseline_first);
        }
    }
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--self-test") {
            require(continuation().size() == ContinuationTokens && prompt(512).size() == 512, "Stage 14 benchmark workload declaration");
            std::cout << "Stage 14 KV benchmark workloads: PASS\n";
            return 0;
        }
        if (argc != 3 || std::string(argv[1]) != "--weights")
            throw std::invalid_argument("usage: bench_kv_cache --weights FILE | --self-test");
        const auto config = DecoderConfig::tiny();
        const ParameterTable parameters(config, load_parameters(argv[2], config));
        benchmark(config, parameters);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "bench_kv_cache: " << error.what() << '\n';
        return 1;
    }
}
