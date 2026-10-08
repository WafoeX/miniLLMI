#include "model/decoder.hpp"
#include "runtime/planned_executor.hpp"

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

Tensor load_expected(const std::string& path, const DecoderConfig& config) {
    auto expected = Tensor::allocate_cpu({4, config.vocab});
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open logits fixture");
    input.read(reinterpret_cast<char*>(expected.data<float>()), static_cast<std::streamsize>(expected.nbytes()));
    require(static_cast<bool>(input) && input.peek() == std::ifstream::traits_type::eof(), "invalid logits fixture size");
    return expected;
}

Tensor ids(std::int64_t length) {
    static constexpr std::int32_t values[] = {256, 0, 1, 257};
    auto result = Tensor::allocate_cpu({length}, DType::INT32);
    for (std::int64_t index = 0; index < length; ++index)
        result.data<std::int32_t>()[index] = values[index % 4];
    return result;
}

void compare(const Tensor& actual, const Tensor& expected) {
    require(actual.shape() == expected.shape(), "logits shape mismatch");
    for (std::size_t index = 0; index < actual.numel(); ++index) {
        const auto value = actual.data<float>()[index], reference = expected.data<float>()[index];
        require(std::isfinite(value) && std::abs(static_cast<double>(value) - reference) <=
                    2e-6 + 2e-5 * std::abs(reference), "logits fixture mismatch");
    }
}

double milliseconds(Clock::time_point begin, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - begin).count();
}

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}
} // namespace

int main(int argc, char** argv) {
    try {
        std::string weights, expected_path;
        for (int index = 1; index < argc; ++index) {
            const std::string argument = argv[index];
            if (argument == "--weights" && index + 1 < argc) weights = argv[++index];
            else if (argument == "--expected" && index + 1 < argc) expected_path = argv[++index];
            else throw std::invalid_argument("usage: bench_decoder --weights FILE --expected FILE");
        }
        if (weights.empty() || expected_path.empty()) throw std::invalid_argument("weights and expected logits are required");
        const auto config = DecoderConfig::tiny();
        const ParameterTable parameters(config, load_parameters(weights, config));
        const auto expected = load_expected(expected_path, config);
        auto token_ids = ids(4);

        const auto prepare_begin = Clock::now();
        const auto decoder = build_decoder_prefill(config, parameters, token_ids);
        PlannedAllocationProvider prepared(decoder.graph);
        const auto prepare_end = Clock::now();
        const auto dynamic = execute_graph(decoder.graph);
        require(dynamic.ok(), dynamic.status.message.c_str());
        compare(dynamic.outputs.at("logits"), expected);

        for (int warmup = 0; warmup < 3; ++warmup) {
            const auto run = execute_planned(decoder.graph, prepared);
            require(run.ok(), run.status.message.c_str());
        }
        std::vector<double> execution_samples;
        ExecutionCounts planned_counts;
        for (int sample = 0; sample < 10; ++sample) {
            const auto begin = Clock::now();
            const auto run = execute_planned(decoder.graph, prepared);
            const auto end = Clock::now();
            require(run.ok(), run.status.message.c_str());
            compare(run.outputs.at("logits"), expected);
            planned_counts = run.counts;
            execution_samples.push_back(milliseconds(begin, end));
        }
        std::vector<double> shape_change_samples;
        for (int sample = 0; sample < 10; ++sample) {
            const auto begin = Clock::now();
            const auto rebuilt = build_decoder_prefill(config, parameters, ids(4));
            PlannedAllocationProvider replanned(rebuilt.graph);
            const auto run = execute_planned(rebuilt.graph, replanned);
            const auto end = Clock::now();
            require(run.ok(), run.status.message.c_str());
            compare(run.outputs.at("logits"), expected);
            shape_change_samples.push_back(milliseconds(begin, end));
        }
        require(planned_counts.allocations == 0 && planned_counts.frees == 0,
                "planned decoder unexpectedly allocated backing storage during execute");
        std::cout << std::setprecision(17)
                  << "{\"schema_version\":1,\"stage\":\"stage13-c3\",\"mode\":\"cpu\","
                  << "\"sequence_length\":4,\"warmups\":3,\"samples\":10,"
                  << "\"prepare_ms\":" << milliseconds(prepare_begin, prepare_end) << ','
                  << "\"execute_median_ms\":" << median(execution_samples) << ','
                  << "\"shape_change_end_to_end_median_ms\":" << median(shape_change_samples) << ','
                  << "\"nodes\":" << decoder.graph.order().size() << ','
                  << "\"dynamic_allocations\":" << dynamic.counts.allocations << ','
                  << "\"planned_execute_allocations\":" << planned_counts.allocations << ','
                  << "\"planned_peak_live_bytes\":" << planned_counts.peak_live_bytes << ','
                  << "\"planned_capacity_bytes\":" << prepared.capacity() << ','
                  << "\"correctness\":\"passed\"}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "bench_decoder: " << error.what() << '\n';
        return 1;
    }
}
