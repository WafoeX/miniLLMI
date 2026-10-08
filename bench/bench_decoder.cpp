#include "model/decoder.hpp"
#include "runtime/planned_executor.hpp"
#include "stage0/build_info.hpp"
#include "stage0/common.hpp"
#ifdef DECODER_CUDA_BUILD
#include "runtime/cuda_backend.hpp"
#include "runtime/scheduler.hpp"
#endif

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
    for (std::int64_t index = 0; index < length; ++index) result.data<std::int32_t>()[index] = values[index % 4];
    return result;
}

void compare(const Tensor& actual, const Tensor& expected, double atol, double rtol) {
    require(actual.shape() == expected.shape(), "logits shape mismatch");
    for (std::size_t index = 0; index < actual.numel(); ++index) {
        const auto value = actual.data<float>()[index], reference = expected.data<float>()[index];
        require(std::isfinite(value) && std::abs(static_cast<double>(value) - reference) <=
                    atol + rtol * std::abs(reference), "logits fixture mismatch");
    }
}

double milliseconds(Clock::time_point begin, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - begin).count();
}
double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}
#ifdef DECODER_CUDA_BUILD
std::size_t capacity(const MemoryPlan& plan, Device device) {
    const auto found = plan.device_capacity_bytes.find(device);
    return found == plan.device_capacity_bytes.end() ? 0 : found->second;
}
#endif

void metadata(const char* mode) {
    std::cout << "{\"schema_version\":1,\"stage\":\"stage13-c4\",\"mode\":" << stage0::json_quote(mode)
              << ",\"commit\":" << stage0::json_quote(stage0::kCommit)
              << ",\"source_digest\":" << stage0::json_quote(stage0::kSourceDigest)
              << ",\"source_dirty\":" << (stage0::kSourceDirty ? "true" : "false")
              << ",\"build_type\":" << stage0::json_quote(stage0::kBuildType)
              << ",\"compiler\":" << stage0::json_quote(stage0::kCompiler)
              << ",\"cuda_compiler\":" << stage0::json_quote(stage0::kCudaCompiler)
              << ",\"cuda_architecture\":" << stage0::json_quote(stage0::kCudaArchitectures)
              << ",\"testing\":";
#ifdef RUNTIME_TESTING
    std::cout << "true";
#else
    std::cout << "false";
#endif
    std::cout << ",\"cuda_enabled\":";
#ifdef DECODER_CUDA_BUILD
    std::cout << "true";
#else
    std::cout << "false";
#endif
}

void cpu_benchmark(const DecoderConfig& config, const ParameterTable& parameters, const Tensor& expected) {
    auto token_ids = ids(4);
    const auto prepare_begin = Clock::now();
    const auto decoder = build_decoder_prefill(config, parameters, token_ids);
    PlannedAllocationProvider prepared(decoder.graph);
    const auto prepare_end = Clock::now();
    const auto dynamic = execute_graph(decoder.graph);
    require(dynamic.ok(), dynamic.status.message.c_str());
    compare(dynamic.outputs.at("logits"), expected, 2e-6, 2e-5);
    for (int warmup = 0; warmup < Warmups; ++warmup) {
        const auto run = execute_planned(decoder.graph, prepared);
        require(run.ok(), run.status.message.c_str());
    }
    std::vector<double> execution_samples;
    ExecutionCounts counts;
    for (int sample = 0; sample < Samples; ++sample) {
        const auto begin = Clock::now();
        const auto run = execute_planned(decoder.graph, prepared);
        const auto end = Clock::now();
        require(run.ok(), run.status.message.c_str());
        compare(run.outputs.at("logits"), expected, 2e-6, 2e-5);
        counts = run.counts;
        execution_samples.push_back(milliseconds(begin, end));
    }
    std::vector<double> shape_change_samples;
    for (int sample = 0; sample < Samples; ++sample) {
        const auto begin = Clock::now();
        const auto rebuilt = build_decoder_prefill(config, parameters, ids(4));
        PlannedAllocationProvider replanned(rebuilt.graph);
        const auto run = execute_planned(rebuilt.graph, replanned);
        const auto end = Clock::now();
        require(run.ok(), run.status.message.c_str());
        compare(run.outputs.at("logits"), expected, 2e-6, 2e-5);
        shape_change_samples.push_back(milliseconds(begin, end));
    }
    require(counts.allocations == 0 && counts.frees == 0, "planned CPU execute allocated backing storage");
    metadata("cpu");
    std::cout << std::setprecision(17)
              << ",\"sequence_length\":4,\"warmups\":" << Warmups << ",\"samples\":" << Samples
              << ",\"prepare_ms\":" << milliseconds(prepare_begin, prepare_end)
              << ",\"execute_median_ms\":" << median(execution_samples)
              << ",\"shape_change_end_to_end_median_ms\":" << median(shape_change_samples)
              << ",\"nodes\":" << decoder.graph.order().size()
              << ",\"dynamic_allocations\":" << dynamic.counts.allocations
              << ",\"execute_allocations\":" << counts.allocations
              << ",\"peak_live_bytes\":" << counts.peak_live_bytes
              << ",\"cpu_capacity_bytes\":" << prepared.capacity()
              << ",\"cuda_capacity_bytes\":0,\"copies\":0,\"copy_bytes\":0"
              << ",\"backend_dispatches\":" << counts.backend_dispatches
              << ",\"backend_switches\":" << counts.backend_switches
              << ",\"correctness\":\"passed\"}\n";
}

#ifdef DECODER_CUDA_BUILD
void mixed_benchmark(const DecoderConfig& config, const ParameterTable& parameters, const Tensor& expected) {
    CudaBackend cuda(0, CudaMatmul::Stage0Naive);
    Scheduler scheduler(default_cpu_backend(), &cuda);
    const auto prepare_begin = Clock::now();
    const auto logical = build_decoder_prefill(config, parameters, ids(4), {cuda.device()});
    const auto scheduled = scheduler.rewrite(logical.graph);
    require(scheduled.ok(), scheduled.status.message.c_str());
    ScheduledAllocationProvider prepared(*scheduled.graph, scheduler);
    const auto prepare_end = Clock::now();
    std::size_t cuda_placements = 0;
    for (const auto& item : scheduled.placements) {
        const auto& descriptor = logical.graph.nodes().at(item.first).descriptor;
        if (descriptor.code() == OpCode::MATMUL && descriptor.backend_hint() &&
            item.second.device == cuda.device()) ++cuda_placements;
    }
    require(cuda_placements == logical.projection_nodes && cuda_placements == 21,
            "every learned projection must execute on CUDA");
    require(!scheduled.inserted_copies.empty(), "mixed decoder requires explicit scheduler copies");
    for (int warmup = 0; warmup < Warmups; ++warmup) {
        const auto run = execute_graph(*scheduled.graph, nullptr, &prepared, nullptr, &scheduler);
        require(run.ok(), run.status.message.c_str());
    }
    std::vector<double> execution_samples;
    ExecutionCounts counts;
    for (int sample = 0; sample < Samples; ++sample) {
        const auto begin = Clock::now();
        const auto run = execute_graph(*scheduled.graph, nullptr, &prepared, nullptr, &scheduler);
        const auto end = Clock::now();
        require(run.ok(), run.status.message.c_str());
        compare(run.outputs.at("logits"), expected, 2e-4, 2e-3);
        counts = run.counts;
        execution_samples.push_back(milliseconds(begin, end));
    }
    std::vector<double> shape_change_samples;
    for (int sample = 0; sample < Samples; ++sample) {
        const auto begin = Clock::now();
        const auto rebuilt = build_decoder_prefill(config, parameters, ids(4), {cuda.device()});
        const auto rewritten = scheduler.rewrite(rebuilt.graph);
        require(rewritten.ok(), rewritten.status.message.c_str());
        ScheduledAllocationProvider replanned(*rewritten.graph, scheduler);
        const auto run = execute_graph(*rewritten.graph, nullptr, &replanned, nullptr, &scheduler);
        const auto end = Clock::now();
        require(run.ok(), run.status.message.c_str());
        compare(run.outputs.at("logits"), expected, 2e-4, 2e-3);
        shape_change_samples.push_back(milliseconds(begin, end));
    }
    require(counts.allocations == 0 && counts.frees == 0 && counts.copies > 0 && counts.copy_bytes > 0,
            "mixed planned execute/copy accounting changed");
    metadata("mixed");
    std::cout << std::setprecision(17)
              << ",\"sequence_length\":4,\"warmups\":" << Warmups << ",\"samples\":" << Samples
              << ",\"prepare_ms\":" << milliseconds(prepare_begin, prepare_end)
              << ",\"execute_median_ms\":" << median(execution_samples)
              << ",\"shape_change_end_to_end_median_ms\":" << median(shape_change_samples)
              << ",\"nodes\":" << scheduled.graph->order().size()
              << ",\"logical_nodes\":" << logical.graph.order().size()
              << ",\"cuda_projection_nodes\":" << cuda_placements
              << ",\"inserted_copy_nodes\":" << scheduled.inserted_copies.size()
              << ",\"dynamic_allocations\":null,\"execute_allocations\":" << counts.allocations
              << ",\"peak_live_bytes\":" << counts.peak_live_bytes
              << ",\"cpu_capacity_bytes\":" << capacity(prepared.plan(), Device{})
              << ",\"cuda_capacity_bytes\":" << capacity(prepared.plan(), cuda.device())
              << ",\"copies\":" << counts.copies << ",\"copy_bytes\":" << counts.copy_bytes
              << ",\"backend_dispatches\":" << counts.backend_dispatches
              << ",\"backend_switches\":" << counts.backend_switches
              << ",\"correctness\":\"passed\"}\n";
}
#endif
} // namespace

int main(int argc, char** argv) {
    try {
        std::string weights, expected_path, mode = "cpu";
        for (int index = 1; index < argc; ++index) {
            const std::string argument = argv[index];
            if (argument == "--weights" && index + 1 < argc) weights = argv[++index];
            else if (argument == "--expected" && index + 1 < argc) expected_path = argv[++index];
            else if (argument == "--mode" && index + 1 < argc) mode = argv[++index];
            else throw std::invalid_argument("usage: bench_decoder --weights FILE --expected FILE [--mode cpu|mixed]");
        }
        if (weights.empty() || expected_path.empty()) throw std::invalid_argument("weights and expected logits are required");
        const auto config = DecoderConfig::tiny();
        const ParameterTable parameters(config, load_parameters(weights, config));
        const auto expected = load_expected(expected_path, config);
        if (mode == "cpu") cpu_benchmark(config, parameters, expected);
        else if (mode == "mixed") {
#ifdef DECODER_CUDA_BUILD
            mixed_benchmark(config, parameters, expected);
#else
            throw std::runtime_error("mixed mode requires ENABLE_CUDA=ON");
#endif
        } else throw std::invalid_argument("mode must be cpu or mixed");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "bench_decoder: " << error.what() << '\n';
        return 1;
    }
}
