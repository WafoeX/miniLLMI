#include "runtime/scheduler.hpp"
#include "runtime/graph_executor.hpp"
#include "stage0/build_info.hpp"
#include "stage0/common.hpp"
#include "scheduler_workload.hpp"
#include <cmath>
#include <iostream>
#ifdef SCHEDULER_CUDA_BUILD
#include "runtime/cuda_backend.hpp"
#include "scheduler_artifacts.hpp"
#include <chrono>
#endif

namespace {
using namespace runtime;
constexpr int Size = 64;
#ifdef SCHEDULER_CUDA_BUILD
constexpr int Warmups = 3, Samples = 10, Batch = 5;
constexpr const char* Boundary = "execute+counts+full_output_check+output_release;synchronous;prepare_oracle_trace_io_excluded";
#endif
std::string input_id(const scheduler_workload::Inputs& in) {
    std::vector<float> values;
    for (const auto* t : {&in.x, &in.bias, &in.weight, &in.scale})
        values.insert(values.end(), t->data<float>(), t->data<float>() + t->numel());
    return stage0::input_hash(values, {});
}
double validate(const ExecutionResult& result, const Tensor& oracle, bool mixed) {
    scheduler_workload::check(result.status);
    const auto& c = result.counts;
    if (c.allocations || c.backend_switches != (mixed ? 2u : 0u) || c.copies != (mixed ? 4u : 0u) ||
        c.copy_bytes != (mixed ? 65536u : 0u) || c.backend_dispatches != (mixed ? 9u : 5u))
        throw std::runtime_error("frozen scheduler workload execution counters changed");
    if (mixed && (c.peak_live_bytes != 65536 || c.live_bytes != 16384))
        throw std::runtime_error("graph-owned live memory accounting changed");
    const auto& value = result.outputs.at("result");
    if (value.shape() != oracle.shape() || value.device() != Device{}) throw std::runtime_error("wrong CPU boundary output");
    double checksum = 0;
    for (std::size_t i = 0; i < value.numel(); ++i) {
        const auto actual = value.data<float>()[i], expected = oracle.data<float>()[i];
        if (!std::isfinite(actual) || std::abs(actual - expected) > .001F + .001F * std::abs(expected))
            throw std::runtime_error("full CPU oracle mismatch");
        checksum += actual;
    }
    return checksum;
}
void probe() {
    std::cout << "{\"schema_version\":1,\"benchmark\":\"scheduler-c4-v1\",\"commit\":" << stage0::json_quote(stage0::kCommit)
        << ",\"source_digest\":" << stage0::json_quote(stage0::kSourceDigest) << ",\"source_dirty\":" << (stage0::kSourceDirty ? "true" : "false")
        << ",\"build_type\":" << stage0::json_quote(stage0::kBuildType) << ",\"compiler\":" << stage0::json_quote(stage0::kCompiler)
        << ",\"cuda_compiler\":" << stage0::json_quote(stage0::kCudaCompiler) << ",\"cuda_architecture\":" << stage0::json_quote(stage0::kCudaArchitectures)
        << ",\"testing\":"
#ifdef RUNTIME_TESTING
        << "true"
#else
        << "false"
#endif
        << ",\"cuda_enabled\":"
#ifdef SCHEDULER_CUDA_BUILD
        << "true"
#else
        << "false"
#endif
        << ",\"input_hash\":" << stage0::json_quote(input_id(scheduler_workload::inputs(Size, Size, Size))) << "}\n";
}
void self_test() {
    const auto in = scheduler_workload::inputs(Size, Size, Size);
    const auto reference = execute_graph(scheduler_workload::graph(in, false, true)); scheduler_workload::check(reference.status);
#ifdef SCHEDULER_CUDA_BUILD
    CudaBackend cuda(0, CudaMatmul::Stage0Naive);
    Scheduler scheduler(default_cpu_backend(), &cuda);
    const auto automatic = scheduler.rewrite(scheduler_workload::graph(in, false)); scheduler_workload::check(automatic.status);
    const auto manual = scheduler_workload::graph(in, true);
    for (const auto* graph : {&*automatic.graph, &manual}) {
        ScheduledAllocationProvider provider(*graph, scheduler);
        const auto result = execute_graph(*graph, nullptr, &provider, nullptr, &scheduler);
        (void)validate(result, reference.outputs.at("result"), true);
    }
    std::cout << "scheduler workload self-test PASS (untimed mixed CUDA)\n";
#else
    Scheduler scheduler;
    const auto automatic = scheduler.rewrite(scheduler_workload::graph(in, false)); scheduler_workload::check(automatic.status);
    ScheduledAllocationProvider provider(*automatic.graph, scheduler);
    for (int repeat = 0; repeat < 2; ++repeat) {
        const auto result = execute_graph(*automatic.graph, nullptr, &provider, nullptr, &scheduler);
        (void)validate(result, reference.outputs.at("result"), false);
    }
    std::cout << "scheduler workload self-test PASS (untimed CPU fallback, NOT CUDA evidence)\n";
#endif
}
#ifdef SCHEDULER_CUDA_BUILD
void benchmark(const std::filesystem::path& output, const std::string& repeat, const std::string& order) {
    if (std::string(stage0::kBuildType) != "Release" || stage0::kSourceDirty) throw std::runtime_error("timing requires clean Release source");
#ifdef RUNTIME_TESTING
    throw std::runtime_error("timing requires BUILD_TESTING=OFF");
#endif
    if (!std::filesystem::create_directory(output)) throw std::invalid_argument("benchmark output directory must be new");
    const auto in = scheduler_workload::inputs(Size, Size, Size); const auto hash = input_id(in);
    const auto reference = execute_graph(scheduler_workload::graph(in, false, true)); scheduler_workload::check(reference.status);
    const auto& oracle = reference.outputs.at("result");
    CudaBackend cuda(0, CudaMatmul::Stage0Naive); Scheduler scheduler(default_cpu_backend(), &cuda);
    const auto automatic = scheduler.rewrite(scheduler_workload::graph(in, false)); scheduler_workload::check(automatic.status);
    const auto manual = scheduler_workload::graph(in, true);
    ScheduledAllocationProvider ap(*automatic.graph, scheduler), mp(manual, scheduler);
    const auto resident = 4 * in.x.nbytes() + oracle.nbytes() + ap.capacity() + mp.capacity();
    const std::vector<std::string> columns = {"schema_version", "repeat", "order", "mode", "sample", "ms", "warmup", "samples", "batch", "m", "k", "n", "copies", "copy_bytes", "backend_dispatches", "backend_switches", "allocations", "peak_live_bytes", "live_bytes", "cpu_capacity", "cuda_capacity", "external_bytes", "oracle_bytes", "workspace_bytes", "runtime_tensor_resident_bytes", "input_hash", "checksum", "commit", "source_digest", "source_dirty", "build_type", "compiler", "cuda_compiler", "cuda_architecture", "cpu_backend", "cuda_backend", "timing_boundary", "correctness"};
    const auto modes = order == "manual,automatic" ? std::vector<std::string>{"manual", "automatic"} : std::vector<std::string>{"automatic", "manual"};
    for (const auto& mode : modes) {
        const auto& graph = mode == "manual" ? manual : *automatic.graph;
        auto& provider = mode == "manual" ? mp : ap;
        const auto capture = [&](const std::string& phase) {
            ExecutionTrace trace;
            const auto result = execute_graph(graph, &trace, &provider, nullptr, &scheduler);
            (void)validate(result, oracle, true);
            if (trace.dropped()) throw std::runtime_error("incomplete scheduler diagnostic trace");
            scheduler_artifacts::snapshot(output, mode + "-" + phase, graph, provider.plan(), result, oracle, trace);
        };
        struct Observation { double checksum; ExecutionCounts counts; };
        const auto once = [&]() {
            const auto result = execute_graph(graph, nullptr, &provider, nullptr, &scheduler);
            return Observation{validate(result, oracle, true), result.counts}; // output release before returning
        };
        capture("initial");
        for (int i = 0; i < Warmups; ++i) (void)once();
        for (int sample = 0; sample < Samples; ++sample) {
            double checksum = 0; ExecutionCounts actual;
            const auto start = std::chrono::steady_clock::now();
            for (int i = 0; i < Batch; ++i) { const auto observed = once(); checksum += observed.checksum; actual = observed.counts; }
            const auto stop = std::chrono::steady_clock::now();
            const auto ms = std::chrono::duration<double, std::milli>(stop - start).count() / Batch;
            stage0::Record row{{"schema_version", "1"}, {"repeat", repeat}, {"order", order}, {"mode", mode}, {"sample", std::to_string(sample)},
                {"ms", stage0::number(ms)}, {"warmup", std::to_string(Warmups)}, {"samples", std::to_string(Samples)}, {"batch", std::to_string(Batch)},
                {"m", std::to_string(Size)}, {"k", std::to_string(Size)}, {"n", std::to_string(Size)},
                {"copies", std::to_string(actual.copies)}, {"copy_bytes", std::to_string(actual.copy_bytes)},
                {"backend_dispatches", std::to_string(actual.backend_dispatches)}, {"backend_switches", std::to_string(actual.backend_switches)},
                {"allocations", std::to_string(actual.allocations)}, {"peak_live_bytes", std::to_string(actual.peak_live_bytes)},
                {"live_bytes", std::to_string(actual.live_bytes)},
                {"cpu_capacity", std::to_string(provider.plan().device_capacity_bytes.at(Device{}))},
                {"cuda_capacity", std::to_string(provider.plan().device_capacity_bytes.at(cuda.device()))},
                {"external_bytes", std::to_string(4 * in.x.nbytes())}, {"oracle_bytes", std::to_string(oracle.nbytes())}, {"workspace_bytes", "0"},
                {"runtime_tensor_resident_bytes", std::to_string(resident)}, {"input_hash", hash}, {"checksum", stage0::number(checksum / Batch)},
                {"commit", stage0::kCommit}, {"source_digest", stage0::kSourceDigest}, {"source_dirty", "false"}, {"build_type", stage0::kBuildType},
                {"compiler", stage0::kCompiler}, {"cuda_compiler", stage0::kCudaCompiler}, {"cuda_architecture", stage0::kCudaArchitectures},
                {"cpu_backend", scheduler.cpu().name()}, {"cuda_backend", cuda.name()}, {"timing_boundary", Boundary}, {"correctness", "passed"}};
            stage0::append_csv(output / "samples.csv", columns, row);
        }
        capture("final");
    }
}
#endif
} // namespace
int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--probe") { probe(); return 0; }
        if (argc == 2 && std::string(argv[1]) == "--self-test") { self_test(); return 0; }
#ifndef SCHEDULER_CUDA_BUILD
        throw std::runtime_error("scheduler timing is T4-only; build ENABLE_CUDA=ON");
#else
        std::filesystem::path output; std::string repeat, order;
        for (int i = 1; i < argc; i += 2) {
            if (i + 1 == argc) throw std::invalid_argument("missing argument value");
            const std::string key = argv[i];
            if (key == "--output") output = argv[i + 1];
            else if (key == "--repeat") repeat = argv[i + 1];
            else if (key == "--order") order = argv[i + 1];
            else throw std::invalid_argument("unknown argument");
        }
        if (output.empty() || (repeat != "1" && repeat != "2" && repeat != "3") || order != (repeat == "2" ? "automatic,manual" : "manual,automatic"))
            throw std::invalid_argument("usage: bench_scheduler --output NEWDIR --repeat 1|2|3 --order manual,automatic|automatic,manual");
        benchmark(output, repeat, order); return 0;
#endif
    } catch (const std::exception& e) { std::cerr << "bench_scheduler: " << e.what() << '\n'; return 1; }
}
