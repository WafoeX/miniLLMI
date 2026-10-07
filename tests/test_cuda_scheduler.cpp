#include "runtime/cuda_backend.hpp"
#include "runtime/scheduler.hpp"
#include "runtime/graph_executor.hpp"
#include "../bench/scheduler_workload.hpp"
#include <cmath>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <stdexcept>

namespace {
using namespace runtime;
void require(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
void equal(const Tensor& value, const Tensor& reference) {
    require(value.shape() == reference.shape(), "shape equality");
    for (std::size_t i = 0; i < value.numel(); ++i) {
        const auto actual = value.data<float>()[i], expected = reference.data<float>()[i];
        require(std::isfinite(actual) && std::abs(actual - expected) <= .001F + .001F * std::abs(expected), "mixed values agree with CPU reference");
    }
}
void snapshot(const std::filesystem::path& directory, const std::string& label, const Graph& graph,
              const MemoryPlan& plan, const ExecutionResult& result, const Tensor& oracle, const ExecutionTrace& trace) {
    if (directory.empty()) return;
    std::ofstream out(directory / (label + ".json")); out.exceptions(std::ios::failbit | std::ios::badbit);
    out << std::setprecision(std::numeric_limits<float>::max_digits10);
    out << "{\"schema_version\":1,\"atol\":0.001,\"rtol\":0.001,\"counts\":{\"copies\":" << result.counts.copies
        << ",\"copy_bytes\":" << result.counts.copy_bytes << ",\"backend_dispatches\":" << result.counts.backend_dispatches
        << ",\"backend_switches\":" << result.counts.backend_switches << ",\"allocations\":" << result.counts.allocations
        << "},\"workspace_bytes\":0,\"device_capacities\":[";
    bool first = true;
    for (const auto& item : plan.device_capacity_bytes) {
        if (!first) out << ','; first = false;
        out << "{\"device\":\"" << (item.first.type() == DeviceType::CPU ? "cpu:" : "cuda:") << item.first.index()
            << "\",\"bytes\":" << item.second << '}';
    }
    out << "],\"nodes\":["; first = true;
    for (auto node : graph.order()) {
        if (!first) out << ','; first = false;
        out << "{\"id\":" << node << ",\"descriptor\":" << graph.nodes().at(node).descriptor.serialize() << '}';
    }
    out << "],\"trace_dropped\":" << trace.dropped() << ",\"trace\":["; first = true;
    for (const auto& event : trace.events()) {
        if (!first) out << ','; first = false;
        out << "{\"kind\":\"" << trace_name(event.kind) << "\",\"bytes\":" << event.bytes << ",\"node\":";
        if (event.node) out << *event.node; else out << "null";
        out << ",\"tensor\":"; if (event.tensor) out << *event.tensor; else out << "null";
        if (event.metadata) out << ",\"device\":\"" << (event.metadata->device.type() == DeviceType::CPU ? "cpu:" : "cuda:")
            << event.metadata->device.index() << "\",\"offset_bytes\":" << event.metadata->offset_bytes << ",\"capacity_bytes\":" << event.metadata->capacity_bytes;
        out << '}';
    }
    const auto values = [&](const Tensor& tensor) {
        out << '['; for (std::size_t i = 0; i < tensor.numel(); ++i) { if (i) out << ','; out << tensor.data<float>()[i]; } out << ']';
    };
    out << "],\"output\":"; values(result.outputs.at("result"));
    out << ",\"cpu_reference\":"; values(oracle); out << "}\n";
}
void mixed(const CudaBackend& cuda, const std::filesystem::path& artifacts) {
    Scheduler scheduler(default_cpu_backend(), &cuda);
    for (const auto& shape : {Shape{17, 13, 11}, Shape{64, 64, 64}, Shape{0, 3, 2}, Shape{2, 0, 3}}) {
        const auto in = scheduler_workload::inputs(shape[0], shape[1], shape[2]);
        const auto logical = scheduler_workload::graph(in, false);
        const auto rewritten = scheduler.rewrite(logical); require(rewritten.ok(), "rewrite");
        auto manual = scheduler_workload::graph(in, true);
        const auto oracle = execute_graph(scheduler_workload::graph(in, false, true)); require(oracle.ok(), "CPU reference");
        ScheduledAllocationProvider automatic(*rewritten.graph, scheduler), baseline(manual, scheduler);
        ExecutionCounts a, b;
        for (int run = 0; run < 2; ++run) {
            ExecutionTrace trace, manual_trace;
            const auto result = execute_graph(*rewritten.graph, &trace, &automatic, nullptr, &scheduler);
            const auto reference = execute_graph(manual, &manual_trace, &baseline, nullptr, &scheduler);
            require(result.ok() && reference.ok(), "manual and automatic execute");
            equal(result.outputs.at("result"), oracle.outputs.at("result")); equal(reference.outputs.at("result"), oracle.outputs.at("result"));
            a = result.counts; b = reference.counts;
            require(a.copies == b.copies && a.copy_bytes == b.copy_bytes && a.backend_switches == b.backend_switches, "actual counters match manual baseline");
            require(a.allocations == 0 && b.allocations == 0 && trace.dropped() == 0, "planned execute no backing allocations / complete trace");
            std::size_t copy_count = 0, copy_bytes = 0, switches = 0; std::optional<TraceKind> previous;
            for (const auto& event : trace.events()) {
                if (event.kind == TraceKind::Copy) { ++copy_count; copy_bytes += event.bytes; }
                if (event.kind == TraceKind::BackendCPU || event.kind == TraceKind::BackendCUDA) {
                    if (previous && *previous != event.kind) ++switches;
                    previous = event.kind;
                }
            }
            require(copy_count == a.copies && copy_bytes == a.copy_bytes && switches == a.backend_switches, "trace independently agrees with executed counts");
            const auto label = std::string(cuda.name()) + "-" + std::to_string(shape[0]) + "-" + std::to_string(shape[1]) + "-" + std::to_string(shape[2]) + "-repeat" + std::to_string(run);
            snapshot(artifacts, label + "-automatic", *rewritten.graph, automatic.plan(), result, oracle.outputs.at("result"), trace);
            snapshot(artifacts, label + "-manual", manual, baseline.plan(), reference, oracle.outputs.at("result"), manual_trace);
            std::cout << label << " copies=" << a.copies << " bytes=" << a.copy_bytes << " switches=" << a.backend_switches << " dispatches=" << a.backend_dispatches << " PASS\n";
            require(!execute_graph(*rewritten.graph, nullptr, &automatic, nullptr, &scheduler).ok(), "live outputs block reuse");
        }
        if (shape[0] && shape[1]) require(a.copies == 4 && a.backend_switches == 2 && a.backend_dispatches == 9, "frozen nonempty workload counters");
    }
}
void state_versions(const CudaBackend& cuda) {
    Scheduler scheduler(default_cpu_backend(), &cuda);
    auto source = Tensor::allocate_cpu({2, 2}), state = Tensor::allocate_cpu({2, 2}), weight = Tensor::allocate_cpu({2, 2});
    for (int i = 0; i < 4; ++i) weight.data<float>()[i] = i % 3 == 0 ? 1.F : 0.F;
    Graph g; g.add_input(0, "source", source); g.add_input(1, "state", state, true); g.add_input(2, "weight", weight);
    for (TensorId id : {3u, 4u, 5u, 6u}) g.add_tensor(id, {2, 2});
    g.add_node(0, OpDesc(OpCode::COPY, {0, 1}, {3}, CopyAttrs{}));
    g.add_node(1, OpDesc(OpCode::MATMUL, {3, 2}, {4}, {}, cuda.device()));
    g.add_node(2, OpDesc(OpCode::COPY, {4, 3}, {5}, CopyAttrs{}));
    g.add_node(3, OpDesc(OpCode::MATMUL, {5, 2}, {6}, {}, cuda.device()));
    g.add_output("result", 6); scheduler_workload::check(g.freeze());
    const auto r = scheduler.rewrite(g); require(r.ok(), "state rewrite");
    ScheduledAllocationProvider prepared(*r.graph, scheduler);
    for (int run = 0; run < 3; ++run) {
        for (int i = 0; i < 4; ++i) source.data<float>()[i] = static_cast<float>(i + run * 10);
        const auto result = execute_graph(*r.graph, nullptr, &prepared, nullptr, &scheduler);
        require(result.ok(), "synchronous state writes and versioned transfers");
        equal(result.outputs.at("result"), source); equal(state, source);
        require(result.counts.copies == 6 && result.counts.copy_bytes == 96, "CPU write + two distinct state H2Ds + weight H2D + write D2H + output D2H");
    }
}
void copies_and_aliases(const CudaBackend& cuda) {
    Scheduler scheduler(default_cpu_backend(), &cuda);
    auto host = Tensor::allocate_cpu({2, 2}); for (int i = 0; i < 4; ++i) host.data<float>()[i] = static_cast<float>(i + 1);
    auto device = cuda.allocate({2, 2}, DType::FP32, cuda.device()); require(device.ok(), "CUDA root"); scheduler_workload::check(cuda.copy(host, *device.tensor));
    Graph g; g.add_input(0, "x", *device.tensor);
    g.add_tensor(1, {2, 2}, DType::FP32, cuda.device()); g.add_tensor(2, {4}, DType::FP32, cuda.device());
    g.add_node(0, OpDesc(OpCode::COPY, {0}, {1}, CopyAttrs{CopyOverlap::RejectExceptExactSelf, cuda.device()}));
    g.add_node(1, OpDesc(OpCode::RESHAPE, {1}, {2}, ReshapeAttrs{Shape{4}})); g.add_output("view", 2); scheduler_workload::check(g.freeze());
    const auto r = scheduler.rewrite(g); require(r.ok(), "D2D + CUDA metadata alias");
    ScheduledAllocationProvider provider(*r.graph, scheduler);
    auto result = execute_graph(*r.graph, nullptr, &provider, nullptr, &scheduler); require(result.ok(), "D2D execution");
    auto out = Tensor::allocate_cpu({4}); scheduler_workload::check(cuda.copy(result.outputs.at("view"), out)); equal(out, host.reshape({4}));
    require(result.counts.copies == 1 && result.counts.backend_dispatches == 1 && result.counts.backend_switches == 0, "alias is not backend dispatch");
    auto escaped = result.outputs.at("view"); result.outputs.clear(); require(!provider.begin().ok(), "escaped CUDA alias pins backing");
    escaped = host.reshape({4}); require(provider.begin().ok(), "CUDA alias released"); provider.end();
    // Output boundary is explicitly copied back when a logical CPU MATMUL is placed on CUDA.
    Graph output; output.add_input(0, "a", host); output.add_tensor(1, {2, 2}); output.add_tensor(2, {4});
    output.add_node(0, OpDesc(OpCode::MATMUL, {0, 0}, {1}, {}, cuda.device()));
    output.add_node(1, OpDesc(OpCode::RESHAPE, {1}, {2}, ReshapeAttrs{Shape{4}})); output.add_output("view", 2); scheduler_workload::check(output.freeze());
    const auto boundary = scheduler.rewrite(output); require(boundary.ok() && boundary.inserted_copies.size() == 2, "dedup same input and explicit output boundary");
    ScheduledAllocationProvider p(*boundary.graph, scheduler); const auto value = execute_graph(*boundary.graph, nullptr, &p, nullptr, &scheduler);
    require(value.ok() && value.outputs.at("view").device() == Device{} && value.counts.copies == 2, "CPU output device preserved");
}
}
int main(int argc, char** argv) {
    try {
        if (argc > 2) throw std::invalid_argument("usage: test_cuda_scheduler [new-artifact-directory]");
        const std::filesystem::path artifacts = argc == 2 ? argv[1] : "";
        if (!artifacts.empty() && !std::filesystem::create_directory(artifacts)) throw std::invalid_argument("artifact directory must be new");
        CudaBackend cuda; mixed(cuda, artifacts); state_versions(cuda); copies_and_aliases(cuda);
        CudaBackend tiled(0, CudaMatmul::Stage9Tiled); mixed(tiled, artifacts);
        std::cout << "CUDA scheduler: PASS (manual/automatic, fanout, state versions, repeated runs, D2D, aliases, empty, v0/v1)\n"; return 0;
    } catch (const std::exception& e) { std::cerr << "test_cuda_scheduler: " << e.what() << '\n'; return 1; }
}
