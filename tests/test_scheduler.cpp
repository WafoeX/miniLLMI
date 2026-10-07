#include "runtime/scheduler.hpp"
#include "runtime/graph_executor.hpp"
#include "../bench/scheduler_workload.hpp"
#include <iostream>
#include <stdexcept>

using namespace runtime;
namespace {
void require(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
// Metadata-only capability double: never pretends to execute GPU work locally.
class MetadataCuda final : public Backend {
public:
    const char* name() const noexcept override { return "metadata-only-cuda"; }
    Device device() const noexcept override { return Device(DeviceType::CUDA); }
    Status capability(OpCode code, Device d, DType dtype) const override {
        if (d == device() && dtype == DType::FP32 && (code == OpCode::MATMUL || code == OpCode::COPY)) return {};
        return Status::failure(StatusCode::Unsupported, "unimplemented metadata capability");
    }
    BackendBuffer allocate(Shape, DType, Device) const override { throw std::logic_error("metadata only"); }
    Status copy(const Tensor&, Tensor&) const override { throw std::logic_error("metadata only"); }
    BackendPreparation prepare(const OpDesc&, const TensorInputs&, const Tensor&) const override { throw std::logic_error("metadata only"); }
    Status execute(const OpDesc&, const TensorInputs&, Tensor&, Workspace) const override { throw std::logic_error("metadata only"); }
};
void placement_tests() {
    const auto& cpu = default_cpu_backend(); MetadataCuda cuda;
    Scheduler scheduler(cpu, &cuda), strict(cpu, &cuda, PlacementFallback::Error), local;
    Layout a{{2, 3}, {3, 1}, DType::FP32, Device{}, 24, 0, 0};
    Layout b{{3, 2}, {2, 1}, DType::FP32, Device{}, 24, 0, 1};
    for (int i = 0; i < 10; ++i) {
        const auto mm = scheduler.place(OpCode::MATMUL, DType::FP32, {a, b}, cuda.device());
        require(mm.ok() && mm.device == cuda.device() && !mm.fell_back, "deterministic CUDA MATMUL placement");
        const auto add = scheduler.place(OpCode::ADD, DType::FP32, {a, a}, cuda.device());
        require(add.ok() && add.device == Device{} && add.fell_back && !add.reason.empty(), "explicit CPU fallback");
    }
    require(!strict.place(OpCode::ADD, DType::FP32, {a, a}, cuda.device()).ok(), "strict unsupported error");
    require(!scheduler.place(OpCode::ATTENTION, DType::FP32, {a, a}, cuda.device()).ok(), "opaque ATTENTION kernel is not implemented on either backend");
    require(local.place(OpCode::MATMUL, DType::FP32, {a, b}, cuda.device()).device == Device{}, "CPU-only fallback");
    require(!Scheduler(cpu, nullptr, PlacementFallback::Error).place(OpCode::MATMUL, DType::FP32, {a, b}, cuda.device()).ok(), "missing CUDA strict error");
    require(scheduler.place(OpCode::MATMUL, DType::FP32, {a, b}, Device(DeviceType::CUDA, 1)).fell_back, "unregistered device fallback");
    auto t = a.transpose(0, 1);
    require(scheduler.place(OpCode::COPY, DType::FP32, {t}, cuda.device()).fell_back, "CUDA strided COPY not claimed");
    auto integer = a; integer.dtype = DType::INT32;
    require(!scheduler.place(OpCode::MATMUL, DType::INT32, {integer}, cuda.device()).ok(), "no integer arithmetic");
    require(scheduler.place(OpCode::COPY, DType::INT32, {integer}, cuda.device()).fell_back, "INT32 CUDA unsupported, CPU COPY available");
}
void rewrite_tests() {
    MetadataCuda cuda; Scheduler scheduler(default_cpu_backend(), &cuda);
    const auto in = scheduler_workload::inputs(17, 13, 11);
    const auto logical = scheduler_workload::graph(in, false);
    const auto before = testing::cpu_allocation_counts().allocations;
    const auto scheduled = scheduler.rewrite(logical);
    require(scheduled.ok(), "mixed rewrite succeeds metadata-only");
    require(testing::cpu_allocation_counts().allocations == before, "rewrite allocates no backing");
    require(scheduled.inserted_copies.size() == 4, "fanout deduplicates both H2D edges");
    require(scheduled.placements.at(7).fell_back && scheduled.placements.at(7).device == Device{}, "ADD explicit CPU placement");
    const auto p = plan_memory(*scheduled.graph); scheduler_workload::check(validate_memory_plan(*scheduled.graph, p));
    require(p.device_capacity_bytes.size() == 2, "separate CPU/CUDA plans");
    require(p.slots.at(4).device == cuda.device(), "MATMUL device plan");
    auto corrupt = p; corrupt.device_capacity_bytes[cuda.device()] = 0;
    require(!validate_memory_plan(*scheduled.graph, corrupt).ok(), "per-device bounds independently checked");
    require(!validate_memory_plan(logical, p).ok(), "old CPU plan cannot validate rewritten graph");
    const auto again = scheduler.rewrite(logical);
    require(again.ok() && plan_memory(*again.graph).graph_signature == p.graph_signature, "deterministic rewrite");
    require(!Scheduler(default_cpu_backend(), &cuda, PlacementFallback::Error).rewrite(logical).ok(), "strict graph placement error");
    require(!scheduler.rewrite(Graph{}).ok(), "editable logical graph rejected");
    // CPU-only fallback runs through the SAME Graph executor and planner.
    Scheduler local; const auto cpu_graph = local.rewrite(logical); require(cpu_graph.ok() && cpu_graph.inserted_copies.empty(), "CPU-only rewrite");
    ScheduledAllocationProvider provider(*cpu_graph.graph, local);
    const auto oracle = execute_graph(scheduler_workload::graph(in, false, true)); require(oracle.ok(), "CPU oracle");
    {
        ExecutionTrace trace; const auto result = execute_graph(*cpu_graph.graph, &trace, &provider, nullptr, &local);
        require(result.ok() && result.counts.allocations == 0 && result.counts.backend_switches == 0, "CPU scheduled execution");
        require(!execute_graph(*cpu_graph.graph, nullptr, &provider, nullptr, &local).ok(), "escaped output blocks reuse");
        for (std::size_t i = 0; i < in.scale.numel(); ++i)
            require(result.outputs.at("result").data<float>()[i] == oracle.outputs.at("result").data<float>()[i], "CPU values equal oracle");
    }
    require(execute_graph(*cpu_graph.graph, nullptr, &provider, nullptr, &local).ok(), "context reusable after output release");
    require(!execute_graph(*cpu_graph.graph, nullptr, nullptr, nullptr, &local).ok(), "scheduled execution requires explicit plan/provider");
    // Allocating CPU COPY, including integer/empty values; no state binding needed.
    for (auto dtype : {DType::FP32, DType::INT32}) for (std::int64_t size : {0, 4}) {
        Graph g; auto x = Tensor::allocate_cpu({size}, dtype);
        for (std::int64_t i = 0; i < size; ++i) {
            if (dtype == DType::FP32) x.data<float>()[i] = static_cast<float>(i + 1);
            else x.data<std::int32_t>()[i] = static_cast<std::int32_t>(i + 1);
        }
        g.add_input(0, "x", x); g.add_tensor(1, {size}, dtype);
        g.add_node(0, OpDesc(OpCode::COPY, {0}, {1}, CopyAttrs{CopyOverlap::RejectExceptExactSelf, Device{}}));
        g.add_output("copy", 1); scheduler_workload::check(g.freeze());
        const auto r = local.rewrite(g); require(r.ok(), "CPU COPY rewrite");
        ScheduledAllocationProvider prepared(*r.graph, local);
        const auto result = execute_graph(*r.graph, nullptr, &prepared, nullptr, &local);
        require(result.ok() && result.counts.copies == (size ? 1u : 0u), "CPU COPY actual counts");
        for (std::int64_t i = 0; i < size; ++i) require(dtype == DType::FP32 ? result.outputs.at("copy").data<float>()[i] == x.data<float>()[i]
            : result.outputs.at("copy").data<std::int32_t>()[i] == x.data<std::int32_t>()[i], "typed COPY values");
    }
}
Graph state_graph(bool stale = false) {
    Graph g; auto source = Tensor::allocate_cpu({2, 2}); auto state = Tensor::allocate_cpu({2, 2});
    auto weight = Tensor::allocate_cpu({2, 2});
    for (int i = 0; i < 4; ++i) { source.data<float>()[i] = static_cast<float>(i + 1); weight.data<float>()[i] = i % 3 == 0 ? 1.F : 0.F; }
    g.add_input(0, "source", source); g.add_input(1, "state", state, true); g.add_input(2, "weight", weight);
    for (TensorId i : {3u, 4u, 5u, 6u}) g.add_tensor(i, {2, 2});
    g.add_node(0, OpDesc(OpCode::COPY, {0, 1}, {3}, CopyAttrs{}));
    g.add_node(1, OpDesc(OpCode::MATMUL, {3, 2}, {4}, {}, Device(DeviceType::CUDA)));
    // GPU result copies back before mutating the SAME external root again.
    g.add_node(2, OpDesc(OpCode::COPY, {4, 3}, {5}, CopyAttrs{}));
    g.add_node(3, OpDesc(OpCode::MATMUL, {stale ? 3u : 5u, 2}, {6}, {}, Device(DeviceType::CUDA)));
    g.add_output("result", 6);
    const auto status = g.freeze();
    if (stale) require(!status.ok() && status.code == StatusCode::Aliasing, "stale state graph rejected before scheduling");
    else scheduler_workload::check(status);
    return g;
}
void versions_tests() {
    MetadataCuda cuda; Scheduler scheduler(default_cpu_backend(), &cuda);
    const auto logical = state_graph(); const auto r = scheduler.rewrite(logical);
    require(r.ok(), "version-correct mixed state rewrite");
    std::size_t old_version = 0, new_version = 0;
    for (const auto& copy : r.inserted_copies) {
        old_version += copy.source == 3 && copy.to == cuda.device();
        new_version += copy.source == 5 && copy.to == cuda.device();
    }
    require(old_version == 1 && new_version == 1, "different state versions never share a cached copy");
    const auto stale = state_graph(true); require(!scheduler.rewrite(stale).ok(), "stale graph cannot be scheduled");
    const auto p = plan_memory(*r.graph); scheduler_workload::check(validate_memory_plan(*r.graph, p));
    Scheduler local; const auto cpu = local.rewrite(logical); require(cpu.ok(), "CPU state rewrite");
    ScheduledAllocationProvider provider(*cpu.graph, local);
    auto source = *logical.tensors().at(0).external;
    for (int run = 0; run < 2; ++run) {
        // Persistent-state contents are fresh inputs on EVERY execute, not
        // reusable value copies cached in a prepared context.
        source.data<float>()[0] = static_cast<float>(run + 9);
        const auto result = execute_graph(*cpu.graph, nullptr, &provider, nullptr, &local);
        require(result.ok() && result.outputs.at("result").data<float>()[0] == static_cast<float>(run + 9), "state versions/repeated execution");
    }
}
}
int main() {
    try { placement_tests(); rewrite_tests(); versions_tests(); std::cout << "scheduler: PASS (CPU execution / CUDA metadata only)\n"; return 0; }
    catch (const std::exception& e) { std::cerr << "test_scheduler: " << e.what() << '\n'; return 1; }
}
