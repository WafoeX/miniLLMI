#include "runtime/planned_executor.hpp"
#include "runtime/copy.hpp"
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {
using namespace runtime;
void require(bool condition, const char* message) { if (!condition) throw std::runtime_error(message); }
void success(Status status) { if (!status.ok()) throw std::runtime_error(status.message); }
Graph graph(bool pin_view = true) {
    Graph g; auto x = Tensor::allocate_cpu({2, 2});
    for (std::size_t i = 0; i < 4; ++i) x.data<float>()[i] = static_cast<float>(i + 1);
    g.add_input(0, "x", x);
    g.add_tensor(1, {2, 2}); g.add_tensor(2, {2, 2}); g.add_tensor(3, {2, 2}); g.add_tensor(4, {2, 2}); g.add_tensor(5, {2, 2});
    g.add_node(0, OpDesc(OpCode::ADD, {0, 0}, {1}));
    g.add_node(1, OpDesc(OpCode::TRANSPOSE, {1}, {2}, TransposeAttrs{}));
    g.add_node(2, OpDesc(OpCode::MATERIALIZE, {2}, {3}));
    g.add_node(3, OpDesc(OpCode::ADD, {3, 3}, {4}));
    g.add_node(4, OpDesc(OpCode::MATMUL, {4, 0}, {5}));
    g.add_output("out", 5); if (pin_view) g.add_output("view", 2);
    success(g.freeze()); return g;
}
void integration() {
    const auto g = graph(); const auto dynamic = execute_graph(g); success(dynamic.status);
    require(dynamic.counts.allocations == 4, "baseline allocates new roots");
    for (auto policy : {PlanPolicy::Reuse, PlanPolicy::NoReuse}) {
        const auto before = testing::cpu_allocation_counts();
        PlannedAllocationProvider prepared(g, policy);
        require(testing::cpu_allocation_counts().allocations == before.allocations + 1, "one prepared backing");
        std::optional<Tensor> alias, snapshot;
        {
            ExecutionTrace trace;
            const auto result = execute_planned(g, prepared, &trace); success(result.status);
            require(result.counts.allocations == 0 && result.counts.frees == 0 && result.counts.allocation_requests == 4, "zero execute backing calls");
            for (const auto& item : result.outputs) for (std::int64_t a = 0; a < 2; ++a) for (std::int64_t b = 0; b < 2; ++b)
                require(item.second.at<float>({a, b}) == dynamic.outputs.at(item.first).at<float>({a, b}), "views and matmul equal baseline");
            for (const auto& event : trace.events()) if (event.kind == TraceKind::BlockAllocate) {
                require(event.metadata->offset_bytes == prepared.plan().slots.at(*event.tensor).offset, "trace uses planned offsets");
                require(event.metadata->capacity_bytes == prepared.capacity(), "trace reservation");
            }
            alias = result.outputs.at("view");
            // Explicit independent snapshot: contiguous() alone can still alias.
            snapshot = Tensor::allocate_cpu(alias->shape());
            require(copy_cpu(*alias, *snapshot) == alias->nbytes(), "independent snapshot copied");
            const auto busy = execute_planned(g, prepared);
            require(busy.status.code == StatusCode::InvalidArgument && busy.outputs.empty(), "held output cannot be overwritten");
        }
        require(execute_planned(g, prepared).status.code == StatusCode::InvalidArgument, "copied output handle pins backing");
        alias.reset();
        for (int i = 0; i < 5; ++i) { const auto result = execute_planned(g, prepared); success(result.status); }
        require(testing::cpu_allocation_counts().allocations == before.allocations + 2, "repeated execute never allocates backing; snapshot is separate");
        require(snapshot->data<float>()[1] == dynamic.outputs.at("view").at<float>({0, 1}), "snapshot survives repeated execution");
        const auto incompatible = graph(false); const auto stale = execute_planned(incompatible, prepared);
        require(stale.status.code == StatusCode::InvalidArgument && stale.counts.nodes_completed == 0, "changed outputs/topology rejected before writes");
    }
    std::optional<Tensor> survivor; std::weak_ptr<Storage> weak;
    { PlannedAllocationProvider prepared(g); const auto result = execute_planned(g, prepared); success(result.status); survivor = result.outputs.at("out"); weak = survivor->storage(); }
    require(!weak.expired() && survivor->data<float>()[0] == dynamic.outputs.at("out").data<float>()[0], "provider destruction preserves outputs");
    survivor.reset(); require(weak.expired(), "final handle frees backing once");
    testing::fail_next_cpu_allocation();
    try { PlannedAllocationProvider failed_prepare(g); throw std::runtime_error("prepare allocation failure ignored"); } catch (const std::bad_alloc&) {}
    PlannedAllocationProvider prepared(g);
    testing::fail_next_cpu_allocation();
    { const auto run = execute_planned(g, prepared); success(run.status); }
    try { Tensor::allocate_cpu({1}); throw std::runtime_error("execute consumed backing allocation hook"); } catch (const std::bad_alloc&) {}
    auto corrupt = plan_memory(g); corrupt.slots.at(5).offset = 0;
    try { PlannedAllocationProvider bad(g, corrupt); throw std::runtime_error("corrupt accepted"); } catch (const std::invalid_argument&) {}
}
void failure_state_zero() {
    Graph g; auto x = Tensor::allocate_cpu({4}); auto state = Tensor::allocate_cpu({4});
    for (std::size_t i = 0; i < 4; ++i) x.data<float>()[i] = 1;
    auto bad = Tensor::allocate_cpu({4}); bad.data<float>()[0] = std::numeric_limits<float>::infinity();
    g.add_input(0, "x", x); g.add_input(1, "state", state, true); g.add_input(2, "bad", bad);
    g.add_tensor(3, {4}); g.add_tensor(4, {4}); g.add_tensor(5, {4});
    g.add_node(0, OpDesc(OpCode::COPY, {0, 1}, {3}, CopyAttrs{}));
    g.add_node(1, OpDesc(OpCode::ADD, {3, 2}, {4}));
    g.add_node(2, OpDesc(OpCode::MUL, {4, 0}, {5}));
    g.add_output("out", 5); success(g.freeze());
    PlannedAllocationProvider prepared(g);
    require(!prepared.plan().slots.count(1) && !prepared.plan().slots.count(3), "persistent state and write aliases excluded");
    const auto fail = execute_planned(g, prepared);
    require(fail.status.code == StatusCode::NonFinite && fail.failed_node == 1 && fail.outputs.empty() && fail.counts.live_bytes == 0, "failed plan execution clears leases");
    require(state.data<float>()[0] == 1, "completed external writes are not rolled back");
    bad.data<float>()[0] = 2; { const auto retry = execute_planned(g, prepared); success(retry.status); }
    Graph zero; zero.add_input(0, "x", Tensor::allocate_cpu({0})); zero.add_tensor(1, {0}); zero.add_node(0, OpDesc(OpCode::ADD, {0, 0}, {1})); zero.add_output("empty", 1); success(zero.freeze());
    const auto before = testing::cpu_allocation_counts();
    PlannedAllocationProvider empty(zero);
    { const auto run = execute_planned(zero, empty); success(run.status); require(run.counts.allocation_requests == 0 && empty.capacity() == 0, "empty plan zero backing");
      require(execute_planned(zero, empty).status.code == StatusCode::InvalidArgument, "even empty planned output pins context"); }
    require(testing::cpu_allocation_counts().allocations == before.allocations, "zero capacity never allocates nonzero backing");
    Graph passthrough; passthrough.add_input(0, "x", x); passthrough.add_output("out", 0); success(passthrough.freeze());
    PlannedAllocationProvider external(passthrough); const auto run = execute_planned(passthrough, external); success(run.status);
    require(run.outputs.at("out").storage() == x.storage() && external.capacity() == 0, "external-only outputs remain external");
}
void randomized_dags() {
    for (std::uint64_t seed = 1; seed <= 6; ++seed) {
        auto state = seed;
        const auto next = [&]() { state ^= state << 13; state ^= state >> 7; state ^= state << 17; return state; };
        for (int trial = 0; trial < 100; ++trial) {
            Graph g; auto x = Tensor::allocate_cpu({4, 4});
            for (std::size_t i = 0; i < 16; ++i) x.data<float>()[i] = static_cast<float>(i) / 16;
            g.add_input(0, "x", x);
            const auto count = static_cast<TensorId>(4 + next() % 28);
            for (TensorId id = 1; id <= count; ++id) {
                g.add_tensor(id, {4, 4});
                const auto a = static_cast<TensorId>(next() % id), b = static_cast<TensorId>(next() % id);
                if (next() % 4 == 0) g.add_node(id, OpDesc(OpCode::RESHAPE, {a}, {id}, ReshapeAttrs{Shape{4, 4}}));
                else g.add_node(id, OpDesc(OpCode::ADD, {a, b}, {id}));
            }
            g.add_output("last", count); g.add_output("branch", static_cast<TensorId>(next() % count)); success(g.freeze());
            const auto baseline = execute_graph(g); success(baseline.status);
            for (auto policy : {PlanPolicy::Reuse, PlanPolicy::NoReuse}) {
                PlannedAllocationProvider provider(g, policy);
                const auto before = testing::cpu_allocation_counts();
                const auto result = execute_planned(g, provider); success(result.status);
                require(testing::cpu_allocation_counts().allocations == before.allocations, "randomized planned zero backing");
                for (const auto& item : result.outputs) for (std::size_t i = 0; i < 16; ++i)
                    require(item.second.data<float>()[i] == baseline.outputs.at(item.first).data<float>()[i], "randomized branch/view/output values");
                require(provider.capacity() >= result.counts.peak_live_bytes, "capacity covers simultaneous payload");
            }
        }
    }
}
void typed_reuse_and_devices() {
    Graph g; g.add_input(0, "x", Tensor::allocate_cpu({8}));
    auto ids = Tensor::allocate_cpu({4}, DType::INT32); ids.data<std::int32_t>()[2] = 73;
    g.add_input(9, "ids", ids); g.add_tensor(1, {8}); g.add_tensor(2, {8});
    g.add_tensor(3, {4}, DType::INT32); g.add_tensor(4, {4}, DType::INT32);
    g.add_node(0, OpDesc(OpCode::ADD, {0, 0}, {1}));
    g.add_node(1, OpDesc(OpCode::MATERIALIZE, {1}, {2}));
    g.add_node(2, OpDesc(OpCode::MATERIALIZE, {9}, {3}));
    g.add_node(3, OpDesc(OpCode::MATERIALIZE, {3}, {4}));
    g.add_output("ids", 4); success(g.freeze());
    PlannedAllocationProvider p(g);
    require(p.plan().slots.at(1).offset == p.plan().slots.at(3).offset, "INT32 reuses expired larger FP32 span");
    for (int repeat = 0; repeat < 5; ++repeat) {
        const auto run = execute_planned(g, p); success(run.status);
        require(run.outputs.at("ids").data<std::int32_t>()[2] == 73, "typed integer object lifetimes restart every execute");
    }
    Graph cuda;
    auto storage = Storage::wrap(Device(DeviceType::CUDA, 0), 0, nullptr, [](void*) {});
    cuda.add_input(0, "cuda-metadata", Tensor(storage, DType::FP32, Shape{0}, Stride{1}));
    cuda.add_output("external", 0); success(cuda.freeze());
    try { plan_memory(cuda); throw std::runtime_error("CUDA planner accepted"); } catch (const std::invalid_argument&) {}
}
}
int main() {
    try { integration(); failure_state_zero(); randomized_dags(); typed_reuse_and_devices(); require(testing::cpu_allocation_counts().live == 0, "planned executor leaks"); std::cout << "planned executor: PASS\n"; return 0; }
    catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
