#include "runtime/graph_executor.hpp"
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {
using namespace runtime;
void require(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
void success(Status status) { if (!status.ok()) throw std::runtime_error(status.message); }
Graph graph() {
    Graph g; auto x = Tensor::allocate_cpu({2, 2}), w = Tensor::allocate_cpu({2, 2});
    for (std::size_t i = 0; i < 4; ++i) { x.data<float>()[i] = static_cast<float>(i + 1); w.data<float>()[i] = static_cast<float>(i + 2); }
    g.add_input(0, "x", x); g.add_input(1, "w", w);
    g.add_tensor(2, {2, 2}); g.add_tensor(3, {4}); g.add_tensor(4, {4}); g.add_tensor(5, {2, 2}); g.add_tensor(6, {2, 2});
    g.add_node(0, OpDesc(OpCode::ADD, {0, 0}, {2}));
    g.add_node(1, OpDesc(OpCode::RESHAPE, {2}, {3}, ReshapeAttrs{Shape{4}}));
    g.add_node(2, OpDesc(OpCode::MATERIALIZE, {3}, {4}));
    g.add_node(3, OpDesc(OpCode::RESHAPE, {4}, {5}, ReshapeAttrs{Shape{2, 2}}));
    g.add_node(4, OpDesc(OpCode::MATMUL, {5, 1}, {6}));
    g.add_output("result", 6); g.add_output("alias", 3); success(g.freeze()); return g;
}
void integration() {
    const auto g = graph(); const auto baseline = execute_graph(g); success(baseline.status);
    const auto before = testing::cpu_allocation_counts();
    ArenaAllocationProvider provider(192); // three simultaneously live 16-byte spans with 64 alignment
    require(testing::cpu_allocation_counts().allocations == before.allocations + 1, "one prepared backing");
    ExecutionTrace trace; std::optional<Tensor> retained;
    {
        const auto result = execute_graph(g, &trace, &provider); success(result.status);
        for (const auto& name : {"result", "alias"}) for (std::size_t i = 0; i < result.outputs.at(name).numel(); ++i)
            require(result.outputs.at(name).data<float>()[i] == baseline.outputs.at(name).data<float>()[i], "same graph values under both policies");
        require(result.counts.allocations == 0 && result.counts.frees == 0 && result.counts.allocation_requests == 3 &&
                result.counts.releases == 1 && result.counts.live_bytes == 32 && result.counts.peak_live_bytes == 48, "logical requests separated from backing calls");
        require(provider.arena().counts().live_blocks == 2 && result.counts.arena_capacity_bytes == 192, "output roots remain leased");
        const auto busy = execute_graph(g, nullptr, &provider);
        require(busy.status.code == StatusCode::InvalidArgument && busy.outputs.empty(), "provider cannot overwrite held output Storage");
        retained = result.outputs.at("alias");
        std::size_t allocs = 0, frees = 0;
        for (const auto& e : trace.events()) {
            require(e.kind != TraceKind::Allocate && e.kind != TraceKind::Free, "arena trace never labels span ops as backing allocation");
            if (e.kind == TraceKind::BlockAllocate) ++allocs;
            if (e.kind == TraceKind::BlockFree) ++frees;
            if (e.tensor == 6 && e.metadata) require(e.metadata->offset_bytes == 128 && e.metadata->capacity_bytes == 192, "trace uses actual arena offsets/capacity");
        }
        require(allocs == 3 && frees == 1, "arena span events");
    }
    require(execute_graph(g, nullptr, &provider).status.code == StatusCode::InvalidArgument, "copied alias, not only result, pins execution context");
    retained.reset();
    { const auto run = execute_graph(g, nullptr, &provider); success(run.status); }
    require(testing::cpu_allocation_counts().allocations == before.allocations + 1, "no execute-time hidden malloc");
    success(provider.begin()); provider.end(); provider.arena().validate();
    require(provider.arena().counts().live_blocks == 0 && provider.arena().free_ranges().size() == 1, "retired outputs reclaimed only after all handles die");
    ArenaAllocationProvider tiny(64);
    const auto failed = execute_graph(g, nullptr, &tiny);
    require(failed.status.code == StatusCode::ResourceExhausted && failed.failed_node == 2 && failed.outputs.empty() &&
            failed.counts.allocations == 0 && failed.counts.allocation_requests == 1 && failed.counts.releases == 1 && tiny.arena().counts().live_blocks == 0,
            "capacity failure stops, frees prior spans, never falls back");
    std::optional<Tensor> survivor; std::weak_ptr<Storage> storage;
    { ArenaAllocationProvider owner(192); const auto result = execute_graph(g, nullptr, &owner); success(result.status);
      survivor = result.outputs.at("result"); storage = survivor->storage(); }
    require(!storage.expired() && survivor->data<float>()[0] == baseline.outputs.at("result").data<float>()[0], "single Storage owner survives provider destruction");
    survivor.reset(); require(storage.expired(), "Storage backing freed with final view, not individual spans");
}
void errors_and_types() {
    Graph g; auto x = Tensor::allocate_cpu({2}); x.data<float>()[0] = std::numeric_limits<float>::infinity();
    g.add_input(0, "x", x); g.add_tensor(1, {2}); g.add_node(0, OpDesc(OpCode::ADD, {0, 0}, {1})); g.add_output("x", 1); success(g.freeze());
    ArenaAllocationProvider provider(64);
    const auto failed = execute_graph(g, nullptr, &provider);
    require(failed.status.code == StatusCode::NonFinite && failed.outputs.empty() && provider.arena().counts().live_blocks == 0, "kernel failure arena cleanup");
    x.data<float>()[0] = 1;
    { const auto retry = execute_graph(g, nullptr, &provider); success(retry.status); }
    Graph ids; auto integers = Tensor::allocate_cpu({4}, DType::INT32); integers.data<std::int32_t>()[2] = 73;
    ids.add_input(0, "ids", integers); ids.add_tensor(1, {4}, DType::INT32);
    ids.add_node(0, OpDesc(OpCode::MATERIALIZE, {0}, {1})); ids.add_output("ids", 1); success(ids.freeze());
    { const auto run = execute_graph(ids, nullptr, &provider); success(run.status); require(run.outputs.at("ids").data<std::int32_t>()[2] == 73, "FP32->INT32 reused typed object lifetime"); }
    Graph zero; zero.add_input(0, "x", Tensor::allocate_cpu({0})); zero.add_tensor(1, {0}); zero.add_node(0, OpDesc(OpCode::ADD, {0, 0}, {1})); zero.add_output("empty", 1); success(zero.freeze());
    ArenaAllocationProvider empty(0); { const auto run = execute_graph(zero, nullptr, &empty); success(run.status); require(run.counts.allocation_requests == 0, "empty arena zero output policy"); }
}
}
int main() {
    try { integration(); errors_and_types(); require(testing::cpu_allocation_counts().live == 0, "provider tests leak"); std::cout << "allocation provider: PASS\n"; return 0; }
    catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
