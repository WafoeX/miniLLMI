#include "runtime/memory_planner.hpp"
#include <iostream>
#include <stdexcept>

namespace {
using namespace runtime;
void require(bool condition, const char* message) { if (!condition) throw std::runtime_error(message); }
void success(Status status) { if (!status.ok()) throw std::runtime_error(status.message); }
Graph chain(std::size_t count = 12, std::int64_t size = 64) {
    Graph g;
    auto x = Tensor::allocate_cpu({size});
    for (std::int64_t i = 0; i < size; ++i) x.data<float>()[i] = static_cast<float>(i) / 16;
    g.add_input(0, "x", x);
    for (std::size_t i = 0; i < count; ++i) {
        g.add_tensor(static_cast<TensorId>(i + 1), {size});
        g.add_node(static_cast<NodeId>(100 + i), OpDesc(OpCode::ADD, {static_cast<TensorId>(i), 0}, {static_cast<TensorId>(i + 1)}));
    }
    g.add_output("out", static_cast<TensorId>(count)); success(g.freeze()); return g;
}
void lifetimes() {
    Graph editable;
    try { analyze_lifetimes(editable); throw std::runtime_error("editable accepted"); }
    catch (const std::invalid_argument&) {}
    const auto g = chain();
    const auto before = testing::cpu_allocation_counts();
    const auto life = analyze_lifetimes(g);
    require(testing::cpu_allocation_counts().allocations == before.allocations, "analysis never allocates buffers");
    require(life.node_count == 12 && life.roots.size() == 13, "chain roots");
    require(life.tensors.at(0).birth == -1 && life.tensors.at(0).consumer_count == 12 && *life.tensors.at(0).first_use == 0, "external lifetime and unique consumers");
    require(life.tensors.at(1).birth == 0 && life.tensors.at(1).last_use == 1, "closed interval");
    require(life.tensors.at(12).last_use == 12 && life.tensors.at(12).output, "output pinned through end");
    Graph aliases;
    aliases.add_input(0, "x", Tensor::allocate_cpu({4}));
    aliases.add_tensor(1, {4}); aliases.add_tensor(2, {2, 2}); aliases.add_tensor(3, {2, 2}); aliases.add_tensor(4, {2, 2});
    aliases.add_node(9, OpDesc(OpCode::ADD, {0, 0}, {1}));
    aliases.add_node(2, OpDesc(OpCode::RESHAPE, {1}, {2}, ReshapeAttrs{Shape{2, 2}}));
    aliases.add_node(7, OpDesc(OpCode::ADD, {2, 2}, {3}));
    aliases.add_node(3, OpDesc(OpCode::MUL, {2, 3}, {4}));
    aliases.add_output("view", 2); aliases.add_output("end", 4); success(aliases.freeze());
    const auto a = analyze_lifetimes(aliases);
    require(a.tensors.at(1).last_use == 1 && a.roots.at(1).last_use == 4 && a.tensors.at(2).alias, "output view extends base");
    require(a.tensors.at(2).consumer_count == 2 && *a.tensors.at(2).first_use == 2, "diamond consumer counts deduplicated");
    require(a.roots.size() == 4 && a.roots.at(0).external, "aliases never become allocations");
}
void slots() {
    const auto g = chain(); const auto reuse = plan_memory(g); const auto no_reuse = plan_memory(g, PlanPolicy::NoReuse);
    success(validate_memory_plan(g, reuse)); success(validate_memory_plan(g, no_reuse));
    require(reuse.capacity_bytes == 512 && no_reuse.capacity_bytes == 3072 && reuse.peak_live_bytes == 512 && reuse.reuse_count == 10, "chain deterministic capacity gate");
    require(reuse.slots.at(1).offset != reuse.slots.at(2).offset && reuse.slots.at(1).offset == reuse.slots.at(3).offset, "producer inputs coexist with output");
    require(plan_memory(g).graph_signature == reuse.graph_signature, "deterministic identity");
    auto corrupt = reuse; corrupt.slots.at(2).offset = 0;
    require(!validate_memory_plan(g, corrupt).ok(), "checker rejects simultaneous overlap");
    corrupt = reuse; corrupt.slots.at(1).bytes += 4;
    require(!validate_memory_plan(g, corrupt).ok(), "checker rejects invalid byte capacity");
    corrupt = reuse; corrupt.slots.at(1).offset = SIZE_MAX;
    require(!validate_memory_plan(g, corrupt).ok(), "bounds check cannot overflow");
    corrupt = reuse; corrupt.reuse_count = 99;
    require(!validate_memory_plan(g, corrupt).ok(), "checker validates counters");
    require(!validate_memory_plan(chain(12, 32), reuse).ok(), "shape change invalidates plan");
    try { plan_memory(g, PlanPolicy::Reuse, 3); throw std::runtime_error("invalid alignment accepted"); } catch (const std::invalid_argument&) {}
    // Different shapes/dtypes reuse byte ranges; an output view pins its root.
    Graph varied; varied.add_input(0, "x", Tensor::allocate_cpu({8}));
    varied.add_input(8, "ids", Tensor::allocate_cpu({4}, DType::INT32));
    varied.add_tensor(1, {8}); varied.add_tensor(2, {2, 4}); varied.add_tensor(3, {2, 4});
    varied.add_tensor(4, {4}, DType::INT32); varied.add_tensor(5, {4}, DType::INT32);
    varied.add_node(0, OpDesc(OpCode::ADD, {0, 0}, {1}));
    varied.add_node(1, OpDesc(OpCode::RESHAPE, {1}, {2}, ReshapeAttrs{Shape{2, 4}}));
    varied.add_node(2, OpDesc(OpCode::MATERIALIZE, {2}, {3}));
    varied.add_node(3, OpDesc(OpCode::MATERIALIZE, {8}, {4}));
    varied.add_node(4, OpDesc(OpCode::MATERIALIZE, {4}, {5}));
    varied.add_output("ids", 5); success(varied.freeze());
    const auto p = plan_memory(varied); success(validate_memory_plan(varied, p));
    require(p.slots.at(4).offset == p.slots.at(1).offset, "smaller integer span reuses expired float root");
    varied = Graph{}; varied.add_input(0, "empty", Tensor::allocate_cpu({0}));
    varied.add_tensor(1, {0}); varied.add_node(0, OpDesc(OpCode::ADD, {0, 0}, {1})); varied.add_output("empty", 1); success(varied.freeze());
    const auto zero = plan_memory(varied); success(validate_memory_plan(varied, zero));
    require(zero.capacity_bytes == 0 && zero.reuse_count == 0, "empty plan reserves nothing");
    for (std::int64_t size : {1, 7, 16, 17, 65}) for (std::size_t count : {1u, 2u, 3u, 25u}) {
        const auto random = chain(count, size);
        for (auto policy : {PlanPolicy::Reuse, PlanPolicy::NoReuse}) for (auto alignment : {32u, 64u, 128u})
            success(validate_memory_plan(random, plan_memory(random, policy, alignment)));
    }
}
}
int main() {
    try { lifetimes(); slots(); require(testing::cpu_allocation_counts().live == 0, "no leaked buffers"); std::cout << "memory planner: PASS\n"; return 0; }
    catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
