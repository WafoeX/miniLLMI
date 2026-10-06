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
}
int main() {
    try { lifetimes(); require(testing::cpu_allocation_counts().live == 0, "no leaked buffers"); std::cout << "memory planner: PASS\n"; return 0; }
    catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
