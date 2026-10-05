#include "runtime/graph.hpp"
#include <iostream>
#include <stdexcept>

namespace {
using namespace runtime;
void require(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
template<class F> void rejects(F function) {
    try { function(); } catch (const std::invalid_argument&) { return; }
    throw std::runtime_error("expected graph builder rejection");
}
void ownership() {
    Graph graph;
    auto input = Tensor::allocate_cpu({2, 2});
    std::weak_ptr<Storage> owner = input.storage();
    graph.add_input(0, "x", input);
    graph.add_input(1, "y", input);
    graph.add_tensor(2, {2, 2});
    const auto before = testing::cpu_allocation_counts();
    graph.add_node(10, OpDesc(OpCode::ADD, {0, 1}, {2}));
    graph.add_output("sum", 2);
    require(graph.tensors().at(2).producer == 10, "producer link");
    require(graph.tensors().at(0).consumers == std::vector<NodeId>{10}, "consumer link");
    require(graph.outputs().at("sum") == 2 && graph.inputs().at("x") == 0, "named boundaries");
    rejects([&] { graph.add_tensor(2, {2, 2}); });
    rejects([&] { graph.add_tensor(INVALID_TENSOR_ID, {}); });
    rejects([&] { graph.add_input(3, "x", input); });
    rejects([&] { graph.add_node(10, OpDesc(OpCode::ADD, {0, 1}, {2})); });
    rejects([&] { graph.add_node(11, OpDesc(OpCode::ADD, {0, 9}, {2})); });
    rejects([&] { graph.add_node(11, OpDesc(OpCode::ADD, {0, 1}, {9})); });
    rejects([&] { graph.add_node(11, OpDesc(OpCode::ADD, {0, 1}, {2})); });
    rejects([&] { graph.add_node(11, OpDesc(OpCode::ADD, {0, 1}, {1})); });
    rejects([&] { graph.add_output("sum", 0); });
    rejects([&] { graph.add_output("bad", 9); });
    require(graph.nodes().size() == 1 && graph.tensors().size() == 3, "failed insertions are transactional");
    require(testing::cpu_allocation_counts().allocations == before.allocations, "building graph allocates no buffers");
    { auto copied = graph; require(copied.nodes().at(10).descriptor.inputs()[0] == 0, "owned topology survives value copy"); }
    require(!owner.expired(), "external storage is retained by graph");
    Graph duplicate;
    duplicate.add_input(0, "a", input);
    duplicate.add_tensor(1, {2, 2});
    duplicate.add_node(0, OpDesc(OpCode::ADD, {0, 0}, {1}));
    require(duplicate.tensors().at(0).consumers.size() == 1, "repeated input is one consumer edge");
}
}
int main() {
    try { ownership(); std::cout << "graph ownership: PASS\n"; return 0; }
    catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
