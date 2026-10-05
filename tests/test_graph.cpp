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
void success(Status status) { if (!status.ok()) throw std::runtime_error(status.message); }
void failure(Graph& graph, StatusCode code, const char* diagnostic) {
    const auto status = graph.freeze();
    require(status.code == code && status.message.find(diagnostic) != std::string::npos, "graph diagnostic classification");
    require(!graph.frozen() && graph.order().empty(), "failed freeze is transactional");
    for (const auto& tensor : graph.tensors()) require(!tensor.second.layout, "failed freeze leaks no inferred layouts");
}
Graph diamond(bool reverse) {
    Graph graph;
    graph.add_input(0, "x", Tensor::allocate_cpu({2, 2}));
    graph.add_input(1, "y", Tensor::allocate_cpu({2, 2}));
    for (TensorId i = 2; i < 6; ++i) graph.add_tensor(i, {2, 2});
    const auto add = [&] { graph.add_node(10, OpDesc(OpCode::ADD, {0, 1}, {2})); };
    const auto left = [&] { graph.add_node(20, OpDesc(OpCode::MUL, {2, 1}, {3})); };
    const auto right = [&] { graph.add_node(30, OpDesc(OpCode::ADD, {2, 0}, {4})); };
    const auto join = [&] { graph.add_node(40, OpDesc(OpCode::ADD, {3, 4}, {5})); };
    if (reverse) { join(); right(); left(); add(); } else { add(); left(); right(); join(); }
    graph.add_output("diamond", 5);
    return graph;
}
void topology() {
    auto first = diamond(false), second = diamond(true);
    const auto before = testing::cpu_allocation_counts();
    success(first.freeze()); success(second.freeze()); success(first.freeze());
    require(first.order() == std::vector<NodeId>({10, 20, 30, 40}) && second.order() == first.order(), "stable topology independent of insertion order");
    require(testing::cpu_allocation_counts().allocations == before.allocations, "freeze is backing-allocation-free");
    try { first.add_output("late", 0); throw std::runtime_error("mutable frozen graph"); } catch (const std::logic_error&) {}
    try { first.add_tensor(99, {}); throw std::runtime_error("mutable frozen tensors"); } catch (const std::logic_error&) {}
    try { first.add_node(99, OpDesc(OpCode::ADD, {0, 1}, {2})); throw std::runtime_error("mutable frozen nodes"); } catch (const std::logic_error&) {}
    try { first.add_input(99, "late", *first.tensors().at(0).external); throw std::runtime_error("mutable frozen inputs"); } catch (const std::logic_error&) {}
    Graph roots;
    roots.add_input(0, "unused", Tensor::allocate_cpu({}));
    roots.add_input(1, "x", Tensor::allocate_cpu({}));
    roots.add_tensor(2, {}); roots.add_tensor(3, {});
    roots.add_node(50, OpDesc(OpCode::ADD, {0, 0}, {2}));
    roots.add_node(5, OpDesc(OpCode::MUL, {1, 1}, {3}));
    roots.add_output("passthrough", 0);
    success(roots.freeze()); require(roots.order() == std::vector<NodeId>({5, 50}), "disconnected roots accepted");
    Graph missing; missing.add_tensor(4, {}); missing.add_output("missing", 4);
    failure(missing, StatusCode::InvalidArgument, "tensor 4 has no producer");
    missing.add_input(0, "x", Tensor::allocate_cpu({}));
    missing.add_node(0, OpDesc(OpCode::MATERIALIZE, {0}, {4})); success(missing.freeze());
    Graph cycle; cycle.add_tensor(0, {}); cycle.add_tensor(1, {});
    cycle.add_node(0, OpDesc(OpCode::MATERIALIZE, {1}, {0}));
    cycle.add_node(1, OpDesc(OpCode::MATERIALIZE, {0}, {1}));
    failure(cycle, StatusCode::InvalidArgument, "cycle");
    Graph wrong; wrong.add_input(0, "x", Tensor::allocate_cpu({2})); wrong.add_tensor(1, {3});
    wrong.add_node(7, OpDesc(OpCode::MATERIALIZE, {0}, {1})); failure(wrong, StatusCode::ShapeMismatch, "node 7 output 1");
    Graph types; types.add_input(0, "x", Tensor::allocate_cpu({2})); types.add_tensor(1, {2}, DType::INT32);
    types.add_node(7, OpDesc(OpCode::MATERIALIZE, {0}, {1})); failure(types, StatusCode::DTypeMismatch, "dtype");
    Graph device; device.add_input(0, "x", Tensor::allocate_cpu({2})); device.add_tensor(1, {2}, DType::FP32, Device(DeviceType::CUDA));
    device.add_node(7, OpDesc(OpCode::MATERIALIZE, {0}, {1})); failure(device, StatusCode::DeviceMismatch, "device");
    Graph invalid_math; invalid_math.add_input(0, "x", Tensor::allocate_cpu({2}, DType::INT32)); invalid_math.add_tensor(1, {2});
    invalid_math.add_node(7, OpDesc(OpCode::ADD, {0, 0}, {1})); failure(invalid_math, StatusCode::DTypeMismatch, "node 7 (ADD)");
    Graph moved; moved.add_input(0, "x", Tensor::allocate_cpu({})); moved.add_tensor(1, {});
    OpDesc desc(OpCode::MATERIALIZE, {0}, {1}); const auto retained = std::move(desc); (void)retained;
    rejects([&] { moved.add_node(1, std::move(desc)); });
}
Graph write_graph(bool ordered, bool state = true, bool stale = false) {
    Graph graph;
    graph.add_input(0, "source", Tensor::allocate_cpu({4}));
    graph.add_input(1, "state", Tensor::allocate_cpu({4}), state);
    graph.add_tensor(2, {4}); graph.add_tensor(3, {4});
    graph.add_node(10, OpDesc(OpCode::COPY, {0, 1}, {2}, CopyAttrs{}));
    graph.add_node(20, OpDesc(OpCode::COPY, {0, ordered ? 2u : 1u}, {3}, CopyAttrs{}));
    graph.add_output("state", stale ? 2 : 3);
    return graph;
}
void aliases_and_state() {
    auto ordered = write_graph(true); success(ordered.freeze());
    require(ordered.tensors().at(3).base == 1 && ordered.tensors().at(3).kind == OutputKind::Write, "write retains physical root and new logical ID");
    auto unordered = write_graph(false); failure(unordered, StatusCode::Aliasing, "unordered");
    auto immutable = write_graph(true, false); failure(immutable, StatusCode::Aliasing, "persistent state");
    auto stale = write_graph(true, true, true); failure(stale, StatusCode::Aliasing, "stale state version");
    Graph views;
    views.add_input(0, "x", Tensor::allocate_cpu({2, 3}));
    views.add_tensor(1, {3, 2}); views.add_tensor(2, {2, 2}); views.add_tensor(3, {2, 2});
    views.add_node(0, OpDesc(OpCode::TRANSPOSE, {0}, {1}, TransposeAttrs{}));
    views.add_node(1, OpDesc(OpCode::SLICE, {1}, {2}, SliceAttrs{0, 1, 2, 1}));
    views.add_node(2, OpDesc(OpCode::MATERIALIZE, {2}, {3})); views.add_output("dense", 3);
    success(views.freeze());
    require(views.tensors().at(2).base == 0 && views.tensors().at(2).layout->offset_bytes == 4 && views.tensors().at(2).layout->stride == Stride{1, 3}, "alias base/byte range");
    Graph range; range.add_input(0, "x", Tensor::allocate_cpu({2})); range.add_tensor(1, {3});
    range.add_node(0, OpDesc(OpCode::VIEW, {0}, {1}, ViewAttrs{Shape{3}, Stride{1}, 0}));
    failure(range, StatusCode::OutOfRange, "node 0");
    Graph split;
    split.add_input(0, "source", Tensor::allocate_cpu({2})); split.add_input(1, "state", Tensor::allocate_cpu({4}), true);
    for (TensorId i = 2; i < 6; ++i) split.add_tensor(i, {2});
    split.add_node(0, OpDesc(OpCode::NARROW, {1}, {2}, SliceAttrs{0, 0, 2, 1}));
    split.add_node(1, OpDesc(OpCode::NARROW, {1}, {3}, SliceAttrs{0, 2, 2, 1}));
    split.add_node(2, OpDesc(OpCode::COPY, {0, 2}, {4}, CopyAttrs{}));
    split.add_node(3, OpDesc(OpCode::COPY, {0, 3}, {5}, CopyAttrs{}));
    split.add_output("left", 4); split.add_output("right", 5); success(split.freeze());
    // Independent readers of an old overlapping state must not be silently
    // ordered just because their numeric IDs happen to put them before a write.
    Graph hazard; hazard.add_input(0, "source", Tensor::allocate_cpu({4}));
    hazard.add_input(1, "state", Tensor::allocate_cpu({4}), true);
    hazard.add_tensor(2, {4}); hazard.add_tensor(3, {4});
    hazard.add_node(0, OpDesc(OpCode::ADD, {1, 1}, {2}));
    hazard.add_node(1, OpDesc(OpCode::COPY, {0, 1}, {3}, CopyAttrs{}));
    failure(hazard, StatusCode::Aliasing, "unordered");
    Graph stale_read; stale_read.add_input(0, "x", Tensor::allocate_cpu({4}));
    stale_read.add_input(1, "state", Tensor::allocate_cpu({4}), true);
    stale_read.add_tensor(2, {4}); stale_read.add_tensor(3, {4});
    stale_read.add_node(0, OpDesc(OpCode::COPY, {0, 1}, {2}, CopyAttrs{}));
    stale_read.add_node(1, OpDesc(OpCode::ADD, {2, 1}, {3}));
    failure(stale_read, StatusCode::Aliasing, "stale state version");
    Graph prior_read; prior_read.add_input(0, "state", Tensor::allocate_cpu({4}), true);
    prior_read.add_tensor(1, {4}); prior_read.add_tensor(2, {4});
    prior_read.add_node(0, OpDesc(OpCode::ADD, {0, 0}, {1}));
    prior_read.add_node(1, OpDesc(OpCode::COPY, {1, 0}, {2}, CopyAttrs{}));
    prior_read.add_output("state", 2); success(prior_read.freeze());
    auto backing = Tensor::allocate_cpu({4});
    auto wrapper = Storage::wrap(Device{}, backing.nbytes(), backing.data<float>(), [owner = backing.storage()](void*) noexcept { (void)owner; });
    Tensor other(wrapper, DType::FP32, Shape{4}, Stride{1});
    require(infer_operator(OpDesc(OpCode::COPY, {0, 1}, {2}, CopyAttrs{}), {backing, other}).status.code == StatusCode::Aliasing,
            "S2 distinct overlapping wrappers retain address check");
    Graph wrapped; wrapped.add_input(0, "owner", backing); wrapped.add_input(1, "wrapper", other);
    failure(wrapped, StatusCode::Aliasing, "one shared Storage");
    Graph shared; shared.add_input(0, "constant", backing); shared.add_input(1, "mutable", backing, true);
    failure(shared, StatusCode::Aliasing, "consistent persistent-state");
}
}
int main() {
    try { ownership(); topology(); aliases_and_state(); std::cout << "graph ownership/topology/state: PASS\n"; return 0; }
    catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
