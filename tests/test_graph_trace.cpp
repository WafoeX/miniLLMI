#include "runtime/graph_executor.hpp"
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace {
using namespace runtime;
void require(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
void success(Status status) { if (!status.ok()) throw std::runtime_error(status.message); }
std::string signature(const ExecutionTrace& trace) {
    std::ostringstream stream;
    for (const auto& event : trace.events()) {
        stream << trace_name(event.kind) << ':';
        if (event.node) stream << *event.node; else stream << '-';
        stream << ':';
        if (event.tensor) stream << *event.tensor; else stream << '-';
        stream << ' ';
    }
    return stream.str();
}
Graph diamond() {
    Graph graph; graph.add_input(0, "x", Tensor::allocate_cpu({2})); graph.add_input(1, "y", Tensor::allocate_cpu({2}));
    for (TensorId i = 2; i < 6; ++i) graph.add_tensor(i, {2});
    graph.add_node(40, OpDesc(OpCode::ADD, {3, 4}, {5}));
    graph.add_node(30, OpDesc(OpCode::ADD, {2, 0}, {4}));
    graph.add_node(20, OpDesc(OpCode::MUL, {2, 1}, {3}));
    graph.add_node(10, OpDesc(OpCode::ADD, {0, 1}, {2}));
    graph.add_output("result", 5); success(graph.freeze()); return graph;
}
void golden() {
    const auto graph = diamond(); ExecutionTrace trace;
    const auto before = testing::cpu_allocation_counts();
    std::string full;
    {
        const auto result = execute_graph(graph, &trace); success(result.status);
        const std::string expected =
            "NodeBegin:10:- Tensor:10:0 Tensor:10:1 Allocate:10:2 Tensor:10:2 NodeEnd:10:- "
            "NodeBegin:20:- Tensor:20:2 Tensor:20:1 Allocate:20:3 Tensor:20:3 NodeEnd:20:- Release:20:1 "
            "NodeBegin:30:- Tensor:30:2 Tensor:30:0 Allocate:30:4 Tensor:30:4 NodeEnd:30:- Release:30:0 Release:30:2 Free:30:2 "
            "NodeBegin:40:- Tensor:40:3 Tensor:40:4 Allocate:40:5 Tensor:40:5 NodeEnd:40:- Release:40:3 Free:40:3 Release:40:4 Free:40:4 Output:-:5 ";
        full = signature(trace); require(full == expected, "diamond full trace golden order");
        std::size_t allocations = 0, frees = 0, bytes = 0;
        for (const auto& event : trace.events()) {
            if (event.tensor) {
                require(event.metadata && event.base == event.tensor && event.metadata->shape == Shape{2} &&
                        event.metadata->stride == Stride{1} && event.metadata->dtype == DType::FP32 &&
                        event.metadata->device == Device{} && event.metadata->offset_bytes == 0, "trace tensor/base/layout metadata");
            }
            if (event.kind == TraceKind::Allocate) { ++allocations; bytes += event.bytes; }
            if (event.kind == TraceKind::Free) ++frees;
        }
        require(allocations == result.counts.allocations && frees == result.counts.frees && bytes == result.counts.allocated_bytes,
                "trace and always-on counters agree");
        require(result.counts.peak_live_bytes == 24 && result.counts.live_bytes == 8, "diamond backing liveness baseline");
        std::cout << "diamond trace: " << full << '\n';
    }
    require(testing::cpu_allocation_counts().live == before.live, "trace must not retain any Storage");
    {
        const auto untraced = execute_graph(graph); success(untraced.status);
        ExecutionTrace limited(3); const auto bounded = execute_graph(graph, &limited); success(bounded.status);
        require(limited.events().size() == 3 && limited.dropped() == trace.events().size() - 3, "first-N trace is bounded with dropped count");
        require(signature(limited) == "NodeBegin:10:- Tensor:10:0 Tensor:10:1 ", "bounded prefix");
        const auto repeated = execute_graph(graph, &limited); success(repeated.status);
        require(limited.events().size() == 3 && limited.dropped() == trace.events().size() - 3, "trace reset at each execution");
        ExecutionTrace disabled(0); const auto zero = execute_graph(graph, &disabled); success(zero.status);
        require(disabled.events().empty() && disabled.dropped() == trace.events().size(), "zero-limit trace stores no events");
        require(untraced.counts.allocations == bounded.counts.allocations && bounded.counts.peak_live_bytes == zero.counts.peak_live_bytes,
                "tracing never changes backing allocation policy");
    }
}
void alias_copy_failure() {
    Graph graph; graph.add_input(0, "x", Tensor::allocate_cpu({2, 2}));
    graph.add_tensor(1, {2, 2}); graph.add_tensor(2, {4}); graph.add_tensor(3, {4});
    graph.add_node(0, OpDesc(OpCode::ADD, {0, 0}, {1}));
    graph.add_node(1, OpDesc(OpCode::RESHAPE, {1}, {2}, ReshapeAttrs{Shape{4}}));
    graph.add_node(2, OpDesc(OpCode::MATERIALIZE, {2}, {3})); graph.add_output("dense", 3); success(graph.freeze());
    ExecutionTrace trace; const auto result = execute_graph(graph, &trace); success(result.status);
    bool alias = false, copy = false, base_free = false;
    for (const auto& event : trace.events()) {
        if (event.kind == TraceKind::Alias) { alias = true; require(event.tensor == 2 && event.base == 1, "alias trace physical root"); }
        if (event.kind == TraceKind::Copy) { copy = true; require(event.node == 2 && event.bytes == 16, "explicit materialize copy trace"); }
        if (event.kind == TraceKind::Free) { base_free = true; require(event.node == 2 && event.tensor == 1, "base frees only after alias last use"); }
    }
    require(alias && copy && base_free, "alias/copy/free events present");
    Graph state; state.add_input(0, "source", Tensor::allocate_cpu({2})); state.add_input(1, "state", Tensor::allocate_cpu({2}), true);
    state.add_tensor(2, {2}); state.add_tensor(3, {2});
    state.add_node(0, OpDesc(OpCode::COPY, {0, 1}, {2}, CopyAttrs{}));
    state.add_node(1, OpDesc(OpCode::COPY, {2, 2}, {3}, CopyAttrs{})); state.add_output("state", 3); success(state.freeze());
    const auto written = execute_graph(state, &trace); success(written.status);
    std::size_t writes = 0, copies = 0;
    for (const auto& event : trace.events()) {
        if (event.kind == TraceKind::StateWrite) ++writes;
        if (event.kind == TraceKind::Copy) ++copies;
    }
    require(writes == 2 && copies == 1 && written.counts.allocations == 0, "self-write produces version but no copy event");
    Graph bad; auto input = Tensor::allocate_cpu({2}); input.data<float>()[1] = std::numeric_limits<float>::infinity();
    bad.add_input(0, "x", input); bad.add_tensor(1, {2}); bad.add_tensor(2, {2});
    bad.add_node(0, OpDesc(OpCode::ADD, {0, 0}, {1})); bad.add_node(1, OpDesc(OpCode::ADD, {1, 1}, {2}));
    bad.add_output("result", 2); success(bad.freeze());
    const auto failed = execute_graph(bad, &trace);
    require(failed.status.code == StatusCode::NonFinite, "failure trace run");
    std::size_t starts = 0, failures = 0, frees = 0;
    for (const auto& event : trace.events()) {
        if (event.kind == TraceKind::NodeBegin) ++starts;
        if (event.kind == TraceKind::Failure) { ++failures; require(event.node == 0 && event.status == StatusCode::NonFinite, "trace classified first failure"); }
        if (event.kind == TraceKind::Free) ++frees;
        require(event.kind != TraceKind::Output && event.kind != TraceKind::NodeEnd, "no completion events after failure");
    }
    require(starts == 1 && failures == 1 && frees == 1, "trace stops then cleans up");
    testing::fail_next_cpu_allocation(); const auto oom = execute_graph(bad, &trace);
    require(oom.status.code == StatusCode::ResourceExhausted, "OOM trace run");
    for (const auto& event : trace.events()) require(event.kind != TraceKind::Allocate && event.kind != TraceKind::Free, "failed buffer allocation is not counted");
    Graph invalid; (void)execute_graph(invalid, &trace);
    require(trace.events().size() == 1 && trace.events()[0].kind == TraceKind::Failure, "invalid graph diagnostic trace");
}
}
int main() {
    try {
        golden(); alias_copy_failure();
        require(testing::cpu_allocation_counts().live == 0, "trace tests leak backing buffers");
        std::cout << "graph trace: PASS\n"; return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
