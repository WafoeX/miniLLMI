#pragma once

#include "runtime/graph.hpp"

namespace runtime {
struct ExecutionCounts {
    std::size_t nodes_completed = 0;
    std::size_t allocations = 0, frees = 0;
    std::size_t allocated_bytes = 0, live_bytes = 0, peak_live_bytes = 0;
    std::size_t copies = 0, copy_bytes = 0;
};
struct ExecutionResult {
    Status status;
    std::optional<NodeId> failed_node;
    std::map<std::string, Tensor> outputs; // only valid on success; pins alias bases
    ExecutionCounts counts;
    bool ok() const noexcept { return status.ok(); }
};
// Intentional dynamic baseline: one new CPU buffer per NewTensor node, inside
// execute, no implicit copies. All nodes execute, even disconnected/dead branches.
// Shared Storage extends alias lifetimes; only named outputs escape the call.
ExecutionResult execute_graph(const Graph& graph);
} // namespace runtime
