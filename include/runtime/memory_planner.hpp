#pragma once

#include "runtime/graph.hpp"

namespace runtime {
// Closed node-index intervals: inputs and outputs coexist at the producing
// node. -1 denotes external birth; order.size() pins named outputs to end.
struct TensorLifetime {
    TensorId id, root;
    std::int64_t birth, last_use;
    std::optional<std::size_t> first_use;
    std::size_t consumer_count;
    bool external, output, alias;
};
struct RootLifetime {
    TensorId root;
    std::int64_t birth, last_use;
    std::size_t bytes;
    Device device;
    bool external;
};
struct LifetimeAnalysis {
    std::map<TensorId, TensorLifetime> tensors;
    std::map<TensorId, RootLifetime> roots;
    std::size_t node_count = 0;
};
// Frozen graph required; metadata-only, no Storage or Tensor allocations.
LifetimeAnalysis analyze_lifetimes(const Graph& graph);
} // namespace runtime
