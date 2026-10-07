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

enum class PlanPolicy { Reuse, NoReuse };
struct PlannedSlot {
    TensorId root;
    std::size_t offset, bytes;
    std::int64_t birth, last_use;
    Device device;
};
struct MemoryPlan {
    std::map<TensorId, PlannedSlot> slots; // NewTensor roots only, including outputs
    std::size_t alignment = 64, capacity_bytes = 0, peak_live_bytes = 0, reuse_count = 0;
    // Offsets are local to each device's backing storage. capacity_bytes is the
    // sum, never a shared address space across CPU/CUDA.
    std::map<Device, std::size_t> device_capacity_bytes;
    PlanPolicy policy = PlanPolicy::Reuse;
    std::string graph_signature; // full structural identity, not a hash/pointer
};
MemoryPlan plan_memory(const Graph& graph, PlanPolicy policy = PlanPolicy::Reuse, std::size_t alignment = 64);
Status validate_memory_plan(const Graph& graph, const MemoryPlan& plan);
} // namespace runtime
