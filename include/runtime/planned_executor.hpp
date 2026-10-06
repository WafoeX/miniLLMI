#pragma once

#include "runtime/graph_executor.hpp"
#include "runtime/memory_planner.hpp"

namespace runtime {
// Explicit prepare/replan. One owner, one aligned Arena backing, fixed offsets.
// No online allocation, resizing, hidden copies or fallback. Not thread-safe.
class PlannedAllocationProvider final : public AllocationProvider {
public:
    explicit PlannedAllocationProvider(const Graph& graph, PlanPolicy policy = PlanPolicy::Reuse, std::size_t alignment = 64);
    PlannedAllocationProvider(const Graph& graph, MemoryPlan plan);
    Status validate_graph(const Graph& graph) const override;
    Status begin() override;
    void end() noexcept override { running_ = false; }
    Tensor allocate(TensorId root, Shape shape, DType dtype) override;
    void release(TensorId root) noexcept override;
    bool backing_per_request() const noexcept override { return false; }
    std::size_t capacity() const noexcept override { return plan_.capacity_bytes; }
    const MemoryPlan& plan() const noexcept { return plan_; }
private:
    struct Binding { Shape shape; Stride stride; DType dtype; bool active = false; };
    MemoryPlan plan_;
    std::map<TensorId, Binding> bindings_;
    Arena arena_;
    Block reservation_; // fixed reservation; lifetime remains owned by shared Storage
    bool running_ = false;
};
inline ExecutionResult execute_planned(const Graph& graph, PlannedAllocationProvider& prepared, ExecutionTrace* trace = nullptr) {
    return execute_graph(graph, trace, &prepared);
}
} // namespace runtime
