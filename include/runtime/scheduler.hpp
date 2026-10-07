#pragma once

#include "runtime/backend.hpp"
#include "runtime/graph.hpp"
#include "runtime/allocation_provider.hpp"
#include "runtime/memory_planner.hpp"

namespace runtime {
enum class PlacementFallback { CPU, Error };
struct PlacementDecision {
    Status status;
    Device device;
    bool fell_back = false;
    std::string reason;
    bool ok() const noexcept { return status.ok(); }
};
struct InsertedCopy {
    NodeId node;
    TensorId source, destination; // logical source ID identifies its state version
    Device from, to;
    std::size_t bytes;
};
struct ScheduledGraph {
    Status status;
    std::optional<Graph> graph;
    std::map<NodeId, PlacementDecision> placements; // original node IDs
    std::map<NodeId, NodeId> rewritten_nodes;
    std::vector<InsertedCopy> inserted_copies;
    bool ok() const noexcept { return status.ok() && graph.has_value(); }
};
// Borrowed backends; deterministic, synchronous, single CUDA device only.
// No value reads, allocations, transfers, cost model or kernel substitution.
class Scheduler {
public:
    explicit Scheduler(const Backend& cpu = default_cpu_backend(), const Backend* cuda = nullptr,
                       PlacementFallback fallback = PlacementFallback::CPU);
    PlacementDecision place(OpCode code, DType output_dtype, const LayoutInputs& inputs,
                            Device requested = Device{}) const;
    ScheduledGraph rewrite(const Graph& logical) const;
    bool has_device(Device device) const noexcept;
    const Backend& backend(Device device) const;
    BackendPreparation prepare_node(const OpDesc&, const TensorInputs&, const Tensor&) const;
    Status execute_node(const OpDesc&, const TensorInputs&, Tensor&, std::optional<Device>& executed) const;
    const Backend& cpu() const noexcept { return cpu_; }
    const Backend* cuda() const noexcept { return cuda_; }
private:
    Status supports(const Backend&, OpCode, DType, const LayoutInputs&) const;
    const Backend& cpu_;
    const Backend* cuda_;
    PlacementFallback fallback_;
};
// Replans the rewritten graph, one backing Storage per device. Uses the same
// executor/provider seam as S5/S8; output aliases block reuse until released.
class ScheduledAllocationProvider final : public AllocationProvider {
public:
    ScheduledAllocationProvider(const Graph&, const Scheduler&, PlanPolicy = PlanPolicy::Reuse);
    Status validate_graph(const Graph&) const override;
    Status begin() override;
    void end() noexcept override { running_ = false; }
    Tensor allocate(TensorId, Shape, DType) override;
    void release(TensorId) noexcept override;
    bool backing_per_request() const noexcept override { return false; }
    std::size_t capacity() const noexcept override { return plan_.capacity_bytes; }
    std::size_t capacity_for(TensorId root) const noexcept override;
    const MemoryPlan& plan() const noexcept { return plan_; }
private:
    struct Binding { Shape shape; DType dtype; Device device; bool active = false; };
    MemoryPlan plan_;
    std::map<Device, std::shared_ptr<Storage>> storage_;
    std::map<TensorId, Binding> bindings_;
    bool running_ = false;
};
} // namespace runtime
