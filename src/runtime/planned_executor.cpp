#include "runtime/planned_executor.hpp"
#include <memory>
#include <stdexcept>

namespace runtime {
namespace {
MemoryPlan checked_plan(const Graph& graph, MemoryPlan plan) {
    const auto status = validate_memory_plan(graph, plan);
    if (!status.ok()) throw std::invalid_argument(status.message);
    for (const auto& item : graph.tensors())
        if (item.second.device.type() != DeviceType::CPU)
            throw std::invalid_argument("CPU planned provider requires CPU graph tensors; use CudaPlannedAllocationProvider");
    return plan;
}
}
PlannedAllocationProvider::PlannedAllocationProvider(const Graph& graph, PlanPolicy policy, std::size_t alignment)
    : PlannedAllocationProvider(graph, plan_memory(graph, policy, alignment)) {}
PlannedAllocationProvider::PlannedAllocationProvider(const Graph& graph, MemoryPlan plan)
    : plan_(checked_plan(graph, std::move(plan))), arena_(plan_.capacity_bytes, plan_.alignment),
      reservation_(arena_.allocate(plan_.capacity_bytes, plan_.alignment)) {
    for (const auto& item : plan_.slots) {
        const auto& record = graph.tensors().at(item.first);
        bindings_.emplace(item.first, Binding{record.shape, contiguous_stride(record.shape), record.dtype, false});
    }
}
Status PlannedAllocationProvider::validate_graph(const Graph& graph) const {
    return validate_memory_plan(graph, plan_);
}
Status PlannedAllocationProvider::begin() {
    if (running_ || arena_.storage().use_count() != 1)
        return Status::failure(StatusCode::InvalidArgument, "planned context busy: release all returned Tensor/Storage aliases or prepare a separate context");
    for (auto& item : bindings_) item.second.active = false;
    running_ = true;
    return Status::success();
}
Tensor PlannedAllocationProvider::allocate(TensorId root, Shape shape, DType dtype) {
    const auto found = bindings_.find(root);
    if (!running_ || found == bindings_.end() || found->second.active || found->second.shape != shape || found->second.dtype != dtype)
        throw std::invalid_argument("invalid planned allocation binding; explicitly prepare again");
    const auto& slot = plan_.slots.at(root);
    if (slot.bytes) {
        auto* pointer = static_cast<unsigned char*>(arena_.storage()->data()) + slot.offset;
        // C++17 typed lifetimes, including float/integer reuse of the same bytes.
        if (dtype == DType::FP32) std::uninitialized_value_construct_n(reinterpret_cast<float*>(pointer), numel(shape));
        else std::uninitialized_value_construct_n(reinterpret_cast<std::int32_t*>(pointer), numel(shape));
    }
    Tensor output(arena_.storage(), dtype, std::move(shape), found->second.stride, slot.offset);
    found->second.active = true;
    return output;
}
void PlannedAllocationProvider::release(TensorId root) noexcept {
    const auto found = bindings_.find(root);
    if (found != bindings_.end()) found->second.active = false;
}
} // namespace runtime
