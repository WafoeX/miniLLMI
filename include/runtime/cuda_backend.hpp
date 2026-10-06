#pragma once

#include "runtime/backend.hpp"
#include "runtime/allocation_provider.hpp"
#include "runtime/memory_planner.hpp"
#include <map>
#include <memory>

namespace runtime {
// CUDA is an explicit backend.  It owns one nonblocking stream; public calls
// synchronize that stream before returning, so pageable host bindings and
// Storage lifetimes are safe without promising overlap.
enum class CudaMatmul { Stage0Naive, CuBlas };
class CudaBackend final : public Backend {
public:
    explicit CudaBackend(int device_index = 0, CudaMatmul matmul = CudaMatmul::Stage0Naive);
    ~CudaBackend() override;
    CudaBackend(const CudaBackend&) = delete;
    CudaBackend& operator=(const CudaBackend&) = delete;
    const char* name() const noexcept override;
    Device device() const noexcept override;
    Status capability(OpCode, Device, DType) const override;
    BackendBuffer allocate(Shape, DType, Device) const override;
    Status copy(const Tensor&, Tensor&) const override;
    BackendPreparation prepare(const OpDesc&, const TensorInputs&, const Tensor&) const override;
    Status execute(const OpDesc&, const TensorInputs&, Tensor&, Workspace = {}) const override;
private:
    struct State;
    std::unique_ptr<State> state_;
};

// Device analogue of S5's prepared provider: one CUDA Storage reservation and
// the already-validated S5 byte-slot plan. It is explicitly CUDA-device-bound.
class CudaPlannedAllocationProvider final : public AllocationProvider {
public:
    CudaPlannedAllocationProvider(const Graph&, const CudaBackend&, PlanPolicy policy = PlanPolicy::Reuse);
    Status validate_graph(const Graph&) const override;
    Status begin() override;
    void end() noexcept override { running_ = false; }
    Tensor allocate(TensorId, Shape, DType) override;
    void release(TensorId) noexcept override;
    bool backing_per_request() const noexcept override { return false; }
    std::size_t capacity() const noexcept override { return plan_.capacity_bytes; }
private:
    struct Binding { Shape shape; DType dtype; bool active = false; };
    const CudaBackend& backend_;
    MemoryPlan plan_;
    std::shared_ptr<Storage> storage_;
    std::map<TensorId, Binding> bindings_;
    bool running_ = false;
};
} // namespace runtime
