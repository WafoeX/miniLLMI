#pragma once

#include "runtime/backend.hpp"
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
} // namespace runtime
