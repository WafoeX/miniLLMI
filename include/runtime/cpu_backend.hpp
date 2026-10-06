#pragma once
#include "runtime/backend.hpp"

namespace runtime {
class CpuBackend final : public Backend {
public:
    const char* name() const noexcept override { return "cpu-reference-fp64"; }
    Device device() const noexcept override { return Device{}; }
    Status capability(OpCode code, Device device, DType dtype) const override;
    BackendBuffer allocate(Shape shape, DType dtype, Device device) const override;
    Status copy(const Tensor& source, Tensor& destination) const override;
    BackendPreparation prepare(const OpDesc&, const TensorInputs&, const Tensor& output) const override;
    Status execute(const OpDesc&, const TensorInputs&, Tensor& output, Workspace workspace = {}) const override;
};
} // namespace runtime
