#pragma once
#include "runtime/backend.hpp"
#include <memory>

namespace runtime {
class ThreadPool;
enum class CpuMatmul { ReferenceFP64, ScalarFP32V0, LoopIKJFP32C1, LoopIKJFifoPoolFP32C3 };
class CpuBackend final : public Backend {
public:
    explicit CpuBackend(CpuMatmul matmul = CpuMatmul::ReferenceFP64, std::size_t workers = 1);
    ~CpuBackend() override;
    const char* name() const noexcept override;
    CpuMatmul matmul() const noexcept { return matmul_; }
    std::size_t workers() const noexcept;
    Device device() const noexcept override { return Device{}; }
    Status capability(OpCode code, Device device, DType dtype) const override;
    BackendBuffer allocate(Shape shape, DType dtype, Device device) const override;
    Status copy(const Tensor& source, Tensor& destination) const override;
    BackendPreparation prepare(const OpDesc&, const TensorInputs&, const Tensor& output) const override;
    Status execute(const OpDesc&, const TensorInputs&, Tensor& output, Workspace workspace = {}) const override;
private:
    CpuMatmul matmul_;
    std::unique_ptr<ThreadPool> pool_;
};
} // namespace runtime
