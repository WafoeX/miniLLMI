#pragma once

#include "runtime/shape_inference.hpp"

namespace runtime {
// Borrowed caller-owned scratch, never retained. CPU scalar/reference paths need
// zero bytes. Future backends must publish requirements through prepare.
struct Workspace { void* data = nullptr; std::size_t bytes = 0; };
struct BackendPreparation {
    Status status;
    std::size_t workspace_bytes = 0;
    bool ok() const noexcept { return status.ok(); }
};
struct BackendBuffer {
    Status status;
    std::optional<Tensor> tensor;
    bool ok() const noexcept { return status.ok() && tensor.has_value(); }
};
// Internal synchronous contract; no CUDA stream, scheduler, or model API.
// All bindings use the one S1 Tensor/Storage representation. Graph allocation
// providers still own output allocation/planning; execute never allocates it.
class Backend {
public:
    virtual ~Backend() = default;
    virtual const char* name() const noexcept = 0;
    virtual Device device() const noexcept = 0;
    // Coarse support only. prepare additionally checks actual shape/attrs/layout.
    virtual Status capability(OpCode code, Device device, DType dtype) const = 0;
    virtual BackendBuffer allocate(Shape shape, DType dtype, Device device) const = 0;
    virtual Status copy(const Tensor& source, Tensor& destination) const = 0;
    // Metadata-only, no buffer allocation or value reads. No cached bindings;
    // execute revalidates, including after mutations to inputs between calls.
    virtual BackendPreparation prepare(const OpDesc&, const TensorInputs&, const Tensor& output) const = 0;
    virtual Status execute(const OpDesc&, const TensorInputs&, Tensor& output, Workspace workspace = {}) const = 0;
};
// Stable S2 FP64-accumulating math, now behind the common contract. Explicit
// CpuBackend selection opts into the immutable S6 FP32 scalar baseline.
const Backend& default_cpu_backend();
} // namespace runtime
