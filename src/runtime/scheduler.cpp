#include "runtime/scheduler.hpp"
#include <stdexcept>

namespace runtime {
Scheduler::Scheduler(const Backend& cpu, const Backend* cuda, PlacementFallback fallback)
    : cpu_(cpu), cuda_(cuda), fallback_(fallback) {
    if (cpu.device() != Device{} || (cuda && cuda->device().type() != DeviceType::CUDA))
        throw std::invalid_argument("scheduler requires CPU:0 and optionally one CUDA backend");
    if (fallback != PlacementFallback::CPU && fallback != PlacementFallback::Error)
        throw std::invalid_argument("unknown placement fallback");
}
Status Scheduler::supports(const Backend& backend, OpCode code, DType dtype, const LayoutInputs& inputs) const {
    const auto capability = backend.capability(code, backend.device(), dtype);
    if (!capability.ok()) return capability;
    for (const auto& item : inputs) {
        const auto& layout = item.get();
        try { layout.validate(); }
        catch (const std::exception& e) { return Status::failure(StatusCode::LayoutMismatch, e.what()); }
        if (backend.device().type() == DeviceType::CUDA &&
            (layout.dtype != DType::FP32 || !layout.is_contiguous()))
            return Status::failure(StatusCode::LayoutMismatch, "CUDA requires contiguous FP32 bindings; MATERIALIZE must be explicit");
    }
    return Status::success();
}
PlacementDecision Scheduler::place(OpCode code, DType dtype, const LayoutInputs& inputs, Device requested) const {
    const Backend* preferred = requested == cpu_.device() ? &cpu_
        : cuda_ && requested == cuda_->device() ? cuda_ : nullptr;
    const auto status = preferred ? supports(*preferred, code, dtype, inputs)
        : Status::failure(StatusCode::Unsupported, "requested device has no registered backend");
    if (status.ok()) return {status, requested, false, "requested placement supported"};
    if (fallback_ == PlacementFallback::CPU && requested != cpu_.device()) {
        const auto fallback = supports(cpu_, code, dtype, inputs);
        if (fallback.ok()) return {fallback, cpu_.device(), true, status.message + "; explicit CPU placement"};
        return {fallback, cpu_.device(), true, status.message + "; CPU also unsupported: " + fallback.message};
    }
    return {status, requested, false, status.message};
}
} // namespace runtime
