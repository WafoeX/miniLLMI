#pragma once

#include "runtime/backend.hpp"
#include "runtime/graph.hpp"

namespace runtime {
enum class PlacementFallback { CPU, Error };
struct PlacementDecision {
    Status status;
    Device device;
    bool fell_back = false;
    std::string reason;
    bool ok() const noexcept { return status.ok(); }
};
// Borrowed backends; deterministic, synchronous, single CUDA device only.
// No value reads, allocations, transfers, cost model or kernel substitution.
class Scheduler {
public:
    explicit Scheduler(const Backend& cpu = default_cpu_backend(), const Backend* cuda = nullptr,
                       PlacementFallback fallback = PlacementFallback::CPU);
    PlacementDecision place(OpCode code, DType output_dtype, const LayoutInputs& inputs,
                            Device requested = Device{}) const;
    const Backend& cpu() const noexcept { return cpu_; }
    const Backend* cuda() const noexcept { return cuda_; }
private:
    Status supports(const Backend&, OpCode, DType, const LayoutInputs&) const;
    const Backend& cpu_;
    const Backend* cuda_;
    PlacementFallback fallback_;
};
} // namespace runtime
