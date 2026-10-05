#pragma once

#include "runtime/tensor.hpp"
#include <cstddef>

namespace runtime {
// CPU leaf copy primitive for the future backend COPY contract; not dispatch.
// Shapes/dtypes must match. Supports positive strided source and destination.
// Rejects intersecting backing-address spans (even across different Storage
// wrappers), except exact self-copy which is a no-op. Never allocates buffers.
// Returns logical bytes actually copied, 0 for empty/exact self-copy.
std::size_t copy_cpu(const Tensor& source, Tensor& destination);
} // namespace runtime
