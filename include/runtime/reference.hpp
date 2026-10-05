#pragma once

#include "runtime/shape_inference.hpp"

namespace runtime::reference {
// Single-thread CPU correctness oracle; not an optimized backend/graph interface.
// Caller owns all output allocation. Metadata/device/alias/input-finiteness errors
// are checked before writes. Arithmetic overflow may leave a partially written
// destination; a failed status never represents a valid output.
Status execute(const OpDesc& descriptor, const TensorInputs& inputs, Tensor& output);
} // namespace runtime::reference
