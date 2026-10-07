#pragma once

#include "runtime/shape_inference.hpp"

namespace runtime::detail {
// Scalar CPU implementations for declared S12 primitives. Bindings and output
// storage are validated/owned by the common backend/executor path.
Status execute_transformer_cpu(const OpDesc&, const TensorInputs&, Tensor&);
} // namespace runtime::detail
