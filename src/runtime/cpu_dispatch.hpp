#pragma once
#include "runtime/shape_inference.hpp"

namespace runtime {
class ThreadPool;
}
namespace runtime::detail {
using MatmulKernel = Status (*)(const Tensor&, const Tensor&, Tensor&, ThreadPool*);
// Shared S2 binding/overlap rules, also used by CPU backend prepare.
Status validate_cpu_core(const OpDesc&, const TensorInputs&, const Tensor&, OutputKind&);
// One source of shape, layout, aliasing, copy and finite-value semantics. Only
// reduction precision varies. Not a public runtime or model-level entry point.
Status execute_cpu_core(const OpDesc&, const TensorInputs&, Tensor&, MatmulKernel, ThreadPool* pool = nullptr);
} // namespace runtime::detail
